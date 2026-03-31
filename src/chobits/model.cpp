#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/chobits.hpp"

#include <thread>

#include "torch/torch.h"
#include "torch/script.h"

static struct ModelState {
    torch::jit::Module model;
    torch::DeviceType  device = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
} model_state;

bool chobits::model::open_model(const std::string& model_path) {
    std::printf("打开模型：%s\n", model_path.c_str());
    try {
        model_state.model = torch::jit::load(model_path);
        model_state.model.to(model_state.device);
        model_state.model.eval();
        return true;
    } catch(const std::exception& e) {
        std::printf("打开模型失败：%s\n", e.what());
    }
    return false;
}

bool chobits::model::stop_model() {
    std::printf("关闭模型\n");
    return true;
}

void chobits::model::run_model() {
    auto audio_memory = torch::randn({ 1, 10, 1024 }, torch::kFloat32).to(model_state.device);
    auto video_memory = torch::randn({ 1, 10, 1024 }, torch::kFloat32).to(model_state.device);
    while(chobits::running) {
        auto [ success, audio, video ] = chobits::media::get_data();
        if(!success) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        std::vector<torch::jit::IValue> input;
        input.push_back(audio.unsqueeze(0).to(model_state.device));
        input.push_back(video.unsqueeze(0).to(model_state.device));
        input.push_back(audio_memory);
        input.push_back(video_memory);
        auto tuple = model_state.model.forward(input).toTuple()->elements();
        auto audio_pred = tuple[0].toTensor();
        auto video_pred = tuple[1].toTensor();
        audio_memory = tuple[2].toTensor();
        video_memory = tuple[3].toTensor();
        chobits::media::set_data(audio_pred, video_pred);
    }
}
