#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <cinttypes>
#include <filesystem>

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
    while(chobits::running) {
        auto [ success, audio, video ] = chobits::media::get_data();
        if(!success) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        std::vector<torch::jit::IValue> input;
        input.push_back(audio.to(model_state.device));
        input.push_back(video.to(model_state.device));
        auto pred = model_state.model.forward(input).toTensor().to(torch::kCPU);
        chobits::media::set_data(pred, video);
    }
}