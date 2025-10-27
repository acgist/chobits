#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <fstream>
#include <cinttypes>
#include <filesystem>

#include "torch/torch.h"

/**
 * 输入：
 *   - 音频（耳朵麦克风）
 *   - 视频（眼睛摄像头）
 * 
 * 输出：
 *   - 音频（嘴巴扬声器）
 *   - 动作（肌肉传感器）
 * 
 * EMA
 */
class ChobitsImpl : public torch::nn::Module {

friend chobits::model::Trainer;

private:
    chobits::nn::AudioHeadBlock audio_head{ nullptr };
    chobits::nn::VideoHeadBlock video_head{ nullptr };
    chobits::nn::MediaMixBlock  media_mix { nullptr };
    chobits::nn::AudioTailBlock audio_tail{ nullptr };

public:
    ChobitsImpl() {
    }
    ~ChobitsImpl() {
        this->unregister_module("audio_head");
        this->unregister_module("video_head");
        this->unregister_module("media_mix");
        this->unregister_module("audio_tail");
    }

public:
    void define() {
        this->audio_head = this->register_module("audio_head", chobits::nn::AudioHeadBlock(           3, 1, std::vector<int>{ 1, 8, 64, 512 }, std::vector<int>{ 4, 4, 3          }));
        this->video_head = this->register_module("video_head", chobits::nn::VideoHeadBlock(400, 1000, 3, 1, std::vector<int>{ 3, 8, 64, 512 }, std::vector<int>{ 3, 4, 3, 4, 2, 2 }));
        this->media_mix  = this->register_module("media_mix",  chobits::nn::MediaMixBlock(1000, 512, 3, 1, 2));
        this->audio_tail = this->register_module("audio_tail", chobits::nn::AudioTailBlock(1000, 3, 1, std::vector<int>{ 512, 64, 8, 1 }, std::vector<double>{ 3, 4, 4 }));
    }
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_head = this->audio_head->forward(audio);
        auto video_head = this->video_head->forward(video);
        auto media_mix  = this->media_mix->forward(audio_head, video_head);
        auto audio_out  = this->audio_tail->forward(media_mix);
        return audio_out;
    }

};

TORCH_MODULE(Chobits);

using AdamWOptimizer = std::shared_ptr<torch::optim::AdamW>;

struct TrainerState {
    float learning_rate            = 0.0001;
    float clip_grad_norm           = 10.0;
    Chobits              model     = nullptr;
    AdamWOptimizer       optimizer = nullptr;
    torch::DeviceType    device    = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
};

static TrainerState trainer_state{};

bool chobits::model::Trainer::save(const std::string& path) {
    if(!trainer_state.model) {
        return false;
    }
    trainer_state.model->eval();
    trainer_state.model->to(torch::DeviceType::CPU);
    std::printf("保存模型：%s\n", path.c_str());
    const std::string save_path = "chobits.ckpt";
    torch::save(trainer_state.model, save_path);
    std::filesystem::rename(save_path, path);
    trainer_state.model->to(trainer_state.device);
    return true;
}

bool chobits::model::Trainer::load(const std::string& path) {
    trainer_state.model = Chobits();
    trainer_state.model->define();
    if(std::filesystem::exists(path)) {
        try {
            std::printf("加载模型：%s\n", path.c_str());
            torch::load(trainer_state.model, path, torch::DeviceType::CPU);
        } catch(const std::exception& e) {
            std::printf("加载模型失败：%s\n", e.what());
        }
    }
    trainer_state.model->to(trainer_state.device);
    trainer_state.model->eval();
    this->info();
    return true;
}

void chobits::model::Trainer::train() {
    try {
        trainer_state.optimizer = std::make_shared<torch::optim::AdamW>(trainer_state.model->parameters(), trainer_state.learning_rate);
        auto scheduler = torch::optim::StepLR(*trainer_state.optimizer, 10, 0.9999);
        for (size_t epoch = 1; epoch <= 100'000'000LL && chobits::running; ++epoch) {
            this->train(epoch);
            scheduler.step();
            if(epoch % 100 == 0) {
                this->save("chobits." + std::to_string(epoch / 100 % 10) + ".ckpt");
            }
        }
    } catch(const std::exception& e) {
        std::printf("训练异常：%s\n", e.what());
    } catch(...) {
        std::printf("训练异常\n");
    }
}

void chobits::model::Trainer::train(const size_t epoch) {
    static const int epoch_count = 10;
    double loss_val = 0.0;
    trainer_state.model->train();
    const auto a = std::chrono::system_clock::now();
    trainer_state.optimizer->zero_grad();
    for (int i = 0; i < epoch_count && chobits::running; ++i) {
        auto [success, audio, video, label] = chobits::media::get_data();
        if(!success) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        audio = audio.to(trainer_state.device);
        video = video.to(trainer_state.device);
        label = label.to(trainer_state.device);
        auto pred = trainer_state.model->forward(audio, video);
        torch::Tensor loss = torch::mse_loss(pred, label);
        loss.backward();
        loss_val += loss.template item<float>();
        if(chobits::play_audio) {
            torch::NoGradGuard no_grad_guard;
            chobits::media::set_data(pred.squeeze().cpu());
        }
    }
    torch::nn::utils::clip_grad_norm_(trainer_state.model->parameters(), trainer_state.clip_grad_norm);
    trainer_state.optimizer->step();
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    std::printf("训练轮次：%" PRIu64 " 预测损失：%f 耗时：%" PRIu64 "\n", epoch, loss_val / epoch_count, duration);
}

void chobits::model::Trainer::eval(const bool save_file) {
    trainer_state.model->eval();
    torch::NoGradGuard no_grad_guard;
    std::ofstream stream;
    if(save_file) {
        stream.open("chobits.pcm", std::ios::binary);
        if(stream.fail()) {
            std::printf("无法创建音频文件\n");
            return;
        }
    }
    while(chobits::running) {
        auto [success, audio, video, label] = chobits::media::get_data(false);
        if(!success) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        audio = audio.to(trainer_state.device);
        video = video.to(trainer_state.device);
        auto pred = trainer_state.model->forward(audio, video);
        auto data = chobits::media::set_data(pred.squeeze().cpu());
        if(save_file) {
            stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(short));
            std::printf("写入音频数据：%" PRIu64 "\n", data.size());
        }
    }
    if(save_file) {
        stream.close();
    }
}

void chobits::model::Trainer::info() {
    int64_t total_numel = 0;
    for(const auto& buffer : trainer_state.model->named_buffers()) {
        int64_t numel = buffer.value().numel();
        total_numel += numel;
        std::printf("缓存参数数量：%s = %" PRIu64 "\n", buffer.key().c_str(), numel);
    }
    std::printf("缓存参数总量：%" PRIu64 "\n", total_numel);
    total_numel = 0;
    for(const auto& parameter : trainer_state.model->named_parameters()) {
        int64_t numel = parameter.value().numel();
        total_numel += numel;
        std::printf("模型参数数量：%s = %" PRIu64 "\n", parameter.key().c_str(), numel);
    }
    std::printf("模型参数总量：%" PRIu64 "\n", total_numel);
}

bool chobits::model::open_model(int argc, char const *argv[]) {
    chobits::model::Trainer trainer;
    trainer.load();
    if(argc >= 2 && std::strcmp("eval", argv[1]) == 0) {
        trainer.eval();
    } else {
        trainer.train();
        trainer.save();
    }
    return true;
}

void chobits::model::stop_all() {
    std::printf("关闭模型\n");
}
