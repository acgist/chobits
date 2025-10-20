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
 * 预测：
 *   - 音频：预测所有音频
 *   - 动作：控制运动、控制镜头（输出两个XY角度分量）
 *     * 视频训练：暂无
 *     * 现实训练：暂无
 *   - 预测一秒后的数据（慢慢的傻傻的也挺可爱）
 * 
 * 记忆：短时记忆、长时记忆、识别模式
 * 
 * EMA
 */
class ChobitsImpl : public torch::nn::Module {

friend chobits::model::Trainer;

private:
    torch::Tensor               memory_;
    chobits::nn::MediaHeadBlock audio_head         { nullptr };
    chobits::nn::MediaHeadBlock video_head         { nullptr };
    chobits::nn::MediaMixBlock  audio_media_mix    { nullptr };
    chobits::nn::AttentionBlock audio_attention_l_1{ nullptr };
    chobits::nn::AttentionBlock audio_attention_l_2{ nullptr };
    chobits::nn::AttentionBlock audio_attention_l_3{ nullptr };
    chobits::nn::AttentionBlock audio_attention_s_1{ nullptr };
    chobits::nn::MemoryBlock    memory             { nullptr };
    chobits::nn::AudioTailBlock audio_tail         { nullptr };

public:
    ChobitsImpl() {
    }
    ~ChobitsImpl() {
        this->unregister_module("audio_head");
        this->unregister_module("video_head");
        this->unregister_module("audio_media_mix");
        this->unregister_module("audio_attention_l_1");
        this->unregister_module("audio_attention_l_2");
        this->unregister_module("audio_attention_l_3");
        this->unregister_module("audio_attention_s_1");
        this->unregister_module("memory");
        this->unregister_module("audio_tail");
    }

public:
    void define() {
        this->memory_             = torch::zeros({ 1, 128, 24, 74 });
        this->audio_head          = this->register_module("audio_head",          chobits::nn::MediaHeadBlock(std::vector<int>{ 2, 8, 32, 128 }, std::vector<int>{ 3, 3, 3 }, std::vector<int>{ 2, 2, 2 }, 24 * 74));
        this->video_head          = this->register_module("video_head",          chobits::nn::MediaHeadBlock(std::vector<int>{ 3, 8, 32, 128 }, std::vector<int>{ 3, 3, 3 }, std::vector<int>{ 3, 2, 2 }, 29 * 52));
        this->audio_media_mix     = this->register_module("audio_media_mix",     chobits::nn::MediaMixBlock(128, 128, 24, 74, 29, 52));
        this->audio_attention_l_1 = this->register_module("audio_attention_l_1", chobits::nn::AttentionBlock(128, 24 * 74));
        this->audio_attention_l_2 = this->register_module("audio_attention_l_2", chobits::nn::AttentionBlock(128, 24 * 74));
        this->audio_attention_l_3 = this->register_module("audio_attention_l_3", chobits::nn::AttentionBlock(128, 24 * 74));
        this->audio_attention_s_1 = this->register_module("audio_attention_s_1", chobits::nn::AttentionBlock(128, 24 * 74));
        this->memory              = this->register_module("memory",              chobits::nn::MemoryBlock(128, 24, 74));
        this->audio_tail          = this->register_module("audio_tail",          chobits::nn::AudioTailBlock(std::vector<int>{ 128, 32, 8, 2 }, std::vector<int>{ 4, 5, 5 }, std::vector<int>{ 2, 2, 2 }, 24 * 74));
    }
    torch::Tensor forward_(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_head  = this->audio_head->forward(audio);
        auto video_head  = this->video_head->forward(video);
        auto audio_mix   = this->audio_media_mix->forward(audio_head, video_head);
        auto audio_out   = this->audio_tail->forward(audio_mix);
        return audio_out;
    }
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_head  = this->audio_head->forward(audio);
        auto video_head  = this->video_head->forward(video);
        auto audio_mix   = this->audio_media_mix->forward(audio_head, video_head);
        auto audio_out_l = this->audio_attention_l_1->forward(audio_mix);
             audio_out_l = this->audio_attention_l_2->forward(audio_out_l);
             audio_out_l = this->audio_attention_l_3->forward(audio_out_l);
        auto audio_out_s = this->audio_attention_s_1->forward(audio_mix);
             audio_mix   = audio_out_l + audio_out_s;
        auto audio_mem   = this->memory->forward(audio_mix, this->memory_);
        auto audio_out   = this->audio_tail->forward(audio_mem);
        {
            torch::NoGradGuard no_grad_guard;
            this->memory_ = torch::sum(audio_mem.detach(), { 0 }, true) / chobits::batch_size;
        }
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
    trainer_state.model->memory_ = trainer_state.model->memory_.to(trainer_state.device);
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
    double loss_val_audio = 0.0;
    double loss_val_label = 0.0;
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
        auto pred_audio = trainer_state.model->forward_(audio, video);
        auto pred_label = trainer_state.model->forward(audio, video);
        torch::Tensor loss_audio = torch::mse_loss(pred_audio, audio);
        torch::Tensor loss_label = torch::mse_loss(pred_label, label);
        torch::Tensor loss       = loss_audio + loss_label;
        loss.backward();
        loss_val_audio += loss_audio.template item<float>();
        loss_val_label += loss_label.template item<float>();
        if(chobits::play_audio) {
            torch::NoGradGuard no_grad_guard;
            chobits::media::set_data(loss_label.squeeze(0).cpu());
        }
    }
    torch::nn::utils::clip_grad_norm_(trainer_state.model->parameters(), trainer_state.clip_grad_norm);
    trainer_state.optimizer->step();
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    std::printf("训练轮次：%" PRIu64 " 音频损失：%f 预测损失：%f 耗时：%" PRIu64 "\n", epoch, loss_val_audio / epoch_count, loss_val_label / epoch_count, duration);
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
        auto data = chobits::media::set_data(pred.squeeze(0).cpu());
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
    if(argc == 1 || (argc >= 2 && std::strcmp("eval", argv[1]) == 0)) {
        chobits::batch_size = 1;
    } else {
        chobits::batch_size = argc >= 4 ? std::atoi(argv[3]) : 10;
    }
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
