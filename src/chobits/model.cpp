#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"

#include <memory>
#include <string>
#include <thread>
#include <filesystem>

#include "torch/torch.h"

static const size_t epoch_count = 0;

/**
 * 输入：
 *   - 音频（耳朵）
 *   - 视频（眼睛）
 *   - 其他（感受）
 * 
 * 输出：
 *   - 音频（嘴巴）
 *   - 动作（肌肉）
 * 
 * 预测：
 *   - 音频：预测所有音频
 *   - 动作：控制运动、控制镜头（输出两个XY角度分量）
 *     * 视频训练：暂无
 *     * 现实训练：暂无
 *   - 预测一秒后的数据（慢慢的傻傻的也挺可爱）
 */
class Chobits : public torch::nn::Module {

private:
    torch::Tensor                       state;
    chobits::nn::AudioHeadBlock         audio_head       { nullptr };
    chobits::nn::VideoHeadBlock         video_head       { nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_1{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_1{ nullptr };
    chobits::nn::MediaMixBlock          audio_media_mix_1{ nullptr };
    chobits::nn::MediaMixBlock          video_media_mix_1{ nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_2{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_2{ nullptr };
    chobits::nn::MediaMixBlock          audio_media_mix_2{ nullptr };
    chobits::nn::MediaMixBlock          video_media_mix_2{ nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_3{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_3{ nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_4{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_4{ nullptr };
    chobits::nn::MediaMixBlock          media_mix        { nullptr };
    chobits::nn::MediaMixBlock          media_mem        { nullptr };
    chobits::nn::AudioTailBlock         audio_tail       { nullptr };

public:
    Chobits() {
    }
    ~Chobits() {
        this->unregister_module("audio_head");
        this->unregister_module("video_head");
    }

public:
    void define() {
        this->state             = this->register_buffer("state",             torch::zeros({ 1, 256, 24, 74 }));
        this->audio_head        = this->register_module("audio_head",        chobits::nn::AudioHeadBlock(2));
        this->video_head        = this->register_module("video_head",        chobits::nn::VideoHeadBlock(3));
        this->audio_resi_attn_1 = this->register_module("audio_resi_attn_1", chobits::nn::ResidualAttentionBlock(64, 64, 24 * 74));
        this->video_resi_attn_1 = this->register_module("video_resi_attn_1", chobits::nn::ResidualAttentionBlock(64, 64, 52 * 30));
        this->audio_media_mix_1 = this->register_module("audio_media_mix_1", chobits::nn::MediaMixBlock(64, 128, 24, 74, 52, 30, 24, 74));
        this->video_media_mix_1 = this->register_module("video_media_mix_1", chobits::nn::MediaMixBlock(64, 128, 24, 74, 52, 30, 52, 30));
        this->audio_resi_attn_2 = this->register_module("audio_resi_attn_2", chobits::nn::ResidualAttentionBlock(128, 128, 24 * 74));
        this->video_resi_attn_2 = this->register_module("video_resi_attn_2", chobits::nn::ResidualAttentionBlock(128, 128, 52 * 30));
        this->audio_media_mix_2 = this->register_module("audio_media_mix_2", chobits::nn::MediaMixBlock(128, 256, 24, 74, 52, 30, 24, 74));
        this->video_media_mix_2 = this->register_module("video_media_mix_2", chobits::nn::MediaMixBlock(128, 256, 24, 74, 52, 30, 52, 30));
        this->audio_resi_attn_3 = this->register_module("audio_resi_attn_3", chobits::nn::ResidualAttentionBlock(256, 256, 24 * 74));
        this->video_resi_attn_3 = this->register_module("video_resi_attn_3", chobits::nn::ResidualAttentionBlock(256, 256, 52 * 30));
        this->audio_resi_attn_4 = this->register_module("audio_resi_attn_4", chobits::nn::ResidualAttentionBlock(64, 256, 24 * 74));
        this->video_resi_attn_4 = this->register_module("video_resi_attn_4", chobits::nn::ResidualAttentionBlock(64, 256, 52 * 30));
        this->media_mix         = this->register_module("media_mix",         chobits::nn::MediaMixBlock(256, 256, 24, 74, 52, 30, 24, 74));
        this->media_mem         = this->register_module("media_mem",         chobits::nn::MediaMixBlock(256, 256, 24, 74, 52, 30, 24, 74));
        this->audio_tail        = this->register_module("audio_tail",        chobits::nn::AudioTailBlock(256, 256, 24 * 74));
    }
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_head = this->audio_head->forward(audio);
        auto video_head = this->video_head->forward(video);
             video_head = torch::nn::functional::pad(video_head, torch::nn::functional::PadFuncOptions({ 0, 1, 0, 0 }));
        auto audio_out  = this->audio_resi_attn_1->forward(audio_head);
        auto video_out  = this->video_resi_attn_1->forward(video_head);
        auto audio_mix  = this->audio_media_mix_1->forward(audio_out, video_out);
        auto video_mix  = this->video_media_mix_1->forward(audio_out, video_out);
             audio_out  = this->audio_resi_attn_2->forward(audio_mix);
             video_out  = this->video_resi_attn_2->forward(video_mix);
             audio_mix  = this->audio_media_mix_2->forward(audio_out, video_out);
             video_mix  = this->video_media_mix_2->forward(audio_out, video_out);
             audio_out  = this->audio_resi_attn_3->forward(audio_mix);
             video_out  = this->video_resi_attn_3->forward(video_mix);
        auto media_mix  = this->media_mix->forward(audio_out, video_out);
        auto audio_mem  = this->audio_resi_attn_4->forward(audio_head);
        auto video_mem  = this->video_resi_attn_4->forward(video_head);
        auto media_mem  = this->media_mem->forward(audio_mem, video_mem);
        auto media_out  = media_mix + media_mem + torch::sigmoid(this->state);
        return this->audio_tail->forward(media_out);
    }

};

struct TrainerState {
    bool running = false;
    std::shared_ptr<Chobits>             model     = nullptr;
    std::shared_ptr<torch::optim::AdamW> optimizer = nullptr;
    torch::DeviceType                    device    = torch::DeviceType::CPU;
};

static TrainerState trainer{};

bool chobits::model::Trainer::save(const std::string& path) {
    if(!trainer.model) {
        return false;
    }
    trainer.model->eval();
    trainer.model->to(torch::DeviceType::CPU);
    std::printf("保存模型：%s\n", path.c_str());
    torch::save(trainer.model, path);
    trainer.model->to(trainer.device);
    return true;
}

bool chobits::model::Trainer::load(const std::string& path, bool load_file) {
    if(load_file && std::filesystem::exists(path)) {
        try {
            std::printf("加载模型：%s\n", path.c_str());
            torch::load(trainer.model, path, torch::DeviceType::CPU);
        } catch(const std::exception& e) {
            std::printf("加载模型失败：%s", e.what());
            return false;
        }
    } else {
        trainer.model = std::make_shared<Chobits>();
        trainer.model->define();
    }
    trainer.model->to(trainer.device);
    trainer.model->eval();
    this->info();
    return true;
}

void chobits::model::Trainer::train() {
    trainer.running = true;
    try {
        trainer.optimizer = std::make_shared<torch::optim::AdamW>(trainer.model->parameters(), 0.0001);
        auto scheduler = torch::optim::StepLR(*trainer.optimizer, 10, 0.9999);
        for (size_t epoch = 1; epoch <= 100'000'000LL && trainer.running; ++epoch) {
            this->train(epoch);
            scheduler.step();
            if(epoch % 100 == 0) {
                this->save("chobits." + std::to_string(epoch % 10) + ".ckpt");
            }
        }
    } catch(const std::exception& e) {
        std::printf("训练异常：%s\n", e.what());
    } catch(...) {
        std::printf("训练异常");
    }
}

void chobits::model::Trainer::train(const size_t epoch) {
    double loss_val = 0.0;
    trainer.model->train();
    const auto a = std::chrono::system_clock::now();
    trainer.optimizer->zero_grad();
    for (int i = 0; i < epoch_count && trainer.running; ++i) {
        auto [audio, video, label] = chobits::media::get_data();
        if(audio.numel() == 0 || video.numel() == 0 || label.numel() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        auto pred = trainer.model->forward(audio, video);
        torch::Tensor loss = torch::mse_loss(pred, label);
        loss.backward();
        loss_val += loss.template item<float>();
    }
    trainer.optimizer->step();
    torch::nn::utils::clip_grad_norm_(trainer.model->parameters(), 1.0);
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    std::printf("训练轮次：%ld 损失：%f 耗时：%ld\n", epoch, loss_val / epoch_count, duration);
}

void chobits::model::Trainer::eval() {
    trainer.model->eval();
    torch::NoGradGuard no_grad_guard;
    while(trainer.running) {
        auto [audio, video, label] = chobits::media::get_data();
        if(audio.numel() == 0 || video.numel() == 0 || label.numel() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        auto pred = trainer.model->forward(audio, video);
        torch::Tensor loss = torch::mse_loss(pred, label);
        std::printf("验证损失：%f\n", loss.template item<float>());
    }
}

void chobits::model::Trainer::test() {
    trainer.model->eval();
    torch::NoGradGuard no_grad_guard;
    auto audio = trainer.model->forward(
        torch::randn({ 1, 2, 201, 601 }),
        torch::randn({ 1, 3, 640, 360 })
    );
    std::cout << audio.sizes() << std::endl;
}

void chobits::model::Trainer::info() {
    size_t total_numel = 0;
    for(const auto& buffer : trainer.model->named_buffers()) {
        size_t numel = buffer.value().numel();
        total_numel += numel;
        std::printf("模型参数数量: %s = %ld\n", buffer.key().c_str(), numel);
    }
    for(const auto& parameter : trainer.model->named_parameters()) {
        size_t numel = parameter.value().numel();
        total_numel += numel;
        std::printf("模型参数数量: %s = %ld\n", parameter.key().c_str(), numel);
    }
    std::printf("模型参数总量: %ld\n", total_numel);
}

void chobits::model::stop_all() {
    if(!trainer.running) {
        return;
    }
    std::printf("关闭模型\n");
    trainer.running = false;
}
