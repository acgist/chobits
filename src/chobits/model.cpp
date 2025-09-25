#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>
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
 *   - 预测五百毫秒后的数据（慢慢的傻傻的也挺可爱）
 * 
 * 记忆：短时记忆、长时记忆、识别模式
 */
class ChobitsImpl : public torch::nn::Module {

friend chobits::model::Trainer;

private:
    torch::Tensor                       memory_l;
    torch::Tensor                       memory_s;
    chobits::nn::AudioHeadBlock         audio_head         { nullptr };
    chobits::nn::VideoHeadBlock         video_head         { nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_l_1{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_l_1{ nullptr };
    chobits::nn::MediaMixBlock          audio_media_mix_l_1{ nullptr };
    chobits::nn::MediaMixBlock          video_media_mix_l_1{ nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_l_2{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_l_2{ nullptr };
    chobits::nn::MediaMixBlock          audio_media_mix_l_2{ nullptr };
    chobits::nn::MediaMixBlock          video_media_mix_l_2{ nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_l_3{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_l_3{ nullptr };
    chobits::nn::MediaMixBlock          audio_video_mix_l_3{ nullptr };
    chobits::nn::ResidualAttentionBlock media_resi_attn_l_4{ nullptr };
    chobits::nn::ResidualAttentionBlock audio_resi_attn_s_1{ nullptr };
    chobits::nn::ResidualAttentionBlock video_resi_attn_s_1{ nullptr };
    chobits::nn::MediaMixBlock          audio_video_mem_s_1{ nullptr };
    chobits::nn::ResidualAttentionBlock media_resi_attn_s_2{ nullptr };
    chobits::nn::AudioTailBlock         audio_tail         { nullptr };

public:
    ChobitsImpl() {
    }
    ~ChobitsImpl() {
        this->unregister_module("audio_head");
        this->unregister_module("video_head");
        this->unregister_module("audio_resi_attn_l_1");
        this->unregister_module("video_resi_attn_l_1");
        this->unregister_module("audio_media_mix_l_1");
        this->unregister_module("video_media_mix_l_1");
        this->unregister_module("audio_resi_attn_l_2");
        this->unregister_module("video_resi_attn_l_2");
        this->unregister_module("audio_media_mix_l_2");
        this->unregister_module("video_media_mix_l_2");
        this->unregister_module("audio_resi_attn_l_3");
        this->unregister_module("video_resi_attn_l_3");
        this->unregister_module("audio_video_mix_l_3");
        this->unregister_module("media_resi_attn_l_4");
        this->unregister_module("audio_resi_attn_s_1");
        this->unregister_module("video_resi_attn_s_1");
        this->unregister_module("audio_video_mem_s_1");
        this->unregister_module("media_resi_attn_s_2");
        this->unregister_module("audio_tail");
    }

public:
    void define() {
        this->memory_l            = torch::zeros({ chobits::batch_size, 128, 24, 74 });
        this->memory_s            = torch::zeros({ chobits::batch_size, 128, 24, 74 });
        this->audio_head          = this->register_module("audio_head",          chobits::nn::AudioHeadBlock(2));
        this->video_head          = this->register_module("video_head",          chobits::nn::VideoHeadBlock(3));
        this->audio_resi_attn_l_1 = this->register_module("audio_resi_attn_l_1", chobits::nn::ResidualAttentionBlock(64, 64, 24 * 74));
        this->video_resi_attn_l_1 = this->register_module("video_resi_attn_l_1", chobits::nn::ResidualAttentionBlock(64, 64, 30 * 52));
        this->audio_media_mix_l_1 = this->register_module("audio_media_mix_l_1", chobits::nn::MediaMixBlock(64, 128, 24, 74, 30, 52, 24, 74));
        this->video_media_mix_l_1 = this->register_module("video_media_mix_l_1", chobits::nn::MediaMixBlock(64, 128, 24, 74, 30, 52, 30, 52));
        this->audio_resi_attn_l_2 = this->register_module("audio_resi_attn_l_2", chobits::nn::ResidualAttentionBlock(128, 128, 24 * 74));
        this->video_resi_attn_l_2 = this->register_module("video_resi_attn_l_2", chobits::nn::ResidualAttentionBlock(128, 128, 30 * 52));
        this->audio_media_mix_l_2 = this->register_module("audio_media_mix_l_2", chobits::nn::MediaMixBlock(128, 256, 24, 74, 30, 52, 24, 74));
        this->video_media_mix_l_2 = this->register_module("video_media_mix_l_2", chobits::nn::MediaMixBlock(128, 256, 24, 74, 30, 52, 30, 52));
        this->audio_resi_attn_l_3 = this->register_module("audio_resi_attn_l_3", chobits::nn::ResidualAttentionBlock(256, 256, 24 * 74));
        this->video_resi_attn_l_3 = this->register_module("video_resi_attn_l_3", chobits::nn::ResidualAttentionBlock(256, 256, 30 * 52));
        this->audio_video_mix_l_3 = this->register_module("audio_video_mix_l_3", chobits::nn::MediaMixBlock(256, 128, 24, 74, 30, 52, 24, 74));
        this->media_resi_attn_l_4 = this->register_module("media_resi_attn_l_4", chobits::nn::ResidualAttentionBlock(128, 128, 24 * 74));
        this->audio_resi_attn_s_1 = this->register_module("audio_resi_attn_s_1", chobits::nn::ResidualAttentionBlock(64, 128, 24 * 74));
        this->video_resi_attn_s_1 = this->register_module("video_resi_attn_s_1", chobits::nn::ResidualAttentionBlock(64, 128, 30 * 52));
        this->audio_video_mem_s_1 = this->register_module("audio_video_mem_s_1", chobits::nn::MediaMixBlock(128, 128, 24, 74, 30, 52, 24, 74));
        this->media_resi_attn_s_2 = this->register_module("media_resi_attn_s_2", chobits::nn::ResidualAttentionBlock(128, 128, 24 * 74));
        this->audio_tail          = this->register_module("audio_tail",          chobits::nn::AudioTailBlock(128, 128, 24 * 74));
    }
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_head = this->audio_head->forward(audio);
        auto video_head = this->video_head->forward(video);
        auto audio_out  = this->audio_resi_attn_l_1->forward(audio_head);
        auto video_out  = this->video_resi_attn_l_1->forward(video_head);
        auto audio_mix  = this->audio_media_mix_l_1->forward(audio_out, video_out);
        auto video_mix  = this->video_media_mix_l_1->forward(audio_out, video_out);
             audio_out  = this->audio_resi_attn_l_2->forward(audio_mix);
             video_out  = this->video_resi_attn_l_2->forward(video_mix);
             audio_mix  = this->audio_media_mix_l_2->forward(audio_out, video_out);
             video_mix  = this->video_media_mix_l_2->forward(audio_out, video_out);
             audio_out  = this->audio_resi_attn_l_3->forward(audio_mix);
             video_out  = this->video_resi_attn_l_3->forward(video_mix);
        auto media_mix  = this->audio_video_mix_l_3->forward(audio_out, video_out);
             media_mix  = this->media_resi_attn_s_2->forward(media_mix + torch::sigmoid(this->memory_l));
        auto audio_mem  = this->audio_resi_attn_s_1->forward(audio_head);
        auto video_mem  = this->video_resi_attn_s_1->forward(video_head);
        auto media_mem  = this->audio_video_mem_s_1->forward(audio_mem, video_mem);
             media_mem  = this->media_resi_attn_s_2->forward(media_mem + torch::sigmoid(this->memory_s));
        this->memory_l  = media_mix.detach();
        this->memory_s  = media_mem.detach();
        auto media_out  = media_mix + media_mem;
        return this->audio_tail->forward(media_out);
    }

};

TORCH_MODULE(Chobits);

using AdamWOptimizer = std::shared_ptr<torch::optim::AdamW>;

struct TrainerState {
    float learning_rate            = 0.0001;
    float clip_grad_norm           = 1.0;
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
    trainer_state.model->memory_l = trainer_state.model->memory_l.to(trainer_state.device);
    trainer_state.model->memory_s = trainer_state.model->memory_s.to(trainer_state.device);
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
        auto [audio, video, label] = chobits::media::get_data();
        if(audio.numel() == 0 || video.numel() == 0 || label.numel() == 0) {
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
        if(chobits::batch_size == 1) {
            torch::NoGradGuard no_grad_guard;
            auto pcm = chobits::media::pcm_istft(pred.squeeze(0).cpu());
            chobits::player::play_audio(pcm.data(), pcm.size() * 2);
        }
    }
    trainer_state.optimizer->step();
    torch::nn::utils::clip_grad_norm_(trainer_state.model->parameters(), trainer_state.clip_grad_norm);
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    std::printf("训练轮次：%" PRIu64 " 损失：%f 耗时：%" PRIu64 "\n", epoch, loss_val / epoch_count, duration);
}

void chobits::model::Trainer::eval() {
    trainer_state.model->eval();
    torch::NoGradGuard no_grad_guard;
    while(chobits::running) {
        auto [audio, video, label] = chobits::media::get_data(false);
        if(audio.numel() == 0 || video.numel() == 0 || label.numel() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        audio = audio.to(trainer_state.device);
        video = video.to(trainer_state.device);
        auto pred = trainer_state.model->forward(audio, video);
        auto pcm  = chobits::media::pcm_istft(pred.squeeze(0).cpu());
        chobits::player::play_audio(pcm.data(), pcm.size() * 2);
    }
}

void chobits::model::Trainer::test() {
    trainer_state.model->eval();
    torch::NoGradGuard no_grad_guard;
    auto audio = trainer_state.model->forward(
        torch::randn({ 10, 2, 201, 601 }),
        torch::randn({ 10, 3, 372, 640 })
    );
    std::cout << audio.sizes() << std::endl;
}

void chobits::model::Trainer::info() {
    int64_t total_numel = 0;
    for(const auto& buffer : trainer_state.model->named_buffers()) {
        int64_t numel = buffer.value().numel();
        total_numel += numel;
        std::printf("缓存参数数量: %s = %" PRIu64 "\n", buffer.key().c_str(), numel);
    }
    std::printf("缓存参数总量: %" PRIu64 "\n", total_numel);
    total_numel = 0;
    for(const auto& parameter : trainer_state.model->named_parameters()) {
        int64_t numel = parameter.value().numel();
        total_numel += numel;
        std::printf("模型参数数量: %s = %" PRIu64 "\n", parameter.key().c_str(), numel);
    }
    std::printf("模型参数总量: %" PRIu64 "\n", total_numel);
}

bool chobits::model::open_model(int argc, char const *argv[]) {
    if(argc == 1 || (argc >= 2 && std::strcmp("eval", argv[1]) == 0)) {
        chobits::batch_size = 1;
    } else {
        chobits::batch_size = 10;
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
