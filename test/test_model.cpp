#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <cinttypes>

static void info(std::shared_ptr<torch::nn::Module> layer) {
    int     layer_size  = 0;
    int64_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        ++layer_size;
        std::printf("参数数量：%32s = %" PRId64 "\n", parameter.key().c_str(), parameter.value().numel());
        total_numel += parameter.value().numel();
    }
    std::printf("层数总量：%d\n", layer_size);
    std::printf("参数总量：%" PRId64 "\n", total_numel);
}

[[maybe_unused]] static void test_res_net_1d() {
    torch::NoGradGuard no_grad_guard;
    // chobits::nn::ResNet1dBlock layer(10, 64, 800);
    chobits::nn::ResNet1dBlock layer(10, 64, 800, 2);
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 10, 800 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_res_net_2d() {
    torch::NoGradGuard no_grad_guard;
    // chobits::nn::ResNet2dBlock layer(10, 64, std::vector<int64_t>{ 360, 640 });
    chobits::nn::ResNet2dBlock layer(10, 64, std::vector<int64_t>{ 360, 640 }, std::vector<int64_t>{ 2, 2 });
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 10, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_gru() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::GRUBlock layer(768, 768, 10);
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 10, 768 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_attention() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AttentionBlock layer(768, 336, 336, 768);
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 256, 768 }),
        torch::randn({ 10, 256, 336 }),
        torch::randn({ 10, 256, 336 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_audio_head() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AudioHeadBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 10, 800 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_video_head() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::VideoHeadBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 10, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_image_head() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::ImageHeadBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 3, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_media_mixer() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MediaMixerBlock layer;
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 256, 256 }),
        torch::randn({ 10, 256, 768 }),
        torch::randn({ 10, 256, 336 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_audio_tail() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AudioTailBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 256, 1024 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_model() {
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.save();
    trainer.load();
    trainer.close();
}

[[maybe_unused]] static void test_model_eval() {
    std::thread media_thread([]() {
        #if _WIN32
        chobits::media::open_file("D:/tmp/video.mp4");
        #else
        chobits::media::open_file("video.mp4");
        #endif
    });
    chobits::mode_save = true;
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.eval();
    media_thread.join();
}

[[maybe_unused]] static void test_model_train() {
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    std::thread media_thread([]() {
        // #if _WIN32
        // chobits::media::open_file("D:/tmp/video.mp4");
        // #else
        // chobits::media::open_file("video/32429377729-1-192.mp4");
        // #endif
        chobits::media::open_device();
    });
    std::thread model_thread([]() {
        chobits::model::Trainer trainer;
        #if _WIN32
        trainer.load("D:/tmp/chobits.ckpt");
        #else
        trainer.load("chobits.ckpt");
        #endif
        trainer.eval();
        // trainer.train();
        // trainer.save();
    });
    media_thread.join();
    model_thread.join();
    player_thread.join();
}

int main() {
    try {
        // test_res_net_1d();
        // test_res_net_2d();
        // test_gru();
        // test_attention();
        // test_audio_head();
        // test_video_head();
        // test_image_head();
        // test_media_mixer();
        // test_audio_tail();
        // test_model();
        test_model_eval();
        // test_model_train();
    } catch(const std::exception& e) {
        std::printf("异常内容：%s", e.what());
    }
    return 0;
}
