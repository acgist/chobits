#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
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

[[maybe_unused]] static void test_rotary() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::RotaryPositionEmbedding layer(128);
    info(layer.ptr());
    auto [ q, k ] = layer->forward(
        torch::randn({ 10, 8, 32, 128 }),
        torch::randn({ 10, 8, 32, 128 })
    );
    std::cout << q.sizes() << std::endl;
    std::cout << k.sizes() << std::endl;
}

[[maybe_unused]] static void test_res_net_1d() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::ResNet1dBlock layer(8, 16, 800);
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 8, 800 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_res_net_2d() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::ResNet2dBlock layer(8, 16, std::vector<int64_t>{ 360, 640 });
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 8, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_attention() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AttentionBlock layer(800, 800, 800, 800);
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 256, 800 }),
        torch::randn({ 10, 256, 800 }),
        torch::randn({ 10, 256, 800 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_audio_head() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AudioHeadBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 32, 800 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_video_head() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::VideoHeadBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 32, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_image_head() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::ImageHeadBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 3, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_media_muxer() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MediaMuxerBlock layer;
    info(layer.ptr());
    auto [ audio, video, muxer ] = layer->forward(
        torch::randn({ 10, 256, 160 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 576 })
    );
    std::cout << audio.sizes() << std::endl;
    std::cout << video.sizes() << std::endl;
    std::cout << muxer.sizes() << std::endl;
}

[[maybe_unused]] static void test_media_mixer() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MediaMixerBlock layer;
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 256, 160 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 576 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_audio_tail() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AudioTailBlock layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 256, 672 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_model_eval() {
    chobits::mode_save    = true;
    chobits::batch_thread = 1;
    std::thread media_thread([]() {
        #if _WIN32
        chobits::media::open_file("D:/tmp/video.mp4");
        #else
        chobits::media::open_file("video.mp4");
        #endif
    });
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.eval();
    media_thread.join();
}

int main() {
    try {
        // test_rotary();
        // test_res_net_1d();
        // test_res_net_2d();
        // test_attention();
        // test_audio_head();
        // test_video_head();
        // test_image_head();
        // test_media_muxer();
        // test_media_mixer();
        // test_audio_tail();
        test_model_eval();
    } catch(const std::exception& e) {
        std::printf("异常内容：%s", e.what());
    }
    return 0;
}
