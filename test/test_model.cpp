#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"

[[maybe_unused]] static void test_trainer() {
    // chobits::media::open_file("D:/tmp/video.mp4");
    chobits::media::open_hardware();
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.save();
    chobits::media::stop_all();
}

[[maybe_unused]] static void test_audio_head() {
    chobits::nn::AudioHeadBlock layer(2);
    auto output = layer->forward(torch::randn({ 1, 2, 201, 601 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_video_head() {
    chobits::nn::VideoHeadBlock layer(3);
    auto output = layer->forward(torch::randn({ 1, 3, 640, 360 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_media_mix() {
    // 64, 24, 74
    // 64, 52, 30
    chobits::nn::MediaMixBlock layer(
        64, 128,
        24, 74,
        52, 30,
        24, 74
    );
    auto output = layer->forward(
        torch::randn({ 1, 64, 24, 74 }),
        torch::randn({ 1, 64, 52, 30 })
    );
    std::cout << output.sizes() << std::endl;
    size_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        total_numel += parameter.value().numel();
    }
    std::cout << total_numel << std::endl;
}

[[maybe_unused]] static void test_residual() {
    chobits::nn::ResidualBlock layer(64, 128);
    auto output = layer->forward(torch::randn({ 1, 64, 20, 20 }));
    std::cout << output.sizes() << std::endl;
    size_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        total_numel += parameter.value().numel();
    }
    std::cout << total_numel << std::endl;
}

[[maybe_unused]] static void test_attention() {
    // 64, 24, 74
    // 64, 52, 30
    chobits::nn::AttentionBlock layer(64, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 64, 24, 74 }));
    // chobits::nn::AttentionBlock layer(64, 52 * 30);
    // auto output = layer->forward(torch::randn({ 1, 64, 52, 30 }));
    std::cout << output.sizes() << std::endl;
    size_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        total_numel += parameter.value().numel();
    }
    std::cout << total_numel << std::endl;
}

[[maybe_unused]] static void test_residual_attention() {
    chobits::nn::ResidualAttentionBlock layer(64, 128, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 64, 24, 74 }));
    std::cout << output.sizes() << std::endl;
    size_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        total_numel += parameter.value().numel();
    }
    std::cout << total_numel << std::endl;
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer(256, 256, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 256, 24, 74 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_model() {
    auto audio = torch::randn({ 1, 2, 201, 601 });
    auto video = torch::randn({ 1, 3, 640, 360 });
    chobits::model::Trainer trainer;
    trainer.load("chobits.pt", false);
    trainer.test();
    // trainer.save("chobits.pt");
}

int main() {
    // test_trainer();
    // test_audio_head();
    // test_video_head();
    // test_media_mix();
    // test_residual();
    // test_attention();
    // test_residual_attention();
    // test_audio_tail();
    test_model();
    return 0;
}
