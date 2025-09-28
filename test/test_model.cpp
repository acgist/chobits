#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"

#include <cinttypes>

static void info(std::shared_ptr<torch::nn::Module> layer) {
    size_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        total_numel += parameter.value().numel();
    }
    std::printf("参数总量：%" PRIu64 "\n", total_numel);
}

[[maybe_unused]] static void test_audio_head() {
    chobits::nn::AudioHeadBlock layer(2);
    auto output = layer->forward(torch::randn({ 1, 2, 201, 601 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_video_head() {
    chobits::nn::VideoHeadBlock layer(3);
    auto output = layer->forward(torch::randn({ 1, 3, 372, 640 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_media_mix() {
    chobits::nn::MediaMixBlock layer(
        64, 128,    
        24, 74,
        30, 52,
        24, 74
    );
    auto output = layer->forward(
        torch::randn({ 1, 64, 24, 74 }),
        torch::randn({ 1, 64, 30, 52 })
    );
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_residual() {
    chobits::nn::ResidualBlock layer(64, 128);
    auto output = layer->forward(torch::randn({ 1, 64, 20, 20 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_attention() {
    // 64, 24, 74
    // 64, 52, 30
    chobits::nn::AttentionBlock layer(64, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 64, 24, 74 }));
    // chobits::nn::AttentionBlock layer(64, 30 * 52);
    // auto output = layer->forward(torch::randn({ 1, 64, 30, 52 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_residual_attention() {
    chobits::nn::ResidualAttentionBlock layer(64, 128, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 64, 24, 74 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer(256, 256, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 256, 24, 74 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_load_save() {
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.save();
    trainer.load();
}

[[maybe_unused]] static void test_model() {
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.test();
    // trainer.save("chobits.pt");
}

[[maybe_unused]] static void test_trainer() {
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    std::thread media_thread([]() {
        // chobits::media::open_file("D:/tmp/video.mp4");
        chobits::media::open_hardware();
    });
    std::thread model_thread([]() {
        chobits::model::Trainer trainer;
        trainer.load();
        // trainer.eval();
        trainer.train();
        trainer.save();
    });
    player_thread.join();
    media_thread.join();
    model_thread.join();
}

int main() {
    // test_audio_head();
    // test_video_head();
    // test_media_mix();
    // test_residual();
    // test_attention();
    // test_residual_attention();
    // test_audio_tail();
    // test_load_save();
    // test_model();
    // test_trainer();
    return 0;
}
