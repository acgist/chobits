#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <cinttypes>

static void info(std::shared_ptr<torch::nn::Module> layer) {
    size_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        total_numel += parameter.value().numel();
    }
    std::printf("参数总量：%" PRIu64 "\n", total_numel);
}

[[maybe_unused]] static void test_attention() {
    chobits::nn::AttentionBlock layer(800, 512);
    auto output = layer->forward(torch::randn({ 1, 512, 800 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_head() {
    chobits::nn::AudioHeadBlock layer(3, 1, std::vector<int>{ 1, 8, 64, 512 }, std::vector<int>{ 4, 4, 3 });
    auto output = layer->forward(torch::randn({ 1, 1, 48000 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_video_head() {
    chobits::nn::VideoHeadBlock layer(400, 1000, 3, 1, std::vector<int>{ 3, 8, 64, 512 }, std::vector<int>{ 3, 4, 3, 4, 2, 2 });
    auto output = layer->forward(torch::randn({ 1, 3, 360, 640 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_media_mix() {
    chobits::nn::MediaMixBlock layer(800, 512, 3, 1, 2);
    auto output = layer->forward(
        torch::randn({ 1, 512, 800 }),
        torch::randn({ 1, 512, 800 })
    );
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer(800, 3, 1, std::vector<int>{ 512, 64, 8, 1 }, std::vector<double>{ 3, 4, 4 });
    auto output = layer->forward(torch::randn({ 1, 512, 800 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_load_save() {
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.save();
    trainer.load();
}

[[maybe_unused]] static void test_model_eval() {
    chobits::batch_size = 1;
    std::thread media_thread([]() {
        #if _WIN32
        chobits::media::open_file("D:/tmp/video.mp4");
        #else
        chobits::media::open_file("video/32429377729-1-192.mp4");
        #endif
    });
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.eval(true);
    // trainer.save("chobits.pt");
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
    chobits::player::stop_player();
    player_thread.join();
}

int main() {
    // test_attention();
    // test_audio_head();
    // test_video_head();
    // test_media_mix();
    // test_audio_tail();
    // test_load_save();
    // test_model_eval();
    // test_model_train();
    return 0;
}
