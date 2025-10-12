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

[[maybe_unused]] static void test_media_head() {
    chobits::nn::MediaHeadBlock layer(16, std::vector<int>{ 2, 8, 32, 64 }, std::vector<int>{ 3, 3, 3 }, std::vector<int>{ 2, 2, 2 });
    auto output = layer->forward(torch::randn({ 1, 2, 201, 601 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_media_mix() {
    chobits::nn::MediaMixBlock layer(
        64, 128,    
        24,  74,
        30,  52,
        30,  52
        // 24,  74
    );
    auto output = layer->forward(
        torch::randn({ 1, 64, 24, 74 }),
        torch::randn({ 1, 64, 30, 52 })
    );
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_attention() {
    chobits::nn::AttentionBlock audio_layer(64, 24 * 74);
    auto audio_output = audio_layer->forward(torch::randn({ 10, 64, 24, 74 }));
    std::cout << audio_output.sizes() << std::endl;
    info(audio_layer.ptr());
    chobits::nn::AttentionBlock video_layer(64, 30 * 52);
    auto video_output = video_layer->forward(torch::randn({ 10, 64, 30, 52 }));
    std::cout << video_output.sizes() << std::endl;
    info(video_layer.ptr());
}

[[maybe_unused]] static void test_memory_prob() {
    chobits::nn::MemoryProbBlock layer(128, 24, 74);
    auto output = layer->forward(torch::randn({ 10, 128, 24, 74 }), torch::randn({ 1, 128, 24, 74 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer(std::vector<int>{ 24 * 74, 48 * 148 }, std::vector<int>{ 8, 8 }, std::vector<int>{ 128, 32, 8, 1 }, std::vector<int>{ 4, 5, 5 }, std::vector<int>{ 2, 2, 2 });
    auto output = layer->forward(torch::randn({ 1, 128, 24, 74 }));
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
    std::thread media_thread([]() {
        chobits::media::open_file("D:/tmp/video.mp4");
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
        // chobits::media::open_file("D:/tmp/video.mp4");
        chobits::media::open_device();
    });
    std::thread model_thread([]() {
        chobits::model::Trainer trainer;
        trainer.load("D:/tmp/chobits.ckpt");
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
    // test_media_head();
    // test_media_mix();
    // test_attention();
    // test_memory_prob();
    // test_audio_tail();
    test_load_save();
    // test_model_eval();
    // test_model_train();
    return 0;
}
