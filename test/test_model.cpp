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

[[maybe_unused]] static void test_attention() {
    chobits::nn::AttentionBlock audio_layer(128, 24 * 74);
    auto audio_output = audio_layer->forward(torch::randn({ 10, 128, 24, 74 }));
    std::cout << audio_output.sizes() << std::endl;
    info(audio_layer.ptr());
    chobits::nn::AttentionBlock video_layer(128, 29 * 52);
    auto video_output = video_layer->forward(torch::randn({ 10, 128, 29, 52 }));
    std::cout << video_output.sizes() << std::endl;
    info(video_layer.ptr());
}

[[maybe_unused]] static void test_media_head() {
    chobits::nn::MediaHeadBlock audio_layer(std::vector<int>{ 2, 8, 32, 128 }, std::vector<int>{ 3, 3, 3 }, std::vector<int>{ 2, 2, 2 }, 24 * 74);
    auto audio_output = audio_layer->forward(torch::randn({ 1, 2, 201, 601 }));
    std::cout << audio_output.sizes() << std::endl;
    info(audio_layer.ptr());
    chobits::nn::MediaHeadBlock video_layer(std::vector<int>{ 3, 8, 32, 128 }, std::vector<int>{ 3, 3, 3 }, std::vector<int>{ 3, 2, 2 }, 29 * 52);
    auto video_output = video_layer->forward(torch::randn({ 1, 3, 360, 640 }));
    std::cout << video_output.sizes() << std::endl;
    info(video_layer.ptr());
}

[[maybe_unused]] static void test_media_mix() {
    chobits::nn::MediaMixBlock layer(
        64, 128,    
        24,  74,
        29,  52
    );
    auto output = layer->forward(
        torch::randn({ 1, 64, 24, 74 }),
        torch::randn({ 1, 64, 29, 52 })
    );
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer(std::vector<int>{ 128, 32, 8, 2 }, std::vector<int>{ 4, 5, 5 }, std::vector<int>{ 2, 2, 2 }, 24 * 74);
    auto output = layer->forward(torch::randn({ 1, 128, 24, 74 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_memory() {
    chobits::nn::MemoryBlock layer(128, 24, 74);
    auto output = layer->forward(torch::randn({ 10, 128, 24, 74 }), torch::randn({ 1, 128, 24, 74 }));
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
    test_media_head();
    // test_media_mix();
    // test_audio_tail();
    // test_memory();
    // test_load_save();
    // test_model_eval();
    // test_model_train();
    return 0;
}
