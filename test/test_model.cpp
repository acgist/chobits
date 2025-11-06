#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <fstream>
#include <cinttypes>

static void info(std::shared_ptr<torch::nn::Module> layer) {
    int64_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        std::printf("参数数量：%s = %" PRId64 "\n", parameter.key().c_str(), parameter.value().numel());
        total_numel += parameter.value().numel();
    }
    std::printf("参数总量：%" PRId64 "\n", total_numel);
}

[[maybe_unused]] static void test_gru() {
    chobits::nn::GRUBlock layer(128 * 2, 128);
    auto output = layer->forward(torch::randn({ 1, 800, 128 * 2 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_res_net() {
    chobits::nn::ResNetBlock layer(800, 800);
    auto output = layer->forward(torch::randn({ 1, 800, 128 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_attention() {
    chobits::nn::AttentionBlock layer;
//  chobits::nn::AttentionBlock layer(800, 128, 8);
    auto output = layer->forward(torch::randn({ 1, 800, 128 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_head() {
    chobits::nn::AudioHeadBlock layer;
    auto output = layer->forward(torch::randn({ 10, 800, 1 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_video_head() {
    chobits::nn::VideoHeadBlock layer(std::vector<int>{ 3, 100, 400, 800 }, std::vector<int>{ 5, 5, 3, 4, 3, 2 });
    auto output = layer->forward(torch::randn({ 1, 3, 360, 640 }));
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_media_mix() {
    chobits::nn::MediaMixBlock layer;
    auto output = layer->forward(
        torch::randn({ 1, 800, 128 }),
        torch::randn({ 1, 800, 128 })
    );
    std::cout << output.sizes() << std::endl;
    info(layer.ptr());
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer;
    auto output = layer->forward(torch::randn({ 1, 800, 128 }));
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
    std::ofstream stream;
    stream.open("chobits.pcm", std::ios::binary);
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.eval([&stream](const std::vector<short>& data) {
        stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(short));
        std::printf("写入音频数据：%" PRIu64 "\n", data.size());
    });
    media_thread.join();
    stream.close();
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
    // test_gru();
    // test_res_net();
    // test_attention();
    // test_audio_head();
    // test_video_head();
    test_media_mix();
    // test_audio_tail();
    // test_load_save();
    // test_model_eval();
    // test_model_train();
    return 0;
}
