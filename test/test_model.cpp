#include "chobits/nn.hpp"
#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <fstream>
#include <cinttypes>

static void info(std::shared_ptr<torch::nn::Module> layer) {
    int     layer_size  = 0;
    int64_t total_numel = 0;
    for(const auto& parameter : layer->named_parameters()) {
        ++layer_size;
        std::printf("参数数量：%64s = %" PRId64 "\n", parameter.key().c_str(), parameter.value().numel());
        total_numel += parameter.value().numel();
    }
    std::printf("层数总量：%d\n", layer_size);
    std::printf("参数总量：%" PRId64 "\n", total_numel);
}

[[maybe_unused]] static void test_gru() {
    chobits::nn::GRUBlock layer(576, 576);
    auto output = layer->forward(torch::randn({ 10, 256, 576 }));
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_attention() {
    chobits::nn::AttentionBlock layer(576, 576, 576, 576);
    auto output = layer->forward(torch::randn({ 10, 256, 576 }), torch::randn({ 10, 256, 576 }), torch::randn({ 10, 256, 576 }));
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_res_net() {
    chobits::nn::ResNetBlock layer(10, 100, 576);
    auto output = layer->forward(torch::randn({ 10, 10, 576 }));
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_res_net_3d() {
    chobits::nn::ResNet3dBlock layer(10, 100, 10, std::vector<int64_t>{ 1, 5, 5 });
    auto output = layer->forward(torch::randn({ 10, 10, 3, 360, 640 }));
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_audio_head() {
    chobits::nn::AudioHeadBlock layer;
    auto output = layer->forward(torch::randn({ 10, 10, 800 }));
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_video_head() {
    chobits::nn::VideoHeadBlock layer;
    auto output = layer->forward(torch::randn({ 10, 10, 3, 360, 640 }));
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_media_mixer() {
    chobits::nn::MediaMixerBlock layer;
    auto output = layer->forward(
        torch::randn({ 10, 256,     416 }),
        torch::randn({ 10, 256, 3 * 576 })
    );
    info(layer.ptr());
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_audio_tail() {
    chobits::nn::AudioTailBlock layer;
    auto output = layer->forward(torch::randn({ 10, 256, 400 }));
    info(layer.ptr());
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
    std::ofstream stream;
    stream.open("chobits.pcm", std::ios::binary);
    chobits::model::Trainer trainer;
    trainer.load();
    auto time_point = std::chrono::system_clock::now();
    trainer.eval([&stream, &time_point](const std::vector<short>& data) {
        stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(short));
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - time_point).count();
        std::printf("写入音频数据：%" PRIu64 " = %" PRId64 "\n", data.size(), duration);
        time_point = std::chrono::system_clock::now();
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
    player_thread.join();
}

int main() {
    try {
        // test_gru();
        // test_attention();
        // test_res_net();
        // test_res_net_3d();
        // test_audio_head();
        // test_video_head();
        // test_media_mixer();
        test_audio_tail();
        // test_model();
        // test_model_eval();
        // test_model_train();
    } catch(const std::exception& e) {
        std::printf("异常内容：%s", e.what());
    }
    return 0;
}
