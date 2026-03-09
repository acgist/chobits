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
        std::printf("参数数量：%64s = %" PRId64 "\n", parameter.key().c_str(), parameter.value().numel());
        total_numel += parameter.value().numel();
    }
    std::printf("层数总量：%d\n", layer_size);
    std::printf("参数总量：%" PRId64 "\n", total_numel);
}

[[maybe_unused]] static void test_moe() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MoE layer(512);
    layer->init();
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 256, 512 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_mha() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MHA layer(512, 512, 512, 512);
    layer->init();
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 512 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_ast() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::AsT layer(
        32 * 800, 512, 1, 256,
        100, 200
    );
    layer->init();
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 1, 32 * 800 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_vit() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::ViT layer(
        360, 640, 512, 32, 512,
        std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 40, 40 }
    );
    layer->init();
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 32, 360, 640 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_mixer() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Mixer layer;
    layer->init();
    info(layer.ptr());
    auto [ audio, video ] = layer->forward(
        torch::randn({ 10, 128, 256 }),
        torch::randn({ 10, 256, 512 })
    );
    std::cout << audio.sizes() << std::endl;
    std::cout << video.sizes() << std::endl;
}

[[maybe_unused]] static void test_talk() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Talk layer;
    layer->init();
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 512, 256 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_chobits() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Chobits layer;
    layer->init();
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 32, 800 }),
        torch::randn({ 10, 32, 3, 360, 640 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_eval() {
    chobits::mode_save = true;
    std::thread media_thread([]() {
        chobits::train_media = "audio";
        #if _WIN32
        // chobits::media::open_file("D:/tmp/video.mp4");
        chobits::media::open_file("D:/tmp/audio.wav");
        #else
        // chobits::media::open_file("video.mp4");
        chobits::media::open_file("audio.wav");
        #endif
    });
    chobits::model::Trainer trainer;
    trainer.load();
    trainer.eval();
    media_thread.join();
}

int main() {
    try {
        // test_moe();
        // test_mha();
        // test_ast();
        // test_vit();
        // test_mixer();
        // test_talk();
        // test_chobits();
        test_eval();
    } catch(const std::exception& e) {
        std::printf("异常内容：%s", e.what());
    }
    return 0;
}
