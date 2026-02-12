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

[[maybe_unused]] static void test_expert() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Expert layer(512);
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 256, 512 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_moe() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MoE layer(512);
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 256, 512 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_mha() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::MHA layer(512, 512, 512, 512);
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 512 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_vit() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::ViT layer(
        11, 201, 32, 128,
        std::vector<int64_t>{ 2, 2 },
        std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 5, 5 },
        std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }
    );
    info(layer.ptr());
    auto output = layer->forward(
        torch::randn({ 10, 32, 11, 201 }),
        torch::randn({ 10, 128, 256 })
    );
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_mixer() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Mixer layer;
    info(layer.ptr());
    auto [ audio, video, image, muxer ] = layer->forward(
        torch::randn({ 10, 256, 256 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 128, 768 })
    );
    std::cout << audio.sizes() << std::endl;
    std::cout << video.sizes() << std::endl;
    std::cout << image.sizes() << std::endl;
    std::cout << muxer.sizes() << std::endl;
}

[[maybe_unused]] static void test_talk() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Talk layer;
    info(layer.ptr());
    auto output = layer->forward(torch::randn({ 10, 256, 768 }));
    std::cout << output.sizes() << std::endl;
}

[[maybe_unused]] static void test_chobits() {
    torch::NoGradGuard no_grad_guard;
    chobits::nn::Chobits layer;
    info(layer.ptr());
    auto [ audio, video, image, muxer ] = layer->forward(
        torch::randn({ 10, 32, 800 }),
        torch::randn({ 10, 32, 3, 360, 640 }),
        torch::randn({ 10, 256, 256 }),
        torch::randn({ 10, 256, 512 }),
        torch::randn({ 10, 256, 512 })
    );
    std::cout << audio.sizes() << std::endl;
    std::cout << video.sizes() << std::endl;
    std::cout << image.sizes() << std::endl;
    std::cout << muxer.sizes() << std::endl;
}

[[maybe_unused]] static void test_eval() {
    chobits::mode_save = true;
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
        // test_expert();
        // test_moe();
        // test_mha();
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
