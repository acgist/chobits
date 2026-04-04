#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>

#include "torch/torch.h"

[[maybe_unused]] static void test_media() {
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    std::thread media_thread([]() {
        chobits::media::open_media();
    });
    while(chobits::running) {
        auto [ success, audio, video ] = chobits::media::get_data();
        std::cout << audio.sizes() << std::endl;
        std::cout << video.sizes() << std::endl;
        if(success) {
            chobits::media::set_data(audio, video);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    player_thread.join();
    media_thread.join();
}

[[maybe_unused]] static void test_model() {
    chobits::open_all("D://download/chobits.pt");
}

int main() {
    test_media();
    // test_model();
    return 0;
}
