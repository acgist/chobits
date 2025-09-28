#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>

#include "ATen/Tensor.h"

[[maybe_unused]] static void test_open_file() {
    chobits::train = false;
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    // file/http/rtmp/rtsp/m3u8
    chobits::media::open_file("D:/tmp/video.mp4");
    chobits::player::stop_player();
    player_thread.join();
}

[[maybe_unused]] static void test_open_hardware() {
    chobits::train = false;
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    chobits::media::open_hardware();
    chobits::player::stop_player();
    player_thread.join();
}

[[maybe_unused]] static void test_get_data() {
    chobits::batch_size = 1;
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    std::thread media_thread([]() {
        chobits::media::open_file("D:/tmp/video.mp4");
        // chobits::media::open_hardware();
    });
    while(chobits::running) {
        auto [audio, video, pred] = chobits::media::get_data(false);
        std::cout << audio.sizes() << std::endl;
        std::cout << video.sizes() << std::endl;
        std::cout << pred .sizes() << std::endl;
        if(audio.numel() == 0 || video.numel() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            chobits::media::set_data(audio.squeeze(0));
        }
    }
    media_thread.join();
    chobits::player::stop_player();
    player_thread.join();
}

int main() {
    // test_open_file();
    // test_open_hardware();
    test_get_data();
    return 0;
}
