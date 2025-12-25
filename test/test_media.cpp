#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <fstream>

#include "torch/torch.h"

[[maybe_unused]] static void test_media() {
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    std::thread media_thread([]() {
        // file/http/rtmp/rtsp/m3u8
        // open_device
        // chobits::media::open_device();
        // open_file
        // #if _WIN32
        // chobits::media::open_file("D:/tmp/video.mp4");
        // #else
        // chobits::media::open_file("video/32429377729-1-192.mp4");
        // #endif
        // open_media
        chobits::mode_file = true;
        #if _WIN32
        chobits::train_dataset = "D:/tmp/video.mp4";
        #else
        chobits::train_dataset = "video/32429377729-1-192.mp4";
        #endif
        chobits::media::open_media();
    });
    while(chobits::running) {
        auto [success, audio, video, label] = chobits::media::get_data();
        std::cout << audio.sizes() << std::endl;
        std::cout << video.sizes() << std::endl;
        std::cout << label.sizes() << std::endl;
        std::cout << label.min().item<float>() << " = " << label.mean().item<float>() << " = " << label.max().item<float>() << std::endl;
        if(success) {
            chobits::media::set_data(audio.cpu());
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    media_thread.join();
    player_thread.join();
}

int main() {
    test_media();
    return 0;
}
