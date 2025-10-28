#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>

#include "ATen/Tensor.h"

int main() {
    chobits::batch_size = 1;
    // chobits::batch_wind = 1;
    // chobits::video_skip = 1;
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    std::thread media_thread([]() {
        // file/http/rtmp/rtsp/m3u8
        #if _WIN32
        chobits::media::open_file("D:/tmp/video.mp4");
        #else
        chobits::media::open_file("video/32429377729-1-192.mp4");
        #endif
        // chobits::media::open_device();
    });
    while(chobits::running) {
        auto [success, audio, video, pred] = chobits::media::get_data(false);
        std::cout << audio.sizes() << std::endl;
        std::cout << video.sizes() << std::endl;
        std::cout << pred .sizes() << std::endl;
        if(success) {
            chobits::media::set_data(audio[0][0].squeeze().cpu());
            // std::this_thread::sleep_for(std::chrono::milliseconds(25));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    media_thread.join();
    chobits::player::stop_player();
    player_thread.join();
    return 0;
}
