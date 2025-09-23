#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <fstream>

#include "torch/torch.h"

[[maybe_unused]] static void test_pcm() {
    std::ifstream input;
    std::ofstream output;
    input .open("D:/tmp/dzht.pcm",      std::ios_base::binary);
    output.open("D:/tmp/dzht-copy.pcm", std::ios_base::binary);
    if(!input.is_open()) {
        input.close();
        return;
    }
    std::vector<short> pcm;
    pcm.resize(9600); // 48000 / 10 * 2
    while(input.read(reinterpret_cast<char*>(pcm.data()), pcm.size() * 2)) {
        auto tensor = chobits::media::pcm_stft(pcm.data(), input.gcount() / 2);
        auto result = chobits::media::pcm_istft(tensor);
        output.write(reinterpret_cast<char*>(result.data()), result.size() * 2);
    }
    input .close();
    output.close();
}

[[maybe_unused]] static void test_open_file() {
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    // file/http/rtmp/rtsp/m3u8
    chobits::media::open_file("D:/tmp/video.mp4");
    player_thread.join();
}

[[maybe_unused]] static void test_open_hardware() {
    std::thread player_thread([]() {
        chobits::player::open_player();
    });
    chobits::media::open_hardware();
    player_thread.join();
}

[[maybe_unused]] static void test_get_data() {
    std::thread thread([]() {
        chobits::media::open_file("D:/tmp/video.mp4");
        // chobits::media::open_hardware();
    });
    while(chobits::running) {
        auto [audio, video, pred] = chobits::media::get_data();
        std::cout << audio.sizes() << std::endl;
        std::cout << video.sizes() << std::endl;
        std::cout << pred .sizes() << std::endl;
        if(audio.numel() == 0 || video.numel() == 0 || pred.numel() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    thread.join();
}

int main() {
    // test_pcm();
    test_open_file();
    // test_open_hardware();
    // test_get_data();
    return 0;
}
