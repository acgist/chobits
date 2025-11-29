#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <fstream>

#include "torch/fft.h"

[[maybe_unused]] static void test_sftf() {
    std::ifstream stream_in ("D:/tmp/dzht.pcm",  std::ios::binary);
    std::ofstream stream_out("D:/tmp/dzht_.pcm", std::ios::binary);
    int size = 800;
    std::vector<short> pcm(size);
    const int n_fft    = 200;
    const int hop_size = 50;
    const int win_size = 200;
    auto wind = torch::hann_window(win_size);
    while(stream_in.read((char*) pcm.data(), sizeof(short) * size)) {
        auto tensor = torch::from_blob(pcm.data(), { 1, size }, torch::kShort).to(torch::kFloat32).div(32768.0);
        auto com = torch::stft(tensor.squeeze(-1), n_fft, hop_size, win_size, wind, true, "reflect", false, std::nullopt, true);
        // 分解幅度和相位
        auto mag = torch::abs(com);
        auto pha = torch::angle(com);
        std::cout << tensor.sizes() << std::endl;
        std::cout << com.sizes() << std::endl;
        std::cout << mag.sizes() << std::endl;
        std::cout << pha.sizes() << std::endl;
        // 合成幅度和相位
        // auto real = mag * torch::cos(pha);
        // auto imag = mag * torch::sin(pha);
        //      com  = torch::complex(real, imag);
             com = torch::polar(mag, pha);
        auto out = torch::istft(com, n_fft, hop_size, win_size, wind, true).unsqueeze(-1).mul(32768.0).to(torch::kShort);
        stream_out.write((char*) out.data_ptr(), sizeof(short) * size);
    }
    stream_in.close();
    stream_out.close();
}

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
        auto [success, stft, audio, video, label] = chobits::media::get_data();
        std::cout << stft .sizes() << std::endl;
        std::cout << audio.sizes() << std::endl;
        std::cout << video.sizes() << std::endl;
        std::cout << label.sizes() << std::endl;
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
    // test_sftf();
    test_media();
    return 0;
}
