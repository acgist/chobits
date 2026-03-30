#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <ctime>
#include <thread>
#include <iostream>

bool chobits::running = true;

int chobits::per_millisecond = 100;

int chobits::audio_sample_rate = 8000;
int chobits::audio_nb_channels = 1;

int chobits::video_width  = 640;
int chobits::video_height = 480;

void chobits::open_all(const std::string& model_path) {
    std::thread player_thread([]() {
        if(!chobits::player::open_player()) {
            chobits::stop_all();
        }
    });
    std::thread media_thread([]() {
        if(!chobits::media::open_media()) {
            chobits::stop_all();
        }
    });
    std::thread model_thread([model_path]() {
        if(chobits::model::open_model(model_path)) {
            chobits::model::run_model();
        } else {
            chobits::stop_all();
        }
    });
    player_thread.join();
    media_thread.join();
    model_thread.join();
}

void chobits::stop_all() {
    std::time_t time = std::time(nullptr);
    std::tm* tm = std::localtime(&time);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
    std::printf("%s\n", buffer);
    std::printf("等待系统关闭...\n");
    chobits::running = false;
    chobits::player::stop_player();
    chobits::media::stop_media();
    chobits::model::stop_model();
}
