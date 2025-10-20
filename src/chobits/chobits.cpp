#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/chobits.hpp"

#include <string>

bool chobits::running           = true;
bool chobits::train             = true;
bool chobits::play_audio        = false;
int  chobits::batch_size        = 1;
int  chobits::audio_sample_rate = 48000;
int  chobits::audio_nb_channels = 1;
int  chobits::video_width       = 640;
int  chobits::video_height      = 360;

void chobits::stop_all() {
    std::printf("等待系统关闭...\n");
    chobits::running = false;
    chobits::media::stop_all();
    chobits::model::stop_all();
}
