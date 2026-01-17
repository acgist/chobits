#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

bool        chobits::running   = true;
bool        chobits::mode_drop = false;
bool        chobits::mode_eval = false;
bool        chobits::mode_file = false;
bool        chobits::mode_play = false;
bool        chobits::mode_save = false;

int         chobits::batch_size    = 1;
int         chobits::batch_length  = 10;
int         chobits::train_epoch   = 1;
std::string chobits::train_dataset = "";

int  chobits::per_wind_second   = 10;
int  chobits::audio_sample_rate = 8000;
int  chobits::audio_nb_channels = 1;
int  chobits::video_width       = 640;
int  chobits::video_height      = 360;

void chobits::stop_all() {
    std::printf("等待系统关闭...\n");
    chobits::running = false;
    chobits::player::stop_player();
    chobits::media::stop_all();
    chobits::model::stop_all();
}
