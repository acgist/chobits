#include "chobits/model.hpp"
#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include "SDL2/SDL.h"

struct PlayerState {
    SDL_AudioDeviceID audio_id = 0;
    SDL_Window  * window   = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture * texture  = nullptr;
    SDL_AudioSpec audio_spec = {
        .freq     = chobits::audio_sample_rate,
        .format   = AUDIO_S16,
        .channels = static_cast<uint8_t>(chobits::audio_nb_channels),
        .silence  = 0,
        .samples  = 4800,
        .callback = nullptr
    };
};

static PlayerState player_state = {};

static bool init_audio_player();
static bool init_video_player();
static void stop_audio_player();
static void stop_video_player();

bool chobits::player::open_player() {
    int ret = SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO);
    if(ret != 0) {
        std::printf("打开播放器失败：%d\n", ret);
        return false;
    }
    if(init_audio_player() && init_video_player()) {
        SDL_Event event;
        std::printf("打开播放器\n");
        while(chobits::running) {
            SDL_WaitEventTimeout(&event, 1000);
            if(event.type == SDL_QUIT) {
                std::printf("关闭播放器\n");
                std::printf("等待系统关闭...\n");
                chobits::running = false;
                chobits::media::stop_all();
                chobits::model::stop_all();
                break;
            }
        }
    } else {
        // -
    }
    stop_audio_player();
    stop_video_player();
    SDL_Quit();
    return true;
}

void chobits::player::stop_player() {
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
}

bool chobits::player::play_audio(const void* data, int len) {
    if(chobits::running && player_state.audio_id != 0) {
        SDL_QueueAudio(player_state.audio_id, data, len);
        return true;
    }
    return false;
}

bool chobits::player::play_video(const void* data, int len) {
    if(chobits::running && player_state.renderer && player_state.texture) {
        SDL_RenderClear(player_state.renderer);
        SDL_UpdateTexture(player_state.texture, nullptr, data, len);
        SDL_RenderCopy(player_state.renderer, player_state.texture, nullptr, nullptr);
        SDL_RenderPresent(player_state.renderer);
        return true;
    }
    return false;
}

static bool init_audio_player() {
    player_state.audio_id = SDL_OpenAudioDevice(nullptr, 0, &player_state.audio_spec, nullptr, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if(player_state.audio_id == 0) {
        std::printf("打开音频失败\n");
        return false;
    }
    SDL_PauseAudioDevice(player_state.audio_id, 0);
    return true;
}

static bool init_video_player() {
    player_state.window = SDL_CreateWindow("Chobits", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, chobits::video_width, chobits::video_height, SDL_WINDOW_OPENGL);
    if(!player_state.window) {
        std::printf("打开窗口失败\n");
        return false;
    }
    player_state.renderer = SDL_CreateRenderer(player_state.window, -1, SDL_RENDERER_ACCELERATED);
    if(!player_state.renderer) {
        std::printf("打开渲染失败\n");
        return false;
    }
    player_state.texture = SDL_CreateTexture(player_state.renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, chobits::video_width, chobits::video_height);
    if(!player_state.texture) {
        std::printf("打开纹理失败\n");
        return false;
    }
    return true;
}

static void stop_audio_player() {
    std::printf("关闭音频播放器\n");
    if(player_state.audio_id != 0) {
        SDL_CloseAudioDevice(player_state.audio_id);
        player_state.audio_id = 0;
    }
}

static void stop_video_player() {
    std::printf("关闭视频播放器\n");
    if(player_state.texture) {
        SDL_DestroyTexture(player_state.texture);
        player_state.texture = nullptr;
    }
    if(player_state.renderer) {
        SDL_DestroyRenderer(player_state.renderer);
        player_state.renderer = nullptr;
    }
    if(player_state.window) {
        SDL_DestroyWindow(player_state.window);
        player_state.window = nullptr;
    }
}
