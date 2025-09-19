#include "chobits/media.hpp"
#include "chobits/model.hpp"

#include "signal.h"

#include <thread>

static void help();
static void stop_all();
static void sigint_handler(int code);

int main(int argc, char const *argv[]) {
    signal(SIGINT, sigint_handler);
    #if _WIN32
    system("chcp 65001");
    #endif
    if(argc >= 2 && std::strcmp("help", argv[1]) == 0) {
        help();
        return 0;
    }
    std::thread media_thread([argc, argv]() {
        if(chobits::media::open_media(argc, argv)) {
            // -
        } else {
            help();
        }
    });
    std::thread model_thread([]() {
        chobits::model::Trainer trainer;
        if(trainer.load()) {
            trainer.train();
            trainer.save();
        } else {
            help();
        }
    });
    media_thread.join();
    model_thread.join();
    stop_all();
    return 0;
}

static void help() {
    std::printf(R"(帮助：chobits help
视频文件训练：chobits file
现实生活训练：chobits\n)");
}

static void stop_all() {
    chobits::media::stop_all();
    chobits::model::stop_all();
}

static void sigint_handler(int code) {
    std::printf("处理信号：%d\n", code);
    if(code == SIGINT) {
        stop_all();
    } else {
    }
}
