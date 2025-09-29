#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <csignal>
#include <cstring>

static void help();
static void signal_handler(int code);

int main(int argc, char const *argv[]) {
    char buffer[1024];
    std::setvbuf(stdout, buffer, _IOLBF, sizeof(buffer));
    #if (defined(_DEBUG) || !defined(NDEBUG)) && defined(_WIN32)
    system("chcp 65001");
    #endif
    if(argc >= 2 && std::strcmp("help", argv[1]) == 0) {
        help();
        std::fflush(stdout);
        return 0;
    }
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::thread media_thread([argc, argv]() {
        chobits::media::open_media(argc, argv);
    });
    std::thread model_thread([argc, argv]() {
        chobits::model::open_model(argc, argv);
    });
    media_thread.join();
    model_thread.join();
    std::fflush(stdout);
    return 0;
}

static void help() {
    std::printf(R"(帮助：chobits[.exe] help
视频文件训练：chobits[.exe] file [训练次数] [训练批次]
现实生活评估：chobits[.exe] eval
现实生活训练：chobits[.exe]
)");
}

static void signal_handler(int code) {
    std::printf("处理信号：%d\n", code);
    if(code == SIGINT || code == SIGTERM) {
        chobits::stop_all();
    } else {
        // -
    }
}
