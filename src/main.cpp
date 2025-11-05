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
    if(argc >= 2 && (
        std::strcmp("?",      argv[1]) == 0 ||
        std::strcmp("-h",     argv[1]) == 0 ||
        std::strcmp("help",   argv[1]) == 0 ||
        std::strcmp("--help", argv[1]) == 0
    )) {
        help();
        return 0;
    }
    if(argc == 1 || (argc >= 2 && std::strcmp("eval", argv[1]) == 0)) {
        chobits::mode_eval   = std::strcmp("eval", argv[1]) == 0;
        chobits::mode_file   = false;
        chobits::drop_busy   = true;
        chobits::play_audio  = true;
        chobits::batch_size  = 1;
        chobits::train_epoch = 1;
        chobits::train_path  = "";
    } else if(argc >= 2) {
        chobits::mode_eval   = false;
        chobits::mode_file   = true;
        chobits::drop_busy   = false;
        chobits::play_audio  = false;
        chobits::batch_size  = argc >= 4 ? std::atoi(argv[3]) : 10;
        chobits::train_epoch = argc >= 3 ? std::atoi(argv[2]) : 1;
        chobits::train_path  = argv[1];
    } else {
        help();
        return 0;
    }
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::thread media_thread([]() {
        chobits::media::open_media();
    });
    std::thread model_thread([]() {
        chobits::model::open_model();
    });
    media_thread.join();
    model_thread.join();
    std::fflush(stdout);
    return 0;
}

static void help() {
    std::printf(R"(帮助：chobits[.exe] [?|help]
视频文件训练：chobits[.exe] file [训练次数] [训练批次]
媒体设备评估：chobits[.exe] eval
媒体设备训练：chobits[.exe]
)");
    std::fflush(stdout);
}

static void signal_handler(int code) {
    std::printf("处理信号：%d\n", code);
    if(code == SIGINT || code == SIGTERM) {
        chobits::stop_all();
    } else {
        // -
    }
}
