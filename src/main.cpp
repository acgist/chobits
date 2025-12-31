#include "chobits/media.hpp"
#include "chobits/model.hpp"
#include "chobits/chobits.hpp"

#include <thread>
#include <csignal>
#include <cstring>

static void help();
static bool init(int argc, char const *argv[]);
static void signal_handler(int code);

int main(int argc, char const *argv[]) {
    char buffer[1024];
    std::setvbuf(stdout, buffer, _IOLBF, sizeof(buffer));
    #if defined(_WIN32)
    system("chcp 65001");
    #endif
    if(!init(argc, argv)) {
        help();
        return 0;
    }
    std::signal(SIGINT,  signal_handler);
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

static bool init(int argc, char const *argv[]) {
    std::printf(
R"(
洛神赋（节选）
曹植
其形也，翩若惊鸿，婉若游龙。荣曜秋菊，华茂春松。
髣髴兮若轻云之蔽月，飘飖兮若流风之回雪。
远而望之，皎若太阳升朝霞；
迫而察之，灼若芙蕖出渌波。
穠纤得衷，修短合度。
肩若削成，腰如约素。
延颈秀项，皓质呈露。
芳泽无加，铅华弗御。
云髻峨峨，修眉联娟。
丹唇外朗，皓齿内鲜。
明眸善睐，靥辅承权。
瓌姿艳逸，仪静体闲。
柔情绰态，媚于语言。
奇服旷世，骨像应图。
披罗衣之璀粲兮，珥瑶碧之华琚。
戴金翠之首饰，缀明珠以耀躯。
践远游之文履，曳雾绡之轻裾。
微幽兰之芳蔼兮，步踟蹰于山隅。
于是忽焉纵体，以遨以嬉。
左倚采旄，右荫桂旗。
攘皓腕于神浒兮，采湍濑之玄芝。
)"
    );
    if(argc == 1) {
        chobits::mode_drop = true;
        chobits::mode_eval = false;
        chobits::mode_file = false;
        chobits::mode_play = true;
    } else {
        if(
            std::strcmp("?",      argv[1]) == 0 ||
            std::strcmp("-h",     argv[1]) == 0 ||
            std::strcmp("--help", argv[1]) == 0
        ) {
            return false;
        } else if(std::strcmp("eval", argv[1]) == 0) {
            chobits::mode_drop = true;
            chobits::mode_eval = true;
            chobits::mode_file = false;
            chobits::mode_play = true;
        } else {
            chobits::mode_drop     = false;
            chobits::mode_eval     = false;
            chobits::mode_file     = true;
            chobits::mode_play     = false;
            chobits::batch_size    = argc >= 4 ? std::atoi(argv[3]) : 10;
            chobits::train_epoch   = argc >= 3 ? std::atoi(argv[2]) : 10;
            chobits::train_dataset = argv[1];
        }
    }
    std::printf(
        "丢弃模式：%d 验证模式：%d 文件模式：%d 播放模式：%d "
        "训练批次大小：%d 训练批次长度：%d 训练批次轮数：%d 训练数据集：%s "
        "音频每秒窗口：%d 音频采样率：%d 音频通道：%d "
        "视频宽度：%d 视频高度：%d\n",
        chobits::mode_drop, chobits::mode_eval, chobits::mode_file, chobits::mode_play,
        chobits::batch_size, chobits::batch_length, chobits::train_epoch, chobits::train_dataset.c_str(),
        chobits::per_wind_second, chobits::audio_sample_rate, chobits::audio_nb_channels,
        chobits::video_width, chobits::video_height
    );
    return true;
}

static void help() {
    std::printf(R"(帮助：chobits[.exe] [?|-h|--help]
视频文件训练：chobits[.exe] file [训练批次大小] [训练批次轮数]
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
