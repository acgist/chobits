#include "chobits/media.hpp"

#include <mutex>
#include <thread>
#include <vector>
#include <iostream>

#include "SDL2/SDL.h"
#include "torch/torch.h"

extern "C" {

#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"

}

#ifndef MEDIA_PREVIEW
#define MEDIA_PREVIEW
#endif

const static int           video_width      = 640;
const static int           video_height     = 360;
const static AVPixelFormat video_pix_format = AV_PIX_FMT_RGB24;

const static AVSampleFormat  audio_format           = AV_SAMPLE_FMT_S16;
const static AVChannelLayout audio_layout           = AV_CHANNEL_LAYOUT_MONO;
const static int             audio_sample_rate      = 48000;
const static int             audio_nb_channels      = audio_layout.nb_channels;
const static int             audio_bytes_per_sample = av_get_bytes_per_sample(audio_format);

const static float NORMALIZATION = 32768.0F;

struct PlayerState {
    bool running = false;
    SDL_AudioDeviceID audio_id;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture * texture  = nullptr;
    std::vector<uint8_t> audio_buffer;
    std::vector<uint8_t> video_buffer;
};

static PlayerState player = {};

static SwrContext* init_audio_swr(AVCodecContext* ctx);
static SwsContext* init_video_sws(int width, int height, AVPixelFormat format);
static bool audio_to_tensor(SwrContext* swr, AVFrame* frame);
static bool video_to_tensor(SwsContext* sws, AVFrame* frame);

/**
 * 短时傅里叶变换
 * 
 * 201 = win_size / 2 + 1
 * 480 = 7 | 4800 = 61 | 48000 = 601
 * [1, 201, 61, 2[实部, 虚部]]
 * 
 * @param pcm_data PCM数据
 * @param pcm_size PCM长度
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return 张量
 */
static torch::Tensor pcm_stft(
    short* pcm_data,
    int pcm_size,
    int n_fft    = 400,
    int hop_size = 80,
    int win_size = 400
);

/**
 * 短时傅里叶逆变换
 * 
 * @param tensor   张量
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return PCM数据
 */
static std::vector<short> pcm_istft(
    const torch::Tensor& tensor,
    int n_fft    = 400,
    int hop_size = 80,
    int win_size = 400
);

bool chobits::media::open_file(const std::string& file) {
    int ret = 0;
    AVFormatContext* format_ctx = avformat_alloc_context();
    ret = avformat_open_input(&format_ctx, file.c_str(), nullptr, nullptr);
    if(ret != 0) {
        std::printf("打开输入文件失败：%d - %s\n", ret, file.c_str());
        return false;
    }
    av_dump_format(format_ctx, 0, format_ctx->url, 0);
    const int audio_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    const int video_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if(audio_index < 0 || video_index < 0) {
        avformat_close_input(&format_ctx);
        std::printf("查找媒体轨道失败：%d - %d\n", audio_index, video_index);
        return false;
    }
    std::printf("打开输入文件成功：%d - %d\n", audio_index, video_index);
    const AVStream* audio_stream    = format_ctx->streams[audio_index];
    const AVCodec * audio_codec     = avcodec_find_decoder(audio_stream->codecpar->codec_id);
    AVCodecContext* audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    const AVStream* video_stream    = format_ctx->streams[video_index];
    const AVCodec * video_codec     = avcodec_find_decoder(video_stream->codecpar->codec_id);
    AVCodecContext* video_codec_ctx = avcodec_alloc_context3(video_codec);
    ret = avcodec_parameters_to_context(audio_codec_ctx, audio_stream->codecpar);
    ret = avcodec_open2(audio_codec_ctx, audio_codec, nullptr);
    if(ret != 0) {
        avcodec_free_context(&audio_codec_ctx);
        avcodec_free_context(&video_codec_ctx);
        avformat_close_input(&format_ctx);
        std::printf("打开音频解码器失败：%d\n", ret);
        return false;
    }
    ret = avcodec_parameters_to_context(video_codec_ctx, video_stream->codecpar);
    ret = avcodec_open2(video_codec_ctx, video_codec, nullptr);
    if(ret != 0) {
        avcodec_free_context(&audio_codec_ctx);
        avcodec_free_context(&video_codec_ctx);
        avformat_close_input(&format_ctx);
        std::printf("打开视频解码器失败：%d\n", ret);
        return false;
    }
    SwrContext* audio_swr = init_audio_swr(audio_codec_ctx);
    SwsContext* video_sws = init_video_sws(video_codec_ctx->width, video_codec_ctx->height, AV_PIX_FMT_YUV420P);
    if(audio_swr == nullptr || video_sws == nullptr) {
        swr_free(&audio_swr);
        sws_freeContext(video_sws);
        avcodec_free_context(&audio_codec_ctx);
        avcodec_free_context(&video_codec_ctx);
        avformat_close_input(&format_ctx);
        std::printf("打开音视频重采样失败");
        return false;
    }
    player.running = true;
    player.audio_buffer.resize(audio_nb_channels * audio_bytes_per_sample * audio_sample_rate * 2);
    player.video_buffer.resize(av_image_get_buffer_size(video_pix_format, video_width, video_height, 1));
    std::thread player_thread([]() {
        open_player();
    });
    double audio_time  = 0;
    double video_time  = 0;
    double audio_base  = av_q2d(audio_stream->time_base);
    double video_base  = av_q2d(video_stream->time_base);
    uint64_t audio_pos = 0;
    uint64_t video_pos = 0;
    uint64_t base_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    uint64_t audio_frame_count = 0;
    uint64_t video_frame_count = 0;
    AVFrame * frame  = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    while(player.running && av_read_frame(format_ctx, packet) == 0) {
        if(packet->stream_index == audio_index) {
            if(avcodec_send_packet(audio_codec_ctx, packet) == 0) {
                while(player.running && avcodec_receive_frame(audio_codec_ctx, frame) == 0) {
                    ++audio_frame_count;
                    if(audio_pos == 0) {
                        audio_pos = frame->pts;
                    }
                    audio_time = (frame->pts - audio_pos) * audio_base;
                    audio_to_tensor(audio_swr, frame);
                    av_frame_unref(frame);
                }
            }
        } else if(packet->stream_index == video_index) {
            if(avcodec_send_packet(video_codec_ctx, packet) == 0) {
                while(player.running && avcodec_receive_frame(video_codec_ctx, frame) == 0) {
                    ++video_frame_count;
                    if(video_pos == 0) {
                        video_pos = frame->pts;
                    }
                    video_time = (frame->pts - video_pos) * video_base;
                    video_to_tensor(video_sws, frame);
                    av_frame_unref(frame);
                    #ifdef MEDIA_PREVIEW
                    // 正常速度
                    uint64_t pts_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    int delay = (int) (video_time * 1000 - (pts_time - base_time));
                    if(delay > 0) {
                        SDL_Delay(delay);
                    }
                    #endif
                }
            }
        } else {
            // -
        }
        av_packet_unref(packet);
    }
    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&audio_swr);
    sws_freeContext(video_sws);
    avcodec_free_context(&audio_codec_ctx);
    avcodec_free_context(&video_codec_ctx);
    avformat_close_input(&format_ctx);
    stop_all();
    player_thread.join();
    std::printf("文件处理完成：%lld - %lld - %f - %f - %s\n", audio_frame_count, video_frame_count, audio_time, video_time, file.c_str());
    return true;
}

bool chobits::media::open_hardware() {
    int ret = 0;
    avdevice_register_all();
    AVFormatContext    * console_format_ctx = avformat_alloc_context();
    AVDictionary       * console_options    = nullptr;
    const AVInputFormat* console_format     = av_find_input_format("dshow");
    av_dict_set(&console_options, "list_devices", "true", 0);
    avformat_open_input(&console_format_ctx, "video=dummy", console_format, &console_options);
    av_dict_free(&console_options);
    avformat_close_input(&console_format_ctx);
    std::string audio_device_name = "麦克风阵列 (适用于数字麦克风的英特尔® 智音技术)";
    std::string video_device_name = "Integrated Camera";
    // std::printf("请选择音频输入设备名称：\n");
    // std::cin >> audio_device_name;
    // std::printf("请选择视频输入设备名称：\n");
    // std::cin >> video_device_name;
    std::printf("打开音频输入设备：%s\n", audio_device_name.c_str());
    std::printf("打开视频输入设备：%s\n", video_device_name.c_str());
    audio_device_name = "audio=" + audio_device_name;
    video_device_name = "video=" + video_device_name;
    AVFormatContext    * audio_format_ctx = avformat_alloc_context();
    AVDictionary       * audio_options    = nullptr;
    const AVInputFormat* audio_format     = av_find_input_format("dshow");
    AVFormatContext    * video_format_ctx = avformat_alloc_context();
    AVDictionary       * video_options    = nullptr;
    const AVInputFormat* video_format     = av_find_input_format("dshow");
    // 音频参数
    av_dict_set(&audio_options, "channels",          "2",         0);
    av_dict_set(&audio_options, "sample_rate",       "48000",     0);
    av_dict_set(&audio_options, "sample_format",     "pcm_s16le", 0);
    av_dict_set(&audio_options, "audio_buffer_size", "100",       0);
    // 视频参数
    av_dict_set(&video_options, "framerate",    "30",      0);
    av_dict_set(&video_options, "video_size",   "640*360", 0);
    av_dict_set(&video_options, "pixel_format", "yuyv422", 0);
    ret = avformat_open_input(&audio_format_ctx, audio_device_name.c_str(), audio_format, &audio_options);
    av_dict_free(&audio_options);
    if(ret != 0) {
        avformat_close_input(&audio_format_ctx);
        avformat_close_input(&video_format_ctx);
        std::printf("打开音频硬件失败：%d - %s", ret, audio_device_name.c_str());
        return false;
    }
    ret = avformat_open_input(&video_format_ctx, video_device_name.c_str(), video_format, &video_options);
    av_dict_free(&video_options);
    if(ret != 0) {
        avformat_close_input(&audio_format_ctx);
        avformat_close_input(&video_format_ctx);
        std::printf("打开视频硬件失败：%d - %s", ret, video_device_name.c_str());
        return false;
    }
    av_dump_format(audio_format_ctx, 0, audio_format_ctx->url, 0);
    av_dump_format(video_format_ctx, 0, video_format_ctx->url, 0);
    player.running = true;
    player.audio_buffer.resize(audio_nb_channels * audio_bytes_per_sample * audio_sample_rate * 2);
    player.video_buffer.resize(av_image_get_buffer_size(video_pix_format, video_width, video_height, 1));
    std::thread player_thread([]() {
        open_player();
    });
    uint64_t audio_frame_count = 0;
    uint64_t video_frame_count = 0;
    std::thread audio_thread([audio_format_ctx, &audio_frame_count]() {
        int ret = 0;
        const AVStream* audio_stream    = audio_format_ctx->streams[0];
        const AVCodec * audio_codec     = avcodec_find_decoder(audio_stream->codecpar->codec_id);
        AVCodecContext* audio_codec_ctx = avcodec_alloc_context3(audio_codec);
        ret = avcodec_parameters_to_context(audio_codec_ctx, audio_stream->codecpar);
        ret = avcodec_open2(audio_codec_ctx, audio_codec, nullptr);
        if(ret != 0) {
            avcodec_free_context(&audio_codec_ctx);
            std::printf("打开音频解码器失败：%d\n", ret);
            return;
        }
        SwrContext* audio_swr = init_audio_swr(audio_codec_ctx);
        if(audio_swr == nullptr) {
            avcodec_free_context(&audio_codec_ctx);
            std::printf("打开音频重采样失败");
            return;
        }
        AVFrame * frame  = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        while(player.running) {
            if(av_read_frame(audio_format_ctx, packet) == 0) {
                if(avcodec_send_packet(audio_codec_ctx, packet) == 0) {
                    while(player.running && avcodec_receive_frame(audio_codec_ctx, frame) == 0) {
                        ++audio_frame_count;
                        audio_to_tensor(audio_swr, frame);
                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(packet);
        }
        av_frame_free(&frame);
        av_packet_free(&packet);
        swr_free(&audio_swr);
        avcodec_free_context(&audio_codec_ctx);
    });
    std::thread video_thread([video_format_ctx, &video_frame_count]() {
        int ret = 0;
        const AVStream* video_stream    = video_format_ctx->streams[0];
        const AVCodec * video_codec     = avcodec_find_decoder(video_stream->codecpar->codec_id);
        AVCodecContext* video_codec_ctx = avcodec_alloc_context3(video_codec);
        ret = avcodec_parameters_to_context(video_codec_ctx, video_stream->codecpar);
        ret = avcodec_open2(video_codec_ctx, video_codec, nullptr);
        if(ret != 0) {
            avcodec_free_context(&video_codec_ctx);
            std::printf("打开视频解码器失败：%d\n", ret);
            return;
        }
        SwsContext* video_sws = init_video_sws(video_codec_ctx->width, video_codec_ctx->height, video_codec_ctx->pix_fmt);
        if(video_sws == nullptr) {
            avcodec_free_context(&video_codec_ctx);
            std::printf("打开视频重采样失败");
            return;
        }
        AVFrame * frame  = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        while(player.running) {
            if(av_read_frame(video_format_ctx, packet) == 0) {
                if(avcodec_send_packet(video_codec_ctx, packet) == 0) {
                    while(player.running && avcodec_receive_frame(video_codec_ctx, frame) == 0) {
                        ++video_frame_count;
                        video_to_tensor(video_sws, frame);
                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(packet);
        }
        av_frame_free(&frame);
        av_packet_free(&packet);
        sws_freeContext(video_sws);
        avcodec_free_context(&video_codec_ctx);
    });
    audio_thread.join();
    video_thread.join();
    stop_all();
    player_thread.join();
    avformat_close_input(&audio_format_ctx);
    avformat_close_input(&video_format_ctx);
    std::printf("文件处理完成：%lld - %lld\n", audio_frame_count, video_frame_count);
    return true;
}

bool chobits::media::open_player() {
    SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Chobits", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, video_width, video_height, SDL_WINDOW_OPENGL);
    SDL_AudioSpec audio_spec = {
        .freq     = audio_sample_rate,
        .format   = AUDIO_S16,
        .channels = 1,
        .silence  = 0,
        .samples  = 4800,
        .callback = nullptr
    };
    player.audio_id = SDL_OpenAudioDevice(nullptr, 0, &audio_spec, nullptr, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    SDL_PauseAudioDevice(player.audio_id, 0);
    player.renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    player.texture  = SDL_CreateTexture(player.renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, video_width, video_height);
    SDL_Event event;
    while(player.running) {
        SDL_WaitEventTimeout(&event, 1000);
        if(event.type == SDL_QUIT) {
            stop_all();
            break;
        }
    }
    SDL_CloseAudioDevice(player.audio_id);
    SDL_DestroyTexture(player.texture);
    SDL_DestroyRenderer(player.renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return true;
}

bool chobits::media::play_audio(const void* data, int len) {
    if(player.running) {
        SDL_QueueAudio(player.audio_id, data, len);
        SDL_Delay(10);
        return true;
    }
    return false;
}

bool chobits::media::play_video(const void* data, int len) {
    if(player.running) {
        SDL_RenderClear(player.renderer);
        SDL_UpdateTexture(player.texture, nullptr, data, len);
        SDL_RenderCopy(player.renderer, player.texture, nullptr, nullptr);
        SDL_RenderPresent(player.renderer);
        return true;
    }
    return false;
}

void chobits::media::stop_all() {
    if(!player.running) {
        return;
    }
    player.running = false;
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
    SDL_Delay(10);
}

static SwrContext* init_audio_swr(AVCodecContext* ctx) {
    int ret = 0;
    SwrContext* swr = swr_alloc();
    ret = swr_alloc_set_opts2(
        &swr,
        &audio_layout,   audio_format,    audio_sample_rate,
        &ctx->ch_layout, ctx->sample_fmt, ctx->sample_rate,
        0, nullptr
    );
    if(ret != 0) {
        swr_free(&swr);
        std::printf("打开音频重采样失败：%d", ret);
        return nullptr;
    }
    ret = swr_init(swr);
    if(ret != 0) {
        swr_free(&swr);
        std::printf("打开音频重采样失败：%d", ret);
        return nullptr;
    }
    return swr;
}

static SwsContext* init_video_sws(int width, int height, AVPixelFormat format) {
    SwsContext* sws = sws_getContext(
        width,       height,       format,
        video_width, video_height, video_pix_format,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    return sws;
}

static bool audio_to_tensor(SwrContext* swr, AVFrame* frame) {
    static int index = 0;
    index = index == 0 ? 1 : 0;
    int size = audio_nb_channels * audio_bytes_per_sample * frame->nb_samples;
    uint8_t* buffer = player.audio_buffer.data() + (index % 2) * size;
    swr_convert(swr, &buffer, frame->nb_samples, (const uint8_t**) frame->data, frame->nb_samples);
    chobits::media::play_audio(buffer, size);
    // auto tensor = pcm_stft(reinterpret_cast<short*>(ptr), size);
    // auto pcm    = pcm_istft(tensor);
    // std::memcpy(ptr, pcm.data(), size);
    // std::cout << tensor.sizes() << std::endl;
    return true;
}

static bool video_to_tensor(SwsContext* sws, AVFrame* frame) {
    int width = video_width * 3;
    uint8_t* const buffer = player.video_buffer.data();
    sws_scale(sws, (const uint8_t* const *) frame->data, frame->linesize, 0, frame->height, &buffer, &width);
    chobits::media::play_video(buffer, width);
    // torch::Tensor tensor = torch::from_blob(buffer, { video_height, video_width, 3 }, torch::kByte)
    //     .permute({ 2, 0, 1 }).to(torch::kFloat32).div(255.0).mul(2.0).sub(1.0).contiguous();
    // auto image_tensor = tensor.add(1.0).div(2.0).mul(255.0).permute({ 1, 2, 0 }).to(torch::kByte).contiguous();
    // auto x = image_tensor.element_size() * image_tensor.numel();
    // std::memcpy(buffer, reinterpret_cast<char*>(image_tensor.data_ptr()), image_tensor.element_size() * image_tensor.numel());
    return true;
}

static torch::Tensor pcm_stft(
    short* pcm_data,
    int pcm_size,
    int n_fft,
    int hop_size,
    int win_size
) {
    auto data = torch::from_blob(pcm_data, { 1, pcm_size }, torch::kShort).to(torch::kFloat32) / NORMALIZATION;
    auto wind = torch::hann_window(win_size);
    auto real = torch::view_as_real(torch::stft(data, n_fft, hop_size, win_size, wind, true, "reflect", false, std::nullopt, true));
    // 幅度: sqrt(x^2 + y^2)
    auto mag = torch::sqrt(real.pow(2).sum(-1));
    // 相位: atan2(y, x)
    auto pha = torch::atan2(real.index({ "...", 1 }), real.index({ "...", 0 }));
    return torch::stack({ mag, pha }, -1).squeeze();
}

static std::vector<short> pcm_istft(
    const torch::Tensor& tensor,
    int n_fft,
    int hop_size,
    int win_size
) {
    auto copy = tensor.unsqueeze(0);
    auto wind = torch::hann_window(win_size);
    auto mag  = copy.index({ "...", 0 });
    auto pha  = copy.index({ "...", 1 });
    auto com  = torch::complex(mag * torch::cos(pha), mag * torch::sin(pha));
    auto ret  = torch::istft(com, n_fft, hop_size, win_size, wind, true) * NORMALIZATION;
    float* data = reinterpret_cast<float*>(ret.data_ptr());
    std::vector<short> pcm;
    pcm.resize(ret.sizes()[1]);
    std::copy_n(data, pcm.size(), pcm.data());
    return pcm;
}
