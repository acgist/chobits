#include "chobits/media.hpp"
#include "chobits/player.hpp"
#include "chobits/chobits.hpp"

#include <mutex>
#include <thread>
#include <numbers>
#include <filesystem>
#include <condition_variable>

#include "torch/fft.h"

extern "C" {

#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"

}

const static AVPixelFormat video_pix_format = AV_PIX_FMT_RGB24;

const static AVSampleFormat  audio_format           = AV_SAMPLE_FMT_S16;
const static AVChannelLayout audio_layout           = AV_CHANNEL_LAYOUT_MONO;
const static int             audio_bytes_per_sample = av_get_bytes_per_sample(audio_format);

struct Dataset {
    bool   discard    = true;  // 是否丢弃数据
    size_t cache_size = 60;    // 缓存数据大小
    size_t audio_size = 48000; // 48000 / 48000 / 2 * 1000 = 500 ms
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<torch::Tensor> audio;
    std::vector<torch::Tensor> video;
};

static Dataset dataset = {};

static SwrContext* init_audio_swr(AVCodecContext* ctx, AVFrame* frame);
static SwsContext* init_video_sws(AVCodecContext* ctx, AVFrame* frame);

static std::string device_name(AVMediaType type, const char* format_name);

static void sws_free(SwsContext** sws);

static bool audio_to_tensor(SwrContext* swr, AVFrame* frame);
static bool video_to_tensor(SwsContext* sws, AVFrame* frame);

static at::Tensor pcm_stft(short* pcm_data, int pcm_size, int n_fft = 400, int hop_size = 80, int win_size = 400);
static std::vector<short> pcm_istft(const at::Tensor& tensor, int n_fft = 400, int hop_size = 80, int win_size = 400);

bool chobits::media::open_media(int argc, char const *argv[]) {
    if(argc == 1 || (argc >= 2 && std::strcmp("eval", argv[1]) == 0)) {
        std::thread player_thread([]() {
            chobits::player::open_player();
        });
        bool ret = chobits::media::open_device();
        chobits::player::stop_player();
        player_thread.join();
        chobits::stop_all();
        return ret;
    } else if(argc >= 2) {
        const auto path = argv[1];
        const auto size = argc >= 3 ? std::atoi(argv[2]) : 1;
        for(int index = 0; index < size && chobits::running; ++index) {
            std::printf("训练轮次：%d\n", index);
            if(std::filesystem::is_directory(path)) {
                const auto iterator = std::filesystem::directory_iterator(path);
                for(const auto& entry : iterator) {
                    auto file_path = entry.path().string();
                    if(chobits::media::open_file(file_path)) {
                        std::printf("文件处理完成：%s\n", file_path.c_str());
                    } else {
                        std::printf("文件处理失败：%s\n", file_path.c_str());
                    }
                }
            } else {
                if(chobits::media::open_file(path)) {
                    std::printf("文件处理完成：%s\n", path);
                } else {
                    std::printf("文件处理失败：%s\n", path);
                }
            }
        }
        chobits::stop_all();
        return true;
    } else {
        std::printf("不支持的媒体\n");
        return false;
    }
}

bool chobits::media::open_file(const std::string& file) {
    int ret = 0;
    AVFormatContext* format_ctx = avformat_alloc_context();
    ret = avformat_open_input(&format_ctx, file.c_str(), nullptr, nullptr);
    if(ret != 0) {
        avformat_close_input(&format_ctx);
        std::printf("打开输入文件失败：%d - %s\n", ret, file.c_str());
        return false;
    }
    av_dump_format(format_ctx, 0, format_ctx->url, 0);
    int audio_index = -1;
    int video_index = -1;
    for(uint32_t i = 0; i < format_ctx->nb_streams; ++i) {
        auto stream = format_ctx->streams[i];
        if(stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_index = stream->index;
        } else if(stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_index = stream->index;
        } else {
            // -
        }
    }
    if(audio_index < 0 || video_index < 0) {
        avformat_close_input(&format_ctx);
        std::printf("查找媒体轨道失败：%d - %d\n", audio_index, video_index);
        return false;
    }
    std::printf("打开输入文件成功：%d - %d - %s\n", audio_index, video_index, file.c_str());
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
    dataset.discard = false;
    double audio_time  = 0;
    double video_time  = 0;
    double audio_base  = av_q2d(audio_stream->time_base);
    double video_base  = av_q2d(video_stream->time_base);
    uint64_t audio_pos = 0;
    uint64_t video_pos = 0;
    uint64_t audio_frame_count = 0;
    uint64_t video_frame_count = 0;
    AVFrame * frame  = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    SwrContext* audio_swr = nullptr;
    SwsContext* video_sws = nullptr;
    while(chobits::running && av_read_frame(format_ctx, packet) == 0) {
        if(packet->stream_index == audio_index) {
            if(avcodec_send_packet(audio_codec_ctx, packet) == 0) {
                while(chobits::running && avcodec_receive_frame(audio_codec_ctx, frame) == 0) {
                    ++audio_frame_count;
                    if(audio_pos == 0) {
                        audio_pos = frame->pts;
                    }
                    audio_time = (frame->pts - audio_pos) * audio_base;
                    if(audio_swr == nullptr) {
                        audio_swr = init_audio_swr(audio_codec_ctx, frame);
                    }
                    if(audio_swr != nullptr) {
                        audio_to_tensor(audio_swr, frame);
                    }
                    av_frame_unref(frame);
                }
            }
        } else if(packet->stream_index == video_index) {
            if(avcodec_send_packet(video_codec_ctx, packet) == 0) {
                while(chobits::running && avcodec_receive_frame(video_codec_ctx, frame) == 0) {
                    ++video_frame_count;
                    if(video_pos == 0) {
                        video_pos = frame->pts;
                    }
                    video_time = (frame->pts - video_pos) * video_base;
                    if(video_sws == nullptr) {
                        video_sws = init_video_sws(video_codec_ctx, frame);
                    }
                    if(video_sws != nullptr) {
                        video_to_tensor(video_sws, frame);
                    }
                    av_frame_unref(frame);
                }
            }
        } else {
            // -
        }
        av_packet_unref(packet);
    }
    swr_free(&audio_swr);
    sws_free(&video_sws);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&audio_codec_ctx);
    avcodec_free_context(&video_codec_ctx);
    avformat_close_input(&format_ctx);
    std::printf("文件处理完成：%" PRIu64 " - %" PRIu64 " - %f - %f - %s\n", audio_frame_count, video_frame_count, audio_time, video_time, file.c_str());
    return true;
}

bool chobits::media::open_device() {
    // ffmpeg -devices
    int ret = 0;
    avdevice_register_all();
    #if _WIN32
    const char* audio_format_name = "dshow";
    const char* video_format_name = "dshow";
    #else
    const char* audio_format_name = "alsa";
    const char* video_format_name = "v4l2";
    #endif
    std::string audio_device_name = device_name(AVMEDIA_TYPE_AUDIO, audio_format_name);
    std::string video_device_name = device_name(AVMEDIA_TYPE_VIDEO, video_format_name);
    std::printf("打开音频输入设备：%s\n", audio_device_name.c_str());
    std::printf("打开视频输入设备：%s\n", video_device_name.c_str());
    #if _WIN32
    audio_device_name = "audio=" + audio_device_name;
    video_device_name = "video=" + video_device_name;
    #endif
    AVFormatContext    * audio_format_ctx = avformat_alloc_context();
    AVDictionary       * audio_options    = nullptr;
    const AVInputFormat* audio_format     = av_find_input_format(audio_format_name);
    AVFormatContext    * video_format_ctx = avformat_alloc_context();
    AVDictionary       * video_options    = nullptr;
    const AVInputFormat* video_format     = av_find_input_format(video_format_name);
    // 音频参数
    av_dict_set(&audio_options, "channels",          "2",         0);
    av_dict_set(&audio_options, "sample_rate",       "48000",     0);
    av_dict_set(&audio_options, "sample_format",     "pcm_s16le", 0);
    av_dict_set(&audio_options, "audio_buffer_size", "100",       0); // 毫秒
    // 视频参数
    av_dict_set(&video_options, "framerate",    "30",      0);
    av_dict_set(&video_options, "video_size",   "640*360", 0);
    av_dict_set(&video_options, "pixel_format", "yuyv422", 0);
    ret = avformat_open_input(&audio_format_ctx, audio_device_name.c_str(), audio_format, &audio_options);
    av_dict_free(&audio_options);
    if(ret != 0) {
        avformat_close_input(&audio_format_ctx);
        avformat_close_input(&video_format_ctx);
        std::printf("打开音频硬件失败：%d - %s\n", ret, audio_device_name.c_str());
        return false;
    }
    ret = avformat_open_input(&video_format_ctx, video_device_name.c_str(), video_format, &video_options);
    av_dict_free(&video_options);
    if(ret != 0) {
        avformat_close_input(&audio_format_ctx);
        avformat_close_input(&video_format_ctx);
        std::printf("打开视频硬件失败：%d - %s\n", ret, video_device_name.c_str());
        return false;
    }
    av_dump_format(audio_format_ctx, 0, audio_format_ctx->url, 0);
    av_dump_format(video_format_ctx, 0, video_format_ctx->url, 0);
    dataset.discard = true;
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
        AVFrame * frame  = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        SwrContext* audio_swr = nullptr;
        while(chobits::running) {
            if(av_read_frame(audio_format_ctx, packet) == 0) {
                if(avcodec_send_packet(audio_codec_ctx, packet) == 0) {
                    while(chobits::running && avcodec_receive_frame(audio_codec_ctx, frame) == 0) {
                        ++audio_frame_count;
                        if(audio_swr == nullptr) {
                            audio_swr = init_audio_swr(audio_codec_ctx, frame);
                        }
                        if(audio_swr != nullptr) {
                            audio_to_tensor(audio_swr, frame);
                        }
                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(packet);
        }
        swr_free(&audio_swr);
        av_frame_free(&frame);
        av_packet_free(&packet);
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
        AVFrame * frame  = av_frame_alloc();
        AVPacket* packet = av_packet_alloc();
        SwsContext* video_sws = nullptr;
        while(chobits::running) {
            if(av_read_frame(video_format_ctx, packet) == 0) {
                if(avcodec_send_packet(video_codec_ctx, packet) == 0) {
                    while(chobits::running && avcodec_receive_frame(video_codec_ctx, frame) == 0) {
                        ++video_frame_count;
                        if(video_sws == nullptr) {
                            video_sws = init_video_sws(video_codec_ctx, frame);
                        }
                        if(video_sws != nullptr) {
                            video_to_tensor(video_sws, frame);
                        }
                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(packet);
        }
        sws_free(&video_sws);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&video_codec_ctx);
    });
    audio_thread.join();
    video_thread.join();
    avformat_close_input(&audio_format_ctx);
    avformat_close_input(&video_format_ctx);
    std::printf("媒体处理完成：%" PRIu64 " - %" PRIu64 "\n", audio_frame_count, video_frame_count);
    return true;
}

void chobits::media::stop_all() {
    std::printf("关闭媒体\n");
    std::unique_lock<std::mutex> lock(dataset.mutex);
    dataset.audio.clear();
    dataset.video.clear();
    dataset.condition.notify_all();
}

std::tuple<bool, at::Tensor, at::Tensor, at::Tensor> chobits::media::get_data(bool train) {
    std::vector<torch::Tensor> audio;
    std::vector<torch::Tensor> video;
    std::vector<torch::Tensor> label;
    {
        std::unique_lock<std::mutex> lock(dataset.mutex);
        dataset.condition.wait(lock, [train]() {
            size_t batch_size = chobits::batch_size;
            if(train) {
                batch_size += 1;
            }
            return
                !(
                    chobits::running &&
                    (
                        dataset.audio.size() < batch_size ||
                        dataset.video.size() < batch_size
                    )
                );
        });
        if(!chobits::running) {
            return {false, {}, {}, {}};
        }
        audio.assign(dataset.audio.begin(), dataset.audio.begin() + chobits::batch_size);
        video.assign(dataset.video.begin(), dataset.video.begin() + chobits::batch_size);
        if(train) {
            label.assign(dataset.audio.begin() + 1, dataset.audio.begin() + chobits::batch_size + 1);
        }
        dataset.audio.erase(dataset.audio.begin(), dataset.audio.begin() + chobits::batch_size);
        dataset.video.erase(dataset.video.begin(), dataset.video.begin() + chobits::batch_size);
        // std::printf("剩余数据：%" PRIu64 " - %" PRIu64 "\n", dataset.audio.size(), dataset.video.size());
        dataset.condition.notify_all();
    }
    if(train) {
        return {
            true,
            torch::stack(audio),
            torch::stack(video),
            torch::stack(label)
        };
    } else {
        return {
            true,
            torch::stack(audio),
            torch::stack(video),
            {}
        };
    }
}

std::vector<short> chobits::media::set_data(const torch::Tensor& tensor) {
    auto pcm = pcm_istft(tensor);
    chobits::player::play_audio(pcm.data(), pcm.size() * sizeof(short));
    return pcm;
}

static SwrContext* init_audio_swr(AVCodecContext* ctx, AVFrame*) {
    SwrContext* swr = swr_alloc();
    if(swr == nullptr) {
        std::printf("打开音频重采样失败\n");
        return nullptr;
    }
    int ret = swr_alloc_set_opts2(
        &swr,
        &audio_layout,   audio_format,    chobits::audio_sample_rate,
        &ctx->ch_layout, ctx->sample_fmt, ctx->sample_rate,
        0, nullptr
    );
    if(ret != 0) {
        swr_free(&swr);
        std::printf("打开音频重采样失败：%d\n", ret);
        return nullptr;
    }
    ret = swr_init(swr);
    if(ret != 0) {
        swr_free(&swr);
        std::printf("打开音频重采样失败：%d\n", ret);
        return nullptr;
    }
    return swr;
}

static SwsContext* init_video_sws(AVCodecContext* ctx, AVFrame* frame) {
    int  width  = ctx->width  != 0 ? ctx->width  : frame->width;
    int  height = ctx->height != 0 ? ctx->height : frame->height;
    auto format = ctx->pix_fmt == AV_PIX_FMT_NONE ? AV_PIX_FMT_YUV420P : ctx->pix_fmt;
    SwsContext* sws = sws_getContext(
        width,                height,                format,
        chobits::video_width, chobits::video_height, video_pix_format,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    if(sws == nullptr) {
        std::printf("打开视频重采样失败\n");
        return nullptr;
    }
    return sws;
}

static std::string device_name(AVMediaType type, const char* format_name) {
    // av_input_audio_device_next()
    // av_input_video_device_next()
    std::string name;
    AVDeviceInfoList   * device_list   = nullptr;
    const AVInputFormat* device_format = av_find_input_format(format_name);
    int ret = avdevice_list_input_sources(device_format, nullptr, nullptr, &device_list);
    if (ret <= 0) {
        std::printf("打开硬件输入失败：%d\n", ret);
        avdevice_free_list_devices(&device_list);
        return name;
    }
    int index = device_list->default_device;
    if(index < 0 && device_list->nb_devices > 0) {
        index = 0;
        for (int i = 0; i < device_list->nb_devices; ++i) {
            AVDeviceInfo* device_info = device_list->devices[i];
            std::printf(
                "所有硬件输入设备：%d = %s = %s = %s\n",
                device_info->nb_media_types,
                av_get_media_type_string(type),
                device_info->device_name,
                device_info->device_description
            );
            for(int j = 0; j < device_info->nb_media_types; ++j) {
                AVMediaType media_type = device_info->media_types[j];
                if(media_type == type) {
                    index = i;
                }
            }
        }
    }
    if(index >= 0) {
        AVDeviceInfo* device_info = device_list->devices[index];
        std::printf(
            "选择硬件输入设备：%d = %s = %s = %s\n",
            device_info->nb_media_types,
            av_get_media_type_string(type),
            device_info->device_name,
            device_info->device_description
        );
        name = device_info->device_name;
    }
    avdevice_free_list_devices(&device_list);
    return name;
}

static void sws_free(SwsContext** sws) {
    sws_freeContext(*sws);
    *sws = nullptr;
}

static bool audio_to_tensor(SwrContext* swr, AVFrame* frame) {
    static size_t remain = 0;
    static std::vector<uint8_t> audio_buffer(chobits::audio_nb_channels * audio_bytes_per_sample * chobits::audio_sample_rate);
    uint8_t* buffer = audio_buffer.data() + remain;
    const int out_samples = swr_convert(swr, &buffer, swr_get_out_samples(swr, frame->nb_samples), (const uint8_t**) frame->data, frame->nb_samples);
    if(out_samples < 0) {
        std::printf("音频重采样失败：%d\n", out_samples);
        return false;
    }
    const size_t size = chobits::audio_nb_channels * audio_bytes_per_sample * out_samples;
    remain += size;
    // chobits::player::play_audio(buffer, size);
    while(remain >= dataset.audio_size) {
        if(chobits::train) {
            bool insert = false;
            std::unique_lock<std::mutex> lock(dataset.mutex);
            if(dataset.audio.size() >= dataset.cache_size) {
                if(dataset.discard) {
                    // std::printf("丢弃音频数据：%" PRIu64 "\n", dataset.audio.size());
                } else {
                    dataset.condition.wait(lock, []() {
                        return !(chobits::running && dataset.audio.size() > dataset.cache_size);
                    });
                    insert = true;
                }
            } else {
                insert = true;
            }
            if(insert) {
                auto tensor = pcm_stft(reinterpret_cast<short*>(audio_buffer.data()), dataset.audio_size);
                dataset.audio.push_back(tensor);
            }
            dataset.condition.notify_all();
        }
        remain -= dataset.audio_size;
        if(remain != 0) {
            std::memcpy(audio_buffer.data(), audio_buffer.data() + dataset.audio_size, remain);
        }
    }
    return true;
}

static bool video_to_tensor(SwsContext* sws, AVFrame* frame) {
    const static int width = chobits::video_width * 3;
    static std::vector<uint8_t> video_buffer(av_image_get_buffer_size(video_pix_format, chobits::video_width, chobits::video_height, 1));
    uint8_t* buffer = video_buffer.data();
    const int height = sws_scale(sws, (const uint8_t* const *) frame->data, frame->linesize, 0, frame->height, &buffer, &width);
    if(height < 0 || chobits::video_height != height) {
        std::printf("视频重采样失败：%d\n", height);
        return false;
    }
    chobits::player::play_video(buffer, width);
    if(chobits::train) {
        bool insert = false;
        std::unique_lock<std::mutex> lock(dataset.mutex);
        if(dataset.audio.size() >= dataset.video.size()) {
            if(dataset.video.size() >= dataset.cache_size) {
                if(dataset.discard) {
                    // std::printf("丢弃视频数据：%" PRIu64 "\n", dataset.video.size());
                } else {
                    dataset.condition.wait(lock, []() {
                        return !(chobits::running && dataset.video.size() > dataset.cache_size);
                    });
                    insert = true;
                }
            } else {
                insert = true;
            }
            if(insert) {
                auto tensor = torch::from_blob(
                    buffer,
                    { chobits::video_height, chobits::video_width, 3 },
                    torch::kUInt8
                ).to(torch::kFloat32).div(255.0).mul(2.0).sub(1.0).permute({ 2, 0, 1 }).contiguous();
                dataset.video.push_back(tensor);
            }
            dataset.condition.notify_all();
        }
    }
    return true;
}

/**
 * 短时傅里叶变换
 * 
 * 201 = win_size / 2 + 1
 * 480 = 7 | 4800 = 61 | 48000 = 601
 * [2(相位, 角度), 201, 601]
 * 
 * @param pcm_data PCM数据
 * @param pcm_size PCM长度
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return 张量
 */
static at::Tensor pcm_stft(short* pcm_data, int pcm_size, int n_fft, int hop_size, int win_size) {
    static auto wind = torch::hann_window(win_size);
    auto data = torch::from_blob(pcm_data, { 1, pcm_size }, torch::kShort).to(torch::kFloat32);
    auto com  = torch::stft(data, n_fft, hop_size, win_size, wind, true, "reflect", false, std::nullopt, true);
    #ifdef CHOBITS_NORMALIZATION
    auto mag  = torch::log10(torch::abs(com) + 1e-4) / 4;
    auto pha  = torch::angle(com) / std::numbers::pi;
    #else
    auto mag  = torch::log10(torch::abs(com) + 1e-4);
    auto pha  = torch::angle(com);
    #endif
    return torch::concat({ mag, pha });
}

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
static std::vector<short> pcm_istft(const at::Tensor& tensor, int n_fft, int hop_size, int win_size) {
    static auto wind = torch::hann_window(win_size);
    #ifdef CHOBITS_NORMALIZATION
    auto mag  = torch::pow(10, tensor[0] * 4) - 1e-4;
    auto pha  = tensor[1] * std::numbers::pi;
    #else
    auto mag  = torch::pow(10, tensor[0]) - 1e-4;
    auto pha  = tensor[1];
    #endif
    auto com  = torch::polar(mag, pha);
    auto ret  = torch::istft(com.unsqueeze(0), n_fft, hop_size, win_size, wind, true);
    float* data = reinterpret_cast<float*>(ret.data_ptr());
    std::vector<short> pcm;
    pcm.resize(ret.sizes()[1]);
    std::copy_n(data, pcm.size(), pcm.data());
    return pcm;
}
