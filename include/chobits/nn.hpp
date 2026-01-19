/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/chobits
 * github: https://github.com/acgist/chobits
 * 
 * 神经网络
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef CHOBITS_NN_HPP
#define CHOBITS_NN_HPP

#include "chobits.hpp"

#include "torch/nn.h"

using layer_act = torch::nn::SiLU;

#ifndef torch_act
#define torch_act torch::silu
#endif

using shp = std::vector<int64_t>;

namespace chobits::nn {

/**
 * Pad
 */
class PadBlockImpl : public torch::nn::Module {

private:
    std::vector<int64_t> pad;

public:
    PadBlockImpl(const shp pad) : pad(pad) {
    }
    ~PadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(this->pad).mode(torch::kReplicate));
    }

};

TORCH_MODULE(PadBlock);

/**
 * 1D残差网络
 */
class ResNet1dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential cv1{ nullptr };
    torch::nn::Sequential cv2{ nullptr };
    torch::nn::Sequential cv3{ nullptr };

public:
    ResNet1dBlockImpl(
        const int in_channels,
        const int out_channels,
        const int features,
        const int stride   = 1,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        this->cv1 = this->register_module("cv1", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ features })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel).padding(padding).dilation(dilation).bias(false).stride(stride)),
            layer_act()
        ));
        this->cv2 = this->register_module("cv2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
        this->cv3 = this->register_module("cv3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
    }
    ~ResNet1dBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto left  = this->cv1->forward(input);
        auto right = this->cv2->forward(left)  + left;
        return       this->cv3->forward(right) + left;
    }

};

TORCH_MODULE(ResNet1dBlock);

/**
 * 2D残差网络
 */
class ResNet2dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential cv1{ nullptr };
    torch::nn::Sequential cv2{ nullptr };
    torch::nn::Sequential cv3{ nullptr };

public:
    ResNet2dBlockImpl(
        const int in_channels,
        const int out_channels,
        const shp shape,
        const shp stride   = std::vector<int64_t>{ 1, 1 },
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 1, 1 },
        const shp dilation = std::vector<int64_t>{ 1, 1 }
    ) {
        this->cv1 = this->register_module("cv1", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel).padding(padding).dilation(dilation).bias(false).stride(stride)),
            layer_act()
        ));
        this->cv2 = this->register_module("cv2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
        this->cv3 = this->register_module("cv3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
    }
    ~ResNet2dBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto left  = this->cv1->forward(input);
        auto right = this->cv2->forward(left)  + left;
        return       this->cv3->forward(right) + left;
    }

};

TORCH_MODULE(ResNet2dBlock);

/**
 * GRU
 */
class GRUBlockImpl : public torch::nn::Module {

private:
    torch::nn::GRU        gru { nullptr };
    torch::nn::Sequential proj{ nullptr };
    torch::nn::Sequential out { nullptr };

public:
    GRUBlockImpl(
        const int input_size,
        const int hidden_size,
        const int time,
        const int num_layers = 1
    ) {
        this->gru = this->register_module("gru", torch::nn::GRU(
            torch::nn::GRUOptions(input_size, hidden_size).num_layers(num_layers).bias(false).batch_first(true)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size).bias(false))
        ));
        this->out = this->register_module("out", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock(time, time, input_size + hidden_size, 2)
        ));
    }
    ~GRUBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto [ output, _ ] = this->gru->forward(input);
        output = this->proj->forward(output);
        return this->out->forward(torch::cat({ input, output }, -1));
    }

};

TORCH_MODULE(GRUBlock);

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential         q   { nullptr };
    torch::nn::Sequential         k   { nullptr };
    torch::nn::Sequential         v   { nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };
    torch::nn::Sequential         proj{ nullptr };
    torch::nn::Sequential         out { nullptr };

public:
    AttentionBlockImpl(
        const int q_dim,
        const int k_dim,
        const int v_dim,
        const int o_dim,
        const int channel   = 256,
        const int num_heads = 8
    ) {
        this->q = this->register_module("q", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(q_dim, o_dim).bias(false))
        ));
        this->k = this->register_module("k", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(k_dim, o_dim).bias(false))
        ));
        this->v = this->register_module("v", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(v_dim, o_dim).bias(false))
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(o_dim, num_heads).bias(false)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(o_dim, o_dim).bias(false))
        ));
        this->out = this->register_module("out", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock(channel, channel, q_dim + o_dim, 2)
        ));
    }
    ~AttentionBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
        auto q = this->q->forward(query.permute({ 1, 0, 2 }));
        auto k = this->k->forward(key  .permute({ 1, 0, 2 }));
        auto v = this->v->forward(value.permute({ 1, 0, 2 }));
        auto [ o, _ ] = this->attn->forward(q, k, v);
        o = o.permute({ 1, 0, 2 });
        o = this->proj->forward(o);
        return this->out->forward(torch::cat({ query, o }, -1));
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };
    torch::nn::Sequential gru { nullptr };
    torch::nn::Sequential conv{ nullptr };

public:
    AudioHeadBlockImpl(
        // TODO: 试试空洞卷积
        const int kernel    = 5,
        const int padding   = 2,
        const int dilation  = 1,
        const int kernel_   = 3,
        const int padding_  = 1,
        const int dilation_ = 1
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock( 1,   4, 800, 4, kernel, padding, dilation),
            chobits::nn::ResNet1dBlock( 4,  16, 200, 4, kernel, padding, dilation),
            chobits::nn::ResNet1dBlock(16,  64,  50, 4, kernel, padding, dilation),
            chobits::nn::ResNet1dBlock(64, 256,  13, 4, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1))
        ));
        this->gru = this->register_module("gru", torch::nn::Sequential(
            chobits::nn::GRUBlock(1024, 1024, 10)
        ));
        this->conv = this->register_module("conv", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock( 10,  64, 1024, 2, kernel_, padding_, dilation_),
            chobits::nn::ResNet1dBlock( 64, 256,  512, 2, kernel_, padding_, dilation_),
            chobits::nn::ResNet1dBlock(256, 256,  256, 1, kernel_, padding_, dilation_)
        ));
    }
    ~AudioHeadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto out = this->head->forward(input.view({ -1, 1, input.size(-1) }));
             out = this->gru->forward(out.view({ input.size(0), input.size(1), -1 }));
             out = this->conv->forward(out);
        return out;
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };
    torch::nn::Sequential gru { nullptr };
    torch::nn::Sequential conv{ nullptr };

public:
    VideoHeadBlockImpl(
        // TODO: 试试空洞卷积
        const shp kernel    = std::vector<int64_t>{ 5, 5 },
        const shp padding   = std::vector<int64_t>{ 2, 2 },
        const shp dilation  = std::vector<int64_t>{ 1, 1 },
        const int kernel_   = 3,
        const int padding_  = 1,
        const int dilation_ = 1
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet2dBlock( 1,   4, std::vector<int64_t>{ 360, 640 }, std::vector<int64_t>{ 4, 4 }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock( 4,  16, std::vector<int64_t>{  90, 160 }, std::vector<int64_t>{ 4, 4 }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(16,  64, std::vector<int64_t>{  23,  40 }, std::vector<int64_t>{ 4, 4 }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(64, 256, std::vector<int64_t>{   6,  10 }, std::vector<int64_t>{ 4, 4 }, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1))
        ));
        this->gru = this->register_module("gru", torch::nn::Sequential(
            chobits::nn::GRUBlock(1536, 1536, 10)
        ));
        this->conv = this->register_module("conv", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock( 10,  64, 1536, 2, kernel_, padding_, dilation_),
            chobits::nn::ResNet1dBlock( 64, 256,  768, 1, kernel_, padding_, dilation_),
            chobits::nn::ResNet1dBlock(256, 256,  768, 1, kernel_, padding_, dilation_)
        ));
    }
    ~VideoHeadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto out = this->head->forward(input.view({ -1, 1, input.size(2), input.size(3) }));
             out = this->gru->forward(out.view({ input.size(0), input.size(1), -1 }));
             out = this->conv->forward(out);
        return out;
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 图片输入
 */
class ImageHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    ImageHeadBlockImpl(
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 1, 1 },
        const shp dilation = std::vector<int64_t>{ 1, 1 }
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet2dBlock(  3,  16, std::vector<int64_t>{ 360, 640 }, std::vector<int64_t>{ 2, 2 }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock( 16,  64, std::vector<int64_t>{ 180, 320 }, std::vector<int64_t>{ 2, 2 }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock( 64, 128, std::vector<int64_t>{  90, 160 }, std::vector<int64_t>{ 2, 2 }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(128, 256, std::vector<int64_t>{  45,  80 }, std::vector<int64_t>{ 2, 2 }, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            chobits::nn::ResNet1dBlock(256, 256, 920)
        ));
    }
    ~ImageHeadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->head->forward(input);
    }

};

TORCH_MODULE(ImageHeadBlock);

/**
 * 媒体混合
 */
class MediaMixerBlockImpl : public torch::nn::Module {

private:
    chobits::nn::AttentionBlock audio_attn{ nullptr };
    chobits::nn::AttentionBlock video_attn{ nullptr };
    chobits::nn::AttentionBlock image_attn{ nullptr };
    chobits::nn::AttentionBlock mixer_attn{ nullptr };

public:
    MediaMixerBlockImpl(
        const int audio_in = 256,
        const int video_in = 768,
        const int image_in = 920
    ) {
        const int mixer_in = audio_in + video_in;
        this->audio_attn = this->register_module("audio_attn", chobits::nn::AttentionBlock(audio_in, video_in, video_in, audio_in));
        this->video_attn = this->register_module("video_attn", chobits::nn::AttentionBlock(video_in, audio_in, audio_in, video_in));
        this->image_attn = this->register_module("image_attn", chobits::nn::AttentionBlock(mixer_in, image_in, image_in, mixer_in));
        this->mixer_attn = this->register_module("mixer_attn", chobits::nn::AttentionBlock(mixer_in, mixer_in, mixer_in, mixer_in));
    }
    ~MediaMixerBlockImpl() {
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video, const torch::Tensor& image) {
        auto audio_o = this->audio_attn->forward(audio, video, video);
        auto video_o = this->video_attn->forward(video, audio, audio);
        auto muxer_o = torch::cat({ audio_o, video_o }, -1);
        auto mixer_o = this->image_attn->forward(muxer_o, image, image);
        return         this->mixer_attn->forward(mixer_o, mixer_o, mixer_o);
    }

};

TORCH_MODULE(MediaMixerBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential tail{ nullptr };

public:
    AudioTailBlockImpl(
        const int in_features  = 1024,
        const int out_features = 800,
        const shp channel      = std::vector<int64_t>{ 256, 64, 16, 4, 1 }
    ) {
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], 3)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], 3)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], 3)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[4], 3)),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            layer_act(),
            torch::nn::Linear(in_features - 2 * 4, out_features)
        ));
    }
    ~AudioTailBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return torch::tanh(this->tail->forward(input));
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
