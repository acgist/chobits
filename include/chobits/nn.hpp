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
 * 残差网络
 */
class ResNetBlock1dImpl : public torch::nn::Module {

private:
    bool use_stride;
    torch::nn::Sequential cv1{ nullptr };
    torch::nn::Sequential cv2{ nullptr };
    torch::nn::Sequential cv3{ nullptr };

public:
    ResNetBlock1dImpl(
        const int in_channels,
        const int out_channels,
        const int shape,
        const int stride   = 0,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        this->use_stride = stride > 0;
        this->cv1 = this->register_module("cv1", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ shape })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel).padding(padding).dilation(dilation).bias(false)),
            layer_act()
        ));
        this->cv2 = this->register_module("cv2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
        if(this->use_stride) {
            this->cv3 = this->register_module("cv3", torch::nn::Sequential(
                torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation).stride(stride)),
                layer_act()
            ));
        } else {
            this->cv3 = this->register_module("cv3", torch::nn::Sequential(
                torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation)),
                layer_act()
            ));
        }
    }
    ~ResNetBlock1dImpl() {
        this->unregister_module("cv1");
        this->unregister_module("cv2");
        this->unregister_module("cv3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(this->use_stride) {
            auto left = this->cv1->forward(input);
                 left = this->cv2->forward(left) + left;
            return      this->cv3->forward(left);
        } else {
            auto left  = this->cv1->forward(input);
            auto right = this->cv2->forward(left)  + left;
            return       this->cv3->forward(right) + left;
        }
    }

};

TORCH_MODULE(ResNetBlock1d);

/**
 * 2D残差网络
 */
class ResNet2dBlockImpl : public torch::nn::Module {

private:
    bool use_stride;
    torch::nn::Sequential cv1{ nullptr };
    torch::nn::Sequential cv2{ nullptr };
    torch::nn::Sequential cv3{ nullptr };

public:
    ResNet2dBlockImpl(
        const int in_channels,
        const int out_channels,
        const shp shape,
        const shp stride   = std::vector<int64_t>{      },
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 2, 2 },
        const shp dilation = std::vector<int64_t>{ 2, 2 }
    ) {
        this->use_stride = !stride.empty();
        if(this->use_stride) {
            this->cv1 = this->register_module("cv1", torch::nn::Sequential(
                torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel).padding(padding).dilation(dilation).bias(false)),
                layer_act(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel).padding(padding).dilation(dilation).stride(stride)),
                layer_act()
            ));
        } else {
            this->cv1 = this->register_module("cv1", torch::nn::Sequential(
                torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel).padding(padding).dilation(dilation).bias(false)),
                layer_act()
            ));
        }
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
        this->unregister_module("cv1");
        this->unregister_module("cv2");
        this->unregister_module("cv3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(this->use_stride) {
            auto left = this->cv1->forward(input);
                 left = this->cv2->forward(left) + left;
            return      this->cv3->forward(left);
        } else {
            auto left  = this->cv1->forward(input);
            auto right = this->cv2->forward(left)  + left;
            return       this->cv3->forward(right) + left;
        }
    }

};

TORCH_MODULE(ResNet2dBlock);

/**
 * GRU
 */
class GRUBlockImpl : public torch::nn::Module {

private:
    torch::Tensor         h0 { nullptr };
    torch::nn::GRU        gru{ nullptr };
    torch::nn::Sequential out{ nullptr };

public:
    GRUBlockImpl(
        const int input_size,
        const int hidden_size,
        const int channel    = 256,
        const int num_layers = 1
    ) {
        this->gru = this->register_module("gru", torch::nn::GRU(
            torch::nn::GRUOptions(input_size, hidden_size).num_layers(num_layers).bias(false).batch_first(true)
        ));
        this->out = this->register_module("out", torch::nn::Sequential(
            chobits::nn::ResNetBlock1d(channel * 2, channel, hidden_size)
        ));
    }
    ~GRUBlockImpl() {
        this->unregister_module("gru");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(!this->h0.defined()) {
            this->h0 = torch::zeros({
                this->gru->options.num_layers(),
                input.size(0),
                this->gru->options.hidden_size()
            }).to(input.device());
        }
        auto [ output, _ ] = this->gru->forward(input, this->h0);
        return this->out->forward(torch::concat({ input, output }, 1));
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
            chobits::nn::ResNetBlock1d(channel * 2, channel, o_dim)
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("q");
        this->unregister_module("k");
        this->unregister_module("v");
        this->unregister_module("attn");
        this->unregister_module("proj");
        this->unregister_module("out");
    }

public:
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
        auto q = this->q->forward(query.permute({ 1, 0, 2 }));
        auto k = this->k->forward(key  .permute({ 1, 0, 2 }));
        auto v = this->v->forward(value.permute({ 1, 0, 2 }));
        auto [ o, _ ] = this->attn->forward(q, k, v);
        o = o.permute({ 1, 0, 2 });
        o = this->proj->forward(o);
        return this->out->forward(torch::concat({ query, o }, 1));
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    AudioHeadBlockImpl(
        const int in_len   = 800,
        const shp channel  = std::vector<int64_t>{ 10, 64, 128, 256 },
        const int pool     = 2,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNetBlock1d(channel[0], channel[1], in_len       , pool, kernel, padding, dilation),
            chobits::nn::ResNetBlock1d(channel[1], channel[1], in_len / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock1d(channel[1], channel[2], in_len / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock1d(channel[2], channel[2], in_len / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock1d(channel[2], channel[3], in_len / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock1d(channel[3], channel[3], in_len / pool,    0, kernel, padding, dilation)
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("head");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->head->forward(input);
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    VideoHeadBlockImpl(
        const int in_channels,
        const int out_len   = 920,
        const int height    = 360,
        const int width     = 640,
        const shp channel   = std::vector<int64_t>{ 32, 64, 128, 256 },
        const shp pool      = std::vector<int64_t>{ 2, 2 },
        const shp kernel    = std::vector<int64_t>{ 3, 3 },
        const shp padding   = std::vector<int64_t>{ 2, 2 },
        const shp dilation  = std::vector<int64_t>{ 2, 2 },
        const shp kernel_   = std::vector<int64_t>{ 3, 3 },
        const shp padding_  = std::vector<int64_t>{ 1, 1 },
        const shp dilation_ = std::vector<int64_t>{ 1, 1 }
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet2dBlock(in_channels, channel[0], std::vector<int64_t>{ int64_t(height / std::pow(pool[0], 0)) + 0, int64_t(width / std::pow(pool[0], 0)) }, pool,  kernel,  padding,  dilation ),
            chobits::nn::ResNet2dBlock(channel[0],  channel[1], std::vector<int64_t>{ int64_t(height / std::pow(pool[0], 1)) + 0, int64_t(width / std::pow(pool[0], 1)) }, pool,  kernel,  padding,  dilation ),
            chobits::nn::ResNet2dBlock(channel[1],  channel[2], std::vector<int64_t>{ int64_t(height / std::pow(pool[0], 2)) + 0, int64_t(width / std::pow(pool[0], 2)) }, pool,  kernel_, padding_, dilation_),
            chobits::nn::ResNet2dBlock(channel[2],  channel[3], std::vector<int64_t>{ int64_t(height / std::pow(pool[0], 3)) + 0, int64_t(width / std::pow(pool[0], 3)) }, pool,  kernel_, padding_, dilation_),
            chobits::nn::ResNet2dBlock(channel[3],  channel[3], std::vector<int64_t>{ int64_t(height / std::pow(pool[0], 4)) + 1, int64_t(width / std::pow(pool[0], 4)) }, shp{}, kernel_, padding_, dilation_),
            chobits::nn::ResNet2dBlock(channel[3],  channel[3], std::vector<int64_t>{ int64_t(height / std::pow(pool[0], 4)) + 1, int64_t(width / std::pow(pool[0], 4)) }, shp{}, kernel_, padding_, dilation_),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            chobits::nn::ResNetBlock1d(channel[3], channel[3], out_len),
            chobits::nn::ResNetBlock1d(channel[3], channel[3], out_len)
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("head");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->head->forward(input);
    }

};

TORCH_MODULE(VideoHeadBlock);

using ImageHeadBlock = VideoHeadBlock;

/**
 * 媒体混合
 */
class MediaMixerBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential       audio  { nullptr };
    torch::nn::Sequential       video  { nullptr };
    chobits::nn::AttentionBlock a_muxer{ nullptr };
    chobits::nn::AttentionBlock v_muxer{ nullptr };
    chobits::nn::AttentionBlock muxer  { nullptr };
    chobits::nn::AttentionBlock mixer  { nullptr };

public:
    MediaMixerBlockImpl(
        const int audio_in = 400,
        const int image_in = 920,
        const int video_in = 920
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::GRUBlock(audio_in, audio_in)
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::GRUBlock(video_in, video_in)
        ));
        this->a_muxer = this->register_module("a_muxer", chobits::nn::AttentionBlock(audio_in, video_in, audio_in,                       audio_in));
        this->v_muxer = this->register_module("v_muxer", chobits::nn::AttentionBlock(video_in, audio_in, video_in,                       video_in));
        this->muxer   = this->register_module("muxer",   chobits::nn::AttentionBlock(audio_in, video_in, audio_in + video_in + image_in, audio_in));
        this->mixer   = this->register_module("mixer",   chobits::nn::AttentionBlock(audio_in, audio_in, audio_in,                       audio_in));
    }
    ~MediaMixerBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("a_muxer");
        this->unregister_module("v_muxer");
        this->unregister_module("muxer");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& image, const torch::Tensor& video) {
        auto audio_v = this->audio->forward(audio);
        auto video_v = this->video->forward(video);
        auto audio_o = this->a_muxer->forward(audio, video, audio_v);
        auto video_o = this->v_muxer->forward(video, audio, video_v);
        auto muxer_o = this->muxer->forward(audio_o, video_o, torch::concat({ audio_v, video_v, image }, -1));
//      auto audio_o = this->a_muxer->forward(audio_v, video_v, audio_v);
//      auto video_o = this->v_muxer->forward(video_v, audio_v, video_v);
//      auto muxer_o = this->muxer->forward(audio_v, video_v, torch::concat({ audio_o, video_o, image }, -1));
        return         this->mixer->forward(muxer_o, muxer_o, muxer_o);
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
        const int stride   = 2,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1,
        const shp channel  = std::vector<int64_t>{ 256, 64, 16, 4, 1 }
    ) {
        // torch::nn::Upsample
        // torch::nn::ConvTranspose1d
        // L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[0], channel[0], kernel).stride(stride).padding(padding).output_padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], 3).padding(1).dilation(1)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], 3).padding(1).dilation(1)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], 3).padding(1).dilation(1)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[4], 3).padding(1).dilation(1)),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("tail");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return torch::tanh(this->tail->forward(input));
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
