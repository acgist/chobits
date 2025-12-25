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

using layer_act = torch::nn::Tanh;

#ifndef torch_act
#define torch_act torch::tanh
#endif

using shp = std::vector<int64_t>;

namespace chobits::nn {

/**
 * GRU
 */
class GRUBlockImpl : public torch::nn::Module {

private:
    torch::Tensor  h0 { nullptr };
    torch::nn::GRU gru{ nullptr };

public:
    GRUBlockImpl(
        const int in,
        const int out,
        const int num_layers = 1
    ) {
        this->gru = this->register_module("gru", torch::nn::GRU(
            torch::nn::GRUOptions(in, out).num_layers(num_layers).bias(false).batch_first(true)
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
        return torch::concat({ input, output }, 1);
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

public:
    AttentionBlockImpl(
        const int q,
        const int k,
        const int v,
        const int o,
        const int num_heads = 8
    ) {
        this->q = this->register_module("q", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(q, o).bias(false))
        ));
        this->k = this->register_module("k", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(k, o).bias(false))
        ));
        this->v = this->register_module("v", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(v, o).bias(false))
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(o, num_heads).bias(false)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(o, o).bias(false)),
            layer_act()
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("q");
        this->unregister_module("k");
        this->unregister_module("v");
        this->unregister_module("attn");
        this->unregister_module("proj");
    }

public:
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
        auto q = this->q->forward(query.permute({ 1, 0, 2 }));
        auto k = this->k->forward(key  .permute({ 1, 0, 2 }));
        auto v = this->v->forward(value.permute({ 1, 0, 2 }));
        auto [ h, w ] = this->attn->forward(q, k, v);
        h = h.permute({ 1, 0, 2 });
        h = this->proj->forward(h);
        return torch::concat({ query, h }, 1);
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 残差网络
 */
class ResNetBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    ResNetBlockImpl(
        const int in,
        const int out,
        const int shape,
        const int pool     = 0,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        if(pool == 0) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ shape })),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ shape })),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
    }
    ~ResNetBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto left  = this->conv_1->forward(input);
        auto right = this->conv_2->forward(left);
             left  = left + right;
             right = this->conv_3->forward(left);
        return left + right;
    }

};

TORCH_MODULE(ResNetBlock);

/**
 * 3D残差网络
 */
class ResNet3dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    ResNet3dBlockImpl(
        const int in,
        const int out,
        const int ch,
        const shp pool     = std::vector<int64_t>{         },
        const shp kernel   = std::vector<int64_t>{ 1, 3, 3 },
        const shp padding  = std::vector<int64_t>{ 0, 2, 2 },
        const shp dilation = std::vector<int64_t>{ 1, 2, 2 }
    ) {
        if(pool.empty()) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(ch)),
                torch::nn::Conv3d(torch::nn::Conv3dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(ch)),
                torch::nn::Conv3d(torch::nn::Conv3dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(pool))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
    }
    ~ResNet3dBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto left  = this->conv_1->forward(input);
        auto right = this->conv_2->forward(left);
             left  = left + right;
             right = this->conv_3->forward(left);
        return left + right;
    }

};

TORCH_MODULE(ResNet3dBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    AudioHeadBlockImpl(
        const int in       = 800,
        const shp channel  = std::vector<int64_t>{ 10, 64, 128, 256 },
        const int pool     = 2,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel[0], channel[1], in       , pool, kernel, padding, dilation),
            chobits::nn::ResNetBlock(channel[1], channel[1], in / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock(channel[1], channel[2], in / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock(channel[2], channel[2], in / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock(channel[2], channel[3], in / pool,    0, kernel, padding, dilation),
            chobits::nn::ResNetBlock(channel[3], channel[3], in / pool,    0, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))
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
        const shp channel  = std::vector<int64_t>{ 10, 64, 128, 256 },
        const shp pool     = std::vector<int64_t>{ 1, 5, 5, 1, 2, 2, 1, 2, 2 },
        const shp kernel   = std::vector<int64_t>{ 1, 3, 3 },
        const shp padding  = std::vector<int64_t>{ 0, 2, 2 },
        const shp dilation = std::vector<int64_t>{ 1, 2, 2 }
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet3dBlock(channel[0], channel[1], channel[0], std::vector<int64_t>{ pool[0], pool[1], pool[2] }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[1], channel[1], channel[1], std::vector<int64_t>{                           }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[1], channel[2], channel[1], std::vector<int64_t>{ pool[3], pool[4], pool[5] }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[2], channel[2], channel[2], std::vector<int64_t>{                           }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[2], channel[3], channel[2], std::vector<int64_t>{ pool[6], pool[7], pool[8] }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[3], channel[3], channel[3], std::vector<int64_t>{                           }, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))
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

/**
 * 媒体混合
 */
class MediaMixerBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential       audio    { nullptr };
    torch::nn::Sequential       video    { nullptr };
    chobits::nn::AttentionBlock a_muxer  { nullptr };
    chobits::nn::AttentionBlock v_muxer  { nullptr };
    chobits::nn::AttentionBlock o_muxer  { nullptr };
    chobits::nn::AttentionBlock o_mixer  { nullptr };
    torch::nn::Sequential       a_muxer_o{ nullptr };
    torch::nn::Sequential       v_muxer_o{ nullptr };
    torch::nn::Sequential       o_muxer_o{ nullptr };
    torch::nn::Sequential       o_mixer_o{ nullptr };

public:
    MediaMixerBlockImpl(
        const int audio_in = 400,
        const int video_in = 576 * 3,
        const int channel  = 256
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel, audio_in),
            chobits::nn::GRUBlock(audio_in, audio_in),
            chobits::nn::ResNetBlock(channel * 2, channel, audio_in)
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel, video_in),
            chobits::nn::GRUBlock(video_in, video_in),
            chobits::nn::ResNetBlock(channel * 2, channel, video_in)
        ));
        this->a_muxer   = this->register_module("a_muxer",   chobits::nn::AttentionBlock(audio_in, video_in, audio_in,            audio_in));
        this->v_muxer   = this->register_module("v_muxer",   chobits::nn::AttentionBlock(video_in, audio_in, video_in,            video_in));
        this->o_muxer   = this->register_module("o_muxer",   chobits::nn::AttentionBlock(audio_in, video_in, audio_in + video_in, audio_in));
        this->o_mixer   = this->register_module("o_mixer",   chobits::nn::AttentionBlock(audio_in, audio_in, audio_in,            audio_in));
        this->a_muxer_o = this->register_module("a_muxer_o", torch::nn::Sequential(chobits::nn::ResNetBlock(channel * 2, channel, audio_in)));
        this->v_muxer_o = this->register_module("v_muxer_o", torch::nn::Sequential(chobits::nn::ResNetBlock(channel * 2, channel, video_in)));
        this->o_muxer_o = this->register_module("o_muxer_o", torch::nn::Sequential(chobits::nn::ResNetBlock(channel * 2, channel, audio_in)));
        this->o_mixer_o = this->register_module("o_mixer_o", torch::nn::Sequential(chobits::nn::ResNetBlock(channel * 2, channel, audio_in)));
    }
    ~MediaMixerBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("a_muxer");
        this->unregister_module("v_muxer");
        this->unregister_module("o_muxer");
        this->unregister_module("o_mixer");
        this->unregister_module("a_muxer_o");
        this->unregister_module("v_muxer_o");
        this->unregister_module("o_muxer_o");
        this->unregister_module("o_mixer_o");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_v = this->audio->forward(audio);
        auto video_v = this->video->forward(video);
        auto audio_o = this->a_muxer_o->forward(this->a_muxer->forward(audio, video, audio_v));
        auto video_o = this->v_muxer_o->forward(this->v_muxer->forward(video, audio, video_v));
        auto muxer_o = this->o_muxer_o->forward(this->o_muxer->forward(audio_o, video_o, torch::concat({ audio_v, video_v }, -1)));
        return         this->o_mixer_o->forward(this->o_mixer->forward(muxer_o, muxer_o, muxer_o));
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
        const int in      = 400,
        const int out     = 800,
        const shp channel = std::vector<int64_t>{ 256, 64, 16, 4, 1 }
    ) {
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[0], channel[0], 2).stride(2)),
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
