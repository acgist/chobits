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

#include "torch/nn.h"

#include "chobits.hpp"

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
        const int in         = 128,
        const int out        = 128,
        const int num_layers = 1
    ) {
        this->gru = this->register_module("gru", torch::nn::GRU(
            torch::nn::GRUOptions(in, out).num_layers(num_layers).bias(false).batch_first(true).bidirectional(false)
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
        auto [ output, hn ] = this->gru->forward(input, this->h0);
        return output;
    }

};

TORCH_MODULE(GRUBlock);

/**
 * 残差网络
 */
class ResNetBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };

public:
    ResNetBlockImpl(
        const int in_channel,
        const int out_channel,
        const int kernel     = 3,
        const int padding    = 1,
        const int num_groups = 16
    ) {
        if(in_channel == out_channel) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channel, out_channel, kernel).padding(padding).bias(false))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channel, out_channel, kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channel, out_channel, kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channel)),
            torch::nn::SiLU()
        ));
    }
    ~ResNetBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output =          this->conv_1->forward(input);
             output = output + this->conv_2->forward(output);
        return output;
    }

};

TORCH_MODULE(ResNetBlock);

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential         qkv { nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };
    torch::nn::Sequential         proj{ nullptr };

public:
    AttentionBlockImpl(
        const int   seq_len    = 800,
        const int   emb_dim    = 128,
        const int   num_heads  = 4,
        const float dropout    = 0.1,
        const int   kernel     = 1,
        const int   padding    = 0,
        const int   num_groups = 16
    ) {
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(seq_len, seq_len * 3, kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, seq_len * 3)),
            torch::nn::SiLU()
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(emb_dim, num_heads).bias(false).dropout(dropout)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(seq_len, seq_len, kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, seq_len)),
            torch::nn::SiLU()
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("qkv");
        this->unregister_module("proj");
        this->unregister_module("attn");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        // N C L = N S L => S N L
        auto qkv = this->qkv->forward(input).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = this->attn->forward(q, k, v);
        h = h.permute({ 1, 0, 2 });
        h = this->proj->forward(h);
        return input + h;
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential embedding{ nullptr };

public:
    AudioHeadBlockImpl(
        const int in         = 1,
        const int out        = 128,
        const int channel    = 800,
        const int num_groups = 16,
        const float dropout  = 0.1
    ) {
        this->embedding = this->register_module("embedding", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::SiLU(),
            chobits::nn::ResNetBlock(channel, channel),
            torch::nn::Linear(torch::nn::LinearOptions(out, out).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::SiLU(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("embedding");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->embedding->forward(input);
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv     { nullptr };
    torch::nn::Sequential embedding{ nullptr };

public:
    VideoHeadBlockImpl(
        std::vector<int> channel,
        std::vector<int> pool,
        const int in         = 128,
        const int kernel     = 3,
        const int padding    = 1,
        const int num_groups = 10,
        const float dropout  = 0.1
    ) {
        this->conv = this->register_module("conv", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[1])),
            torch::nn::SiLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[0], pool[1] })),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[2])),
            torch::nn::SiLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[2], pool[3] })),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3])),
            torch::nn::SiLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[4], pool[5] }))
        ));
        this->embedding = this->register_module("embedding", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel[3], channel[3]),
            torch::nn::Linear(torch::nn::LinearOptions(in, in).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3])),
            torch::nn::SiLU(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("conv");
        this->unregister_module("embedding");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv->forward(input);
        return this->embedding->forward(output.flatten(2));
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 音频合并
 */
class AudioMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential video{ nullptr };
    torch::nn::Sequential mixer{ nullptr };

public:
    AudioMixBlockImpl(
        const int in      = 128,
        const int channel = 800
    ) {
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in)
        ));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel),
            chobits::nn::AttentionBlock(channel, in),
            torch::nn::SiLU()
        ));
    }
    ~AudioMixBlockImpl() {
        this->unregister_module("video");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        return this->mixer->forward(audio + this->video->forward(video));
    }

};

TORCH_MODULE(AudioMixBlock);

/**
 * 视频合并
 */
class VideoMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential audio{ nullptr };
    torch::nn::Sequential mixer{ nullptr };

public:
    VideoMixBlockImpl(
        const int in      = 128,
        const int channel = 800
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in)
        ));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel),
            chobits::nn::AttentionBlock(channel, in),
            torch::nn::SiLU()
        ));
    }
    ~VideoMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        return this->mixer->forward(video + this->audio->forward(audio));
    }

};

TORCH_MODULE(VideoMixBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential audio{ nullptr };
    torch::nn::Sequential video{ nullptr };
    torch::nn::Sequential mixer{ nullptr };
    AudioMixBlock audio_mix{ nullptr };
    VideoMixBlock video_mix{ nullptr };

public:
    MediaMixBlockImpl(
        const int in      = 128,
        const int channel = 800
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in)
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in)
        ));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in * 2),
            chobits::nn::ResNetBlock(channel, channel),
            chobits::nn::GRUBlock(in * 2, in),
            chobits::nn::ResNetBlock(channel, channel),
            chobits::nn::AttentionBlock(channel, in)
        ));
        this->audio_mix = this->register_module("audio_mix", AudioMixBlock());
        this->video_mix = this->register_module("video_mix", VideoMixBlock());
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("mixer");
        this->unregister_module("audio_mix");
        this->unregister_module("video_mix");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        auto audio_mix = this->audio_mix->forward(audio_out, video_out);
        auto video_mix = this->video_mix->forward(audio_out, video_out);
        return this->mixer->forward(torch::concat({ audio_out + audio_mix, video_out + video_mix }, -1));
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential tail{ nullptr };

public:
    AudioTailBlockImpl(
        const int in  = 128,
        const int out = 1
    ) {
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(in, in)),
            torch::nn::SiLU(),
            torch::nn::Linear(torch::nn::LinearOptions(in, out))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("tail");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        #if CHOBITS_NORM == 0
        return torch::sigmoid(this->tail->forward(input));
        #elif CHOBITS_NORM == 1
        return torch::tanh(this->tail->forward(input));
        #else
        return this->tail->forward(input);
        #endif
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
