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
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out_channel)),
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
        const int   num_heads  = 8,
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
            torch::nn::MultiheadAttentionOptions(emb_dim, num_heads).bias(false)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(seq_len, seq_len, kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, seq_len)),
            torch::nn::SiLU()
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("qkv");
        this->unregister_module("attn");
        this->unregister_module("proj");
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
        const int num_groups = 16
    ) {
        this->embedding = this->register_module("embedding", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::SiLU(),
            // -
            chobits::nn::ResNetBlock(channel, channel),
            // -
            torch::nn::Linear(torch::nn::LinearOptions(out, out).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::SiLU()
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
    torch::nn::Sequential embedding{ nullptr };

public:
    VideoHeadBlockImpl(
        std::vector<int> channel = std::vector<int>{ 1, 10, 100, 800  },
        std::vector<int> pool    = std::vector<int>{ 5, 5, 3, 4, 3, 2 },
        const int in             = 128,
        const int kernel         = 3,
        const int padding        = 1,
        const int num_groups     = 10
    ) {
        this->embedding = this->register_module("embedding", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel[1])),
            torch::nn::SiLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[0], pool[1] })),
            // -
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[2])),
            torch::nn::SiLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[2], pool[3] })),
            // -
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3])),
            torch::nn::SiLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[4], pool[5] })),
            // -
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            // -
            chobits::nn::ResNetBlock(channel[3], channel[3]),
            // -
            torch::nn::Linear(torch::nn::LinearOptions(in, in).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3])),
            torch::nn::SiLU()
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("embedding");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->embedding->forward(input);
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 媒体混合
 */
class MediaProbBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential media_1{ nullptr };
    torch::nn::Sequential media_2{ nullptr };
    torch::nn::Sequential mixer  { nullptr };
    torch::nn::Sequential mprob  { nullptr };

public:
    MediaProbBlockImpl(
        const int in      = 128,
        const int channel = 800
    ) {
        this->media_1 = this->register_module("media_1", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel)
        ));
        this->media_2 = this->register_module("media_2", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel),
            torch::nn::Linear(torch::nn::LinearOptions(in, 1))
        ));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in),
            chobits::nn::ResNetBlock(channel, channel)
        ));
        this->mprob = this->register_module("mprob", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in),
            torch::nn::Linear(torch::nn::LinearOptions(in, 1)),
            torch::nn::Sigmoid()
        ));
    }
    ~MediaProbBlockImpl() {
        this->unregister_module("media_1");
        this->unregister_module("media_2");
        this->unregister_module("mixer");
        this->unregister_module("mprob");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& media_1, const torch::Tensor& media_2) {
        auto media_1_out = this->media_1->forward(media_1);
        auto media_2_out = this->media_2->forward(media_2);
        auto media_mix   = media_1_out + media_2_out;
             media_mix   = this->mixer->forward(media_mix);
             media_mix   = media_mix * this->mprob->forward(media_mix);
        return media_mix;
    }

};

TORCH_MODULE(MediaProbBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential       audio{ nullptr };
    torch::nn::Sequential       video{ nullptr };
    chobits::nn::MediaProbBlock aprob{ nullptr };
    chobits::nn::MediaProbBlock vprob{ nullptr };
    torch::nn::Sequential       mixer{ nullptr };

public:
    MediaMixBlockImpl(
        const int in      = 128,
        const int channel = 800
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::GRUBlock(in, in)
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::GRUBlock(in, in)
        ));
        this->aprob = this->register_module("aprob", chobits::nn::MediaProbBlock());
        this->vprob = this->register_module("vprob", chobits::nn::MediaProbBlock());
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, in),
            chobits::nn::ResNetBlock(channel, channel),
            chobits::nn::GRUBlock(in, in),
            chobits::nn::ResNetBlock(channel, channel),
            chobits::nn::AttentionBlock(channel, in)
        ));
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("aprob");
        this->unregister_module("vprob");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        auto audio_mix = this->aprob->forward(audio_out, video_out);
        auto video_mix = this->vprob->forward(video_out, audio_out);
        auto media_mix = audio_mix + video_mix;
        return this->mixer->forward(media_mix);
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
