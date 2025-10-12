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

namespace chobits::nn {

/**
 * 媒体输入
 */
class MediaHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    MediaHeadBlockImpl(
        const int num_groups,
        std::vector<int> channel,
        std::vector<int> kernel,
        std::vector<int> stride
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel[0]).stride(stride[0]).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel[1]).stride(stride[1]).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel[2]).stride(stride[2]).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3])),
            torch::nn::SiLU()
        ));
    }
    ~MediaHeadBlockImpl() {
        this->unregister_module("head");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->head->forward(input);
    }

};

TORCH_MODULE(MediaHeadBlock);

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential         qkv { nullptr };
    torch::nn::Sequential         proj{ nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };

public:
    AttentionBlockImpl(
        const int channels,
        const int embed_dim,
        const int num_heads  = 8,
        const int num_groups = 16,
        const float dropout  = 0.3
    ) {
        const int qkv_channels = channels * 3;
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, qkv_channels, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, qkv_channels)),
            torch::nn::SiLU()
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).bias(false).dropout(dropout))
        );
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channels)),
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
        const int N = input.size(0);
        const int C = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        auto qkv = this->qkv->forward(input).reshape({ N, -1, H * W }).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = this->attn->forward(q, k, v);
        h = this->proj->forward(h.permute({ 1, 0, 2 }).reshape({ N, C, H, W }));
        return input + h;
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    bool audio_;
    torch::nn::Sequential audio{ nullptr };
    torch::nn::Sequential video{ nullptr };

public:
    MediaMixBlockImpl(
        int in,      int out,
        int audio_w, int audio_h,
        int video_w, int video_h,
        int out_w,   int out_h
    ) {
        if(audio_w == out_w && audio_h == out_h) {
            this->audio_ = true;
            this->audio = this->register_module("audio", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false))
            ));
        } else {
            this->audio = this->register_module("audio", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)),
                torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
                torch::nn::Linear(torch::nn::LinearOptions(audio_w * audio_h, out_w).bias(false)),
                torch::nn::SiLU()
            ));
        }
        if(video_w == out_w && video_h == out_h) {
            this->audio_ = false;
            this->video = this->register_module("video", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false))
            ));
        } else {
            this->video = this->register_module("video", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)),
                torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
                torch::nn::Linear(torch::nn::LinearOptions(video_w * video_h, out_w).bias(false)),
                torch::nn::SiLU()
            ));
        }
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        return this->audio_ ? (audio_out + audio_out * video_out.unsqueeze(-1)) : (video_out + video_out * audio_out.unsqueeze(-1));
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 记忆概率
 */
class MemoryProbBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential prob{ nullptr };

public:
    MemoryProbBlockImpl(
        const int channel,
        const int w,
        const int h,
        int num_groups = 16
    ) {
        this->prob = this->register_module("prob", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel, channel, 3).padding(1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            torch::nn::Linear(torch::nn::LinearOptions(channel * w * h, 1).bias(false)),
            torch::nn::Sigmoid()
        ));
    }
    ~MemoryProbBlockImpl() {
        this->unregister_module("prob");
    }

public:
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& memory) {
        return input + this->prob->forward(input).unsqueeze(-1).unsqueeze(-1) * memory;
    }

};

TORCH_MODULE(MemoryProbBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential mag{ nullptr };
    torch::nn::Sequential pha{ nullptr };

public:
    AudioTailBlockImpl(
        std::vector<int> embed_dim,
        std::vector<int> num_heads,
        std::vector<int> channels,
        std::vector<int> kernel,
        std::vector<int> stride
    ) {
        this->mag = this->register_module("mag", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channels[0], embed_dim[0], num_heads[0]),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[0], channels[1], kernel[0]).stride(stride[0]).padding(1).bias(false)),
            // chobits::nn::AttentionBlock(channels[1], embed_dim[1], num_heads[1]),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[1], channels[2], kernel[1]).stride(stride[1]).bias(false)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[2], channels[3], kernel[2]).stride(stride[2]).bias(false))
        ));
        this->pha = this->register_module("pha", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channels[0], embed_dim[0], num_heads[0]),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[0], channels[1], kernel[0]).stride(stride[0]).padding(1).bias(false)),
            // chobits::nn::AttentionBlock(channels[1], embed_dim[1], num_heads[1]),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[1], channels[2], kernel[1]).stride(stride[1]).bias(false)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[2], channels[3], kernel[2]).stride(stride[2]).bias(false))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("mag");
        this->unregister_module("pha");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return torch::concat({
            this->mag->forward(input),
            this->pha->forward(input)
        }, 1);
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
