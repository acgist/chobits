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
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential         qkv { nullptr };
    torch::nn::Sequential         proj{ nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };

public:
    AttentionBlockImpl(
        const int channel,
        const int embed_dim,
        const int num_heads  = 4,
        const int num_groups = 16,
        const float dropout  = 0.3
    ) {
        const int qkv_channel = channel * 3;
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel, qkv_channel, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, qkv_channel)),
            torch::nn::SiLU()
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout))
        );
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel, channel, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
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
 * 媒体输入
 */
class MediaHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d conv_1{ nullptr };
    torch::nn::Conv2d conv_2{ nullptr };
    torch::nn::Conv2d conv_3{ nullptr };

public:
    MediaHeadBlockImpl(
        std::vector<int> channel,
        std::vector<int> kernel,
        std::vector<int> stride,
        const int embed_dim
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel[0]).stride(stride[0]).bias(false)));
        this->conv_2 = this->register_module("conv_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel[1]).stride(stride[1]).bias(false)));
        this->conv_3 = this->register_module("conv_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel[2]).stride(stride[2]).bias(false)));
    }
    ~MediaHeadBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = torch::sigmoid(output) * output;
             output = this->conv_2->forward(output);
             output = torch::sigmoid(output) * output;
             output = this->conv_3->forward(output);
             output = torch::sigmoid(output) * output;
        return output;
    }

};

TORCH_MODULE(MediaHeadBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d     audio{ nullptr };
    torch::nn::Conv2d     video{ nullptr };
    torch::nn::Linear     map_h{ nullptr };
    torch::nn::Linear     map_w{ nullptr };
    torch::nn::Sequential prob { nullptr };

public:
    MediaMixBlockImpl(
        int in,      int out,
        int audio_h, int audio_w,
        int video_h, int video_w,
        const int num_groups = 16
    ) {
        this->audio = this->register_module("audio", torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)));
        this->video = this->register_module("video", torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)));
        this->map_h = this->register_module("map_h", torch::nn::Linear(torch::nn::LinearOptions(video_h, audio_h).bias(false)));
        this->map_w = this->register_module("map_w", torch::nn::Linear(torch::nn::LinearOptions(video_w, audio_w).bias(false)));
        this->prob  = this->register_module("prob", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out)),
            torch::nn::SiLU(),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            torch::nn::Linear(torch::nn::LinearOptions(out * audio_h * audio_w, 1).bias(false)),
            torch::nn::Sigmoid()
        ));
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("map_h");
        this->unregister_module("map_w");
        this->unregister_module("prob");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
             video_out = this->map_w->forward(video_out);
             video_out = this->map_h->forward(video_out.permute({ 0, 1, 3, 2 })).permute({ 0, 1, 3, 2 });
        return audio_out + video_out * this->prob->forward(audio_out + video_out).unsqueeze(-1).unsqueeze(-1);
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::ConvTranspose2d conv_1{ nullptr };
    torch::nn::ConvTranspose2d conv_2{ nullptr };
    torch::nn::ConvTranspose2d conv_3{ nullptr };

public:
    AudioTailBlockImpl(
        std::vector<int> channel,
        std::vector<int> kernel,
        std::vector<int> stride,
        const int embed_dim
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channel[0], channel[1], kernel[0]).stride(stride[0]).bias(false).padding(1)));
        this->conv_2 = this->register_module("conv_2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channel[1], channel[2], kernel[1]).stride(stride[1]).bias(false)));
        this->conv_3 = this->register_module("conv_3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channel[2], channel[3], kernel[2]).stride(stride[2]).bias(false)));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = torch::sigmoid(output) * output;
             output = this->conv_2->forward(output);
             output = torch::sigmoid(output) * output;
             output = this->conv_3->forward(output);
             output = torch::sigmoid(output) * output;
        return output;
    }

};

TORCH_MODULE(AudioTailBlock);

/**
 * 历史记忆
 */
class MemoryBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential memory{ nullptr };

public:
    MemoryBlockImpl(
        const int channel,
        const int w,
        const int h,
        int num_groups = 16
    ) {
        this->memory = this->register_module("memory", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel, channel, 3).padding(1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::SiLU(),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            torch::nn::Linear(torch::nn::LinearOptions(channel * w * h, 1).bias(false)),
            torch::nn::Sigmoid()
        ));
    }
    ~MemoryBlockImpl() {
        this->unregister_module("memory");
    }

public:
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& memory) {
        return input + this->memory->forward(input).unsqueeze(-1).unsqueeze(-1) * memory;
    }

};

TORCH_MODULE(MemoryBlock);    

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
