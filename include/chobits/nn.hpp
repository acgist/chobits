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
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, qkv_channel, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, qkv_channel)),
            torch::nn::ReLU()
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, channel, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::ReLU()
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout)));
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
        const int L = input.size(2);
        auto qkv = this->qkv->forward(input).reshape({ N, -1, L }).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = this->attn->forward(q, k, v);
        h = this->proj->forward(h.permute({ 1, 0, 2 }).reshape({ N, C, L }));
        return input + h;
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Conv1d conv_1{ nullptr };
    torch::nn::Conv1d conv_2{ nullptr };
    torch::nn::Conv1d conv_3{ nullptr };

public:
    AudioHeadBlockImpl(
        std::vector<int> channel,
        std::vector<int> kernel,
        std::vector<int> stride,
        std::vector<int> padding
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], kernel[0]).stride(stride[0]).padding(padding[0]).bias(false)));
        this->conv_2 = this->register_module("conv_2", torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], kernel[1]).stride(stride[1]).padding(padding[1]).bias(false)));
        this->conv_3 = this->register_module("conv_3", torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], kernel[2]).stride(stride[2]).padding(padding[2]).bias(false)));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = torch::tanh(output) + output;
             output = this->conv_2->forward(output);
             output = torch::tanh(output) + output;
             output = this->conv_3->forward(output);
             output = torch::tanh(output) + output;
        return output;
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d conv_1{ nullptr };
    torch::nn::Conv2d conv_2{ nullptr };
    torch::nn::Conv2d conv_3{ nullptr };
    torch::nn::Linear linear{ nullptr };

public:
    VideoHeadBlockImpl(
        const int in,
        const int out,
        std::vector<int> channel,
        std::vector<int> kernel,
        std::vector<int> stride,
        std::vector<int> padding
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel[0]).stride(stride[0]).padding(padding[0]).bias(false)));
        this->conv_2 = this->register_module("conv_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel[1]).stride(stride[1]).padding(padding[1]).bias(false)));
        this->conv_3 = this->register_module("conv_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel[2]).stride(stride[2]).padding(padding[2]).bias(false)));
        this->linear = this->register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false)));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
        this->unregister_module("linear");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = torch::relu(output) + output;
             output = this->conv_2->forward(output);
             output = torch::relu(output) + output;
             output = this->conv_3->forward(output);
             output = torch::relu(output) + output;
        return this->linear->forward(output.flatten(2));
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Conv1d     audio{ nullptr };
    torch::nn::Conv1d     video{ nullptr };
    torch::nn::Sequential mixer{ nullptr };

public:
    MediaMixBlockImpl(
        int in,
        int channel,
        int kernel,
        int stride,
        int padding,
        const int num_groups = 16
    ) {
        const int out = in / stride;
        const int out_channel = channel * 2;
        this->audio = this->register_module("audio", torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, out_channel, kernel).stride(stride).padding(padding).bias(false)));
        this->video = this->register_module("video", torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, out_channel, kernel).stride(stride).padding(padding).bias(false)));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(out_channel, out),
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(out_channel, channel, kernel).stride(stride).padding(padding).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::Tanh()
        ));
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        return audio + this->mixer->forward(audio_out + video_out);
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::ConvTranspose1d conv_1{ nullptr };
    torch::nn::ConvTranspose1d conv_2{ nullptr };
    torch::nn::ConvTranspose1d conv_3{ nullptr };

public:
    AudioTailBlockImpl(
        std::vector<int> channel,
        std::vector<int> kernel,
        std::vector<int> stride,
        std::vector<int> padding
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[0], channel[1], kernel[0]).stride(stride[0]).padding(padding[0]).bias(false)));
        this->conv_2 = this->register_module("conv_2", torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[1], channel[2], kernel[1]).stride(stride[1]).padding(padding[1]).bias(false)));
        this->conv_3 = this->register_module("conv_3", torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[2], channel[3], kernel[2]).stride(stride[2]).padding(padding[2]).bias(false)));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = torch::tanh(output) + output;
             output = this->conv_2->forward(output);
             output = torch::tanh(output) + output;
             output = this->conv_3->forward(output);
             output = torch::tanh(output);
            //  output = torch::tanh(output) + output;
        return output;
    }

};

TORCH_MODULE(AudioTailBlock);

/**
 * 历史记忆
 */
class MemoryBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential prob{ nullptr };

public:
    MemoryBlockImpl(
        const int in,
        const int channel,
        int num_groups = 16
    ) {
        this->prob = this->register_module("prob", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, channel, 3).padding(1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::ReLU(),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            torch::nn::Linear(torch::nn::LinearOptions(channel * in, 1).bias(false)),
            torch::nn::Sigmoid()
        ));
    }
    ~MemoryBlockImpl() {
        this->unregister_module("prob");
    }

public:
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& memory) {
        return input + this->prob->forward(input - memory).unsqueeze(-1) * memory;
    }

};

TORCH_MODULE(MemoryBlock);    

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
