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
    torch::nn::MultiheadAttention attn{ nullptr };

public:
    AttentionBlockImpl(
        const int in,
        const int channel,
        const int num_groups = 16,
        const int num_heads  = 4,
        const float dropout  = 0.3
    ) {
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, channel * 3, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel * 3)),
            torch::nn::Tanh()
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(in, num_heads).bias(false).dropout(dropout)));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("qkv");
        this->unregister_module("attn");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        // N C L => C N L
        auto qkv = this->qkv->forward(input).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = this->attn->forward(q, k, v);
        return input + torch::tanh(h.permute({ 1, 0, 2 }));
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
    torch::nn::Sequential conv_4{ nullptr };

public:
    ResNetBlockImpl(
        const int in,
        const int out,
        const int kernel  = 3,
        const int padding = 1
    ) {
        if(in == out) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).padding(padding).bias(false))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).padding(padding).bias(false)),
            torch::nn::Tanh(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).bias(false)),
            torch::nn::Tanh()
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).bias(false)),
            torch::nn::Tanh(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).bias(false)),
            torch::nn::Tanh()
        ));
        this->conv_4 = this->register_module("conv_4", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).bias(false)),
            torch::nn::Tanh(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).bias(false)),
            torch::nn::Tanh()
        ));
    }
    ~ResNetBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
        this->unregister_module("conv_4");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = output + this->conv_2->forward(input);
             output = output + this->conv_3->forward(output);
             output = output + this->conv_4->forward(output);
        return output;
    }

};

TORCH_MODULE(ResNetBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential norm  { nullptr };
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };
    torch::nn::Sequential conv_4{ nullptr };
    torch::nn::Sequential conv_5{ nullptr };
    torch::nn::Sequential conv_6{ nullptr };

public:
    AudioHeadBlockImpl(
        std::vector<int> channel,
        const int kernel     = 3,
        const int padding    = 1,
        const int num_groups = 16
    ) {
        this->norm = this->register_module("norm", torch::nn::Sequential(
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3]))
        ));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], kernel).padding(padding).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], kernel).padding(padding).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], kernel).padding(padding).bias(false))
        ));
        this->conv_4 = this->register_module("conv_4", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::Tanh(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::Tanh()
        ));
        this->conv_5 = this->register_module("conv_5", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::Tanh(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::Tanh()
        ));
        this->conv_6 = this->register_module("conv_6", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::Tanh(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::Tanh()
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("norm");
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
        this->unregister_module("conv_4");
        this->unregister_module("conv_5");
        this->unregister_module("conv_6");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = this->conv_4->forward(output) + output;
             output = this->conv_2->forward(output);
             output = this->conv_5->forward(output) + output;
             output = this->conv_3->forward(output);
             output = this->conv_6->forward(output) + output;
             output = this->norm->forward(output);
        return output;
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential norm  { nullptr };
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };
    torch::nn::Sequential conv_4{ nullptr };
    torch::nn::Sequential conv_5{ nullptr };
    torch::nn::Sequential conv_6{ nullptr };

public:
    VideoHeadBlockImpl(
        std::vector<int> channel,
        std::vector<int> pool,
        const int in         = 800,
        const int kernel     = 3,
        const int padding    = 1,
        const int num_groups = 16
    ) {
        this->norm = this->register_module("norm", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(in, in)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel[3])),
            torch::nn::Tanh()
        ));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[0], pool[1] }))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[2], pool[3] }))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[4], pool[5] }))
        ));
        this->conv_4 = this->register_module("conv_4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::ReLU()
        ));
        this->conv_5 = this->register_module("conv_5", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::ReLU()
        ));
        this->conv_6 = this->register_module("conv_6", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::ReLU()
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("norm");
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
        this->unregister_module("conv_4");
        this->unregister_module("conv_5");
        this->unregister_module("conv_6");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = this->conv_4->forward(output) + output;
             output = this->conv_2->forward(output);
             output = this->conv_5->forward(output) + output;
             output = this->conv_3->forward(output);
             output = this->conv_6->forward(output) + output;
             output = this->norm->forward(output.flatten(2));
        return output;
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential prob { nullptr };
    torch::nn::Sequential audio{ nullptr };
    torch::nn::Sequential video{ nullptr };
    torch::nn::Sequential mixer{ nullptr };

public:
    MediaMixBlockImpl(
        const int in_channel,
        const int out_channel,
        const int in         = 800,
        const int kernel     = 3,
        const int padding    = 1,
        const int num_groups = 16
    ) {
        this->prob = this->register_module("prob", torch::nn::Sequential(
            chobits::nn::AttentionBlock(in, out_channel, num_groups),
            torch::nn::Sigmoid()
        ));
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::ResNetBlock(in_channel, out_channel, kernel, padding),
            chobits::nn::AttentionBlock(in, out_channel, num_groups)
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::ResNetBlock(in_channel, out_channel, kernel, padding),
            chobits::nn::AttentionBlock(in, out_channel, num_groups)
        ));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::ResNetBlock(out_channel, out_channel, kernel, padding),
            chobits::nn::AttentionBlock(in, out_channel, num_groups)
        ));
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("prob");
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("mixer");
    }
    
public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        auto media_mix = this->mixer->forward(audio_out + video_out);
             media_mix = media_mix * this->prob->forward(media_mix);
        return { audio_out, video_out, media_mix };
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    AudioTailBlockImpl(
        std::vector<int> channel,
        const int kernel  = 3,
        const int padding = 1
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], kernel).padding(padding).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], kernel).padding(padding).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], kernel).padding(padding).bias(false))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = this->conv_2->forward(output);
             output = this->conv_3->forward(output);
        return output;
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
