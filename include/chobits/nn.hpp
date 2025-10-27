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
        const int num_heads  = 4,
        const int num_groups = 16,
        const float dropout  = 0.3
    ) {
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, channel * 3, 1).bias(false))
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(in, num_heads).bias(false).dropout(dropout)));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("qkv");
        this->unregister_module("attn");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        const int N = input.size(0);
//      const int C = input.size(1);
        const int L = input.size(2);
        // N C L => C N L
        auto qkv = this->qkv->forward(input).reshape({ N, -1, L }).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = this->attn->forward(q, k, v);
        return input + h.permute({ 1, 0, 2 });
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential down_1{ nullptr };
    torch::nn::Sequential down_2{ nullptr };
    torch::nn::Sequential down_3{ nullptr };
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    AudioHeadBlockImpl(
        const int kernel,
        const int padding,
        std::vector<int> channel,
        std::vector<int> pool,
        const int num_groups = 16
    ) {
        this->down_1 = this->register_module("down_1", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool[0]))
        ));
        this->down_2 = this->register_module("down_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool[1]))
        ));
        this->down_3 = this->register_module("down_3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool[2]))
        ));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[1], kernel).padding(padding).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[2], kernel).padding(padding).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[3], kernel).padding(padding).bias(false))
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("down_1");
        this->unregister_module("down_2");
        this->unregister_module("down_3");
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->down_1->forward(input);
             output = this->conv_1->forward(output) + output;
             output = this->down_2->forward(output);
             output = this->conv_2->forward(output) + output;
             output = this->down_3->forward(output);
             output = this->conv_3->forward(output) + output;
        return output;
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential down_1{ nullptr };
    torch::nn::Sequential down_2{ nullptr };
    torch::nn::Sequential down_3{ nullptr };
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };
    torch::nn::Sequential linear{ nullptr };

public:
    VideoHeadBlockImpl(
        const int in,
        const int out,
        const int kernel,
        const int padding,
        std::vector<int> channel,
        std::vector<int> pool,
        const int num_groups = 16
    ) {
        this->down_1 = this->register_module("down_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ pool[0], pool[1] }))
        ));
        this->down_2 = this->register_module("down_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ pool[2], pool[3] }))
        ));
        this->down_3 = this->register_module("down_3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ pool[4], pool[5] }))
        ));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[1], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[1], kernel).padding(padding).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[2], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[2], kernel).padding(padding).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[3], channel[3], kernel).padding(padding).bias(false)),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[3], channel[3], kernel).padding(padding).bias(false))
        ));
        this->linear = this->register_module("linear", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false))
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("down_1");
        this->unregister_module("down_2");
        this->unregister_module("down_3");
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
        this->unregister_module("linear");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->down_1->forward(input);
             output = this->conv_1->forward(output) + output;
             output = this->down_2->forward(output);
             output = this->conv_2->forward(output) + output;
             output = this->down_3->forward(output);
             output = this->conv_3->forward(output) + output;
        return this->linear->forward(output.flatten(2));
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential audio  { nullptr };
    torch::nn::Sequential video  { nullptr };
    torch::nn::Sequential mixer_1{ nullptr };
    torch::nn::Sequential mixer_2{ nullptr };
    torch::nn::Sequential mixer_3{ nullptr };

public:
    MediaMixBlockImpl(
        const int in,
        const int channel,
        const int kernel,
        const int padding,
        const double pool,
        const int num_groups = 16
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, channel * pool, kernel).padding(padding).bias(false)),
            torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool))
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, channel * pool, kernel).padding(padding).bias(false)),
            torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool))
        ));
        this->mixer_1 = this->register_module("mixer_1", torch::nn::Sequential(
            chobits::nn::AttentionBlock(in / pool, channel * pool),
            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{ pool }).mode(torch::kLinear).align_corners(true)),
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel * pool, channel, kernel).padding(padding).bias(false))
        ));
        this->mixer_2 = this->register_module("mixer_2", torch::nn::Sequential(
            chobits::nn::AttentionBlock(in, channel)
        ));
        this->mixer_3 = this->register_module("mixer_3", torch::nn::Sequential(
            chobits::nn::AttentionBlock(in, channel),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channel)),
            torch::nn::SiLU()
        ));
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("mixer_1");
        this->unregister_module("mixer_2");
        this->unregister_module("mixer_3");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        auto media_mix = this->mixer_1->forward(audio_out + video_out);
             media_mix = torch::sigmoid(this->mixer_2->forward(media_mix)) * media_mix;
             media_mix = this->mixer_3->forward(audio + media_mix);
        return media_mix;
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::Tensor  h0 { nullptr };
    torch::nn::GRU gru{ nullptr };
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    AudioTailBlockImpl(
        const int in,
        const int kernel,
        const int padding,
        std::vector<int> channel,
        std::vector<double> pool,
        const int num_layers = 1
    ) {
        this->gru = this->register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(in, in).num_layers(num_layers).bias(false).batch_first(true).bidirectional(false)));
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{ pool[0] }).mode(torch::kLinear).align_corners(true)),
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[0], channel[1], kernel).padding(padding).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{ pool[1] }).mode(torch::kLinear).align_corners(true)),
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[1], channel[2], kernel).padding(padding).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{ pool[2] }).mode(torch::kLinear).align_corners(true)),
            torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(channel[2], channel[3], kernel).padding(padding).bias(false))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("gru");
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(!this->h0.defined()) {
            this->h0 = torch::zeros({ this->gru->options.num_layers(), input.size(0), this->gru->options.hidden_size() }).to(input.device());
        }
        auto [output, hn] = this->gru->forward(input, this->h0);
             output = this->conv_1->forward(output);
             output = this->conv_2->forward(output);
             output = this->conv_3->forward(output);
        return output;
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
