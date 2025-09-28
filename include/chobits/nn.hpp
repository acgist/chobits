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
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    AudioHeadBlockImpl(
        int in,
        int num_groups = 16,
        std::vector<int> out    = { 8, 32, 64 },
        std::vector<int> kernel = { 3,  3,  3 },
        std::vector<int> stride = { 2,  2,  2 }
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out[0], kernel[0]).stride(stride[0]).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out[0], out[1], kernel[1]).stride(stride[1]).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out[1], out[2], kernel[2]).stride(stride[2]).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out[2]))
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = output + torch::silu(output);
             output = this->conv_2->forward(output);
             output = output + torch::silu(output);
             output = this->conv_3->forward(output);
        return torch::silu(output);
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };
    torch::nn::Sequential conv_3{ nullptr };

public:
    VideoHeadBlockImpl(
        int in,
        int num_groups = 16,
        std::vector<int> out    = { 8, 32, 64 },
        std::vector<int> kernel = { 3,  3,  3 },
        std::vector<int> stride = { 3,  2,  2 }
    ) {
        this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out[0], kernel[0]).stride(stride[0]).bias(false))
        ));
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out[0], out[1], kernel[1]).stride(stride[1]).bias(false))
        ));
        this->conv_3 = this->register_module("conv_3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out[1], out[2], kernel[2]).stride(stride[2]).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out[2]))
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
        this->unregister_module("conv_3");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->conv_1->forward(input);
             output = output + torch::silu(output);
             output = this->conv_2->forward(output);
             output = output + torch::silu(output);
             output = this->conv_3->forward(output);
        return torch::silu(output);
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 媒体混合
 */
class MediaMixBlockImpl : public torch::nn::Module {

private:
    int out_w;
    int out_h;
    torch::nn::Sequential audio{ nullptr };
    torch::nn::Sequential video{ nullptr };
    torch::nn::Sequential media{ nullptr };

public:
    MediaMixBlockImpl(
        int in,      int out,
        int audio_w, int audio_h,
        int video_w, int video_h,
        int out_w,   int out_h,
        int num_groups = 16
    ) : out_w(out_w), out_h(out_h) {
        if(audio_w == out_w && audio_h == out_h) {
            this->audio = this->register_module("audio", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)),
                torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))
            ));
        } else {
            this->audio = this->register_module("audio", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)),
                torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
                torch::nn::Linear(torch::nn::LinearOptions(audio_w * audio_h, out_w * out_h).bias(false))
            ));
        }
        if(video_w == out_w && video_h == out_h) {
            this->video = this->register_module("video", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)),
                torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))
            ));
        } else {
            this->video = this->register_module("video", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).bias(false)),
                torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
                torch::nn::Linear(torch::nn::LinearOptions(video_w * video_h, out_w * out_h).bias(false))
            ));
        }
        this->media = this->register_module("media", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out))
        ));
    }
    ~MediaMixBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("media");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto audio_out = this->audio->forward(audio);
        auto video_out = this->video->forward(video);
        auto media_out = torch::silu(audio_out) + torch::silu(video_out);
//      auto media_out = torch::concat({ audio_out, video_out });
        const int N = media_out.size(0);
        const int C = media_out.size(1);
        media_out = this->media->forward(media_out.reshape({ N, C, this->out_w, this->out_h }));
        return torch::silu(media_out);
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 残差网络
 */
class ResidualBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential justify { nullptr };
    torch::nn::Sequential conv_1_1{ nullptr };
    torch::nn::Sequential conv_1_2{ nullptr };
    torch::nn::Sequential conv_1_3{ nullptr };
    torch::nn::Sequential conv_1_4{ nullptr };
    torch::nn::Sequential conv_2_1{ nullptr };
    torch::nn::Sequential conv_2_2{ nullptr };
    torch::nn::Sequential conv_3_1{ nullptr };

public:
    ResidualBlockImpl(const int in, const int out, const int num_groups = 16) {
        if(in == out) {
            this->justify = this->register_module("justify", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->justify = this->register_module("justify", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, { 1, 1 }).bias(false))
            ));
        }
        this->conv_1_1 = this->register_module("conv_1_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false))
        ));
        this->conv_1_2 = this->register_module("conv_1_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false))
        ));
        this->conv_1_3 = this->register_module("conv_1_3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false))
        ));
        this->conv_1_4 = this->register_module("conv_1_4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false))
        ));
        this->conv_2_1 = this->register_module("conv_2_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false))
        ));
        this->conv_2_2 = this->register_module("conv_2_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).bias(false))
        ));
        this->conv_3_1 = this->register_module("conv_3_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out * 2, out, 3).padding(1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, out))
        ));
    }
    ~ResidualBlockImpl() {
        this->unregister_module("justify");
        this->unregister_module("conv_1_1");
        this->unregister_module("conv_1_2");
        this->unregister_module("conv_1_3");
        this->unregister_module("conv_1_4");
        this->unregister_module("conv_2_1");
        this->unregister_module("conv_2_2");
        this->unregister_module("conv_3_1");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto result       = this->justify->forward(input);
        auto output_1     = result;
        auto output_1_1   = this->conv_1_1->forward(output_1);
        auto output_1_1_1 = output_1_1 * torch::silu(output_1_1);
             output_1_1   = this->conv_1_2->forward(output_1_1_1);
             output_1_1_1 = output_1_1 * torch::silu(output_1_1);
             output_1     = output_1 + output_1_1_1;
             output_1_1   = this->conv_1_3->forward(output_1);
             output_1_1_1 = output_1_1 * torch::silu(output_1_1);
             output_1_1   = this->conv_1_4->forward(output_1_1_1);
             output_1_1_1 = output_1_1 * torch::silu(output_1_1);
             output_1     = output_1 + output_1_1_1;
        auto output_2     = result;
        auto output_2_1   = this->conv_2_1->forward(output_2);
        auto output_2_1_1 = output_2_1 * torch::silu(output_2_1);
             output_2_1   = this->conv_2_2->forward(output_2_1_1);
             output_2_1_1 = output_2_1 * torch::silu(output_2_1);
             output_2     = output_2 + output_2_1_1;
        auto output_3     = torch::concat({ output_1, output_2 }, 1);
             output_3     = this->conv_3_1->forward(output_3);
        return torch::silu(result + output_3);
    }

};

TORCH_MODULE(ResidualBlock);

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential         qkv { nullptr };
    torch::nn::Sequential         proj{ nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };

public:
    AttentionBlockImpl(const int channels, const int embed_dim, const int num_heads = 8, const int num_groups = 16, const float dropout = 0.3) {
        const int qkv_channels = channels * 3;
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, qkv_channels, 1).bias(false))
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout))
        );
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1).bias(false)),
            torch::nn::GroupNorm(torch::nn::GroupNormOptions(num_groups, channels))
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
//      const int C = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        auto qkv = this->qkv->forward(input).reshape({ N, -1, H * W }).permute({ 1, 0, 2 }).chunk(3, 0);
        auto q   = qkv[0];
        auto k   = qkv[1];
        auto v   = qkv[2];
        auto [ h, w ] = this->attn->forward(q, k, v);
        h = h.permute({ 1, 0, 2 }).reshape({ N, -1, H, W });
        h = this->proj->forward(h);
        return torch::silu(h + input);
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 残差自注意力
 */
class ResidualAttentionBlockImpl : public torch::nn::Module {

private:
    chobits::nn::ResidualBlock resi{ nullptr };
    torch::nn::ModuleDict      attn{ nullptr };

public:
    ResidualAttentionBlockImpl(const int in, const int out, const int embed_dim, int num_attns = 1) {
        torch::OrderedDict<std::string, std::shared_ptr<Module>> attn;
        this->resi = this->register_module("resi", chobits::nn::ResidualBlock(in, out));
        for(int i = 0; i < num_attns; ++i) {
            attn.insert("attn_" + std::to_string(i), chobits::nn::AttentionBlock(out, embed_dim).ptr());
        }
        this->attn = this->register_module("attn", torch::nn::ModuleDict(attn));
    }
    ~ResidualAttentionBlockImpl() {
        this->unregister_module("resi");
        this->unregister_module("attn");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->resi->forward(input);
        for (const auto& value : this->attn->items()) {
            auto layer = value.second;
            if (typeid(*layer) == typeid(chobits::nn::AttentionBlockImpl)) {
                output = layer->as<chobits::nn::AttentionBlock>()->forward(output);
            } else {
                // -
            }
        }
        return output;
    }

};

TORCH_MODULE(ResidualAttentionBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    chobits::nn::ResidualAttentionBlock resi_attn{ nullptr };
    torch::nn::Sequential               output   { nullptr };

public:
    AudioTailBlockImpl(
        const int in,
        const int out,
        const int embed_dim,
        std::vector<int> channels = { 128, 16, 2 },
        std::vector<int> kernel   = {   3,  3, 5 }
    ) {
        this->resi_attn = this->register_module("resi_attn", chobits::nn::ResidualAttentionBlock(in, out, embed_dim, 2));
        this->output = this->register_module("output", torch::nn::Sequential(
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(out,         channels[0], kernel[0]).stride(2).bias(false)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[0], channels[1], kernel[1]).stride(2).bias(false)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(channels[1], channels[2], kernel[2]).stride(2).bias(false))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("resi_attn");
        this->unregister_module("output");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output = this->resi_attn->forward(input);
        output = this->output->forward(output);
        return torch::tanh(output);
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
