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

using shp = std::vector<int64_t>;

using layer_act = torch::nn::SiLU;

namespace chobits::nn {

/**
 * 位置嵌入
 */
class RotaryPositionEmbeddingImpl : public torch::nn::Module {

private:
    int64_t dim       = 0;
    int64_t num_heads = 8;
    torch::Tensor sin_cached{ nullptr };
    torch::Tensor cos_cached{ nullptr };

public:
    RotaryPositionEmbeddingImpl(int64_t dim, int64_t max_len = 512, int64_t num_heads = 8) : dim(dim), num_heads(num_heads) {
        torch::Tensor inv_freq = 1.0 / torch::pow(10000.0, torch::arange(0, dim, 2, torch::kFloat) / static_cast<double>(dim));
        torch::Tensor t = torch::arange(0, max_len, torch::kFloat);
        torch::Tensor freqs = torch::outer(t, inv_freq);
        torch::Tensor emb = torch::cat({freqs, freqs}, -1);
        this->sin_cached = this->register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0));
        this->cos_cached = this->register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0));
    }

private:
    torch::Tensor rotate_half(const torch::Tensor& x) {
        int64_t d = x.size(-1);
        // auto x1 = x.slice(-1, 0, d / 2);
        // auto x2 = x.slice(-1, d / 2, d);
        auto x1 = x.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, d / 2) });
        auto x2 = x.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(d / 2, torch::indexing::None) });
        return torch::cat({ -x2, x1 }, -1);
    }

public:
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& query, const torch::Tensor& key) {
        auto q = query.view({ query.size(0), query.size(1), this->num_heads, this->dim }).transpose(1, 2);
        auto k = key  .view({ key  .size(0), key  .size(1), this->num_heads, this->dim }).transpose(1, 2);
        // auto cos_slice = this->cos_cached.slice(2, 0, q.size(2));
        // auto sin_slice = this->sin_cached.slice(2, 0, k.size(2));
        auto cos_slice = this->cos_cached.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, q.size(2)), torch::indexing::Slice() });
        auto sin_slice = this->sin_cached.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, k.size(2)), torch::indexing::Slice() });
        auto cos_expanded = cos_slice.expand_as(q);
        auto sin_expanded = sin_slice.expand_as(k);
        auto q_rotated = this->rotate_half(q);
        auto k_rotated = this->rotate_half(k);
        auto q_embed = (q * cos_expanded) + (q_rotated * sin_expanded);
        auto k_embed = (k * cos_expanded) + (k_rotated * sin_expanded);
        return std::make_tuple(
            q_embed.transpose(1, 2).view({ query.size(0), query.size(1), -1 }),
            k_embed.transpose(1, 2).view({ key  .size(0), key  .size(1), -1 })
        );
    }

};

TORCH_MODULE(RotaryPositionEmbedding);

/**
 * 1D残差网络
 */
class ResNet1dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential cv1{ nullptr };
    torch::nn::Sequential cv2{ nullptr };
    torch::nn::Sequential cv3{ nullptr };
    torch::nn::Sequential cv4{ nullptr };

public:
    ResNet1dBlockImpl(
        const int in_channels,
        const int out_channels,
        const int features,
        const int stride   = 1,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        this->cv1 = this->register_module("cv1", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels * 2, out_channels, kernel).padding(padding).dilation(dilation).bias(false).stride(stride)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ features })),
            layer_act()
        ));
        this->cv2 = this->register_module("cv2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, in_channels, 3).padding(1).dilation(1)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, in_channels, 3).padding(1).dilation(1)),
            layer_act()
        ));
        this->cv3 = this->register_module("cv3", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, in_channels, 3).padding(2).dilation(2)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, in_channels, 3).padding(2).dilation(2)),
            layer_act()
        ));
        this->cv4 = this->register_module("cv4", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, in_channels, 3).padding(2).dilation(2)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, in_channels, 3).padding(2).dilation(2)),
            layer_act()
        ));
    }
    ~ResNet1dBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto cv2 = this->cv2->forward(input);
        auto cv3 = this->cv3->forward(cv2);
        auto cv4 = this->cv4->forward(input);
        return this->cv1->forward(torch::cat({ cv4 + cv3, cv2 }, 1));
    }

};

TORCH_MODULE(ResNet1dBlock);

/**
 * 2D残差网络
 */
class ResNet2dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential cv1{ nullptr };
    torch::nn::Sequential cv2{ nullptr };
    torch::nn::Sequential cv3{ nullptr };
    torch::nn::Sequential cv4{ nullptr };

public:
    ResNet2dBlockImpl(
        const int in_channels,
        const int out_channels,
        const shp features,
        const shp stride   = std::vector<int64_t>{ 1, 1 },
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 1, 1 },
        const shp dilation = std::vector<int64_t>{ 1, 1 }
    ) {
        this->cv1 = this->register_module("cv1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels * 2, out_channels, kernel).padding(padding).dilation(dilation).bias(false).stride(stride)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(features)),
            layer_act()
        ));
        this->cv2 = this->register_module("cv2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, std::vector<int64_t>{ 3, 3 }).padding(std::vector<int64_t>{ 1, 1 }).dilation(std::vector<int64_t>{ 1, 1 })),
            layer_act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, std::vector<int64_t>{ 3, 3 }).padding(std::vector<int64_t>{ 1, 1 }).dilation(std::vector<int64_t>{ 1, 1 })),
            layer_act()
        ));
        this->cv3 = this->register_module("cv3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, std::vector<int64_t>{ 3, 3 }).padding(std::vector<int64_t>{ 2, 2 }).dilation(std::vector<int64_t>{ 2, 2 })),
            layer_act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, std::vector<int64_t>{ 3, 3 }).padding(std::vector<int64_t>{ 2, 2 }).dilation(std::vector<int64_t>{ 2, 2 })),
            layer_act()
        ));
        this->cv4 = this->register_module("cv4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, std::vector<int64_t>{ 3, 3 }).padding(std::vector<int64_t>{ 2, 2 }).dilation(std::vector<int64_t>{ 2, 2 })),
            layer_act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, std::vector<int64_t>{ 3, 3 }).padding(std::vector<int64_t>{ 2, 2 }).dilation(std::vector<int64_t>{ 2, 2 })),
            layer_act()
        ));
    }
    ~ResNet2dBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto cv2 = this->cv2->forward(input);
        auto cv3 = this->cv3->forward(cv2);
        auto cv4 = this->cv4->forward(input);
        return this->cv1->forward(torch::cat({ cv4 + cv3, cv2 }, 1));
    }

};

TORCH_MODULE(ResNet2dBlock);

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Linear                    q   { nullptr };
    torch::nn::Linear                    k   { nullptr };
    torch::nn::Linear                    v   { nullptr };
    chobits::nn::RotaryPositionEmbedding rope{ nullptr };
    torch::nn::MultiheadAttention        attn{ nullptr };
    torch::nn::Linear                    proj{ nullptr };
    torch::nn::Sequential                ffn { nullptr };

public:
    AttentionBlockImpl(
        const int q_dim,
        const int k_dim,
        const int v_dim,
        const int o_dim,
        const int h_dim     = 1024,
        const int num_heads = 8,
        const int max_len   = 512
    ) {
        this->q    = this->register_module("q", torch::nn::Linear(torch::nn::LinearOptions(q_dim, h_dim).bias(false)));
        this->k    = this->register_module("k", torch::nn::Linear(torch::nn::LinearOptions(k_dim, h_dim).bias(false)));
        this->v    = this->register_module("v", torch::nn::Linear(torch::nn::LinearOptions(v_dim, h_dim).bias(false)));
        this->rope = this->register_module("rope", chobits::nn::RotaryPositionEmbedding(h_dim / num_heads, max_len));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(h_dim, num_heads).bias(false)));
        this->proj = this->register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(h_dim, o_dim).bias(false)));
        this->ffn  = this->register_module("ffn",  torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })),
            torch::nn::Linear(o_dim, o_dim),
            layer_act(),
            torch::nn::Linear(o_dim, o_dim)
        ));
    }
    ~AttentionBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
        auto q = this->q->forward(query.permute({ 1, 0, 2 }));
        auto k = this->k->forward(key  .permute({ 1, 0, 2 }));
        auto v = this->v->forward(value.permute({ 1, 0, 2 }));
        // auto [ q_embed, k_embed ] = this->rope->forward(q, k);
        // auto [ o, _ ] = this->attn->forward(q_embed, k_embed, v);
        auto [ o, _ ] = this->attn->forward(q, k, v);
        return this->ffn->forward(query + this->proj->forward(o.permute({ 1, 0, 2 })));
    }

};

TORCH_MODULE(AttentionBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    const int n_fft    = 400;
    const int hop_size = 80;
    const int win_size = 400;
    torch::Tensor window{ nullptr };
    torch::nn::Sequential       head{ nullptr };
    chobits::nn::AttentionBlock attn{ nullptr };

public:
    AudioHeadBlockImpl() {
        this->head = this->register_module("head", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, std::vector<int64_t>{ 5, 5 }).padding(std::vector<int64_t>{ 0, 0 }).dilation(std::vector<int64_t>{ 1, 1 }).stride(std::vector<int64_t>{ 5, 5 })),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ 40, 64 })),
            chobits::nn::ResNet2dBlock( 32,  32, std::vector<int64_t>{ 40, 64 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 32,  64, std::vector<int64_t>{ 20, 32 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 64,  64, std::vector<int64_t>{ 20, 32 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 64, 128, std::vector<int64_t>{ 10, 16 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock(128, 128, std::vector<int64_t>{ 10, 16 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            chobits::nn::ResNet1dBlock(128, 256, 160, 1, 3, 1, 1)
        ));
        this->attn = this->register_module("attn", chobits::nn::AttentionBlock(160, 160, 160, 160, 512));
    }
    ~AudioHeadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(!this->window.defined()) {
            this->window = torch::hann_window(this->win_size).to(input.device());
        }
        auto com = torch::stft(
            input.view({ input.size(0), -1 }),
            this->n_fft,
            this->hop_size,
            this->win_size,
            this->window,
            true,
            "reflect",
            false,
            std::nullopt,
            true
        );
        auto mag = torch::abs(com);
//      auto pha = torch::angle(com);
             mag = mag.unsqueeze(1).contiguous();
        auto out = this->head->forward(mag);
        return this->attn->forward(out, out, out);
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential       head{ nullptr };
    torch::nn::Sequential       conv{ nullptr };
    chobits::nn::AttentionBlock attn{ nullptr };

public:
    VideoHeadBlockImpl() {
        this->head = this->register_module("head", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, std::vector<int64_t>{ 5, 5 }).padding(std::vector<int64_t>{ 0, 0 }).dilation(std::vector<int64_t>{ 1, 1 }).stride(std::vector<int64_t>{ 5, 5 })),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ 72, 128 })),
            chobits::nn::ResNet2dBlock( 16,  16, std::vector<int64_t>{ 72, 128 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 16,  64, std::vector<int64_t>{ 18,  32 }, std::vector<int64_t>{ 4, 4 }, std::vector<int64_t>{ 4, 4 }, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 64,  64, std::vector<int64_t>{ 18,  32 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 64, 256, std::vector<int64_t>{  4,   8 }, std::vector<int64_t>{ 4, 4 }, std::vector<int64_t>{ 4, 4 }, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock(256, 256, std::vector<int64_t>{  4,   8 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))
        ));
        this->conv = this->register_module("conv", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock( 32,  64, 2048, 4, 4, 0, 1),
            chobits::nn::ResNet1dBlock( 64, 128,  512, 4, 4, 0, 1),
            chobits::nn::ResNet1dBlock(128, 256,  512, 1, 3, 1, 1)
        ));
        this->attn = this->register_module("attn", AttentionBlock(512, 512, 512, 512));
    }
    ~VideoHeadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto out = this->head->forward(input.view({ -1, 1, input.size(2), input.size(3) }));
             out = this->conv->forward(out.view({ input.size(0), input.size(1), -1 }));
        return this->attn->forward(out, out, out);
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 图片输入
 */
class ImageHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential       head{ nullptr };
    chobits::nn::AttentionBlock attn{ nullptr };

public:
    ImageHeadBlockImpl() {
        this->head = this->register_module("head", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, std::vector<int64_t>{ 5, 5 }).padding(std::vector<int64_t>{ 0, 0 }).dilation(std::vector<int64_t>{ 1, 1 }).stride(std::vector<int64_t>{ 5, 5 })),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ 72, 128 })),
            chobits::nn::ResNet2dBlock( 32,  32, std::vector<int64_t>{ 72, 128 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 32,  64, std::vector<int64_t>{ 36,  64 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 64,  64, std::vector<int64_t>{ 36,  64 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock( 64, 128, std::vector<int64_t>{ 18,  32 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }),
            chobits::nn::ResNet2dBlock(128, 128, std::vector<int64_t>{ 18,  32 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 3, 3 }, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 1, 1 }),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            chobits::nn::ResNet1dBlock(128, 256, 576, 1, 3, 1, 1)
        ));
        this->attn = this->register_module("attn", AttentionBlock(576, 576, 576, 576));
    }
    ~ImageHeadBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto out = this->head->forward(input);
        return this->attn(out, out, out);
    }

};

TORCH_MODULE(ImageHeadBlock);

/**
 * 媒体混合
 */
class MediaMuxerBlockImpl : public torch::nn::Module {

private:
    chobits::nn::AttentionBlock audio_attn{ nullptr };
    chobits::nn::AttentionBlock video_attn{ nullptr };
    chobits::nn::AttentionBlock muxer_attn{ nullptr };
    chobits::nn::AttentionBlock mixer_attn{ nullptr };
    torch::nn::Sequential       muxer_conv{ nullptr };

public:
    MediaMuxerBlockImpl(
        const int audio_in = 160,
        const int video_in = 512,
        const int image_in = 576,
        const int channels = 256
    ) {
        const int muxer_in = audio_in + video_in;
        this->audio_attn = this->register_module("audio_attn", chobits::nn::AttentionBlock(audio_in, video_in, video_in, audio_in));
        this->video_attn = this->register_module("video_attn", chobits::nn::AttentionBlock(video_in, audio_in, audio_in, video_in));
        this->muxer_attn = this->register_module("muxer_attn", chobits::nn::AttentionBlock(muxer_in, image_in, image_in, muxer_in));
        this->mixer_attn = this->register_module("mixer_attn", chobits::nn::AttentionBlock(muxer_in, muxer_in, muxer_in, muxer_in));
        this->muxer_conv = this->register_module("muxer_conv", torch::nn::Sequential(
            chobits::nn::ResNet1dBlock(channels, channels, muxer_in, 1, 3, 1, 1)
        ));
    }
    ~MediaMuxerBlockImpl() {
    }
    
public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& audio,
        const torch::Tensor& video,
        const torch::Tensor& image,
        const torch::Tensor& muxer = torch::Tensor{ nullptr }
    ) {
        auto audio_o = this->audio_attn->forward(audio, video, video);
        auto video_o = this->video_attn->forward(video, audio, audio);
        auto muxer_c = this->muxer_conv->forward(torch::cat({ audio_o, video_o }, -1));
        auto muxer_o = this->muxer_attn->forward(muxer_c, image, image);
        if(!muxer.defined()) {
            auto mixer_o = this->mixer_attn->forward(muxer_o, muxer_o, muxer_o);
            return { audio_o, video_o, mixer_o };
        } else {
            auto mixer_o = this->mixer_attn->forward(muxer_o, muxer, muxer);
            return { audio_o, video_o, mixer_o };
        }
    }

};

TORCH_MODULE(MediaMuxerBlock);

/**
 * 媒体混合
 */
class MediaMixerBlockImpl : public torch::nn::Module {

private:
    chobits::nn::MediaMuxerBlock mixer_1{ nullptr };
    chobits::nn::MediaMuxerBlock mixer_2{ nullptr };
    chobits::nn::MediaMuxerBlock mixer_3{ nullptr };

public:
    MediaMixerBlockImpl(
        const int audio_in = 160,
        const int video_in = 512,
        const int image_in = 576,
        const int channels = 256
    ) {
        this->mixer_1 = this->register_module("mixer_1", chobits::nn::MediaMuxerBlock(audio_in, video_in, image_in, channels));
        this->mixer_2 = this->register_module("mixer_2", chobits::nn::MediaMuxerBlock(audio_in, video_in, image_in, channels));
        this->mixer_3 = this->register_module("mixer_3", chobits::nn::MediaMuxerBlock(audio_in, video_in, image_in, channels));
    }
    ~MediaMixerBlockImpl() {
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video, const torch::Tensor& image) {
        auto [ audio_1, video_1, mixer_1 ] = this->mixer_1->forward(audio,   video,   image);
        auto [ audio_2, video_2, mixer_2 ] = this->mixer_2->forward(audio_1, video_1, image, mixer_1);
        auto [ audio_3, video_3, mixer_3 ] = this->mixer_3->forward(audio_2, video_2, image, mixer_2);
        return mixer_3;
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
        const int in_features  = 672,
        const int out_features = 800,
        const shp channel      = std::vector<int64_t>{ 256, 64, 16, 4, 1 }
    ) {
        // 注意：AI必须透明绝对不能隐藏想法
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], 3)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], 3)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[2], channel[3], 3)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[3], channel[4], 3)),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            layer_act(),
            torch::nn::Linear(in_features - 2 * 4, out_features),
            layer_act(),
            torch::nn::Linear(out_features, out_features)
        ));
    }
    ~AudioTailBlockImpl() {
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return torch::tanh(this->tail->forward(input));
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
