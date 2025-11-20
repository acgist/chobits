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

using act = torch::nn::GELU; // torch::nn::SiLU
using shp = std::vector<int64_t>;

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
        auto [ output, _ ] = this->gru->forward(input, this->h0);
        return output;
    }

};

TORCH_MODULE(GRUBlock);

/**
 * LSTM
 */
class LSTMBlockImpl : public torch::nn::Module {

private:
    torch::Tensor   h0  { nullptr };
    torch::Tensor   c0  { nullptr };
    torch::nn::LSTM lstm{ nullptr };

public:
    LSTMBlockImpl(
        const int in,
        const int out,
        const int num_layers = 1
    ) {
        this->lstm = this->register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(in, out).num_layers(num_layers).bias(false).batch_first(true).bidirectional(false)
        ));
    }
    ~LSTMBlockImpl() {
        this->unregister_module("lstm");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(!this->h0.defined()) {
            this->h0 = torch::zeros({
                this->lstm->options.num_layers(),
                input.size(0),
                this->lstm->options.hidden_size()
            }).to(input.device());
        }
        if(!this->c0.defined()) {
            this->c0 = torch::zeros({
                this->lstm->options.num_layers(),
                input.size(0),
                this->lstm->options.hidden_size()
            }).to(input.device());
        }
        auto [ output, _ ] = this->lstm->forward(input, std::make_tuple(this->h0, this->c0));
        return output;
    }

};

TORCH_MODULE(LSTMBlock);

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
        const shp shape,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        if(in_channel == out_channel) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channel, out_channel, kernel).dilation(dilation).padding(padding).bias(false))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channel, out_channel, kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channel, out_channel, kernel).dilation(dilation).padding(padding).bias(false)),
            act()
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
        const int seq_len,
        const int emb_dim,
        const int num_heads = 8,
        const int kernel    = 1,
        const int padding   = 0,
        const int dilation  = 1
    ) {
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ emb_dim })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(seq_len, seq_len * 3, kernel).dilation(dilation).padding(padding).bias(false)),
            act()
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(emb_dim, num_heads).bias(false)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ emb_dim })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(seq_len, seq_len, kernel).dilation(dilation).padding(padding).bias(false)),
            act()
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("qkv");
        this->unregister_module("attn");
        this->unregister_module("proj");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
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
    const int n_fft;
    const int hop_size;
    const int win_size;
    torch::Tensor wind;
    torch::nn::Sequential mag{ nullptr };
    torch::nn::Sequential pha{ nullptr };
    torch::nn::Sequential mix{ nullptr };

public:
    AudioHeadBlockImpl(
        const int len      = 800,
        const int out      = 64,
        const shp channel  = std::vector<int64_t>{ 100, 200, 400 },
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1,
        const int n_fft    = 128,
        const int hop_size = 32,
        const int win_size = 128
    ) : n_fft(n_fft), hop_size(hop_size), win_size(win_size) {
        const int64_t in = n_fft / 2 + 1;
        const int64_t ch = len / hop_size + 1;
        this->mag = this->register_module("mag", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(ch, channel[0], kernel).dilation(dilation).padding(padding).bias(false)),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false)),
            act(),
            torch::nn::Linear(torch::nn::LinearOptions(out, out).bias(false))
        ));
        this->pha = this->register_module("pha", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(ch, channel[0], kernel).dilation(dilation).padding(padding).bias(false)),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false)),
            act(),
            torch::nn::Linear(torch::nn::LinearOptions(out, out).bias(false))
        ));
        this->mix = this->register_module("mix", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel[2], channel[2], std::vector<int64_t>{ out }),
            chobits::nn::AttentionBlock(channel[2], out)
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("mag");
        this->unregister_module("pha");
        this->unregister_module("mix");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        if(!this->wind.defined()) {
            this->wind = torch::hann_window(this->win_size).to(input.device());
        }
        auto com = torch::stft(input, this->n_fft, this->hop_size, this->win_size, this->wind, true, "reflect", false, std::nullopt, true);
        auto mag = torch::abs(com);
        auto pha = torch::angle(com);
             mag = this->mag->forward(mag.permute({ 0, 2, 1 }));
             pha = this->pha->forward(pha.permute({ 0, 2, 1 }));
        return this->mix->forward(mag + pha);
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
        const shp channel  = std::vector<int64_t>{ 3, 100, 200, 400 },
        const shp pool     = std::vector<int64_t>{ 5, 5, 2, 2, 2, 2 },
        const int kernel   = 3,
        const int padding  = 2,
        const int dilation = 2,
        const int height   = 360,
        const int width    = 640
    ) {
        const int64_t out = height * width / std::accumulate(pool.begin(), pool.end(), 1, std::multiplies<int64_t>());
        this->embedding = this->register_module("embedding", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ pool[0], pool[1] })),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ channel[1], height / pool[0], width / pool[1] })),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ pool[2], pool[3] })),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ channel[2], height / pool[0] / pool[2], width / pool[1] / pool[3] })),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ pool[4], pool[5] })),
            // -
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ out })),
            torch::nn::Linear(torch::nn::LinearOptions(out, out).bias(false)),
            act(),
            torch::nn::Linear(torch::nn::LinearOptions(out, out).bias(false)),
            // -
            chobits::nn::ResNetBlock(channel[3], channel[3], std::vector<int64_t>{ out }),
            chobits::nn::AttentionBlock(channel[3], out)
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
        const int media_1_in,
        const int media_2_in,
        const int channel = 400
    ) {
        this->media_1 = this->register_module("media_1", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel, std::vector<int64_t>{ media_1_in })
        ));
        this->media_2 = this->register_module("media_2", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel, std::vector<int64_t>{ media_2_in }),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ media_2_in })),
            torch::nn::Linear(torch::nn::LinearOptions(media_2_in, 1)),
            act()
        ));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(channel, media_1_in)
        ));
        this->mprob = this->register_module("mprob", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel, std::vector<int64_t>{ media_1_in }),
            chobits::nn::AttentionBlock(channel, media_1_in),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ media_1_in })),
            torch::nn::Linear(torch::nn::LinearOptions(media_1_in, 1)),
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
        const int audio_in    = 64,
        const int video_in    = 576,
        const int in_channel  = 400,
        const int out_channel = 800
    ) {
        this->audio = this->register_module("audio", torch::nn::Sequential(
            chobits::nn::GRUBlock(audio_in, audio_in)
        ));
        this->video = this->register_module("video", torch::nn::Sequential(
            chobits::nn::GRUBlock(video_in, video_in)
        ));
        this->aprob = this->register_module("aprob", chobits::nn::MediaProbBlock(audio_in, video_in));
        this->vprob = this->register_module("vprob", chobits::nn::MediaProbBlock(video_in, audio_in));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(in_channel, audio_in + video_in),
            chobits::nn::ResNetBlock(in_channel, in_channel, std::vector<int64_t>{ audio_in + video_in }),
            chobits::nn::LSTMBlock(audio_in + video_in, audio_in),
            chobits::nn::ResNetBlock(in_channel, out_channel, std::vector<int64_t>{ audio_in }),
            chobits::nn::AttentionBlock(out_channel, audio_in)
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
        auto media_mix = torch::concat({ audio_mix, video_mix }, -1);
        return this->mixer->forward(media_mix);
    }

};

TORCH_MODULE(MediaMixBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential prob{ nullptr };
    torch::nn::Sequential tail{ nullptr };

public:
    AudioTailBlockImpl(
        const int in      = 64,
        const int out     = 800,
        const int channel = 800
    ) {
        this->prob = this->register_module("prob", torch::nn::Sequential(
            chobits::nn::ResNetBlock(channel, channel, std::vector<int64_t>{ in }),
            chobits::nn::AttentionBlock(channel, in),
            torch::nn::Sigmoid()
        ));
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::AdaptiveAvgPool1d(1),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            torch::nn::Linear(torch::nn::LinearOptions(out, out)),
            act(),
            torch::nn::Linear(torch::nn::LinearOptions(out, out))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("prob");
        this->unregister_module("tail");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto prob = this->prob->forward(input);
        auto tail = this->tail->forward(input * prob);
        #if CHOBITS_NORM == 0
        return torch::sigmoid(tail);
        #elif CHOBITS_NORM == 1
        return torch::tanh(tail);
        #else
        return tail;
        #endif
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
