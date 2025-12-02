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

using act = torch::nn::GELU; // ELU GELU ReLU SiLU Tanh

using shp = std::vector<int64_t>;

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
        const int in,
        const int out,
        const int num_layers = 1
    ) {
        this->gru = this->register_module("gru", torch::nn::GRU(
            torch::nn::GRUOptions(in, out).num_layers(num_layers).bias(false).batch_first(true)
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
        if(input.size(2) == output.size(2)) {
            return input + output;
        } else {
            return output;
        }
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
            torch::nn::LSTMOptions(in, out).num_layers(num_layers).bias(false).batch_first(true)
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
        if(input.size(2) == output.size(2)) {
            return input + output;
        } else {
            return output;
        }
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
        const int in,
        const int out,
        const shp shape,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        if(in == out) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).dilation(dilation).padding(padding).bias(false))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(shape)),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).dilation(dilation).padding(padding).bias(false)),
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
        const int emb_dim,
        const int num_heads = 8
    ) {
        // Conv1d: 局部特征
        // Linear: 全局特征
        this->qkv = this->register_module("qkv", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, 3 * emb_dim).bias(false))
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(emb_dim, num_heads).bias(false)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false))
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("qkv");
        this->unregister_module("attn");
        this->unregister_module("proj");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto qkv = this->qkv->forward(input).reshape({ input.size(0), input.size(1), 3, -1 }).permute({ 2, 1, 0, 3 });
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
    torch::nn::Sequential head{ nullptr };

public:
    AudioHeadBlockImpl(
        const shp channel  = std::vector<int64_t>{ 10, 100, 200, 400 },
        const shp pool     = std::vector<int64_t>{ 2, 2, 2, 2, 2, 2 },
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 1, 1 },
        const shp dilation = std::vector<int64_t>{ 1, 1 },
        const int height   = 26,
        const int width    = 65
    ) {
        int64_t out = (height / pool[0] / pool[2] / pool[4]) * (width / pool[1] / pool[3] / pool[5]);
        this->head = this->register_module("head", torch::nn::Sequential(
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel[0])),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[0], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[0], pool[1] })),
            // -
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel[1])),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[1], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[2], pool[3] })),
            // -
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel[2])),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[2], channel[3], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channel[3], channel[3], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ pool[4], pool[5] })),
            // -
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ out })),
            chobits::nn::AttentionBlock(out),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ out })),
            chobits::nn::GRUBlock(out, out),
            act()
        ));
    }
    ~AudioHeadBlockImpl() {
        this->unregister_module("head");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->head->forward(input);
    }

};

TORCH_MODULE(AudioHeadBlock);

/**
 * 视频输入
 */
class VideoHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    VideoHeadBlockImpl(
        const shp channel  = std::vector<int64_t>{ 10, 50, 100, 200, 400 },
        const shp pool     = std::vector<int64_t>{ 1, 5, 5, 1, 2, 2, 1, 2, 2, 1, 2, 2 },
        const shp kernel   = std::vector<int64_t>{ 1, 3, 3 },
        const shp padding  = std::vector<int64_t>{ 0, 2, 2 },
        const shp dilation = std::vector<int64_t>{ 1, 2, 2 },
        const int ch       = 3,
        const int height   = 360,
        const int width    = 640
    ) {
        const int64_t out = ch * height * width / std::accumulate(pool.begin(), pool.end(), 1, std::multiplies<int64_t>());
        this->head = this->register_module("head", torch::nn::Sequential(
            torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(channel[0])),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[0], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[1], channel[1], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ pool[0], pool[1], pool[2] })),
            // -
            torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(channel[1])),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[1], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[2], channel[2], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ pool[3], pool[4], pool[5] })),
            // -
            torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(channel[2])),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[2], channel[3], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[3], channel[3], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ pool[6], pool[7], pool[8] })),
            // -
            torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(channel[3])),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[3], channel[4], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(channel[4], channel[4], kernel).dilation(dilation).padding(padding).bias(false)),
            act(),
            torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ pool[9], pool[10], pool[11] })),
            // -
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2)),
            // -
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ out })),
            chobits::nn::AttentionBlock(out),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ out })),
            chobits::nn::GRUBlock(out, out),
            act()
        ));
    }
    ~VideoHeadBlockImpl() {
        this->unregister_module("head");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        return this->head->forward(input);
    }

};

TORCH_MODULE(VideoHeadBlock);

/**
 * 媒体混合
 */
class MediaMuxerBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential embedding{ nullptr };
    torch::nn::Sequential muxer    { nullptr };

public:
    MediaMuxerBlockImpl(
        const int media_1_in,
        const int media_2_in
    ) {
        this->embedding = this->register_module("embedding", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ media_2_in })),
            chobits::nn::AttentionBlock(media_2_in),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ media_2_in })),
            torch::nn::Linear(torch::nn::LinearOptions(media_2_in, 1).bias(false)),
            act()
        ));
        this->muxer = this->register_module("muxer", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ media_1_in })),
            chobits::nn::AttentionBlock(media_1_in),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ media_1_in })),
            chobits::nn::GRUBlock(media_1_in, media_1_in),
            act()
        ));
    }
    ~MediaMuxerBlockImpl() {
        this->unregister_module("embedding");
        this->unregister_module("muxer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& media_1, const torch::Tensor& media_2) {
        return this->muxer->forward(media_1 + this->embedding->forward(media_2)) + media_1;
    }

};

TORCH_MODULE(MediaMuxerBlock);

/**
 * 媒体混合
 */
class MediaMixerBlockImpl : public torch::nn::Module {

private:
    torch::nn::ModuleDict        audio{ nullptr };
    torch::nn::ModuleDict        video{ nullptr };
    chobits::nn::MediaMuxerBlock muxer{ nullptr };
    torch::nn::Sequential        mixer{ nullptr };

public:
    MediaMixerBlockImpl(
        const int audio_in   = 24,
        const int video_in   = 432,
        const int num_layers = 5
    ) {
        torch::OrderedDict<std::string, std::shared_ptr<Module>> audio;
        torch::OrderedDict<std::string, std::shared_ptr<Module>> video;
        for(int i = 0; i < num_layers; ++i) {
            audio.insert(
                "audio_muxer_" + std::to_string(i),
                chobits::nn::MediaMuxerBlock(audio_in, video_in).ptr()
            );
            video.insert(
                "video_muxer_" + std::to_string(i),
                chobits::nn::MediaMuxerBlock(video_in, audio_in).ptr()
            );
        }
        this->audio = this->register_module("audio", torch::nn::ModuleDict(audio));
        this->video = this->register_module("video", torch::nn::ModuleDict(video));
        this->muxer = this->register_module("muxer", chobits::nn::MediaMuxerBlock(audio_in, video_in));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ audio_in })),
            chobits::nn::AttentionBlock(audio_in),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ audio_in })),
            chobits::nn::LSTMBlock(audio_in, audio_in),
            act()
        ));
    }
    ~MediaMixerBlockImpl() {
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("muxer");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        torch::Tensor audio_in = audio;
        torch::Tensor video_in = video;
        auto audios = this->audio->items();
        auto videos = this->video->items();
        for (
            auto
            audio_iter  = audios.begin(), video_iter  = videos.begin();
            audio_iter != audios.end() && video_iter != videos.end()  ;
            ++audio_iter,
            ++video_iter
        ) {
            auto audio_out = audio_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(audio_in, video_in);
            auto video_out = video_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(video_in, audio_in);
            audio_in = audio_out;
            video_in = video_out;
        }
        return this->mixer->forward(this->muxer->forward(audio_in, video_in));
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
        const int in          = 24,
        const int out         = 1,
        const int in_channel  = 400,
        const int out_channel = 800
    ) {
        this->tail = this->register_module("tail", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            chobits::nn::ResNetBlock(in_channel, out_channel, std::vector<int64_t>{ in }),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            chobits::nn::AttentionBlock(in),
            act(),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ in })),
            chobits::nn::GRUBlock(in, in),
            act(),
            torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(false)),
            // -
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1))
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("tail");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto out = this->tail->forward(input);
        #if CHOBITS_NORM == 0
        return torch::sigmoid(out);
        #elif CHOBITS_NORM == 1
        return torch::tanh(out);
        #endif
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
