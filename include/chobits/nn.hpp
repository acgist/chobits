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

using layer_act = torch::nn::Tanh;

#ifndef torch_act
#define torch_act torch::tanh
#endif

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
            return torch_act(input + output);
        } else {
            return torch_act(output);
        }
    }

};

TORCH_MODULE(GRUBlock);

/**
 * 自注意力
 */
class AttentionBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential         q   { nullptr };
    torch::nn::Sequential         k   { nullptr };
    torch::nn::Sequential         v   { nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };
    torch::nn::Sequential         proj{ nullptr };

public:
    AttentionBlockImpl(
        const int emb_dim,
        const int num_heads = 8
    ) {
        this->q = this->register_module("q", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false))
        ));
        this->k = this->register_module("k", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false))
        ));
        this->v = this->register_module("v", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false))
        ));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(emb_dim, num_heads).bias(false)
        ));
        this->proj = this->register_module("proj", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false)),
            layer_act(),
            torch::nn::Linear(torch::nn::LinearOptions(emb_dim, emb_dim).bias(false)),
            layer_act()
        ));
    }
    ~AttentionBlockImpl() {
        this->unregister_module("q");
        this->unregister_module("k");
        this->unregister_module("v");
        this->unregister_module("attn");
        this->unregister_module("proj");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto o = input.permute({ 1, 0, 2 });
        auto q = this->q->forward(o);
        auto k = this->k->forward(o);
        auto v = this->v->forward(o);
        auto [ h, w ] = this->attn->forward(q, k, v);
        h = h.permute({ 1, 0, 2 });
        h = this->proj->forward(h);
        return input + h;
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

public:
    ResNetBlockImpl(
        const int in,
        const int out,
        const int shape,
        const int pool     = 0,
        const int kernel   = 3,
        const int padding  = 1,
        const int dilation = 1
    ) {
        if(in == out) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else if(pool == 0) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ shape })),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
                layer_act()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ shape })),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(pool))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
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
        return torch_act(output);
    }

};

TORCH_MODULE(ResNetBlock);

/**
 * 2D残差网络
 */
class ResNet2dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };

public:
    ResNet2dBlockImpl(
        const int in,
        const int out,
        const int ch,
        const shp pool     = std::vector<int64_t>{      },
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 1, 1 },
        const shp dilation = std::vector<int64_t>{ 1, 1 }
    ) {
        if(in == out) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else if(pool.empty()) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(ch)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, kernel).padding(padding).dilation(dilation)),
                layer_act()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(ch)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, kernel).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(pool))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
    }
    ~ResNet2dBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output =          this->conv_1->forward(input);
             output = output + this->conv_2->forward(output);
        return torch_act(output);
    }

};

TORCH_MODULE(ResNet2dBlock);

/**
 * 3D残差网络
 */
class ResNet3dBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential conv_1{ nullptr };
    torch::nn::Sequential conv_2{ nullptr };

public:
    ResNet3dBlockImpl(
        const int in,
        const int out,
        const int ch,
        const shp pool     = std::vector<int64_t>{         },
        const shp kernel   = std::vector<int64_t>{ 1, 3, 3 },
        const shp padding  = std::vector<int64_t>{ 0, 2, 2 },
        const shp dilation = std::vector<int64_t>{ 1, 2, 2 }
    ) {
        if(in == out) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::Identity()
            ));
        } else if(pool.empty()) {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(ch)),
                torch::nn::Conv3d(torch::nn::Conv3dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
                layer_act()
            ));
        } else {
            this->conv_1 = this->register_module("conv_1", torch::nn::Sequential(
                torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(ch)),
                torch::nn::Conv3d(torch::nn::Conv3dOptions(in, out, kernel).bias(false).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
                layer_act(),
                torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(pool))
            ));
        }
        this->conv_2 = this->register_module("conv_2", torch::nn::Sequential(
            torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act(),
            torch::nn::Conv3d(torch::nn::Conv3dOptions(out, out, kernel).padding(padding).dilation(dilation)),
            layer_act()
        ));
    }
    ~ResNet3dBlockImpl() {
        this->unregister_module("conv_1");
        this->unregister_module("conv_2");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto output =          this->conv_1->forward(input);
             output = output + this->conv_2->forward(output);
        return torch_act(output);
    }

};

TORCH_MODULE(ResNet3dBlock);

/**
 * 音频输入
 */
class AudioHeadBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential head{ nullptr };

public:
    AudioHeadBlockImpl(
        const shp channel  = std::vector<int64_t>{ 10, 50, 200, 400 },
        const shp pool     = std::vector<int64_t>{ 2, 2, 2, 2 },
        const shp kernel   = std::vector<int64_t>{ 3, 3 },
        const shp padding  = std::vector<int64_t>{ 1, 1 },
        const shp dilation = std::vector<int64_t>{ 1, 1 }
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet2dBlock(channel[0], channel[1], channel[0], std::vector<int64_t>{ pool[0], pool[1] }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(channel[1], channel[1], channel[1], std::vector<int64_t>{                  }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(channel[1], channel[2], channel[1], std::vector<int64_t>{ pool[2], pool[3] }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(channel[2], channel[2], channel[2], std::vector<int64_t>{                  }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(channel[2], channel[3], channel[2], std::vector<int64_t>{                  }, kernel, padding, dilation),
            chobits::nn::ResNet2dBlock(channel[3], channel[3], channel[3], std::vector<int64_t>{                  }, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))
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
        const shp channel  = std::vector<int64_t>{ 10, 50, 200, 400 },
        const shp pool     = std::vector<int64_t>{ 1, 5, 5, 1, 4, 4, 1, 2, 2 },
        const shp kernel   = std::vector<int64_t>{ 1, 3, 3 },
        const shp padding  = std::vector<int64_t>{ 0, 2, 2 },
        const shp dilation = std::vector<int64_t>{ 1, 2, 2 }
    ) {
        this->head = this->register_module("head", torch::nn::Sequential(
            chobits::nn::ResNet3dBlock(channel[0], channel[1], channel[0], std::vector<int64_t>{ pool[0], pool[1], pool[2] }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[1], channel[1], channel[1], std::vector<int64_t>{                           }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[1], channel[2], channel[1], std::vector<int64_t>{ pool[3], pool[4], pool[5] }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[2], channel[2], channel[2], std::vector<int64_t>{                           }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[2], channel[3], channel[2], std::vector<int64_t>{ pool[6], pool[7], pool[8] }, kernel, padding, dilation),
            chobits::nn::ResNet3dBlock(channel[3], channel[3], channel[3], std::vector<int64_t>{                           }, kernel, padding, dilation),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(3))
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
    torch::nn::Sequential embed{ nullptr };
    torch::nn::Sequential muxer{ nullptr };

public:
    MediaMuxerBlockImpl(
        const int media_1_in,
        const int media_2_in
    ) {
        this->embed = this->register_module("embed", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(media_2_in, media_1_in)),
            layer_act()
        ));
        this->muxer = this->register_module("muxer", torch::nn::Sequential(
            chobits::nn::AttentionBlock(media_1_in)
        ));
    }
    ~MediaMuxerBlockImpl() {
        this->unregister_module("embed");
        this->unregister_module("muxer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& media_1, const torch::Tensor& media_2) {
        return this->muxer->forward(media_1 + this->embed->forward(media_2));
    }

};

TORCH_MODULE(MediaMuxerBlock);

/**
 * 媒体混合
 */
class MediaMixerBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential a_gru{ nullptr };
    torch::nn::Sequential r_gru{ nullptr };
    torch::nn::Sequential g_gru{ nullptr };
    torch::nn::Sequential b_gru{ nullptr };
    torch::nn::ModuleDict audio{ nullptr };
    torch::nn::ModuleDict video{ nullptr };
    torch::nn::Sequential mixer{ nullptr };

public:
    MediaMixerBlockImpl(
        const int audio_in    = 96,
        const int video_in    = 144,
        const int num_layers  = 3,
        const int in_channel  = 400,
        const int out_channel = 400
    ) {
        torch::OrderedDict<std::string, std::shared_ptr<Module>> audio;
        torch::OrderedDict<std::string, std::shared_ptr<Module>> video;
        for(int i = 0; i < num_layers; ++i) {
            audio.insert("audio_muxer_r_" + std::to_string(i), chobits::nn::MediaMuxerBlock(audio_in, video_in).ptr());
            audio.insert("audio_muxer_g_" + std::to_string(i), chobits::nn::MediaMuxerBlock(audio_in, video_in).ptr());
            audio.insert("audio_muxer_b_" + std::to_string(i), chobits::nn::MediaMuxerBlock(audio_in, video_in).ptr());
            video.insert("video_muxer_r_" + std::to_string(i), chobits::nn::MediaMuxerBlock(video_in, audio_in).ptr());
            video.insert("video_muxer_g_" + std::to_string(i), chobits::nn::MediaMuxerBlock(video_in, audio_in).ptr());
            video.insert("video_muxer_b_" + std::to_string(i), chobits::nn::MediaMuxerBlock(video_in, audio_in).ptr());
        }
        this->a_gru = this->register_module("a_gru", torch::nn::Sequential(
            chobits::nn::GRUBlock(audio_in, audio_in)
        ));
        this->r_gru = this->register_module("r_gru", torch::nn::Sequential(
            chobits::nn::GRUBlock(video_in, video_in)
        ));
        this->g_gru = this->register_module("g_gru", torch::nn::Sequential(
            chobits::nn::GRUBlock(video_in, video_in)
        ));
        this->b_gru = this->register_module("b_gru", torch::nn::Sequential(
            chobits::nn::GRUBlock(video_in, video_in)
        ));
        this->audio = this->register_module("audio", torch::nn::ModuleDict(audio));
        this->video = this->register_module("video", torch::nn::ModuleDict(video));
        this->mixer = this->register_module("mixer", torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ audio_in })),
            chobits::nn::ResNetBlock(in_channel, out_channel, audio_in),
            chobits::nn::ResNetBlock(in_channel, out_channel, audio_in),
            chobits::nn::AttentionBlock(audio_in)
        ));
    }
    ~MediaMixerBlockImpl() {
        this->unregister_module("a_gru");
        this->unregister_module("r_gru");
        this->unregister_module("g_gru");
        this->unregister_module("b_gru");
        this->unregister_module("audio");
        this->unregister_module("video");
        this->unregister_module("mixer");
    }
    
public:
    torch::Tensor forward(const torch::Tensor& audio, const torch::Tensor& video) {
        auto rgb = video.permute({ 2, 0, 1, 3 });
        auto vol = this->a_gru->forward(audio);
        torch::Tensor audio_in_r = vol;
        torch::Tensor audio_in_g = vol;
        torch::Tensor audio_in_b = vol;
        torch::Tensor video_in_r = this->r_gru->forward(rgb[0]);
        torch::Tensor video_in_g = this->g_gru->forward(rgb[1]);
        torch::Tensor video_in_b = this->b_gru->forward(rgb[2]);
        auto audios = this->audio->items();
        auto videos = this->video->items();
        for (
            auto
            audio_iter  = audios.begin(), video_iter  = videos.begin();
            audio_iter != audios.end() && video_iter != videos.end()  ;
        ) {
            video_in_r = video_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(video_in_r, audio_in_r); ++video_iter;
            video_in_g = video_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(video_in_g, audio_in_g); ++video_iter;
            video_in_b = video_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(video_in_b, audio_in_b); ++video_iter;
            audio_in_r = audio_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(audio_in_g, video_in_b); ++audio_iter;
            audio_in_g = audio_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(audio_in_b, video_in_r); ++audio_iter;
            audio_in_b = audio_iter->second->as<chobits::nn::MediaMuxerBlock>()->forward(audio_in_r, video_in_g); ++audio_iter;
        }
        return this->mixer->forward(audio_in_r + audio_in_g + audio_in_b);
    }

};

TORCH_MODULE(MediaMixerBlock);

/**
 * 音频输出
 */
class AudioTailBlockImpl : public torch::nn::Module {

private:
    torch::nn::Sequential tail  { nullptr };
    torch::nn::Sequential volume{ nullptr };

public:
    AudioTailBlockImpl(
        const int in        = 96,
        const int pool      = 4,
        const shp o_channel = std::vector<int64_t>{ 400, 800 },
        const shp v_channel = std::vector<int64_t>{ 400, 100 }
    ) {
        this->tail = this->register_module("tail", torch::nn::Sequential(
            chobits::nn::ResNetBlock(o_channel[0], o_channel[0], in),
            chobits::nn::ResNetBlock(o_channel[0], o_channel[1], in, pool, 1, 0, 1),
            torch::nn::Linear(torch::nn::LinearOptions(in / pool, 1)),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1))
        ));
        this->volume = this->register_module("volume", torch::nn::Sequential(
            chobits::nn::ResNetBlock(v_channel[0], v_channel[0], in),
            chobits::nn::ResNetBlock(v_channel[0], v_channel[1], in, pool),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            torch::nn::Linear(torch::nn::LinearOptions(in / pool * v_channel[1], 1)),
            torch::nn::Sigmoid()
        ));
    }
    ~AudioTailBlockImpl() {
        this->unregister_module("tail");
        this->unregister_module("volume");
    }

public:
    torch::Tensor forward(const torch::Tensor& input) {
        auto out = this->tail->forward(input);
        auto vol = this->volume->forward(input);
        #if CHOBITS_NORM == 0
        return vol * torch::sigmoid(out);
        #elif CHOBITS_NORM == 1
        return vol * torch::tanh(out);
        #endif
    }

};

TORCH_MODULE(AudioTailBlock);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
