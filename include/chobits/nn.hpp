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

#include "model.hpp"
#include "chobits.hpp"

#include "torch/nn.h"

using shape_t = std::vector<int64_t>;

namespace chobits::nn {

class ActivationImpl : public torch::nn::Module {

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        return torch::relu(input);
    }

};

TORCH_MODULE(Activation);

class ExpertImpl : public torch::nn::Module {

private:
    torch::nn::Sequential fc{ nullptr };

public:
    ExpertImpl(
        const int64_t embed_dim,
        const int64_t scale   = 2,
        const double  dropout = 0.1
    ) {
        this->fc = this->register_module("fc", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim * scale)),
            chobits::nn::Activation(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout)),
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim * scale, embed_dim))
        ));
    }

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        return this->fc->forward(input);
    }

};

TORCH_MODULE(Expert);

class MoEImpl : public torch::nn::Module {

private:
    torch::nn::Linear     gate   { nullptr };
    torch::nn::ModuleList experts{         };

public:
    MoEImpl(
        const int64_t embed_dim,
        const int64_t num_experts = 2
    ) {
        torch::nn::ModuleList experts;
        for (int i = 0; i < num_experts; ++i) {
            experts->push_back(chobits::nn::Expert(embed_dim));
        }
        this->gate    = this->register_module("gate",    torch::nn::Linear(embed_dim, num_experts));
        this->experts = this->register_module("experts", experts);
    }

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        auto flat_input   = input.view({ -1, input.size(2) });
        auto gate_logits  = this->gate->forward(flat_input);
        auto gate_weights = torch::softmax(gate_logits, -1);
        std::vector<torch::Tensor> expert_outs;
        for (auto iter = this->experts->begin(); iter != this->experts->end(); ++iter) {
            auto expert_out = (*iter)->as<chobits::nn::Expert>()->forward(flat_input);
            expert_outs.push_back(expert_out);
        }
        auto stacked_expert_outs  = torch::stack(expert_outs, 1);
        auto weighted_expert_outs = stacked_expert_outs * gate_weights.unsqueeze(-1);
        auto final_output = weighted_expert_outs.sum(1);
        return final_output.view(input.sizes());
    }

};

TORCH_MODULE(MoE);

/**
 * MHA MQA GQA MLA
 */
class MHAImpl : public torch::nn::Module {

private:
    torch::nn::Linear             q    { nullptr };
    torch::nn::Linear             k    { nullptr };
    torch::nn::Linear             v    { nullptr };
    torch::nn::MultiheadAttention attn { nullptr };
    torch::nn::Linear             proj { nullptr };
    chobits::nn::MoE              ffn  { nullptr };
//  chobits::nn::Expert           ffn  { nullptr };
    torch::nn::LayerNorm          norm1{ nullptr };
    torch::nn::LayerNorm          norm2{ nullptr };

public:
    MHAImpl(
        const int64_t q_dim,
        const int64_t k_dim,
        const int64_t v_dim,
        const int64_t o_dim,
        const int64_t h_dim     = 1024,
        const int64_t num_heads = 8,
        const double  dropout   = 0.1
    ) {
        this->q     = this->register_module("q",     torch::nn::Linear(torch::nn::LinearOptions(q_dim, h_dim).bias(false)));
        this->k     = this->register_module("k",     torch::nn::Linear(torch::nn::LinearOptions(k_dim, h_dim).bias(false)));
        this->v     = this->register_module("v",     torch::nn::Linear(torch::nn::LinearOptions(v_dim, h_dim).bias(false)));
        this->attn  = this->register_module("attn",  torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(h_dim, num_heads).bias(false).dropout(dropout)));
        this->proj  = this->register_module("proj",  torch::nn::Linear(torch::nn::LinearOptions(h_dim, o_dim).bias(false)));
        this->ffn   = this->register_module("ffn",   chobits::nn::MoE(o_dim));
//      this->ffn   = this->register_module("ffn",   chobits::nn::Expert(o_dim));
        this->norm1 = this->register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
        this->norm2 = this->register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
    }

public:
    torch::Tensor forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value
    ) {
        // [ N S L ] -> [ S N L ]
        auto q = this->q->forward(query.transpose(0, 1));
        auto k = this->k->forward(key  .transpose(0, 1));
        auto v = this->v->forward(value.transpose(0, 1));
        auto [ o, _ ] = this->attn->forward(q, k, v);
             o = query + this->proj->forward(o.transpose(0, 1));
             o = this->norm1->forward(o);
             o = o + this->ffn->forward(o);
        return this->norm2->forward(o);
    }

};

TORCH_MODULE(MHA);

class ViTImpl : public torch::nn::Module {

private:
    torch::nn::Sequential patch_s  { nullptr };
    torch::nn::Sequential patch_l  { nullptr };
    torch::Tensor         pos_embed{ nullptr };
    torch::nn::LayerNorm  norm     { nullptr };
    chobits::nn::MHA      mha      { nullptr };

public:
    ViTImpl(
        const int64_t h,
        const int64_t w,
        const int64_t i_channels,
        const int64_t o_channels,
        const shape_t stride,
        const shape_t kernel_s,
        const shape_t kernel_l,
        const shape_t padding_s  = std::vector<int64_t>{ 0, 0 },
        const shape_t padding_l  = std::vector<int64_t>{ 0, 0 },
        const shape_t dilation_s = std::vector<int64_t>{ 1, 1 },
        const shape_t dilation_l = std::vector<int64_t>{ 1, 1 },
        const int64_t num_heads  = 8,
        const shape_t stride_h   = std::vector<int64_t>{ 1, 1 },
        const shape_t kernel_h   = std::vector<int64_t>{ 3, 3 },
        const shape_t padding_h  = std::vector<int64_t>{ 1, 1 },
        const shape_t dilation_h = std::vector<int64_t>{ 1, 1 }
    ) {
        const int64_t dim     = o_channels;
        const int64_t o_dim   = dim   * 2;
        const int64_t h_dim   = o_dim * 2;
        const int64_t seq_len = (h / stride[0]) * (w / stride[1]);
        this->patch_s = this->register_module("patch_s", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(i_channels, o_channels, kernel_s).padding(padding_s).dilation(dilation_s).stride(stride)),
            chobits::nn::Activation(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(o_channels, o_channels, kernel_h).padding(padding_h).dilation(dilation_h).stride(stride_h))
        ));
        this->patch_l = this->register_module("patch_l", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(i_channels, o_channels, kernel_l).padding(padding_l).dilation(dilation_l).stride(stride)),
            chobits::nn::Activation(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(o_channels, o_channels, kernel_h).padding(padding_h).dilation(dilation_h).stride(stride_h))
        ));
        this->pos_embed = this->register_parameter("pos_embed", torch::zeros({ 1, seq_len, o_dim }));
        this->norm      = this->register_module   ("norm",      torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
        this->mha       = this->register_module   ("mha",       chobits::nn::MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads));
    }
    
public:
    torch::Tensor forward(
        const torch::Tensor& input,
        const torch::Tensor& memory
    ) {
        auto input_s = this->patch_s->forward(input).flatten(2).transpose(1, 2);
        auto input_l = this->patch_l->forward(input).flatten(2).transpose(1, 2);
        auto out = torch::cat({ input_s, input_l }, -1);
             out = out + this->pos_embed;
             out = this->norm->forward(out);
             out = this->mha->forward(memory, out, out);
        return out;
    }

};

TORCH_MODULE(ViT);

/**
 * 音频视频作为动作传入
 * 图片作为一种知识传入
 * 可以添加文字等等知识
 */
class MixerImpl : public torch::nn::Module {

private:
    torch::Tensor    audio_embed{ nullptr };
    torch::Tensor    video_embed{ nullptr };
    torch::Tensor    image_embed{ nullptr };
    chobits::nn::MHA muxer_mha  { nullptr };
    chobits::nn::MHA image_mha  { nullptr };
    chobits::nn::MHA mixer_mha  { nullptr };

public:
    MixerImpl(
        const int64_t audio_dim = 256,
        const int64_t video_dim = 512,
        const int64_t image_dim = 512,
        const int64_t num_heads = 8
    ) {
        const int64_t muxer_dim = audio_dim + video_dim;
        this->audio_embed = this->register_parameter("audio_embed", torch::zeros({ 1, 1, audio_dim }));
        this->video_embed = this->register_parameter("video_embed", torch::zeros({ 1, 1, video_dim }));
        this->image_embed = this->register_parameter("image_embed", torch::zeros({ 1, 1, image_dim }));
        this->muxer_mha = this->register_module("muxer_mha", chobits::nn::MHA(muxer_dim, muxer_dim, muxer_dim, muxer_dim, muxer_dim * 2, num_heads));
        this->image_mha = this->register_module("image_mha", chobits::nn::MHA(muxer_dim, image_dim, image_dim, muxer_dim, muxer_dim * 2, num_heads));
        this->mixer_mha = this->register_module("mixer_mha", chobits::nn::MHA(muxer_dim, muxer_dim, muxer_dim, muxer_dim, muxer_dim * 2, num_heads));
    }
    
public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& audio,
        const torch::Tensor& video,
        const torch::Tensor& image,
        const torch::Tensor& memory
    ) {
        auto audio_o = audio + this->audio_embed;
        auto video_o = video + this->video_embed;
        auto image_o = image + this->image_embed;
        auto muxer_c = torch::cat({ audio_o, video_o }, -1);
        auto muxer_o = this->muxer_mha->forward(muxer_c, muxer_c, muxer_c);
             muxer_o = this->image_mha->forward(muxer_o, image_o, image_o);
        if(!memory.defined()) {
            auto mixer_o = this->mixer_mha->forward(muxer_o, muxer_o, muxer_o);
            return { audio_o, video_o, image_o, mixer_o };
        } else {
            auto mixer_o = this->mixer_mha->forward(memory, muxer_o, muxer_o);
            return { audio_o, video_o, image_o, mixer_o };
        }
    }

};

TORCH_MODULE(Mixer);

class TalkImpl : public torch::nn::Module {

private:
    torch::Tensor         embed{ nullptr };
    chobits::nn::MHA      mha  { nullptr };
    torch::nn::Sequential talk { nullptr };

public:
    TalkImpl(
        const int64_t i_features = 768,
        const int64_t o_features = 800,
        const int64_t num_heads  = 8
    ) {
        this->embed = this->register_parameter("embed", torch::zeros({ 1, 1, i_features }));
        this->mha   = this->register_module   ("mha",   chobits::nn::MHA(i_features, i_features, i_features, i_features, i_features * 2, num_heads));
        this->talk  = this->register_module   ("talk",  torch::nn::Sequential(
            torch::nn::Linear(i_features, o_features * 2),
            chobits::nn::Activation(),
            torch::nn::Linear(o_features * 2, o_features)
        ));
    }

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        auto out = torch::cat({ this->embed.expand({ input.size(0), -1, -1 }), input }, 1);
             out = this->mha->forward(out, out, out);
             out = out.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice() }).flatten(1);
        return torch::tanh(this->talk->forward(out));
    }

};

TORCH_MODULE(Talk);

class ChobitsImpl : public torch::nn::Module {

friend chobits::model::Trainer;

private:
    const int64_t seq_len  = 1;
    const int64_t n_fft    = 400;
    const int64_t hop_size = 80;
    const int64_t win_size = 400;
    torch::Tensor         window   { nullptr };
    chobits::nn::ViT      audio_vit{ nullptr };
    chobits::nn::ViT      video_vit{ nullptr };
    chobits::nn::ViT      image_vit{ nullptr };
    torch::nn::ModuleList mixer    {         };
    chobits::nn::Talk     talk     { nullptr };

public:
    ChobitsImpl() {
        this->audio_vit = this->register_module("audio_vit", chobits::nn::ViT(
            11, 201, 32, 128,
            std::vector<int64_t>{ 2, 2 },
            std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 5, 5 },
            std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }
        ));
        this->video_vit = this->register_module("video_vit", chobits::nn::ViT(
            360, 640, 32, 256,
            std::vector<int64_t>{ 20, 20 },
            std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 40, 40 },
            std::vector<int64_t>{  0,  0 }, std::vector<int64_t>{ 10, 10 }
        ));
        this->image_vit = this->register_module("image_vit", chobits::nn::ViT(
            360, 640, 3, 256,
            std::vector<int64_t>{ 20, 20 },
            std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 40, 40 },
            std::vector<int64_t>{  0,  0 }, std::vector<int64_t>{ 10, 10 }
        ));
        torch::nn::ModuleList mixer;
        for (int i = 0; i < 3; ++i) {
            mixer->push_back(chobits::nn::Mixer(256, 512, 512));
        }
        this->mixer = this->register_module("mixer", mixer);
        this->talk  = this->register_module("talk",  chobits::nn::Talk());
    }

public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& audio,
        const torch::Tensor& video,
        const torch::Tensor& audio_memory,
        const torch::Tensor& video_memory,
        const torch::Tensor& image_memory
    ) {
        if(!this->window.defined()) {
            this->window = torch::hann_window(this->win_size).to(audio.device());
        }
        auto com = torch::stft(
            audio.view({ -1, audio.size(2) }),
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
        // mag理论范围：0~200
        auto m = torch::abs(com);
        // pha理论范围：-PI~PI
//      auto p = torch::angle(com);
             m = torch::log10(m + 1e-8) / 8;
             m = m.view({ audio.size(0), audio.size(1), m.size(1), m.size(2) }).transpose(2, 3).contiguous();
        auto v = video.select(2,  0);
        auto i = video.select(1, -1);
        auto audio_o = this->audio_vit->forward(m, audio_memory);
        auto video_o = this->video_vit->forward(v, video_memory);
        auto image_o = this->image_vit->forward(i, image_memory);
        auto mixer_o = torch::Tensor{ nullptr };
        auto audio_m = audio_o.clone().detach();
        auto video_m = video_o.clone().detach();
        auto image_m = image_o.clone().detach();
        for (auto iter = this->mixer->begin(); iter != this->mixer->end(); ++iter) {
            auto [ audio_x, video_x, image_x, mixer_x ] = (*iter)->as<chobits::nn::Mixer>()->forward(audio_o, video_o, image_o, mixer_o);
            audio_o = audio_x;
            video_o = video_x;
            image_o = image_x;
            mixer_o = mixer_x;
        }
        auto out = this->talk->forward(mixer_o);
        return { audio_m, video_m, image_m, out };
    }

};

TORCH_MODULE(Chobits);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
