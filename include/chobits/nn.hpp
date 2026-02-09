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
        return torch::silu(input);
    }

};

TORCH_MODULE(Activation);

class ExpertImpl : public torch::nn::Module {

private:
    torch::nn::Sequential fc{ nullptr };

public:
    ExpertImpl(
        const int64_t embed_dim,
        const int64_t scale = 2
    ) {
        this->fc = this->register_module("fc", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim * scale)),
            chobits::nn::Activation(),
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
            expert_outs.push_back(expert_out.unsqueeze(1));
        }
        auto stacked_expert_outs  = torch::cat(expert_outs, 1);
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
        const int64_t num_heads = 8
    ) {
        this->q     = this->register_module("q",     torch::nn::Linear(torch::nn::LinearOptions(q_dim, h_dim).bias(false)));
        this->k     = this->register_module("k",     torch::nn::Linear(torch::nn::LinearOptions(k_dim, h_dim).bias(false)));
        this->v     = this->register_module("v",     torch::nn::Linear(torch::nn::LinearOptions(v_dim, h_dim).bias(false)));
        this->attn  = this->register_module("attn",  torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(h_dim, num_heads).bias(false)));
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
    int64_t o_seq_len = 0;
    torch::Tensor        out_embed{ nullptr };
    torch::Tensor        pos_embed{ nullptr };
    torch::nn::Conv2d    patch_s  { nullptr };
    torch::nn::Conv2d    patch_l  { nullptr };
    torch::nn::LayerNorm norm     { nullptr };
    chobits::nn::MHA     mha      { nullptr };

public:
    ViTImpl(
        const int64_t h,
        const int64_t w,
        const int64_t i_channels,
        const int64_t o_channels,
        const int64_t o_seq_len,
        const shape_t stride,
        const shape_t kernel_s,
        const shape_t kernel_l,
        const shape_t padding_s  = std::vector<int64_t>{ 0, 0 },
        const shape_t padding_l  = std::vector<int64_t>{ 0, 0 },
        const shape_t dilation_s = std::vector<int64_t>{ 1, 1 },
        const shape_t dilation_l = std::vector<int64_t>{ 1, 1 },
        const int64_t num_heads  = 8
    ) :o_seq_len(o_seq_len) {
        const int64_t dim     = o_channels;
        const int64_t o_dim   = dim   * 2;
        const int64_t h_dim   = o_dim * 2;
        const int64_t seq_len = (h / stride[0]) * (w / stride[1]);
        this->out_embed = this->register_parameter("out_embed", torch::zeros({ 1,           o_seq_len, o_dim }));
        this->pos_embed = this->register_parameter("pos_embed", torch::zeros({ 1, seq_len + o_seq_len, o_dim }));
        this->patch_s   = this->register_module   ("patch_s",   torch::nn::Conv2d(torch::nn::Conv2dOptions(i_channels, o_channels, kernel_s).padding(padding_s).dilation(dilation_s).stride(stride)));
        this->patch_l   = this->register_module   ("patch_l",   torch::nn::Conv2d(torch::nn::Conv2dOptions(i_channels, o_channels, kernel_l).padding(padding_l).dilation(dilation_l).stride(stride)));
        this->norm      = this->register_module   ("norm",      torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
        this->mha       = this->register_module   ("mha",       chobits::nn::MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads));
    }
    
public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        auto input_s = this->patch_s->forward(input).flatten(2).transpose(1, 2);
        auto input_l = this->patch_l->forward(input).flatten(2).transpose(1, 2);
        auto out = torch::cat({ input_s, input_l }, -1);
             out = torch::cat({ this->out_embed.expand({ out.size(0), -1, -1 }), out }, 1);
             out = out + this->pos_embed;
             out = this->norm->forward(out);
             out = this->mha->forward(out, out, out);
        return out.index({ torch::indexing::Slice(), torch::indexing::Slice(0, this->o_seq_len), torch::indexing::Slice() });
    }

};

TORCH_MODULE(ViT);

class MuxerImpl : public torch::nn::Module {

private:
    chobits::nn::MHA audio_mha{ nullptr };
    chobits::nn::MHA video_mha{ nullptr };
    chobits::nn::MHA muxer_mha{ nullptr };
    chobits::nn::MHA media_mha{ nullptr };
    chobits::nn::MHA mixer_mha{ nullptr };

public:
    MuxerImpl(
        const int64_t audio_in  = 512,
        const int64_t video_in  = 512,
        const int64_t image_in  = 512,
        const int64_t num_heads = 8
    ) {
        const int64_t muxer_in = audio_in + video_in;
        this->audio_mha = this->register_module("audio_mha", chobits::nn::MHA(audio_in, video_in, video_in, audio_in, audio_in * 2, num_heads));
        this->video_mha = this->register_module("video_mha", chobits::nn::MHA(video_in, audio_in, audio_in, video_in, video_in * 2, num_heads));
        this->muxer_mha = this->register_module("muxer_mha", chobits::nn::MHA(muxer_in, muxer_in, muxer_in, muxer_in, muxer_in * 2, num_heads));
        this->media_mha = this->register_module("media_mha", chobits::nn::MHA(muxer_in, image_in, image_in, muxer_in, muxer_in * 2, num_heads));
        this->mixer_mha = this->register_module("mixer_mha", chobits::nn::MHA(muxer_in, muxer_in, muxer_in, muxer_in, muxer_in * 2, num_heads));
    }
    
public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& audio,
        const torch::Tensor& video,
        const torch::Tensor& image,
        const torch::Tensor& muxer = torch::Tensor{ nullptr }
    ) {
        auto audio_o = this->audio_mha->forward(audio, video, video);
        auto video_o = this->video_mha->forward(video, audio, audio);
        auto muxer_c = torch::cat({ audio_o, video_o }, -1);
        auto muxer_o = this->muxer_mha->forward(muxer_c, muxer_c, muxer_c);
        auto media_o = this->media_mha->forward(muxer_o, image,   image  );
        if(!muxer.defined()) {
            auto mixer_o = this->mixer_mha->forward(media_o, media_o, media_o);
            return { audio_o, video_o, mixer_o };
        } else {
            auto mixer_o = this->mixer_mha->forward(media_o, media_o, media_o) + muxer;
            return { audio_o, video_o, mixer_o };
        }
    }

};

TORCH_MODULE(Muxer);

// 不能叫做Tail估计是命名有冲突
class TalkImpl : public torch::nn::Module {

private:
    int64_t h_seq_len = 0;
    torch::Tensor        embed{ nullptr };
    chobits::nn::MHA     mha  { nullptr };
    torch::nn::Sequential talk{ nullptr };

public:
    TalkImpl(
        const int64_t h_seq_len  = 1,
        const int64_t i_features = 1024,
        const int64_t o_features = 800,
        const int64_t num_heads  = 8
    ) :h_seq_len(h_seq_len) {
        const int64_t h_dim = i_features * 2;
        this->embed = this->register_parameter("embed", torch::zeros({ 1, h_seq_len, i_features }));
        this->mha   = this->register_module   ("mha",   chobits::nn::MHA(i_features, i_features, i_features, i_features, h_dim, num_heads));
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
             out = out.index({ torch::indexing::Slice(), torch::indexing::Slice(0, this->h_seq_len), torch::indexing::Slice() }).flatten(1);
        return torch::tanh(this->talk->forward(out));
    }

};

TORCH_MODULE(Talk);

class ChobitsImpl : public torch::nn::Module {

friend chobits::model::Trainer;

private:
    const int64_t n_fft    = 400;
    const int64_t hop_size =  80;
    const int64_t win_size = 400;
    torch::Tensor window { nullptr };
    chobits::nn::ViT      audio_vit{ nullptr };
    chobits::nn::ViT      video_vit{ nullptr };
    chobits::nn::ViT      image_vit{ nullptr };
    torch::nn::ModuleList muxer    {         };
    chobits::nn::Talk     talk     { nullptr };

public:
    ChobitsImpl() {
        this->audio_vit = this->register_module("audio_vit", chobits::nn::ViT(
            11, 201, 32, 256, 256,
            std::vector<int64_t>{ 2, 2 },
            std::vector<int64_t>{ 2, 2 }, std::vector<int64_t>{ 5, 5 },
            std::vector<int64_t>{ 0, 0 }, std::vector<int64_t>{ 1, 1 }
        ));
        this->video_vit = this->register_module("video_vit", chobits::nn::ViT(
            360, 640, 32, 256, 256,
            std::vector<int64_t>{ 20, 20 },
            std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 40, 40 },
            std::vector<int64_t>{  0,  0 }, std::vector<int64_t>{ 10, 10 }
        ));
        this->image_vit = this->register_module("image_vit", chobits::nn::ViT(
            360, 640, 3, 256, 256,
            std::vector<int64_t>{ 20, 20 },
            std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 40, 40 },
            std::vector<int64_t>{  0,  0 }, std::vector<int64_t>{ 10, 10 }
        ));
        torch::nn::ModuleList muxer;
        for (int i = 0; i < 3; ++i) {
            muxer->push_back(chobits::nn::Muxer(512, 512, 512));
        }
        this->muxer = this->register_module("muxer", muxer);
        this->talk  = this->register_module("talk",  chobits::nn::Talk());
    }

public:
    torch::Tensor forward(
        const torch::Tensor& audio,
        const torch::Tensor& video
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
        auto mag = torch::abs(com)   / 100.0;
//      auto pha = torch::angle(com) / 3.142;
             mag = mag.view({ audio.size(0), audio.size(1), mag.size(1), mag.size(2) }).transpose(2, 3).contiguous();
        auto v = video.select(2,  0);
        auto i = video.select(1, -1);
        auto audio_o = this->audio_vit->forward(mag);
        auto video_o = this->video_vit->forward(v);
        auto image_o = this->image_vit->forward(i);
        auto muxer_o = torch::Tensor{ nullptr };
        for (auto iter = this->muxer->begin(); iter != this->muxer->end(); ++iter) {
            auto [ audio_x, video_x, mixer_x ] = (*iter)->as<chobits::nn::Muxer>()->forward(audio_o, video_o, image_o, muxer_o);
            audio_o = audio_x;
            video_o = video_x;
            muxer_o = mixer_x;
        }
        return this->talk->forward(muxer_o);
    }

};

TORCH_MODULE(Chobits);

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
