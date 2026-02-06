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

class RoPEImpl : public torch::nn::Module {

private:
    int64_t embed_dim = 0;
    int64_t num_heads = 8;
    torch::Tensor sin_cached{ nullptr };
    torch::Tensor cos_cached{ nullptr };

public:
    RoPEImpl(
        const int64_t embed_dim,
        const int64_t num_heads   = 8,
        const int64_t max_seq_len = 512
    ) : embed_dim(embed_dim), num_heads(num_heads) {
        torch::Tensor inv_freq = 1.0 / torch::pow(10000.0, torch::arange(0, embed_dim, 2, torch::kFloat) / embed_dim);
        torch::Tensor t        = torch::arange(0, max_seq_len, torch::kFloat);
        torch::Tensor freqs    = torch::outer(t, inv_freq);
        torch::Tensor emb      = torch::cat({ freqs, freqs }, -1);
        this->sin_cached = this->register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0));
        this->cos_cached = this->register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0));
    }

private:
    torch::Tensor rotate_half(
        const torch::Tensor& x
    ) {
        int64_t d = x.size(-1);
        auto x1 = x.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, d / 2) });
        auto x2 = x.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(d / 2, torch::indexing::None) });
        return torch::cat({ -x2, x1 }, -1);
    }

public:
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& query,
        const torch::Tensor& key
    ) {
        auto q = query.view({ query.size(0), query.size(1), this->num_heads, this->embed_dim }).permute({ 1, 2, 0, 3 });
        auto k = key  .view({ key  .size(0), key  .size(1), this->num_heads, this->embed_dim }).permute({ 1, 2, 0, 3 });
        auto cos = this->cos_cached.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, q.size(2)), torch::indexing::Slice() }).expand_as(q);
        auto sin = this->sin_cached.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, k.size(2)), torch::indexing::Slice() }).expand_as(k);
        auto q_rotated = this->rotate_half(q);
        auto k_rotated = this->rotate_half(k);
        auto q_embed = (q * cos) + (q_rotated * sin);
        auto k_embed = (k * cos) + (k_rotated * sin);
        return std::make_tuple(
            q_embed.permute({ 2, 0, 1, 3 }).view({ query.size(0), query.size(1), -1 }),
            k_embed.permute({ 2, 0, 1, 3 }).view({ key  .size(0), key  .size(1), -1 })
        );
    }

};

TORCH_MODULE(RoPE);

class ExpertImpl : public torch::nn::Module {

private:
    torch::nn::Sequential fc{ nullptr };

public:
    ExpertImpl(
        int64_t dim,
        int64_t scale = 2
    ) {
        this->fc = this->register_module("fc", torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(dim, dim * scale)),
            chobits::nn::Activation(),
            torch::nn::Linear(torch::nn::LinearOptions(dim * scale, dim))
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
        const int64_t dim,
        const int64_t num_experts = 2
    ) {
        torch::nn::ModuleList experts;
        for (int i = 0; i < num_experts; ++i) {
            experts->push_back(chobits::nn::Expert(dim));
        }
        this->gate    = this->register_module("gate",    torch::nn::Linear(dim, num_experts));
        this->experts = this->register_module("experts", experts);
    }

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        auto flat_input   = input.view({ -1, input.size(2) });
        auto gate_logits  = this->gate->forward(flat_input);
        auto gate_weights = torch::softmax(gate_logits, -1);
        std::vector<torch::Tensor> expert_outputs;
        for (auto iter = this->experts->begin(); iter != this->experts->end(); ++iter) {
            auto expert_output = (*iter)->as<chobits::nn::Expert>()->forward(flat_input);
            expert_outputs.push_back(expert_output.unsqueeze(1));
        }
        auto stacked_expert_outs  = torch::cat(expert_outputs, 1);
        auto weighted_expert_outs = stacked_expert_outs * gate_weights.unsqueeze(-1);
        auto final_output = weighted_expert_outs.sum(1);
        return final_output.view(input.sizes());
    }

};

TORCH_MODULE(MoE);

class MHAImpl : public torch::nn::Module {

private:
    torch::nn::Linear             q   { nullptr };
    torch::nn::Linear             k   { nullptr };
    torch::nn::Linear             v   { nullptr };
    chobits::nn::RoPE             rope{ nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };
    torch::nn::Linear             proj{ nullptr };
    torch::nn::LayerNorm          norm{ nullptr };
//  chobits::nn::Expert           ffn { nullptr };
    chobits::nn::MoE              ffn { nullptr };

public:
    MHAImpl(
        const int64_t q_dim,
        const int64_t k_dim,
        const int64_t v_dim,
        const int64_t o_dim,
        const int64_t h_dim     = 1024,
        const int64_t num_heads = 8,
        const int64_t seq_len   = 256
    ) {
        this->q    = this->register_module("q",    torch::nn::Linear(torch::nn::LinearOptions(q_dim, h_dim).bias(false)));
        this->k    = this->register_module("k",    torch::nn::Linear(torch::nn::LinearOptions(k_dim, h_dim).bias(false)));
        this->v    = this->register_module("v",    torch::nn::Linear(torch::nn::LinearOptions(v_dim, h_dim).bias(false)));
        this->rope = this->register_module("rope", chobits::nn::RoPE(h_dim / num_heads, num_heads, seq_len));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(h_dim, num_heads).bias(false)));
        this->proj = this->register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(h_dim, o_dim).bias(false)));
        this->norm = this->register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
//      this->ffn  = this->register_module("ffn",  chobits::nn::Expert(o_dim));
        this->ffn  = this->register_module("ffn",  chobits::nn::MoE(o_dim));
    }

public:
    torch::Tensor forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value
    ) {
        auto q = this->q->forward(query).transpose(0, 1);
        auto k = this->k->forward(key  ).transpose(0, 1);
        auto v = this->v->forward(value).transpose(0, 1);
        auto [ q_embed, k_embed ] = this->rope->forward(q, k);
        auto [ o, _ ] = this->attn->forward(q_embed, k_embed, v);
//      auto [ o, _ ] = this->attn->forward(q, k, v);
        o = this->norm->forward(query + this->proj->forward(o.transpose(0, 1)));
        return o + this->ffn->forward(o);
    }

};

TORCH_MODULE(MHA);

class ViTImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d    patch{ nullptr };
    torch::nn::LayerNorm norm { nullptr };
    torch::Tensor        embed{ nullptr };
    chobits::nn::MHA     mha  { nullptr };

public:
    ViTImpl(
        const int64_t in_channels,
        const int64_t embed_dim,
        const shape_t kernel,
        const shape_t stride,
        const int64_t height,
        const int64_t width,
        const shape_t padding   = std::vector<int64_t>{ 0, 0 },
        const shape_t dilation  = std::vector<int64_t>{ 1, 1 },
        const int64_t num_heads = 8
    ) {
        const int64_t h_dim   = embed_dim * 2;
        const int64_t seq_len = (height / stride[0]) * (width / stride[1]);
        this->patch = this->register_module   ("patch", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, embed_dim, kernel).padding(padding).dilation(dilation).stride(stride)));
        this->embed = this->register_parameter("embed", torch::zeros({ 1, seq_len, embed_dim }));
        this->norm  = this->register_module   ("norm",  torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ embed_dim })));
        this->mha   = this->register_module   ("mha",   chobits::nn::MHA(embed_dim, embed_dim, embed_dim, embed_dim, h_dim, num_heads, seq_len));
    }

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        auto out = this->patch->forward(input).flatten(2);
             out = out.transpose(1, 2);
             out = out + this->embed;
             out = this->norm->forward(out);
        return this->mha->forward(out, out, out);
    }

};

TORCH_MODULE(ViT);

class MixerImpl : public torch::nn::Module {

private:
    chobits::nn::MHA      mha  { nullptr };
//  chobits::nn::MHA      a_mha{ nullptr };
//  chobits::nn::MHA      z_mha{ nullptr };
    torch::nn::Sequential conv { nullptr };

public:
    MixerImpl(
        const int64_t dim_a,
        const int64_t dim_z,
        const int64_t in_seq_len,
        const int64_t out_seq_len = 256,
        const int64_t num_heads   = 8
    ) {
        const int64_t dim   = dim_a + dim_z;
        const int64_t h_dim = std::max(dim_a, dim_z) * 2;
        this->mha   = this->register_module("mha",   chobits::nn::MHA(dim,   dim,   dim,   dim,   h_dim, num_heads, in_seq_len));
//      this->s_mha = this->register_module("s_mha", chobits::nn::MHA(dim_s, dim_l, dim_l, dim_s, h_dim, num_heads, in_seq_len));
//      this->l_mha = this->register_module("l_mha", chobits::nn::MHA(dim_l, dim_s, dim_s, dim_l, h_dim, num_heads, in_seq_len));
        this->conv = this->register_module ("conv",  torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(in_seq_len, out_seq_len, 1).padding(0).dilation(1).bias(false).stride(1)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ dim })),
            chobits::nn::Activation()
        ));
    }
    
public:
    torch::Tensor forward(
        const torch::Tensor& s,
        const torch::Tensor& l
    ) {
        auto o = torch::cat({ s, l }, -1);
             o = this->mha->forward(o, o, o);
//      auto s_o = this->s_mha->forward(s, l, l);
//      auto l_o = this->l_mha->forward(l, s, s);
//      auto o   = torch::cat({ s_o, l_o }, -1);
        return this->conv->forward(o);
    }

};

TORCH_MODULE(Mixer);

class MuxerImpl : public torch::nn::Module {

private:
    chobits::nn::MHA    audio_mha{ nullptr };
    chobits::nn::MHA    video_mha{ nullptr };
    chobits::nn::MHA    muxer_mha{ nullptr };
    chobits::nn::MHA    mixer_mha{ nullptr };
    chobits::nn::Expert muxer_ffn{ nullptr };

public:
    MuxerImpl(
        const int64_t audio_in  = 512,
        const int64_t video_in  = 512,
        const int64_t image_in  = 512,
        const int64_t num_heads = 8,
        const int64_t seq_len   = 256
    ) {
        const int64_t h_dim    = std::max(std::max(audio_in, video_in), image_in) * 2;
        const int64_t muxer_in = audio_in + video_in;
        this->audio_mha = this->register_module("audio_mha", chobits::nn::MHA(audio_in, video_in, video_in, audio_in, h_dim, num_heads, seq_len));
        this->video_mha = this->register_module("video_mha", chobits::nn::MHA(video_in, audio_in, audio_in, video_in, h_dim, num_heads, seq_len));
        this->muxer_mha = this->register_module("muxer_mha", chobits::nn::MHA(muxer_in, image_in, image_in, muxer_in, h_dim, num_heads, seq_len));
        this->mixer_mha = this->register_module("mixer_mha", chobits::nn::MHA(muxer_in, muxer_in, muxer_in, muxer_in, h_dim, num_heads, seq_len));
        this->muxer_ffn = this->register_module("muxer_ffn", chobits::nn::Expert(muxer_in));
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
        auto muxer_c = this->muxer_ffn->forward(torch::cat({ audio_o, video_o }, -1));
        auto muxer_o = this->muxer_mha->forward(muxer_c, image, image);
        if(!muxer.defined()) {
            auto mixer_o = this->mixer_mha->forward(muxer_o, muxer_o, muxer_o);
            return { audio_o, video_o, mixer_o };
        } else {
            auto mixer_o = this->mixer_mha->forward(muxer,   muxer_o, muxer_o);
//          auto mixer_o = this->mixer_mha->forward(muxer_o, muxer,   muxer  );
//          auto mixer_o = this->mixer_mha->forward(muxer_o, muxer_o, muxer_o) + muxer;
            return { audio_o, video_o, mixer_o };
        }
    }

};

TORCH_MODULE(Muxer);

// 不能叫做Tail估计是命名有冲突
class TalkImpl : public torch::nn::Module {

private:
    torch::nn::Sequential talk{ nullptr };

public:
    TalkImpl(
        const int64_t in_features  = 1024,
        const int64_t out_features = 800,
        const shape_t channel      = std::vector<int64_t>{ 256, 16, 1 }
    ) {
        // 注意：AI必须透明绝对不能隐藏想法
        this->talk = this->register_module("talk", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[0], channel[1], 3)),
            chobits::nn::Activation(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(channel[1], channel[2], 3)),
            torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
            chobits::nn::Activation(),
            torch::nn::Linear(in_features - 2 * 2, out_features),
            chobits::nn::Activation(),
            torch::nn::Linear(out_features, out_features)
        ));
    }

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        return torch::tanh(this->talk->forward(input));
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
    chobits::nn::ViT      audio_vit_s{ nullptr };
    chobits::nn::ViT      audio_vit_l{ nullptr };
    chobits::nn::ViT      video_vit_s{ nullptr };
    chobits::nn::ViT      video_vit_l{ nullptr };
    chobits::nn::ViT      image_vit_s{ nullptr };
    chobits::nn::ViT      image_vit_l{ nullptr };
    chobits::nn::Mixer    audio_mixer{ nullptr };
    chobits::nn::Mixer    video_mixer{ nullptr };
    chobits::nn::Mixer    image_mixer{ nullptr };
    torch::nn::ModuleList muxer      {         };
    chobits::nn::Talk     talk       { nullptr };

public:
    ChobitsImpl() {
        this->audio_vit_s = this->register_module("audio_vit_s", chobits::nn::ViT(32, 256, std::vector<int64_t>{  2,  2 }, std::vector<int64_t>{  2,  2 },  11, 201));
        this->audio_vit_l = this->register_module("audio_vit_l", chobits::nn::ViT(32, 256, std::vector<int64_t>{  5,  5 }, std::vector<int64_t>{  2,  2 },  11, 201, std::vector<int64_t>{  1,  1 }));
        this->video_vit_s = this->register_module("video_vit_s", chobits::nn::ViT(32, 256, std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 20, 20 }, 360, 640));
        this->video_vit_l = this->register_module("video_vit_l", chobits::nn::ViT(32, 256, std::vector<int64_t>{ 40, 40 }, std::vector<int64_t>{ 40, 40 }, 360, 640, std::vector<int64_t>{ 10, 10 }));
        this->image_vit_s = this->register_module("image_vit_s", chobits::nn::ViT( 3, 256, std::vector<int64_t>{ 20, 20 }, std::vector<int64_t>{ 20, 20 }, 360, 640));
        this->image_vit_l = this->register_module("image_vit_l", chobits::nn::ViT( 3, 256, std::vector<int64_t>{ 40, 40 }, std::vector<int64_t>{ 40, 40 }, 360, 640, std::vector<int64_t>{ 10, 10 }));
        this->audio_mixer = this->register_module("audio_mixer", chobits::nn::Mixer(256, 256, 500));
        this->video_mixer = this->register_module("video_mixer", chobits::nn::Mixer(256, 256, 576));
        this->image_mixer = this->register_module("image_mixer", chobits::nn::Mixer(256, 256, 576));
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
        auto mag = torch::abs(com);
//      auto pha = torch::angle(com);
             mag = mag.view({ audio.size(0), audio.size(1), mag.size(1), mag.size(2) }).transpose(2, 3).contiguous();
        auto v = video.select(2,  0);
        auto i = video.select(1, -1);
        auto audio_s = this->audio_vit_s->forward(mag);
        auto audio_l = this->audio_vit_l->forward(mag);
        auto video_s = this->video_vit_s->forward(v);
        auto video_l = this->video_vit_l->forward(v);
        auto image_s = this->image_vit_s->forward(i);
        auto image_l = this->image_vit_l->forward(i);
        auto audio_o = this->audio_mixer->forward(audio_s, audio_l);
        auto video_o = this->video_mixer->forward(video_s, video_l);
        auto image_o = this->image_mixer->forward(image_s, image_l);
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
