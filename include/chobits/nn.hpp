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

static void init_weight(std::shared_ptr<torch::nn::Module> layer);

class ActivationImpl : public torch::nn::Module {

public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
//      return torch::relu(input);
//      return torch::silu(input);
        return torch::leaky_relu(input);
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

    void init() {
        init_weight(this->fc.ptr());
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
        return final_output.view(input.sizes()) + input;
    }

    void init() {
        init_weight(this->gate.ptr());
        init_weight(this->experts.ptr());
    }

};

TORCH_MODULE(MoE);

class MHAImpl : public torch::nn::Module {

private:
    torch::nn::Linear             q   { nullptr };
    torch::nn::Linear             k   { nullptr };
    torch::nn::Linear             v   { nullptr };
    torch::nn::MultiheadAttention attn{ nullptr };
    torch::nn::Linear             proj{ nullptr };
    chobits::nn::MoE              ffn { nullptr };
//  chobits::nn::Expert           ffn { nullptr };
    torch::nn::LayerNorm          norm{ nullptr };

public:
    MHAImpl(
        const int64_t q_dim,
        const int64_t k_dim,
        const int64_t v_dim,
        const int64_t o_dim,
        const int64_t h_dim     = 1024,
        const int64_t num_heads = 8
    ) {
        this->q    = this->register_module("q",    torch::nn::Linear(torch::nn::LinearOptions(q_dim, h_dim).bias(false)));
        this->k    = this->register_module("k",    torch::nn::Linear(torch::nn::LinearOptions(k_dim, h_dim).bias(false)));
        this->v    = this->register_module("v",    torch::nn::Linear(torch::nn::LinearOptions(v_dim, h_dim).bias(false)));
        this->attn = this->register_module("attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(h_dim, num_heads).bias(false)));
        this->proj = this->register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(h_dim, o_dim).bias(false)));
        this->ffn  = this->register_module("ffn",  chobits::nn::MoE(o_dim));
//      this->ffn  = this->register_module("ffn",  chobits::nn::Expert(o_dim));
        this->norm = this->register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
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
        auto [ o, _ ] = this->attn->forward(q, k, v);
             o = query + this->proj->forward(o).transpose(0, 1);
             o = this->norm->forward(o);
        return this->ffn->forward(o);
    }

    void init() {
        init_weight(this->q.ptr());
        init_weight(this->k.ptr());
        init_weight(this->v.ptr());
        init_weight(this->attn.ptr());
        init_weight(this->proj.ptr());
        init_weight(this->ffn.ptr());
        init_weight(this->norm.ptr());
    }

};

TORCH_MODULE(MHA);

class ViTImpl : public torch::nn::Module {

private:
    torch::Tensor         embed_s{ nullptr };
    torch::Tensor         embed_l{ nullptr };
    torch::nn::Sequential patch_s{ nullptr };
    torch::nn::Sequential patch_l{ nullptr };
    torch::nn::LayerNorm  norm   { nullptr };
    chobits::nn::MHA      mha    { nullptr };

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
        const shape_t stride_h   = std::vector<int64_t>{ 1, 1 },
        const shape_t kernel_h   = std::vector<int64_t>{ 3, 3 },
        const shape_t padding_h  = std::vector<int64_t>{ 1, 1 },
        const shape_t dilation_h = std::vector<int64_t>{ 1, 1 },
        const int64_t num_heads  = 8
    ) {
        const int64_t dim     = o_channels;
        const int64_t o_dim   = dim   * 2;
        const int64_t h_dim   = o_dim * 2;
        const int64_t seq_len = (h / stride[0]) * (w / stride[1]);
        this->embed_s = this->register_parameter("embed_s", torch::zeros({ 1, seq_len, dim }));
        this->embed_l = this->register_parameter("embed_l", torch::zeros({ 1, seq_len, dim }));
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
        this->norm = this->register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{ o_dim })));
        this->mha  = this->register_module("mha",  chobits::nn::MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads));
    }
    
public:
    torch::Tensor forward(
        const torch::Tensor& input
    ) {
        auto input_s = this->patch_s->forward(input).flatten(2).transpose(1, 2);
        auto input_l = this->patch_l->forward(input).flatten(2).transpose(1, 2);
             input_s = input_s + this->embed_s;
             input_l = input_l + this->embed_l;
        auto out = torch::cat({ input_s, input_l }, -1);
             out = this->norm->forward(out);
        return this->mha->forward(out, out, out);
    }

    void init() {
        init_weight(this->patch_s.ptr());
        init_weight(this->patch_l.ptr());
        init_weight(this->norm.ptr());
        init_weight(this->mha.ptr());
    }

};

TORCH_MODULE(ViT);

class MixerImpl : public torch::nn::Module {

private:
    chobits::nn::MHA audio_mha{ nullptr };
    chobits::nn::MHA video_mha{ nullptr };
    chobits::nn::MHA muxer_mha{ nullptr };
    chobits::nn::MHA mixer_mha{ nullptr };

public:
    MixerImpl(
        const int64_t audio_dim = 256,
        const int64_t video_dim = 512,
        const int64_t h_dim     = 1024,
        const int64_t num_heads = 8
    ) {
        this->audio_mha = this->register_module("audio_mha", chobits::nn::MHA(audio_dim, video_dim, video_dim, audio_dim, h_dim, num_heads));
        this->video_mha = this->register_module("video_mha", chobits::nn::MHA(video_dim, audio_dim, audio_dim, video_dim, h_dim, num_heads));
        this->muxer_mha = this->register_module("muxer_mha", chobits::nn::MHA(audio_dim, video_dim, video_dim, audio_dim, h_dim, num_heads));
        this->mixer_mha = this->register_module("mixer_mha", chobits::nn::MHA(audio_dim, audio_dim, audio_dim, audio_dim, h_dim, num_heads));
    }
    
public:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& audio,
        const torch::Tensor& video,
        const torch::Tensor& input
    ) {
        auto audio_o = this->audio_mha->forward(audio,   video,   video  );
        auto video_o = this->video_mha->forward(video,   audio,   audio  );
        auto muxer_o = this->muxer_mha->forward(audio_o, video_o, video_o);
        if(input.defined()) {
            auto mixer_o = this->mixer_mha->forward(muxer_o, input, input);
            return { audio_o, video_o, mixer_o };
        } else {
            auto mixer_o = this->mixer_mha->forward(muxer_o, muxer_o, muxer_o);
            return { audio_o, video_o, mixer_o };
        }
    }

    void init() {
        init_weight(this->audio_mha.ptr());
        init_weight(this->video_mha.ptr());
        init_weight(this->muxer_mha.ptr());
        init_weight(this->mixer_mha.ptr());
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
        const int64_t i_features = 256,
        const int64_t o_features = 800,
        const int64_t num_heads  = 8
    ) {
        this->embed = this->register_parameter("embed", torch::zeros({ 1, 4, i_features }));
        this->mha   = this->register_module   ("mha",   chobits::nn::MHA(i_features, i_features, i_features, i_features, i_features * 2, num_heads));
        this->talk  = this->register_module   ("talk",  torch::nn::Sequential(
            torch::nn::Linear(i_features * 4, o_features * 2),
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
             out = out.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4), torch::indexing::Slice() }).flatten(1);
        return torch::tanh(this->talk->forward(out));
    }

    void init() {
        init_weight(this->mha.ptr());
        init_weight(this->talk.ptr());
    }

};

TORCH_MODULE(Talk);

class ChobitsImpl : public torch::nn::Module {

friend chobits::model::Trainer;

private:
    const int64_t n_fft    = 400;
    const int64_t hop_size = 80;
    const int64_t win_size = 400;
    torch::Tensor         window   { nullptr };
    chobits::nn::ViT      audio_vit{ nullptr };
    chobits::nn::ViT      video_vit{ nullptr };
    chobits::nn::ViT      image_vit{ nullptr };
    chobits::nn::MHA      video_mha{ nullptr };
    torch::nn::ModuleList mixers   {         };
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
        torch::nn::ModuleList mixers;
        for (int i = 0; i < 3; ++i) {
            mixers->push_back(chobits::nn::Mixer(256, 512, 512));
        }
        this->video_mha = this->register_module("video_mha", chobits::nn::MHA(512, 512, 512, 512, 1024, 8));
        this->mixers    = this->register_module("mixers",    mixers);
        this->talk      = this->register_module("talk",      chobits::nn::Talk());
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
        // mag理论范围：0~200
        auto m = torch::abs(com);
        // pha理论范围：-PI~PI
//      auto p = torch::angle(com);
             m = torch::log10(m + 1e-8) / 8;
             m = m.view({ audio.size(0), audio.size(1), m.size(1), m.size(2) }).transpose(2, 3).contiguous();
        auto v = video.select(2,  0);
        auto i = video.select(1, -1);
        auto audio_o = this->audio_vit->forward(m);
        auto video_o = this->video_vit->forward(v);
        auto image_o = this->image_vit->forward(i);
             video_o = this->video_mha->forward(video_o, image_o, image_o);
        torch::Tensor mixer_o{ nullptr };
        for (auto iter = this->mixers->begin(); iter != this->mixers->end(); ++iter) {
            auto [ audio_x, video_x, mixer_x ] = (*iter)->as<chobits::nn::Mixer>()->forward(audio_o, video_o, mixer_o);
            audio_o = audio_x;
            video_o = video_x;
            mixer_o = mixer_x;
        }
        return this->talk->forward(mixer_o);
    }

    void init() {
        init_weight(this->audio_vit.ptr());
        init_weight(this->video_vit.ptr());
        init_weight(this->image_vit.ptr());
        init_weight(this->video_mha.ptr());
        init_weight(this->mixers.ptr());
        init_weight(this->talk.ptr());
    }

};

TORCH_MODULE(Chobits);

static void init_weight(std::shared_ptr<torch::nn::Module> module) {
    if(auto* layer = module->as<torch::nn::Conv2d>()) {
        layer->reset_parameters();
    } else if(auto* layer = module->as<torch::nn::Linear>()) {
        layer->reset_parameters();
    } else if(auto* layer = module->as<torch::nn::LayerNorm>()) {
        layer->reset_parameters();
    } else if(auto* layer = module->as<torch::nn::MultiheadAttention>()) {
        layer->_reset_parameters();
    } else if(auto* layer = module->as<chobits::nn::MoE>()) {
        layer->init();
    } else if(auto* layer = module->as<chobits::nn::MHA>()) {
        layer->init();
    } else if(auto* layer = module->as<chobits::nn::ViT>()) {
        layer->init();
    } else if(auto* layer = module->as<chobits::nn::Talk>()) {
        layer->init();
    } else if(auto* layer = module->as<chobits::nn::Mixer>()) {
        layer->init();
    } else if(auto* layer = module->as<chobits::nn::Expert>()) {
        layer->init();
    } else if(auto* layer = module->as<torch::nn::Sequential>()) {
        for(auto value : layer->children()) {
            init_weight(value);
        }
    } else if(auto* layer = module->as<torch::nn::ModuleList>()) {
        for(auto value : layer->children()) {
            init_weight(value);
        }
    }
}

} // END OF chobits::nn

#endif // CHOBITS_NN_HPP
