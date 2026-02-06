import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

class Activation(nn.Module):
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        return F.silu(input)

class RoPE(nn.Module):
    def __init__(
        self,
        embed_dim  : int,
        num_heads  : int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.dim       = embed_dim
        self.num_heads = num_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, embed_dim, 2, dtype = torch.float) / embed_dim))
        t        = torch.arange(0, max_seq_len, dtype = torch.float)
        freqs    = torch.einsum("i,j->ij", t, inv_freq)
        emb      = torch.cat((freqs, freqs), dim = -1)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent = False)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent = False)

    def rotate_half(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat((-x2, x1), dim = -1)

    def forward(
        self,
        query: torch.Tensor,
        key  : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = query.view(query.size(0), query.size(1), self.num_heads, self.dim).permute(1, 2, 0, 3)
        k = key  .view(key  .size(0), key  .size(1), self.num_heads, self.dim).permute(1, 2, 0, 3)
        cos = self.cos_cached[:, :, :q.size(2), :].expand_as(q)
        sin = self.sin_cached[:, :, :k.size(2), :].expand_as(k)
        q_rotated = self.rotate_half(q)
        k_rotated = self.rotate_half(k)
        q_embed = (q * cos) + (q_rotated * sin)
        k_embed = (k * cos) + (k_rotated * sin)
        return (
            q_embed.permute(2, 0, 1, 3).view(query.size(0), query.size(1), -1),
            k_embed.permute(2, 0, 1, 3).view(key  .size(0), key  .size(1), -1)
        )
    
class Expert(nn.Module):
    def __init__(
        self,
        dim  : int,
        scale: int = 2,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * scale),
            Activation(),
            nn.Linear(dim * scale, dim),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(input)

class MoE(nn.Module):
    def __init__(
        self,
        dim        : int,
        num_experts: int = 2,
    ):
        super().__init__()
        self.gate    = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        flat_input   = input.view(-1, input.size(2))
        gate_logits  = self.gate(flat_input)
        gate_weights = F.softmax(gate_logits, dim = -1)
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(flat_input)
            expert_outputs.append(expert_output.unsqueeze(1))
        stacked_expert_outs  = torch.cat(expert_outputs, dim = 1)
        weighted_expert_outs = stacked_expert_outs * gate_weights.unsqueeze(-1)
        final_output = weighted_expert_outs.sum(dim = 1)
        return final_output.view(input.shape)

class MHA(nn.Module):
    def __init__(
        self,
        q_dim    : int,
        k_dim    : int,
        v_dim    : int,
        o_dim    : int,
        h_dim    : int = 1024,
        num_heads: int = 8,
        seq_len  : int = 256,
    ):
        super().__init__()
        self.q    = nn.Linear(q_dim, h_dim, bias = False)
        self.k    = nn.Linear(k_dim, h_dim, bias = False)
        self.v    = nn.Linear(v_dim, h_dim, bias = False)
        self.rope = RoPE(h_dim // num_heads, num_heads = num_heads, max_seq_len = seq_len)
        self.attn = nn.MultiheadAttention(h_dim, num_heads, bias = False, batch_first = False)
        self.proj = nn.Linear(h_dim, o_dim, bias = False)
        self.norm = nn.LayerNorm(o_dim)
#       self.ffn  = Expert(o_dim)
        self.ffn  = MoE(o_dim)

    def forward(
        self,
        query: torch.Tensor,
        key  : torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q(query).transpose(0, 1)
        k = self.k(key  ).transpose(0, 1)
        v = self.v(value).transpose(0, 1)
        q_embed, k_embed = self.rope(q, k)
        o, _ = self.attn(q_embed, k_embed, v)
#       o, _ = self.attn(q, k, v)
        o = self.norm(query + self.proj(o.transpose(0, 1)))
        return o + self.ffn(o)

class MQA(nn.Module):
    pass

class GQA(nn.Module):
    pass

class MLA(nn.Module):
    pass

class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim  : int,
        kernel     : List[int],
        stride     : List[int],
        height     : int,
        width      : int,
        padding    : List[int] = [ 0, 0 ],
        dilation   : List[int] = [ 1, 1 ],
        num_heads  : int       = 8,
    ):
        super().__init__()
        h_dim   = embed_dim * 2
        seq_len = (height // stride[0]) * (width // stride[1])
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel, padding = padding, dilation = dilation, stride = stride)
        self.embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.norm  = nn.LayerNorm(embed_dim)
        self.mha   = MHA(embed_dim, embed_dim, embed_dim, embed_dim, h_dim, num_heads, seq_len)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = self.patch(input).flatten(2)
        out = out.transpose(1, 2)
        out = out + self.embed
        out = self.norm(out)
        return self.mha(out, out, out)

class Mixer(nn.Module):
    def __init__(
        self,
        dim_s: int,
        dim_l: int,
        in_seq_len : int,
        out_seq_len: int = 256,
        num_heads  : int = 8,
    ):
        super().__init__()
        dim   = dim_s + dim_l
        h_dim = max(dim_s, dim_l) * 2
        self.mha   = MHA(dim,   dim,   dim,   dim  , h_dim, num_heads, in_seq_len)
#       self.s_mha = MHA(dim_s, dim_l, dim_l, dim_s, h_dim, num_heads, in_seq_len)
#       self.l_mha = MHA(dim_l, dim_s, dim_s, dim_l, h_dim, num_heads, in_seq_len)
        self.conv  = nn.Sequential(
            nn.Conv1d(in_seq_len, out_seq_len, 1, padding = 0, dilation = 1, bias = False, stride = 1),
            nn.LayerNorm(dim),
            Activation(),
        )

    def forward(
        self,
        s: torch.Tensor,
        l: torch.Tensor,
    ) -> torch.Tensor:
        o = torch.cat([ s, l ], dim = -1)
        o = self.mha(o, o, o)
#       s_o = self.s_mha(s, l, l)
#       l_o = self.l_mha(l, s, s)
#       o   = torch.cat([ s_o, l_o ], dim = -1)
        return self.conv(o)

class Muxer(nn.Module):
    def __init__(
        self,
        audio_in : int = 512,
        video_in : int = 512,
        image_in : int = 512,
        num_heads: int = 8,
        seq_len  : int = 256,
    ):
        super().__init__()
        h_dim    = max(max(audio_in, video_in), image_in) * 2
        muxer_in = audio_in + video_in
        self.audio_mha = MHA(audio_in, video_in, video_in, audio_in, h_dim, num_heads, seq_len)
        self.video_mha = MHA(video_in, audio_in, audio_in, video_in, h_dim, num_heads, seq_len)
        self.muxer_mha = MHA(muxer_in, image_in, image_in, muxer_in, h_dim, num_heads, seq_len)
        self.mixer_mha = MHA(muxer_in, muxer_in, muxer_in, muxer_in, h_dim, num_heads, seq_len)
        self.muxer_ffn = Expert(muxer_in)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
        muxer: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_o = self.audio_mha(audio, video, video)
        video_o = self.video_mha(video, audio, audio)
        muxer_c = self.muxer_ffn(torch.cat([ audio_o, video_o ], dim = -1))
        muxer_o = self.muxer_mha(muxer_c, image, image)
        if muxer is None:
            mixer_o = self.mixer_mha(muxer_o, muxer_o, muxer_o)
            return (audio_o, video_o, mixer_o)
        else:
            mixer_o = self.mixer_mha(muxer,   muxer_o, muxer_o)
#           mixer_o = self.mixer_mha(muxer_o, muxer,   muxer  )
#           mixer_o = self.mixer_mha(muxer_o, muxer_o, muxer_o) + muxer
            return (audio_o, video_o, mixer_o)

class Talk(nn.Module):
    def __init__(
        self,
        in_features : int       = 1024,
        out_features: int       = 800,
        channels    : List[int] = [ 256, 16, 1 ],
    ):
        super().__init__()
        # 注意：AI必须透明绝对不能隐藏想法
        self.talk = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], 3),
            Activation(),
            nn.Conv1d(channels[1], channels[2], 3),
            nn.Flatten(start_dim = 1),
            Activation(),
            nn.Linear(in_features - 2 * 2, out_features),
            Activation(),
            nn.Linear(out_features, out_features),
        )

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(self.talk(input))
    
class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft      = 400
        self.hop_length =  80
        self.win_length = 400
        self.window = torch.hann_window(self.win_length)
        self.audio_vit_s = ViT(32, 256, [  2,  2 ], [  2,  2 ],  11, 201)
        self.audio_vit_l = ViT(32, 256, [  5,  5 ], [  2,  2 ],  11, 201, [  1,  1 ])
        self.video_vit_s = ViT(32, 256, [ 20, 20 ], [ 20, 20 ], 360, 640)
        self.video_vit_l = ViT(32, 256, [ 40, 40 ], [ 20, 20 ], 360, 640, [ 10, 10 ])
        self.image_vit_s = ViT( 3, 256, [ 20, 20 ], [ 20, 20 ], 360, 640)
        self.image_vit_l = ViT( 3, 256, [ 40, 40 ], [ 20, 20 ], 360, 640, [ 10, 10 ])
        self.audio_mixer = Mixer(256, 256, 500)
        self.video_mixer = Mixer(256, 256, 576)
        self.image_mixer = Mixer(256, 256, 576)
        self.muxer       = nn.ModuleList([Muxer(512, 512, 512) for _ in range(3)])
        self.talk        = Talk()

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:
        com = torch.stft(
            audio.view(-1, audio.size(2)),
            n_fft          = self.n_fft,
            hop_length     = self.hop_length,
            win_length     = self.win_length,
            window         = self.window,
            center         = True,
            return_complex = True,
        )
        mag = torch.abs(com)
#       pha = torch.angle(com)
        mag = mag.view(audio.size(0), audio.size(1), mag.size(1), mag.size(2)).transpose(2, 3).contiguous()
        v = video.select(2,  0)
        i = video.select(1, -1)
        audio_s = self.audio_vit_s(mag)
        audio_l = self.audio_vit_l(mag)
        video_s = self.video_vit_s(v)
        video_l = self.video_vit_l(v)
        image_s = self.image_vit_s(i)
        image_l = self.image_vit_l(i)
        audio_o = self.audio_mixer(audio_s, audio_l)
        video_o = self.video_mixer(video_s, video_l)
        image_o = self.image_mixer(image_s, image_l)
        muxer_o = None
        for layer in self.muxer:
            audio_x, video_x, muxer_x = layer(audio_o, video_o, image_o, muxer_o)
            audio_o = audio_x
            video_o = video_x
            muxer_o = muxer_x
        return self.talk(muxer_o)

# model = RoPE(512 // 8, 8, 256)
# input = (
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
# )
# q, k = model(*input)
# print(q.shape)
# print(k.shape)

# model = Expert(512)
# input = torch.randn(10, 256, 512)
# print(model(input).shape)

# model = MoE(512)
# input = torch.randn(10, 256, 512)
# print(model(input).shape)

# model = MHA(512, 512, 512, 512)
# input = (
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
# )
# print(model(*input).shape)

# model = ViT(32, 256, [ 2, 2 ], [ 2, 2 ], 11, 201)
# input = torch.randn(10, 32, 11, 201)
# model = ViT(32, 256, [ 5, 5 ], [ 2, 2 ], 11, 201, [ 1, 1 ])
# input = torch.randn(10, 32, 11, 201)
# model = ViT(32, 256, [ 20, 20 ], [ 20, 20 ], 360, 640)
# input = torch.randn(10, 32, 360, 640)
# model = ViT(32, 256, [ 40, 40 ], [ 20, 20 ], 360, 640, [ 10, 10 ])
# input = torch.randn(10, 32, 360, 640)
# model = ViT(3, 256, [ 20, 20 ], [ 20, 20 ], 360, 640)
# input = torch.randn(10, 3, 360, 640)
# model = ViT(3, 256, [ 40, 40 ], [ 20, 20 ], 360, 640, [ 10, 10 ])
# input = torch.randn(10, 3, 360, 640)
# print(model(input).shape)

# model = Mixer(256, 256, 576)
# input = (
#     torch.randn(10, 576, 256),
#     torch.randn(10, 576, 256),
# )
# print(model(*input).shape)

# model = Muxer()
# input = (
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
# )
# audio, video, muxer = model(*input)
# print(audio.shape)
# print(video.shape)
# print(muxer.shape)

# model = Talk()
# input = torch.randn(10, 256, 1024)
# print(model(input).shape)

model = Chobits()
model.eval()
input = (
    torch.randn(10, 32, 800),
    torch.randn(10, 32, 3, 360, 640),
)
print(model(*input).shape)

# 直接保存
# torch.save(model, "D:/download/chobits.pt")

# JIT保存
torch.jit.save(torch.jit.trace(model, input), "D:/download/chobits.pt")

# ONNX保存
# batch = torch.export.Dim("batch", min = 1)
# torch.onnx.export(
#     model,
#     input,
#     "D:/download/chobits.onnx",
#     dynamo         = True,
#     opset_version  = 18,
#     input_names    = [ "audio", "video" ],
#     output_names   = [ "output" ],
#     dynamic_shapes = (
#         { 0: batch },
#         { 0: batch },
#     )
# )
