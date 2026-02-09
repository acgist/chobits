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

class Expert(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        scale    : int = 2,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * scale),
            Activation(),
            nn.Linear(embed_dim * scale, embed_dim),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(input)

class MoE(nn.Module):
    def __init__(
        self,
        embed_dim  : int,
        num_experts: int = 2,
    ):
        super().__init__()
        self.gate    = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([Expert(embed_dim) for _ in range(num_experts)])

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        flat_input   = input.view(-1, input.size(2))
        gate_logits  = self.gate(flat_input)
        gate_weights = F.softmax(gate_logits, dim = -1)
        expert_outs  = []
        for expert in self.experts:
            expert_out = expert(flat_input)
            expert_outs.append(expert_out.unsqueeze(1))
        stacked_expert_outs  = torch.cat(expert_outs, dim = 1)
        weighted_expert_outs = stacked_expert_outs * gate_weights.unsqueeze(-1)
        final_output = weighted_expert_outs.sum(dim = 1)
        return final_output.view(input.shape)

"""
MHA MQA GQA MLA
"""
class MHA(nn.Module):
    def __init__(
        self,
        q_dim    : int,
        k_dim    : int,
        v_dim    : int,
        o_dim    : int,
        h_dim    : int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.q     = nn.Linear(q_dim, h_dim, bias = False)
        self.k     = nn.Linear(k_dim, h_dim, bias = False)
        self.v     = nn.Linear(v_dim, h_dim, bias = False)
        self.attn  = nn.MultiheadAttention(h_dim, num_heads, bias = False, batch_first = False)
        self.proj  = nn.Linear(h_dim, o_dim, bias = False)
        self.ffn   = MoE(o_dim)
#       self.ffn   = Expert(o_dim)
        self.norm1 = nn.LayerNorm(o_dim)
        self.norm2 = nn.LayerNorm(o_dim)

    def forward(
        self,
        query: torch.Tensor,
        key  : torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # [ N S L ] -> [ S N L ]
        q = self.q(query.transpose(0, 1))
        k = self.k(key  .transpose(0, 1))
        v = self.v(value.transpose(0, 1))
        o, _ = self.attn(q, k, v)
        o = query + self.proj(o.transpose(0, 1))
        o = self.norm1(o)
        o = o + self.ffn(o)
        return self.norm2(o)

class ViT(nn.Module):
    def __init__(
        self,
        h         : int,
        w         : int,
        i_channels: int,
        o_channels: int,
        o_seq_len : int,
        stride    : List[int],
        kernel_s  : List[int],
        kernel_l  : List[int],
        padding_s : List[int] = [ 0, 0 ],
        padding_l : List[int] = [ 0, 0 ],
        dilation_s: List[int] = [ 1, 1 ],
        dilation_l: List[int] = [ 1, 1 ],
        num_heads : int       = 8,
    ):
        super().__init__()
        dim     = o_channels
        o_dim   = dim   * 2
        h_dim   = o_dim * 2
        seq_len = (h // stride[0]) * (w // stride[1])
        self.o_seq_len = o_seq_len
        self.out_embed = nn.Parameter(torch.zeros(1,           o_seq_len, o_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + o_seq_len, o_dim))
        self.patch_s   = nn.Conv2d(i_channels, o_channels, kernel_s, padding = padding_s, dilation = dilation_s, stride = stride)
        self.patch_l   = nn.Conv2d(i_channels, o_channels, kernel_l, padding = padding_l, dilation = dilation_l, stride = stride)
        self.norm      = nn.LayerNorm(o_dim)
        self.mha       = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input_s = self.patch_s(input).flatten(2).transpose(1, 2)
        input_l = self.patch_l(input).flatten(2).transpose(1, 2)
        out = torch.cat([ input_s, input_l ], dim = -1)
        out = torch.cat((self.out_embed.expand(out.size(0), -1, -1), out), dim = 1)
        out = out + self.pos_embed
        out = self.norm(out)
        out = self.mha(out, out, out)
        return out[:, 0:self.o_seq_len, :]

class Muxer(nn.Module):
    def __init__(
        self,
        audio_in : int = 512,
        video_in : int = 512,
        image_in : int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        muxer_in = audio_in + video_in
        self.audio_mha = MHA(audio_in, video_in, video_in, audio_in, audio_in * 2, num_heads)
        self.video_mha = MHA(video_in, audio_in, audio_in, video_in, video_in * 2, num_heads)
        self.muxer_mha = MHA(muxer_in, muxer_in, muxer_in, muxer_in, muxer_in * 2, num_heads)
        self.media_mha = MHA(muxer_in, image_in, image_in, muxer_in, muxer_in * 2, num_heads)
        self.mixer_mha = MHA(muxer_in, muxer_in, muxer_in, muxer_in, muxer_in * 2, num_heads)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
        muxer: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_o = self.audio_mha(audio, video, video)
        video_o = self.video_mha(video, audio, audio)
        muxer_c = torch.cat([ audio_o, video_o ], dim = -1)
        muxer_o = self.muxer_mha(muxer_c, muxer_c, muxer_c)
        media_o = self.media_mha(muxer_o, image,   image  )
        if muxer is None:
            mixer_o = self.mixer_mha(media_o, media_o, media_o)
            return (audio_o, video_o, mixer_o)
        else:
            mixer_o = self.mixer_mha(media_o, media_o, media_o) + muxer
            return (audio_o, video_o, mixer_o)

class Talk(nn.Module):
    def __init__(
        self,
        h_seq_len : int = 1,
        i_features: int = 1024,
        o_features: int = 800,
        num_heads : int = 8,
    ):
        super().__init__()
        h_dim = i_features * 2
        self.h_seq_len = h_seq_len
        self.embed = nn.Parameter(torch.zeros(1, h_seq_len, i_features))
        self.mha   = MHA(i_features, i_features, i_features, i_features, h_dim, num_heads)
        self.talk  = nn.Sequential(
            nn.Linear(i_features, o_features * 2),
            Activation(),
            nn.Linear(o_features * 2, o_features),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.cat((self.embed.expand(input.size(0), -1, -1), input), dim = 1)
        out = self.mha(out, out, out)
        out = out[:, 0:self.h_seq_len, :].flatten(1)
        return torch.tanh(self.talk(out))
    
class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft      = 400
        self.hop_length =  80
        self.win_length = 400
        self.window = torch.hann_window(self.win_length)
        self.audio_vit = ViT(
            11, 201, 32, 256, 256,
            [ 2, 2 ],
            [ 2, 2 ], [ 5, 5 ],
            [ 0, 0 ], [ 1, 1 ],
        )
        self.video_vit = ViT(
            360, 640, 32, 256, 256,
            [ 20, 20 ],
            [ 20, 20 ], [ 40, 40 ],
            [  0,  0 ], [ 10, 10 ],
        )
        self.image_vit = ViT(
            360, 640, 3, 256, 256,
            [ 20, 20 ],
            [ 20, 20 ], [ 40, 40 ],
            [  0,  0 ], [ 10, 10 ],
        )
        self.muxer = nn.ModuleList([Muxer(512, 512, 512) for _ in range(3)])
        self.talk  = Talk()

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
        mag = torch.abs(com)   / 100.0
#       pha = torch.angle(com) / 3.142
        mag = mag.view(audio.size(0), audio.size(1), mag.size(1), mag.size(2)).transpose(2, 3).contiguous()
        v = video.select(2,  0)
        i = video.select(1, -1)
        audio_o = self.audio_vit(mag)
        video_o = self.video_vit(v)
        image_o = self.image_vit(i)
        muxer_o = None
        for layer in self.muxer:
            audio_x, video_x, muxer_x = layer(audio_o, video_o, image_o, muxer_o)
            audio_o = audio_x
            video_o = video_x
            muxer_o = muxer_x
        return self.talk(muxer_o)

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

# model = ViT(
#     11, 201, 32, 256, 256,
#     [ 2, 2 ],
#     [ 2, 2 ], [ 5, 5 ],
#     [ 0, 0 ], [ 1, 1 ],
# )
# input = (torch.randn(10, 32, 11, 201))
# print(model(input).shape)

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
