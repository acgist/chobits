import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

class Activation(nn.Module):
    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
#       return F.relu(input)
        return F.silu(input)
#       return F.leaky_relu(input)

class Patch(nn.Module):
    def __init__(
        self,
        h         : int,
        w         : int,
        i_channels: int,
        o_channels: int,
        kernel_p  : List[int],
        stride_p  : List[int],
        padding_p : List[int] = [ 0, 0 ],
        dilation_p: List[int] = [ 1, 1 ],
        kernel_h  : List[int] = [ 3, 3 ],
        stride_h  : List[int] = [ 1, 1 ],
        padding_h : List[int] = [ 1, 1 ],
        dilation_h: List[int] = [ 1, 1 ],
    ):
        super().__init__()
        seq_len = (h // stride_p[0]) * (w // stride_p[1])
        self.embed = torch.zeros(1, seq_len, o_channels)
        self.norm  = nn.LayerNorm(o_channels)
        self.conv  = nn.Sequential(
            nn.Conv2d(i_channels, o_channels, kernel_p, stride = stride_p, padding = padding_p, dilation = dilation_p),
            Activation(),
            nn.Conv2d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv(input).flatten(2).transpose(1, 2)
        out = out + self.embed
        return self.norm(out)

class Patch1d(nn.Module):
    def __init__(
        self,
        l         : int,
        i_channels: int,
        o_channels: int,
        kernel_p  : int,
        stride_p  : int,
        padding_p : int = 0,
        dilation_p: int = 1,
        kernel_h  : int = 3,
        stride_h  : int = 1,
        padding_h : int = 1,
        dilation_h: int = 1,
    ):
        super().__init__()
        seq_len = l // kernel_p
        self.embed = torch.zeros(1, seq_len, o_channels)
        self.norm  = nn.LayerNorm(o_channels)
        self.conv  = nn.Sequential(
            nn.Conv1d(i_channels, o_channels, kernel_p, stride = stride_p, padding = padding_p, dilation = dilation_p),
            Activation(),
            nn.Conv1d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv(input).transpose(1, 2)
        out = out + self.embed
        return self.norm(out)
    
class Query(nn.Module):
    def __init__(
        self,
        i_channels: int,
        o_channels: int,
        dim       : int,
        kernel    : int = 3,
        padding   : int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(i_channels, o_channels, kernel_size = kernel, padding = padding),
            Activation(),
            nn.Conv1d(o_channels, o_channels, kernel_size = kernel, padding = padding),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(input)

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
        self.norm    = nn.LayerNorm(embed_dim)
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
            expert_outs.append(expert_out)
        stacked_expert_outs  = torch.stack(expert_outs, dim = 1)
        weighted_expert_outs = stacked_expert_outs * gate_weights.unsqueeze(-1)
        final_output = weighted_expert_outs.sum(dim = 1)
        return self.norm(final_output.view(input.shape) + input)

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
        self.q    = nn.Linear(q_dim, h_dim, bias = False)
        self.k    = nn.Linear(k_dim, h_dim, bias = False)
        self.v    = nn.Linear(v_dim, h_dim, bias = False)
        self.attn = nn.MultiheadAttention(h_dim, num_heads, bias = False, batch_first = False)
        self.proj = nn.Linear(h_dim, o_dim, bias = False)
        self.ffn  = MoE(o_dim)
#       self.ffn  = Expert(o_dim)
        self.norm = nn.LayerNorm(o_dim)

    def forward(
        self,
        query: torch.Tensor,
        key  : torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q(query).transpose(0, 1)
        k = self.k(key  ).transpose(0, 1)
        v = self.v(value).transpose(0, 1)
        o, _ = self.attn(q, k, v)
        o = query + self.proj(o).transpose(0, 1)
        o = self.norm(o)
        return self.ffn(o)

class AsT(nn.Module):
    def __init__(
        self,
        l         : int,
        channels  : int,
        i_channels: int,
        o_channels: int,
        kernel_s  : int,
        kernel_l  : int,
        num_heads : int = 8,
    ):
        super().__init__()
        o_dim   = o_channels
        h_dim   = o_dim * 2
        seq_len = l // kernel_s
        self.patch_s = Patch1d(l, i_channels, o_channels, kernel_s, kernel_s)
        self.patch_l = Patch1d(l, i_channels, o_channels, kernel_l, kernel_l)
        self.query = Query(seq_len, channels, o_dim)
        self.mha   = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)
        self.out   = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input_s = self.patch_s(input)
        input_l = self.patch_l(input)
        out     = self.mha(input_s, input_l, input_l)
        query   = self.query(out)
        return self.out(query, out, out)

class ViT(nn.Module):
    def __init__(
        self,
        h         : int,
        w         : int,
        channels  : int,
        i_channels: int,
        o_channels: int,
        kernel_s  : List[int],
        kernel_l  : List[int],
        num_heads : int = 8,
    ):
        super().__init__()
        o_dim   = o_channels
        h_dim   = o_dim * 2
        seq_len = (h // kernel_s[0]) * (w // kernel_s[1])
        self.patch_s = Patch(h, w, i_channels, o_channels, kernel_s, kernel_s)
        self.patch_l = Patch(h, w, i_channels, o_channels, kernel_l, kernel_l)
        self.query = Query(seq_len, channels, o_dim)
        self.mha   = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)
        self.out   = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input_s = self.patch_s(input)
        input_l = self.patch_l(input)
        out     = self.mha(input_s, input_l, input_l)
        query   = self.query(out)
        return self.out(query, out, out)

class Mixer(nn.Module):
    def __init__(
        self,
        audio_dim: int = 256,
        video_dim: int = 512,
        h_dim    : int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.audio_mha = MHA(audio_dim, video_dim, video_dim, audio_dim, h_dim, num_heads)
        self.video_mha = MHA(video_dim, audio_dim, audio_dim, video_dim, h_dim, num_heads)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_o = self.audio_mha(audio, video, video)
        video_o = self.video_mha(video, audio, audio)
        return (audio_o, video_o)

class Talk(nn.Module):
    def __init__(
        self,
        channels  : int = 512,
        i_features: int = 256,
        o_features: int = 800,
        num_heads : int = 8,
    ):
        super().__init__()
        self.query = Query(channels, 4, i_features)
        self.mha   = MHA(i_features, i_features, i_features, i_features, i_features * 2, num_heads)
        self.out   = nn.Sequential(
            nn.Linear(i_features * 4, o_features * 2),
            nn.Tanh(),
            nn.Linear(o_features * 2, o_features),
        )

    def forward(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(audio)
        out = self.mha(query, audio, audio)
        out = out.flatten(1)
        return torch.tanh(self.out(out))
    
class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_ast = AsT(
            32 * 800, 512, 1, 256,
            100, 200,
        )
        self.video_vit = ViT(
            360, 640, 512, 32, 512,
            [ 20, 20 ], [ 40, 40 ],
        )
        self.image_vit = ViT(
            360, 640, 512, 3, 512,
            [ 20, 20 ], [ 40, 40 ],
        )
        self.image_mha = MHA(512, 512, 512, 512, 1024, 8)
        self.mixer_mha = MHA(256, 512, 512, 256, 1024, 8)
        self.mixers    = nn.ModuleList([Mixer(256, 512, 1024, 8) for _ in range(3)])
        self.talk      = Talk()

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:
        audio_o = self.audio_ast(audio.view(audio.size(0), 1, -1))
        video_i = video.select(2,  0)
        image_i = video.select(1, -1)
        video_o = self.video_vit(video_i)
        image_o = self.image_vit(image_i)
        video_o = self.image_mha(video_o, image_o, image_o)
        for mixer in self.mixers:
            audio_x, video_x = mixer(audio_o, video_o)
            audio_o = audio_x
            video_o = video_x
        audio_o = self.mixer_mha(audio_o, video_o, video_o)
        return self.talk(audio_o)

# model = MoE(512)
# model.eval()
# input = torch.randn(10, 256, 512)
# print(model(input).shape)

# model = MHA(512, 512, 512, 512)
# model.eval()
# input = (
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
# )
# print(model(*input).shape)

# model = AsT(
#     32 * 800, 512, 1, 256,
#     100, 200
# )
# model.eval()
# input = torch.randn(10, 1, 32 * 800)
# print(model(input).shape)

# model = ViT(
#     360, 640, 512, 32, 512,
#     [ 20, 20 ], [ 40, 40 ],
# )
# model.eval()
# input = torch.randn(10, 32, 360, 640)
# print(model(input).shape)

# model = Mixer()
# model.eval()
# input = (
#     torch.randn(10, 128, 256),
#     torch.randn(10, 256, 512),
# )
# audio, video = model(*input)
# print(audio.shape)
# print(video.shape)

# model = Talk()
# model.eval()
# input = torch.randn(10, 512, 256)
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
