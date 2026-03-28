import torch
import torch.nn as nn
import torch.nn.functional as F

"""
RAT = reaction
ACE ACD = audio encoder -> audio decoder = audio
VCE VCD = video encoder -> video decoder = video
ATE TAE = audio encoder -> time embed + memory -> time audio encoder
VTE TAE = video encoder -> time embed + memory -> time video encoder
ATE VTE -> ACD = audio
"""

class Activation(nn.Module):
    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return F.silu(input)
    
class FFN(nn.Module):
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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(self.fc(input) + input)

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
        self.ffn  = FFN(o_dim)
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
        self.embed = nn.Parameter(torch.zeros(1, seq_len, o_channels))
        self.norm1 = nn.LayerNorm([h // stride_p[0], w // stride_p[1]])
        self.norm2 = nn.LayerNorm(o_channels)
        self.patch = nn.Sequential(
            nn.Conv2d(i_channels, o_channels, kernel_p, stride = stride_p, padding = padding_p, dilation = dilation_p),
            Activation(),
            nn.Conv2d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
            Activation(),
            nn.Conv2d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = self.patch(input)
        out = self.norm1(out)
        out = out + self.conv(out)
        out = out.flatten(2).transpose(1, 2)
        out = out + self.embed
        return self.norm2(out)

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
        seq_len = l // stride_p
        self.embed = nn.Parameter(torch.zeros(1, seq_len, o_channels))
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(o_channels)
        self.patch = nn.Sequential(
            nn.Conv1d(i_channels, o_channels, kernel_p, stride = stride_p, padding = padding_p, dilation = dilation_p),
            Activation(),
            nn.Conv1d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
            Activation(),
            nn.Conv1d(o_channels, o_channels, kernel_h, stride = stride_h, padding = padding_h, dilation = dilation_h),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        out = self.patch(input)
        out = self.norm1(out)
        out = out + self.conv(out)
        out = out.transpose(1, 2)
        out = out + self.embed
        return self.norm2(out)

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
        self.mha  = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)
        self.conv = nn.Sequential(
            nn.Conv1d(seq_len, channels, 3, padding = 1),
            Activation(),
            nn.Conv1d(channels, channels, 3, padding = 1),
            nn.LayerNorm(o_dim),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input_s = self.patch_s(input)
        input_l = self.patch_l(input)
        out = self.mha(input_s, input_l, input_l)
        return self.conv(out)

class ViT(nn.Module):
    def __init__(
        self,
        h         : int,
        w         : int,
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
        seq_len = (h // kernel_s) * (w // kernel_s)
        self.patch_s = Patch(h, w, i_channels, o_channels, [kernel_s, kernel_s], [kernel_s, kernel_s])
        self.patch_l = Patch(h, w, i_channels, o_channels, [kernel_l, kernel_l], [kernel_l, kernel_l])
        self.mha  = MHA(o_dim, o_dim, o_dim, o_dim, h_dim, num_heads)
        self.conv = nn.Sequential(
            nn.Conv1d(seq_len, channels, 3, padding = 1),
            Activation(),
            nn.Conv1d(channels, channels, 3, padding = 1),
            nn.LayerNorm(o_dim),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input_s = self.patch_s(input)
        input_l = self.patch_l(input)
        out = self.mha(input_s, input_l, input_l)
        return self.conv(out)

class Mixer(nn.Module):
    def __init__(
        self,
        audio_dim: int = 256,
        video_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.audio_mha = MHA(audio_dim, video_dim, video_dim, audio_dim, audio_dim * 2, num_heads)
        self.video_mha = MHA(video_dim, audio_dim, audio_dim, video_dim, video_dim * 2, num_heads)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_o = self.audio_mha(audio, video, video)
        video_o = self.video_mha(video, audio, audio)
        return (audio_o, video_o)

class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_ast = AsT(32 * 800, 128,  1, 256, 40, 80)
        self.video_vit = ViT(360, 640, 128, 32, 512, 20, 40)
        self.image_vit = ViT(360, 640, 128,  3, 512, 20, 40)
        self.image_mha = MHA(512, 512, 512, 512, 1024, 8)
        self.mixer_mha = MHA(256, 512, 512, 256, 1024, 8)
        self.mixers    = nn.ModuleList([Mixer() for _ in range(3)])
        self.voice     = Voice()

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:
        audio_o = self.audio_ast(audio.view(audio.size(0), 1, -1))
        video_i = video.select(2, -1)
        image_i = video.select(1, -1)
        video_o = self.video_vit(video_i)
        image_o = self.image_vit(image_i)
        video_o = self.image_mha(video_o, image_o, image_o)
        for mixer in self.mixers:
            audio_x, video_x = mixer(audio_o, video_o)
            audio_o = audio_x
            video_o = video_x
        audio_o = self.mixer_mha(audio_o, video_o, video_o)
        # TODO 输出前一秒后一秒
        return self.voice(audio_o)

# model = MHA(512, 512, 512, 512)
# model.eval()
# input = (
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
# )
# print(model(*input).shape)

# model = Mixer()
# model.eval()
# input = (
#     torch.randn(10, 128, 256),
#     torch.randn(10, 256, 512),
# )
# audio, video = model(*input)
# print(audio.shape)
# print(video.shape)

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
