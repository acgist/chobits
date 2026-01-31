import math
import torch
import torch.nn as nn

from typing import List, Tuple

layer_act = nn.SiLU

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim      : int,
        max_len  : int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
        t = torch.arange(0, max_len, dtype = torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent = False)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent = False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat((-x2, x1), dim = -1)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = query.view(query.size(0), query.size(1), self.num_heads, self.dim).transpose(1, 2)
        k = key  .view(key  .size(0), key  .size(1), self.num_heads, self.dim).transpose(1, 2)
        cos = self.cos_cached[:, :, :q.size(2), :].expand_as(q)
        sin = self.sin_cached[:, :, :k.size(2), :].expand_as(k)
        q_rotated = self.rotate_half(q)
        k_rotated = self.rotate_half(k)
        q_embed = (q * cos) + (q_rotated * sin)
        k_embed = (k * cos) + (k_rotated * sin)
        return \
            (
                q_embed.transpose(1, 2).view(query.size(0), query.size(1), -1),
                k_embed.transpose(1, 2).view(key  .size(0), key  .size(1), -1)
            )

class ResNet1dBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        features    : int,
        stride      : int = 1,
        kernel      : int = 3,
        padding     : int = 1,
        dilation    : int = 1,
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            nn.LayerNorm(features),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding = 1, dilation = 1),
            layer_act(),
            nn.Conv1d(in_channels, in_channels, 3, padding = 1, dilation = 1),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(in_channels, in_channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(in_channels, in_channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv2 = self.cv2(input)
        cv3 = self.cv3(cv2)
        cv4 = self.cv4(input)
        return self.cv1(torch.cat([ cv4 + cv3, cv2 ], dim = 1))
    
class ResNet2dBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        features    : List[int],
        stride      : List[int] = [ 1, 1 ],
        kernel      : List[int] = [ 3, 3 ],
        padding     : List[int] = [ 1, 1 ],
        dilation    : List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            nn.LayerNorm(features),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, [ 3, 3 ], padding = [ 1, 1 ], dilation = [ 1, 1 ]),
            layer_act(),
            nn.Conv2d(in_channels, in_channels, [ 3, 3 ], padding = [ 1, 1 ], dilation = [ 1, 1 ]),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(in_channels, in_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(in_channels, in_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv2 = self.cv2(input)
        cv3 = self.cv3(cv2)
        cv4 = self.cv4(input)
        return self.cv1(torch.cat([ cv4 + cv3, cv2 ], dim = 1))

class AttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim    : int,
        k_dim    : int,
        v_dim    : int,
        o_dim    : int,
        h_dim    : int = 1024,
        num_heads: int = 8,
        max_len  : int = 512,

    ):
        super().__init__()
        self.q    = nn.Linear(q_dim, h_dim, bias = False)
        self.k    = nn.Linear(k_dim, h_dim, bias = False)
        self.v    = nn.Linear(v_dim, h_dim, bias = False)
        self.rope = RotaryPositionEmbedding(h_dim // num_heads, max_len = max_len)
        self.attn = nn.MultiheadAttention(h_dim, num_heads, bias = False, batch_first = False)
        self.proj = nn.Linear(h_dim, o_dim, bias = False)
        self.norm = nn.LayerNorm(o_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q(query.permute(1, 0, 2))
        k = self.k(key  .permute(1, 0, 2))
        v = self.v(value.permute(1, 0, 2))
        q_embed, k_embed = self.rope(q, k)
        o, _ = self.attn(q_embed, k_embed, v)
        return self.norm(query + self.proj(o.permute(1, 0, 2)))

class AudioHeadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft      = 400
        self.hop_length = 80
        self.win_length = 400
        self.window     = torch.hann_window(self.win_length)
        self.head = nn.Sequential(
            nn.Conv2d(1, 32, [ 5, 5 ], padding = [ 0, 0 ], dilation = [ 1, 1 ], stride = [ 5, 5 ]),
            nn.LayerNorm([ 40, 64 ]),
            ResNet2dBlock( 32,  32, [ 40, 64 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            ResNet2dBlock( 32,  64, [ 20, 32 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock( 64,  64, [ 20, 32 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            ResNet2dBlock( 64, 128, [ 10, 16 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock(128, 256, [ 10, 16 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            nn.Flatten(start_dim = 2),
        )
        self.attn = AttentionBlock(160, 160, 160, 160, 512)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        com = torch.stft(
            input.view(input.size(0), -1),
            n_fft          = self.n_fft,
            hop_length     = self.hop_length,
            win_length     = self.win_length,
            window         = self.window,
            center         = True,
            return_complex = True,
        )
        mag = torch.abs(com)
#       pha = torch.angle(com)
        mag = mag.unsqueeze(1).contiguous()
        out = self.head(mag)
        return self.attn(out, out, out)

class VideoHeadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, 32, [ 5, 5 ], padding = [ 0, 0 ], dilation = [ 1, 1 ], stride = [ 5, 5 ]),
            nn.LayerNorm([ 72, 128 ]),
            ResNet2dBlock( 32,  32, [ 72, 128 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            ResNet2dBlock( 32,  64, [ 18,  32 ], [ 4, 4 ], [ 4, 4 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock( 64,  64, [ 18,  32 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            ResNet2dBlock( 64, 128, [  4,   8 ], [ 4, 4 ], [ 4, 4 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock(128, 128, [  4,   8 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            nn.Flatten(start_dim = 2),
        )
        self.conv = nn.Sequential(
            ResNet1dBlock( 32,  64, 2048, 2, 2, 0, 1),
            ResNet1dBlock( 64, 128, 1024, 2, 2, 0, 1),
            ResNet1dBlock(128, 256, 1024, 1, 3, 1, 1),
        )
        self.attn = AttentionBlock(1024, 1024, 1024, 1024)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(2), input.size(3))).view(input.size(0), input.size(1), -1)
        out = self.conv(out)
        return self.attn(out, out, out)

class ImageHeadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, [ 5, 5 ], padding = [ 0, 0 ], dilation = [ 1, 1 ], stride = [ 5, 5 ]),
            nn.LayerNorm([ 72, 128 ]),
            ResNet2dBlock( 32,  32, [ 72, 128 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            ResNet2dBlock( 32,  64, [ 36,  64 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock( 64,  64, [ 36,  64 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            ResNet2dBlock( 64, 128, [ 18,  32 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock(128, 256, [ 18,  32 ], [ 1, 1 ], [ 3, 3 ], [ 1, 1 ], [ 1, 1 ]),
            nn.Flatten(start_dim = 2),
        )
        self.attn = AttentionBlock(576, 576, 576, 576)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input)
        return self.attn(out, out, out)

class MediaMuxerBlock(nn.Module):
    def __init__(
        self,
        audio_in: int =  160,
        video_in: int = 1024,
        image_in: int =  576,
        channels: int =  256,
    ):
        super().__init__()
        muxer_in = audio_in + video_in
        self.audio_attn = AttentionBlock(audio_in, video_in, video_in, audio_in)
        self.video_attn = AttentionBlock(video_in, audio_in, audio_in, video_in)
        self.muxer_attn = AttentionBlock(muxer_in, image_in, image_in, muxer_in)
        self.mixer_attn = AttentionBlock(muxer_in, muxer_in, muxer_in, muxer_in)
        self.muxer_conv = nn.Sequential(
            ResNet1dBlock(channels, channels, muxer_in, 1, 3, 1, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
        muxer: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_o = self.audio_attn(audio, video, video)
        video_o = self.video_attn(video, audio, audio)
        muxer_c = self.muxer_conv(torch.cat([ audio_o, video_o ], dim = -1))
        muxer_o = self.muxer_attn(muxer_c, image, image)
        if muxer is None:
            mixer_o = self.mixer_attn(muxer_o, muxer_o, muxer_o)
        else:
            mixer_o = self.mixer_attn(muxer, muxer_o, muxer_o)
        return (audio_o, video_o, mixer_o)

class MediaMixerBlock(nn.Module):
    def __init__(
        self,
        audio_in: int =  160,
        video_in: int = 1024,
        image_in: int =  576,
        channels: int =  256,
    ):
        super().__init__()
        self.muxer_1 = MediaMuxerBlock(audio_in, video_in, image_in, channels)
        self.muxer_2 = MediaMuxerBlock(audio_in, video_in, image_in, channels)
        self.muxer_3 = MediaMuxerBlock(audio_in, video_in, image_in, channels)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        [ audio_1, video_1, mixer_1 ] = self.muxer_1(audio,   video,   image)
        [ audio_2, video_2, mixer_2 ] = self.muxer_2(audio_1, video_1, image, mixer_1)
        [ audio_3, video_3, mixer_3 ] = self.muxer_3(audio_2, video_2, image, mixer_2)
        return mixer_3

class AudioTailBlock(nn.Module):
    def __init__(
        self,
        in_features : int = 1184,
        out_features: int = 800,
        channels    : List[int] = [ 256, 64, 16, 4, 1 ],
    ):
        super().__init__()
        # 注意：AI必须透明绝对不能隐藏想法
        self.tail = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], 3),
            layer_act(),
            nn.Conv1d(channels[1], channels[2], 3),
            layer_act(),
            nn.Conv1d(channels[2], channels[3], 3),
            layer_act(),
            nn.Conv1d(channels[3], channels[4], 3),
            nn.Flatten(start_dim = 1),
            layer_act(),
            nn.Linear(in_features - 2 * 4, out_features),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.tail(input))
    
class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = AudioHeadBlock ()
        self.video = VideoHeadBlock ()
        self.image = ImageHeadBlock ()
        self.mixer = MediaMixerBlock()
        self.tail  = AudioTailBlock ()

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_out = self.audio(audio)
        video_out = self.video(video.select(2,  0))
        image_out = self.image(video.select(1, -1))
        mixer_out = self.mixer(audio_out, video_out, image_out)
        return self.tail(mixer_out)

# embedding = RotaryPositionEmbedding(128)
# q = torch.randn(10, 8, 32, 128)
# k = torch.randn(10, 8, 32, 128)
# q, k = embedding(q, k)
# print(q.shape)
# print(k.shape)

# model = ResNet1dBlock(8, 16, 800)
# input = torch.randn(10, 8, 800)
# print(model(input).shape)

# model = ResNet2dBlock(8, 16, [ 360, 640 ])
# input = torch.randn(10, 8, 360, 640)
# print(model(input).shape)

# model = AttentionBlock(800, 800, 800, 800)
# input = (
#     torch.randn(10, 256, 800),
#     torch.randn(10, 256, 800),
#     torch.randn(10, 256, 800),
# )
# print(model(*input).shape)

# model = AudioHeadBlock()
# input = torch.randn(10, 32, 800)
# print(model(input).shape)

# model = VideoHeadBlock()
# input = torch.randn(10, 32, 360, 640)
# print(model(input).shape)

# model = ImageHeadBlock()
# input = torch.randn(10, 3, 360, 640)
# print(model(input).shape)

# model = MediaMuxerBlock()
# input = (
#     torch.randn(10, 256,  160),
#     torch.randn(10, 256, 1024),
#     torch.randn(10, 256,  576),
# )
# audio, video, muxer = model(*input)
# print(audio.shape)
# print(video.shape)
# print(muxer.shape)

# model = MediaMixerBlock()
# input = (
#     torch.randn(10, 256,  160),
#     torch.randn(10, 256, 1024),
#     torch.randn(10, 256,  576),
# )
# print(model(*input).shape)

# model = AudioTailBlock()
# input = torch.randn(10, 256, 1184)
# print(model(input).shape)

model = Chobits()
model.eval()
input = (torch.randn(10, 32, 800), torch.randn(10, 32, 3, 360, 640))
print(model(*input).shape)

# 直接保存
# torch.save(model, "D:/download/chobits.pt")

# JIT保存
torch.jit.save(torch.jit.trace(model, input), "D:/download/chobits.pt")

# ONNX保存
# batch = torch.export.Dim("batch", min = 1)
# torch.onnx.export(
#     model,
#     (torch.randn(1, 10, 800), torch.randn(1, 10, 3, 360, 640)),
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
