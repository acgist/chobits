import torch
import torch.nn as nn

from typing import List

layer_act = nn.SiLU
# layer_act = nn.GELU

class PadBlock(nn.Module):
    def __init__(
        self,
        pad : List[int] = [],
    ):
        super().__init__()
        self.pad = pad
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(input, pad = self.pad, mode = "replicate")

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
        kernel_     : int = 2,
        padding_    : int = 0,
        dilation_   : int = 1
    ):
        super().__init__()
        if stride == 2:
            self.cv1 = nn.Sequential(
                nn.LayerNorm([features]),
                nn.Conv1d(in_channels, out_channels, kernel_, padding = padding_, dilation = dilation_, bias = False, stride = stride),
                layer_act(),
                nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
                layer_act(),
            )
        else:
            self.cv1 = nn.Sequential(
                nn.LayerNorm([features]),
                nn.Conv1d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
                layer_act(),
                nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
                layer_act(),
            )
        self.cv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        left = self.cv1(input)
        return self.cv2(left) + left

class ResNet2dBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        shape       : List[int],
        stride      : List[int] = [ 1, 1 ],
        kernel      : List[int] = [ 3, 3 ],
        padding     : List[int] = [ 1, 1 ],
        dilation    : List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.LayerNorm(shape),
            nn.Conv2d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        left = self.cv1(input)
        return self.cv2(left) + left

class GRUBlock(nn.Module):
    def __init__(
        self,
        input_size : int,
        hidden_size: int,
        num_layers : int = 1,
    ):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first = True, bias = False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(input)
        output = self.proj(output)
        return torch.cat([ input, output ], dim = -1)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim    : int,
        k_dim    : int,
        v_dim    : int,
        o_dim    : int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.q    = nn.Linear(q_dim, o_dim, bias = False)
        self.k    = nn.Linear(k_dim, o_dim, bias = False)
        self.v    = nn.Linear(v_dim, o_dim, bias = False)
        self.attn = nn.MultiheadAttention(o_dim, num_heads, bias = False, batch_first = False)
        self.proj = nn.Linear(o_dim, o_dim, bias = False)
        self.out  = nn.Sequential(
            ResNet1dBlock(256, 256, q_dim + o_dim, 2, 3, 1, 1),
            ResNet1dBlock(256, 256, o_dim,         1, 3, 2, 2),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q(query.permute(1, 0, 2))
        k = self.k(key  .permute(1, 0, 2))
        v = self.v(value.permute(1, 0, 2))
        o, _ = self.attn(q, k, v)
        o = o.permute(1, 0, 2)
        o = self.proj(o)
        return self.out(torch.cat([ query, o ], dim = -1))

class AudioHeadBlock(nn.Module):
    def __init__(
        self,
        kernel  : int = 3,
        padding : int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNet1dBlock(  1,   4, 800, 3, kernel, padding, dilation),
            ResNet1dBlock(  4,  16, 267, 3, kernel, padding, dilation),
            ResNet1dBlock( 16,  64,  89, 3, kernel, padding, dilation),
            ResNet1dBlock( 64, 256,  30, 3, kernel, padding, dilation),
            ResNet1dBlock(256, 256,  10, 3, kernel, padding, dilation),
            nn.Flatten(start_dim = 1),
        )
        self.gru = GRUBlock(1024, 1024)
        self.conv = nn.Sequential(
            ResNet1dBlock( 10,  32, 2048, 2, 3, 1, 1),
            ResNet1dBlock( 32,  64, 1024, 2, 3, 1, 1),
            ResNet1dBlock( 64, 128,  512, 2, 3, 1, 1),
            ResNet1dBlock(128, 256,  256, 1, 3, 2, 2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(-1)))
        out = self.gru(out.view(input.size(0), input.size(1), -1))
        out = self.conv(out)
        return out

class VideoHeadBlock(nn.Module):
    def __init__(
        self,
        kernel  : List[int] = [ 3, 3 ],
        padding : List[int] = [ 1, 1 ],
        dilation: List[int] = [ 1, 1 ]
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dBlock(  1,   4, [ 360, 640 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock(  4,  16, [ 120, 214 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock( 16,  64, [  40,  72 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock( 64, 256, [  14,  24 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock(256, 256, [   5,   8 ], [ 3, 3 ], kernel, padding, dilation),
            nn.Flatten(start_dim = 1),
        )
        self.gru = GRUBlock(1536, 1536)
        self.conv = nn.Sequential(
            ResNet1dBlock( 10,  32, 3072, 2, 3, 1, 1),
            ResNet1dBlock( 32,  64, 1536, 2, 3, 1, 1),
            ResNet1dBlock( 64, 128,  768, 2, 3, 1, 1),
            ResNet1dBlock(128, 256,  384, 1, 3, 2, 2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(2), input.size(3)))
        out = self.gru(out.view(input.size(0), input.size(1), -1))
        out = self.conv(out)
        return out

class ImageHeadBlock(nn.Module):
    def __init__(
        self,
        kernel  : List[int] = [ 3, 3 ],
        padding : List[int] = [ 1, 1 ],
        dilation: List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dBlock(  3,  16, [ 360, 640 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock( 16,  64, [ 120, 214 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock( 64, 256, [  40,  72 ], [ 3, 3 ], kernel, padding, dilation),
            ResNet2dBlock(256, 256, [  14,  24 ], [ 1, 1 ], kernel, padding, dilation),
            nn.Flatten(start_dim = 2),
            ResNet1dBlock(256, 256, 336, 1, 3, 1, 1),
            ResNet1dBlock(256, 256, 336, 1, 3, 2, 2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.head(input)

class MediaMixerBlock(nn.Module):
    def __init__(
        self,
        audio_in: int = 256,
        video_in: int = 384,
        image_in: int = 336,
    ):
        super().__init__()
        muxer_in = audio_in + video_in
        self.audio_attn = AttentionBlock(audio_in, video_in, video_in, audio_in)
        self.video_attn = AttentionBlock(video_in, audio_in, audio_in, video_in)
        self.image_attn = AttentionBlock(muxer_in, image_in, image_in, muxer_in)
        self.mixer_attn = AttentionBlock(muxer_in, muxer_in, muxer_in, muxer_in)
        self.muxer_conv = nn.Sequential(
            ResNet1dBlock(256, 256, muxer_in, 1, 3, 2, 2),
            ResNet1dBlock(256, 256, muxer_in, 1, 3, 2, 2),
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        audio_o = self.audio_attn(audio, video, video)
        video_o = self.video_attn(video, audio, audio)
        muxer_o = self.muxer_conv(torch.cat([audio_o, video_o], dim = -1))
        mixer_o = self.image_attn(muxer_o, image, image)
        return    self.mixer_attn(mixer_o, mixer_o, mixer_o)

class AudioTailBlock(nn.Module):
    def __init__(
        self,
        in_features : int = 640,
        out_features: int = 800,
        channels    : List[int] = [256, 64, 16, 4, 1],
    ):
        super().__init__()
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

# # model = ResNet1dBlock(10, 64, 800)
# model = ResNet1dBlock(10, 64, 800, 2)
# input = torch.randn(10, 10, 800)
# print(model(input).shape)

# # model = ResNet2dBlock(10, 64, [360, 640])
# model = ResNet2dBlock(10, 64, [360, 640], [ 2, 2 ])
# input = torch.randn(10, 10, 360, 640)
# print(model(input).shape)

# model = GRUBlock(384, 384)
# input = torch.randn(10, 10, 384)
# print(model(input).shape)

# model = AttentionBlock(384, 336, 336, 384)
# input = (torch.randn(10, 256, 384), torch.randn(10, 256, 336), torch.randn(10, 256, 336))
# print(model(*input).shape)

# model = AudioHeadBlock()
# input = torch.randn(10, 10, 800)
# print(model(input).shape)

# model = VideoHeadBlock()
# input = torch.randn(10, 10, 360, 640)
# print(model(input).shape)

# model = ImageHeadBlock()
# input = torch.randn(10, 3, 360, 640)
# print(model(input).shape)

# model = MediaMixerBlock()
# input = (torch.randn(10, 256, 256), torch.randn(10, 256, 384), torch.randn(10, 256, 336))
# print(model(*input).shape)

# model = AudioTailBlock()
# input = torch.randn(10, 256, 640)
# print(model(input).shape)

model = Chobits()
model.eval();
input = (torch.randn(10, 10, 800), torch.randn(10, 10, 3, 360, 640))
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
