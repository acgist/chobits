import torch
import torch.nn as nn

from typing import List

layer_act = nn.SiLU

class PadBlock(nn.Module):
    def __init__(
        self,
        pad : List[int] = [],
    ):
        super().__init__()
        self.pad = pad
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(input, pad = self.pad, value = 0.0)

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
            nn.LayerNorm([features]),
            nn.Conv1d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        left  = self.cv1(input)
        right = self.cv2(left) + left
        return  self.cv4(torch.cat([self.cv3(right), left], dim = 1))

class ResNet2dBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        shape       : List[int],
        stride      : List[int] = [1, 1],
        kernel      : List[int] = [3, 3],
        padding     : List[int] = [1, 1],
        dilation    : List[int] = [1, 1],
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.LayerNorm(shape),
            nn.Conv2d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        left  = self.cv1(input)
        right = self.cv2(left) + left
        return  self.cv4(torch.cat([self.cv3(right), left], dim = 1))

class GRUBlock(nn.Module):
    def __init__(
        self,
        input_size : int,
        hidden_size: int,
        time,
        stride     : int = 2,
        num_layers : int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first = True, bias = False)
        self.out = ResNet1dBlock(time, time, input_size + hidden_size, stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(input)
        # TODO: 映射
        return self.out(torch.cat([input, output], dim = -1))

class AttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim    : int,
        k_dim    : int,
        v_dim    : int,
        o_dim    : int,
        channel  : int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.q    = nn.Linear(q_dim, o_dim, bias = False)
        self.k    = nn.Linear(k_dim, o_dim, bias = False)
        self.v    = nn.Linear(v_dim, o_dim, bias = False)
        self.attn = nn.MultiheadAttention(o_dim, num_heads, bias = False, batch_first = False)
        self.proj = nn.Linear(o_dim, o_dim, bias = False)
        self.out  = ResNet1dBlock(channel, channel, q_dim + o_dim, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q(query.permute(1, 0, 2))
        k = self.k(key  .permute(1, 0, 2))
        v = self.v(value.permute(1, 0, 2))
        o, _ = self.attn(q, k, v)
        o = o.permute(1, 0, 2)
        o = self.proj(o)
        return self.out(torch.cat([query, o], dim = -1))

class AudioHeadBlock(nn.Module):
    def __init__(
        self,
        kernel   : int = 5,
        padding  : int = 2,
        dilation : int = 1,
        kernel_  : int = 3,
        padding_ : int = 1,
        dilation_: int = 1,
    ):
        super().__init__()
        # self.embed = nn.Sequential(
        #     nn.Linear(1, 4),
        #     GRUBlock(1, 4, 800),
        #     nn.Flatten(start_dim = 1),
        # )
        # TODO: 试试空洞卷积
        self.head = nn.Sequential(
            ResNet1dBlock( 1,   4, 800, 4, kernel, padding, dilation),
            ResNet1dBlock( 4,  16, 200, 4, kernel, padding, dilation),
            ResNet1dBlock(16,  64,  50, 4, kernel, padding, dilation),
            ResNet1dBlock(64, 256,  13, 4, kernel, padding, dilation),
            nn.Flatten(start_dim = 1),
        )
        self.gru = GRUBlock(1024, 1024, 10)
        self.conv = nn.Sequential(
            ResNet1dBlock( 10,  64, 1024, 2, kernel_, padding_, dilation_),
            ResNet1dBlock( 64, 256,  512, 2, kernel_, padding_, dilation_),
            ResNet1dBlock(256, 256,  256, 1, kernel_, padding_, dilation_),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # out = self.embed(input.view(-1, input.size(-1), 1))
        # out = self.head(out.view(-1, 1, out.size(-1)))
        # out = self.gru (out.view(input.size(0), input.size(1), -1))
        # out = self.conv(out)
        # return out
        out = self.head(input.view(-1, 1, input.size(-1)))
        out = self.gru (out.view(input.size(0), input.size(1), -1))
        out = self.conv(out)
        return out

class VideoHeadBlock(nn.Module):
    def __init__(
        self,
        kernel   : List[int] = [5, 5],
        padding  : List[int] = [2, 2],
        dilation : List[int] = [1, 1],
        kernel_  : int = 3,
        padding_ : int = 1,
        dilation_: int = 1,
    ):
        super().__init__()
        # TODO: 试试空洞卷积
        self.head = nn.Sequential(
            ResNet2dBlock( 1,   4, [360, 640], [4, 4], kernel, padding, dilation),
            ResNet2dBlock( 4,  16, [ 90, 160], [4, 4], kernel, padding, dilation),
            ResNet2dBlock(16,  64, [ 23,  40], [4, 4], kernel, padding, dilation),
            ResNet2dBlock(64, 256, [  6,  10], [4, 4], kernel, padding, dilation),
            nn.Flatten(start_dim = 1),
        )
        self.gru = GRUBlock(1536, 1536, 10)
        self.conv = nn.Sequential(
            ResNet1dBlock( 10,  64, 1536, 2, kernel_, padding_, dilation_),
            ResNet1dBlock( 64, 256,  768, 1, kernel_, padding_, dilation_),
            ResNet1dBlock(256, 256,  768, 1, kernel_, padding_, dilation_),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(2), input.size(3)))
        out = self.gru(out.view(input.size(0), input.size(1), -1))
        out = self.conv(out)
        return out

class ImageHeadBlock(nn.Module):
    def __init__(
        self,
        kernel  : List[int] = [3, 3],
        padding : List[int] = [1, 1],
        dilation: List[int] = [1, 1],
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dBlock(  3,  32, [ 360, 640 ], [2, 2], kernel, padding, dilation),
            ResNet2dBlock( 32,  64, [ 180, 320 ], [2, 2], kernel, padding, dilation),
            ResNet2dBlock( 64, 128, [  90, 160 ], [2, 2], kernel, padding, dilation),
            ResNet2dBlock(128, 256, [  45,  80 ], [2, 2], kernel, padding, dilation),
            PadBlock([0, 0, 1, 0]),
            ResNet2dBlock(256, 256, [ 24, 40 ], [1, 1], kernel, padding, dilation),
            nn.Flatten(start_dim = 2),
            ResNet1dBlock(256, 256, 960),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.head(input)

class MediaMixerBlock(nn.Module):
    def __init__(
        self,
        audio_in: int = 256,
        video_in: int = 768,
        image_in: int = 960,
    ):
        super().__init__()
        mixer_in = audio_in + video_in
        self.audio_attn = AttentionBlock(audio_in, video_in, video_in, audio_in)
        self.video_attn = AttentionBlock(video_in, audio_in, audio_in, video_in)
        self.image_attn = AttentionBlock(mixer_in, image_in, image_in, mixer_in)
        self.mixer_attn = AttentionBlock(mixer_in, mixer_in, mixer_in, mixer_in)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        audio_o = self.audio_attn(audio, video, video)
        video_o = self.video_attn(video, audio, audio)
        muxer_o = torch.cat([audio_o, video_o], dim = -1)
        mixer_o = self.image_attn(muxer_o, image, image)
        return    self.mixer_attn(mixer_o, mixer_o, mixer_o)

class AudioTailBlock(nn.Module):
    def __init__(
        self,
        in_features : int = 1024,
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

# model = GRUBlock(768, 768, 10)
# input = torch.randn(10, 10, 768)
# print(model(input).shape)

# model = AttentionBlock(768, 960, 960, 768)
# input = (torch.randn(10, 256, 768), torch.randn(10, 256, 960), torch.randn(10, 256, 960))
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
# input = (torch.randn(10, 256, 256), torch.randn(10, 256, 768), torch.randn(10, 256, 960))
# print(model(*input).shape)

# model = AudioTailBlock()
# input = torch.randn(10, 256, 1024)
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
