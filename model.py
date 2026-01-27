import torch
import torch.nn as nn

from typing import List

layer_act = nn.SiLU

class PadBlock(nn.Module):
    def __init__(
        self,
        pad: List[int] = [],
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
    
class ResNet1dCatBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        features: int,
        stride  : int = 1,
        kernel  : int = 3,
        padding : int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv1d(channels * 4, channels * 4, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            nn.LayerNorm(features),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding = 1, dilation = 1),
            layer_act(),
            nn.Conv1d(channels, channels, 3, padding = 1, dilation = 1),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(channels, channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(channels, channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv2 = self.cv2(input)
        cv3 = self.cv3(cv2)
        cv4 = self.cv4(input)
        return self.cv1(torch.cat([ input, cv2, cv3, cv4 ], dim = 1))

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

class ResNet2dCatBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        features: List[int],
        stride  : List[int] = [ 1, 1 ],
        kernel  : List[int] = [ 3, 3 ],
        padding : List[int] = [ 1, 1 ],
        dilation: List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            nn.LayerNorm(features),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(channels, channels, [ 3, 3 ], padding = [ 1, 1 ], dilation = [ 1, 1 ]),
            layer_act(),
            nn.Conv2d(channels, channels, [ 3, 3 ], padding = [ 1, 1 ], dilation = [ 1, 1 ]),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(channels, channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(channels, channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(channels, channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(channels, channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv2 = self.cv2(input)
        cv3 = self.cv3(cv2)
        cv4 = self.cv4(input)
        return self.cv1(torch.cat([ input, cv2, cv3, cv4 ], dim = 1))

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
        return self.proj(output)

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
            ResNet1dBlock(256, 256, q_dim + o_dim, 1, 3, 1, 1),
            nn.MaxPool1d(2),
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
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            ResNet1dBlock   ( 1,   4, 160, 5, 5, 0, 1),
            ResNet1dBlock   ( 4,  16,  40, 4, 4, 0, 1),
            ResNet1dBlock   (16,  32,  20, 2, 2, 0, 1),
            ResNet1dCatBlock(32,        5, 4, 4, 0, 1),
        )
        self.gru = GRUBlock(640, 640)
        self.conv = nn.Sequential(
            ResNet1dBlock   (32, 64, 640, 2, 2, 0, 1),
            ResNet1dCatBlock(64,     320, 2, 2, 0, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(-1))).view(input.size(0), input.size(1), -1)
        gru = self.gru(out)
        return self.conv(torch.cat([ out, gru ], dim = -1))

class VideoHeadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dBlock   ( 1,  4, [ 72, 128 ], [ 5, 5 ], [ 5, 5 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock   ( 4, 16, [ 18,  32 ], [ 4, 4 ], [ 4, 4 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock   (16, 32, [  9,  16 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dCatBlock(32,     [  2,   4 ], [ 4, 4 ], [ 4, 4 ], [ 0, 0 ], [ 1, 1 ]),
            nn.Flatten(start_dim = 2),
        )
        self.gru = GRUBlock(1024, 1024)
        self.conv = nn.Sequential(
            ResNet1dBlock   (32, 64, 1024, 2, 2, 0, 1),
            ResNet1dCatBlock(64,      512, 2, 2, 0, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(2), input.size(3))).view(input.size(0), input.size(1), -1)
        gru = self.gru(out)
        return self.conv(torch.cat([ out, gru ], dim = -1))

class ImageHeadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dBlock   ( 3, 16, [ 72, 128 ], [ 5, 5 ], [ 5, 5 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock   (16, 32, [ 36,  64 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dBlock   (32, 64, [ 18,  32 ], [ 2, 2 ], [ 2, 2 ], [ 0, 0 ], [ 1, 1 ]),
            ResNet2dCatBlock(64,     [ 18,  32 ]),
            nn.Flatten(start_dim = 2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.head(input)

class MediaMixerBlock(nn.Module):
    def __init__(
        self,
        audio_in: int = 320,
        video_in: int = 512,
        image_in: int = 576,
    ):
        super().__init__()
        muxer_in = audio_in + video_in
        self.audio_attn = AttentionBlock(audio_in, video_in, video_in, audio_in)
        self.video_attn = AttentionBlock(video_in, audio_in, audio_in, video_in)
        self.muxer_attn = AttentionBlock(muxer_in, image_in, image_in, muxer_in)
        self.mixer_attn = AttentionBlock(muxer_in, muxer_in, muxer_in, muxer_in)
        self.muxer_conv = nn.Sequential(
            ResNet1dBlock(256, 256, muxer_in, 1, 3, 1, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        audio_o = self.audio_attn(audio, video, video)
        video_o = self.video_attn(video, audio, audio)
        media_o = self.muxer_conv(torch.cat([ audio_o, video_o ], dim = -1))
        muxer_o = self.muxer_attn(media_o, image,   image  )
        return    self.mixer_attn(muxer_o, muxer_o, muxer_o)

class AudioTailBlock(nn.Module):
    def __init__(
        self,
        in_features : int = 832,
        out_features: int = 800,
        channels    : List[int] = [ 256, 64, 16, 4, 1 ],
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

# model = ResNet1dBlock(8, 16, 800)
# input = torch.randn(10, 8, 800)
# print(model(input).shape)

# model = ResNet1dCatBlock(8, 800)
# input = torch.randn(10, 8, 800)
# print(model(input).shape)

# model = ResNet2dBlock(8, 16, [ 360, 640 ])
# input = torch.randn(10, 8, 360, 640)
# print(model(input).shape)

# model = ResNet2dCatBlock(8, [ 360, 640 ])
# input = torch.randn(10, 8, 360, 640)
# print(model(input).shape)

# model = GRUBlock(800, 800)
# input = torch.randn(10, 32, 800)
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

# model = MediaMixerBlock()
# input = (
#     torch.randn(10, 256, 320),
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 576),
# )
# print(model(*input).shape)

# model = AudioTailBlock()
# input = torch.randn(10, 256, 832)
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
