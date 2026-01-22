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
        return torch.nn.functional.pad(input, pad = self.pad, mode = "replicate")

class ResNet1dBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        kernel      : int = 3,
        padding     : int = 1,
        dilation    : int = 1,
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding = 1, dilation = 1),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv1 = self.cv1(input)
        cv2 = self.cv2(cv1)
        cv3 = self.cv3(cv1)
        cv4 = self.cv4(cv2)
        return cv1 + cv2 + cv3 + cv4
    
class ResNet1dCatBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        kernel      : int = 3,
        padding     : int = 1,
        dilation    : int = 1,
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding = 1, dilation = 1),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, 3, padding = 2, dilation = 2),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv1 = self.cv1(input)
        cv2 = self.cv2(cv1)
        cv3 = self.cv3(cv1)
        cv4 = self.cv4(cv2)
        return torch.cat([cv1, cv2, cv3, cv4], dim = 1)

class ResNet2dBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : List[int] = [ 1, 1 ],
        kernel      : List[int] = [ 3, 3 ],
        padding     : List[int] = [ 1, 1 ],
        dilation    : List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 1, 1 ], dilation = [ 1, 1 ]),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv1 = self.cv1(input)
        cv2 = self.cv2(cv1)
        cv3 = self.cv3(cv1)
        cv4 = self.cv4(cv2)
        return cv1 + cv2 + cv3 + cv4

class ResNet2dCatBlock(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : List[int] = [ 1, 1 ],
        kernel      : List[int] = [ 3, 3 ],
        padding     : List[int] = [ 1, 1 ],
        dilation    : List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False, stride = stride),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 1, 1 ], dilation = [ 1, 1 ]),
            layer_act(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
            nn.Conv2d(out_channels, out_channels, [ 3, 3 ], padding = [ 2, 2 ], dilation = [ 2, 2 ]),
            layer_act(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cv1 = self.cv1(input)
        cv2 = self.cv2(cv1)
        cv3 = self.cv3(cv1)
        cv4 = self.cv4(cv2)
        return torch.cat([cv1, cv2, cv3, cv4], dim = 1)

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
            ResNet1dBlock(256, 256, 2, 2, 0, 1), # q_dim + o_dim
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
            ResNet1dCatBlock(  1,   1, 3, kernel, padding, dilation), # 800
            ResNet1dCatBlock(  4,   4, 3, kernel, padding, dilation), # 267
            ResNet1dCatBlock( 16,  16, 3, kernel, padding, dilation), #  89
            ResNet1dCatBlock( 64,  64, 3, kernel, padding, dilation), #  30
            ResNet1dBlock   (256, 256, 3, kernel, padding, dilation), #  10
            nn.Flatten(start_dim = 1),
        )
        self.gru = GRUBlock(1024, 1024)
        self.conv = nn.Sequential(
            nn.LayerNorm(1024),
            ResNet1dCatBlock( 20,  20, 2, 2, 0, 1), # 1024
            ResNet1dBlock   ( 80, 160, 2, 2, 0, 1), #  512
            ResNet1dBlock   (160, 256, 1, 3, 1, 1), #  256
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(-1))).view(input.size(0), input.size(1), -1)
        gru = self.gru(out)
        return self.conv(torch.cat([ out, gru ], dim = 1))

class VideoHeadBlock(nn.Module):
    def __init__(
        self,
        kernel  : List[int] = [ 3, 3 ],
        padding : List[int] = [ 1, 1 ],
        dilation: List[int] = [ 1, 1 ]
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dCatBlock(  1,   1, [ 3, 3 ], kernel, padding, dilation), # [ 360, 640 ]
            ResNet2dCatBlock(  4,   4, [ 3, 3 ], kernel, padding, dilation), # [ 120, 214 ]
            ResNet2dCatBlock( 16,  16, [ 3, 3 ], kernel, padding, dilation), # [  40,  72 ]
            ResNet2dCatBlock( 64,  64, [ 3, 3 ], kernel, padding, dilation), # [  14,  24 ]
            ResNet2dBlock   (256, 256, [ 3, 3 ], kernel, padding, dilation), # [   5,   8 ]
            nn.Flatten(start_dim = 1),
        )
        self.gru = GRUBlock(1536, 1536)
        self.conv = nn.Sequential(
            nn.LayerNorm(1536),
            ResNet1dCatBlock( 20,  20, 2, 2, 0, 1), # 1536
            ResNet1dBlock   ( 80, 160, 2, 2, 0, 1), #  768
            ResNet1dBlock   (160, 256, 1, 3, 1, 1), #  384
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.head(input.view(-1, 1, input.size(2), input.size(3))).view(input.size(0), input.size(1), -1)
        gru = self.gru(out)
        return self.conv(torch.cat([ out, gru ], dim = 1))

class ImageHeadBlock(nn.Module):
    def __init__(
        self,
        kernel  : List[int] = [ 3, 3 ],
        padding : List[int] = [ 1, 1 ],
        dilation: List[int] = [ 1, 1 ],
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNet2dCatBlock(  3,   3, [ 3, 3 ], kernel, padding, dilation), # [ 360, 640 ]
            ResNet2dBlock   ( 12,  16, [ 1, 1 ], kernel, padding, dilation), # [ 120, 214 ]
            ResNet2dCatBlock( 16,  16, [ 3, 3 ], kernel, padding, dilation), # [ 120, 214 ]
            ResNet2dCatBlock( 64,  64, [ 3, 3 ], kernel, padding, dilation), # [  40,  72 ]
            ResNet2dBlock   (256, 256, [ 1, 1 ], kernel, padding, dilation), # [  14,  24 ]
            nn.Flatten(start_dim = 2),
            nn.LayerNorm(336),
            ResNet1dBlock(256, 256, 1, 3, 1, 1), # 336
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
            ResNet1dBlock(256, 256, 1, 3, 1, 1), # muxer_in
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
            nn.LayerNorm(in_features),
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

# # model = ResNet1dBlock(10, 64)
# model = ResNet1dBlock(10, 64, 2)
# input = torch.randn(10, 10, 800)
# print(model(input).shape)

# # model = ResNet1dCatBlock(10, 64)
# model = ResNet1dCatBlock(10, 64, 2)
# input = torch.randn(10, 10, 800)
# print(model(input).shape)

# # model = ResNet2dBlock(10, 64)
# model = ResNet2dBlock(10, 64, [ 2, 2 ])
# input = torch.randn(10, 10, 360, 640)
# print(model(input).shape)

# # model = ResNet2dCatBlock(10, 64)
# model = ResNet2dCatBlock(10, 64, [ 2, 2 ])
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
