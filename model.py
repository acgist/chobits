import torch
import torch.nn as nn

from typing import List

layer_act = nn.SiLU

class ResNetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        shape       : int,
        stride      : int = 0,
        kernel      : int = 3,
        padding     : int = 1,
        dilation    : int = 1,
    ):
        super().__init__()
        self.use_stride = stride > 0
        self.cv1 = nn.Sequential(
            nn.LayerNorm([shape]),
            nn.Conv1d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False),
            layer_act(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
            nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
            layer_act(),
        )
        if self.use_stride:
            self.cv3 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
                layer_act(),
                nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation, stride = stride),
                layer_act(),
            )
        else:
            self.cv3 = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
                layer_act(),
                nn.Conv1d(out_channels, out_channels, kernel, padding = padding, dilation = dilation),
                layer_act(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_stride:
            left = self.cv1(x)
            left = self.cv2(left) + left
            return self.cv3(left)
        else:
            left  = self.cv1(x)
            right = self.cv2(left)  + left
            return  self.cv3(right) + left

class ResNetBlock2d(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        shape       : List[int],
        stride      : List[int] = None,
        kernel      : List[int] = [3, 3],
        padding     : List[int] = [2, 2],
        dilation    : List[int] = [2, 2],
    ):
        super().__init__()
        self.use_stride = stride is not None and len(stride) > 0
        if self.use_stride:
            self.cv1 = nn.Sequential(
                nn.LayerNorm(shape),
                nn.Conv2d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False),
                layer_act(),
                nn.Conv2d(out_channels, out_channels, kernel, padding = padding, dilation = dilation, stride = stride),
                layer_act(),
            )
        else:
            self.cv1 = nn.Sequential(
                nn.LayerNorm(shape),
                nn.Conv2d(in_channels, out_channels, kernel, padding = padding, dilation = dilation, bias = False),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_stride:
            left = self.cv1(x)
            left = self.cv2(left) + left
            return self.cv3(left)
        else:
            left  = self.cv1(x)
            right = self.cv2(left)  + left
            return  self.cv3(right) + left

class GRUBlock(nn.Module):
    def __init__(
        self,
        input_size : int,
        hidden_size: int,
        channel    : int = 256,
        num_layers : int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first = True, bias = False)
        self.out = ResNetBlock1d(channel * 2, channel, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0         = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device = x.device)
        output, _  = self.gru(x, h0)
        return self.out(torch.cat([x, output], dim = 1))

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
        self.out  = ResNetBlock1d(channel * 2, channel, o_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q(query.permute(1, 0, 2))
        k = self.k(key  .permute(1, 0, 2))
        v = self.v(value.permute(1, 0, 2))
        o, _ = self.attn(q, k, v)
        o = o.permute(1, 0, 2)
        o = self.proj(o)
        return self.out(torch.cat([query, o], dim = 1))

class AudioHeadBlock(nn.Module):
    def __init__(
        self,
        in_len  : int       = 800,
        channels: List[int] = [10, 64, 128, 256],
        pool    : int       = 2,
        kernel  : int       = 3,
        padding : int       = 1,
        dilation: int       = 1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNetBlock1d(channels[0], channels[1], in_len,         pool, kernel, padding, dilation),
            ResNetBlock1d(channels[1], channels[1], in_len // pool,    0, kernel, padding, dilation),
            ResNetBlock1d(channels[1], channels[2], in_len // pool,    0, kernel, padding, dilation),
            ResNetBlock1d(channels[2], channels[2], in_len // pool,    0, kernel, padding, dilation),
            ResNetBlock1d(channels[2], channels[3], in_len // pool,    0, kernel, padding, dilation),
            ResNetBlock1d(channels[3], channels[3], in_len // pool,    0, kernel, padding, dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class VideoHeadBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_len    : int = 920,
        height     : int = 360,
        width      : int = 640,
        channels   : List[int] = [32, 64, 128, 256],
        pool       : List[int] = [2, 2],
        kernel     : List[int] = [3, 3],
        padding    : List[int] = [2, 2],
        dilation   : List[int] = [2, 2],
        kernel_    : List[int] = [3, 3],
        padding_   : List[int] = [1, 1],
        dilation_  : List[int] = [1, 1],
    ):
        super().__init__()
        self.head = nn.Sequential(
            ResNetBlock2d(in_channels, channels[0], [ height // pow(pool[0], 0) + 0, width // pow(pool[1], 0) ], pool, kernel,  padding,  dilation ),
            ResNetBlock2d(channels[0], channels[1], [ height // pow(pool[0], 1) + 0, width // pow(pool[1], 1) ], pool, kernel,  padding,  dilation ),
            ResNetBlock2d(channels[1], channels[2], [ height // pow(pool[0], 2) + 0, width // pow(pool[1], 2) ], pool, kernel_, padding_, dilation_),
            ResNetBlock2d(channels[2], channels[3], [ height // pow(pool[0], 3) + 0, width // pow(pool[1], 3) ], pool, kernel_, padding_, dilation_),
            ResNetBlock2d(channels[3], channels[3], [ height // pow(pool[0], 4) + 1, width // pow(pool[1], 4) ],   [], kernel_, padding_, dilation_),
            ResNetBlock2d(channels[3], channels[3], [ height // pow(pool[0], 4) + 1, width // pow(pool[1], 4) ],   [], kernel_, padding_, dilation_),
            nn.Flatten(start_dim = 2),
            ResNetBlock1d(channels[3], channels[3], out_len),
            ResNetBlock1d(channels[3], channels[3], out_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

ImageHeadBlock = VideoHeadBlock

class MediaMixerBlock(nn.Module):
    def __init__(
        self,
        audio_in: int = 400,
        image_in: int = 920,
        video_in: int = 920,
    ):
        super().__init__()
        muxer_in = audio_in + video_in
        self.audio_gru  = GRUBlock(audio_in, audio_in)
        self.video_gru  = GRUBlock(video_in, video_in)
        self.audio_attn = AttentionBlock(audio_in, video_in, video_in, audio_in)
        self.video_attn = AttentionBlock(video_in, audio_in, audio_in, video_in)
        self.image_attn = AttentionBlock(muxer_in, image_in, image_in, muxer_in)
        self.muxer_attn = AttentionBlock(muxer_in, muxer_in, muxer_in, muxer_in)
        self.mixer_attn = AttentionBlock(muxer_in, muxer_in, muxer_in, muxer_in)

    def forward(
        self,
        audio: torch.Tensor,
        image: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:
        audio_v = self.audio_gru(audio)
        video_v = self.video_gru(video)
        audio_o = self.audio_attn(audio_v, video_v, video_v)
        video_o = self.video_attn(video_v, audio_v, audio_v)
        muxer_o = torch.cat([audio_o, video_o], dim = -1)
        image_o = self.image_attn(muxer_o, image, image)
        mixer_o = self.muxer_attn(muxer_o, image_o, image_o)
        return    self.mixer_attn(mixer_o, mixer_o, mixer_o)

class AudioTailBlock(nn.Module):
    def __init__(
        self,
        in_features : int = 1320,
        out_features: int = 800,
        channels    : List[int] = [256, 64, 16, 4, 1],
    ):
        super().__init__()
        self.tail = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], 3, padding = 1),
            layer_act(),
            nn.Conv1d(channels[1], channels[2], 3, padding = 1),
            layer_act(),
            nn.Conv1d(channels[2], channels[3], 3, padding = 1),
            layer_act(),
            nn.Conv1d(channels[3], channels[4], 3, padding = 1),
            nn.Flatten(start_dim = 1),
            layer_act(),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.tail(x))
    
class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = AudioHeadBlock (  )
        self.image = ImageHeadBlock ( 3)
        self.video = VideoHeadBlock (10)
        self.mixer = MediaMixerBlock(  )
        self.tail  = AudioTailBlock (  )

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_out = self.audio(audio)
        image_out = self.image(video.select(1, -1))
        video_out = self.video(video.select(2,  0))
        mixer_out = self.mixer(audio_out, image_out, video_out)
        return self.tail(mixer_out)

# model = ResNetBlock1d(10, 64, 800, 2)
# input = torch.randn(10, 10, 800)
# print(model(input).shape)

# model = ResNetBlock2d(10, 64, [360, 640], [ 2, 2 ])
# input = torch.randn(10, 10, 360, 640)
# print(model(input).shape)

# model = GRUBlock(920, 920)
# input = torch.randn(10, 256, 920)
# print(model(torch.randn(10, 256, 920)).shape)

# model = AttentionBlock(920, 920, 920, 920)
# input = (torch.randn(10, 256, 920), torch.randn(10, 256, 920), torch.randn(10, 256, 920))
# print(model(*input).shape)

# model = AudioHeadBlock()
# input = torch.randn(10, 10, 800)
# print(model(input).shape)

# model = VideoHeadBlock(3)
# input = torch.randn(10, 3, 360, 640)
# print(model(input).shape)

# model = MediaMixerBlock()
# input = (torch.randn(10, 256, 400), torch.randn(10, 256, 920), torch.randn(10, 256, 920))
# print(model(*input).shape)

# model = AudioTailBlock()
# input = torch.randn(10, 256, 400)
# print(model(torch.randn(10, 256, 400)).shape)

model = Chobits()
model.eval();
input = (torch.randn(10, 10, 800), torch.randn(10, 10, 3, 360, 640))
print(model(*input).shape)

# torch.save(model, "D:/download/chobits.pt")

torch.jit.save(torch.jit.trace(model, input), "D:/download/chobits.pt")

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
