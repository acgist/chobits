import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        scale    : int = 2,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * scale),
            nn.SiLU(),
            nn.Linear(embed_dim * scale, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self) -> None:
        initialize_weights(self)

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
        h_dim    : int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.q    = nn.Linear(q_dim, h_dim, bias = False)
        self.k    = nn.Linear(k_dim, h_dim, bias = False)
        self.v    = nn.Linear(v_dim, h_dim, bias = False)
        self.attn = nn.MultiheadAttention(h_dim, num_heads, bias = False, batch_first = True)
        self.proj = nn.Linear(h_dim, o_dim, bias = False)
        self.norm = nn.LayerNorm(o_dim)
        self.ffn  = FFN(o_dim)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        query: torch.Tensor,
        key  : torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q(query)
        k = self.k(key  )
        v = self.v(value)
        o, _ = self.attn(q, k, v)
        o = query + self.proj(o)
        o = self.norm(o)
        return self.ffn(o)

class BasicBlock1dDownsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
    ):
        super().__init__()
        self.need_downsample = in_channels != out_channels
        if self.need_downsample:
            self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
            self.conv2 = nn.Conv1d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.bn3 = nn.BatchNorm1d(out_channels)
        else:
            self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_downsample:
            x = self.bn1(self.conv1(input))
            y = self.bn2(self.conv2(input))
            y = F.leaky_relu(y)
            y = self.bn3(self.conv3(y))
            return F.leaky_relu(x + y)
        else:
            x = self.bn1(self.conv1(input))
            x = F.leaky_relu(x)
            x = self.bn2(self.conv2(x))
            return F.leaky_relu(x + input)

class BasicBlock1dUpsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
    ):
        super().__init__()
        self.need_upsample = in_channels != out_channels
        if self.need_upsample:
            self.deconv1 = nn.ConvTranspose1d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose1d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, output_padding = 0, bias = False)
            self.deconv3 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.bn3 = nn.BatchNorm1d(out_channels)
        else:
            self.deconv1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_upsample:
            x = self.bn1(self.deconv1(input))
            y = self.bn2(self.deconv2(input))
            y = F.leaky_relu(y)
            y = self.bn3(self.deconv3(y))
            return F.leaky_relu(x + y)
        else:
            x = self.bn1(self.deconv1(input))
            x = F.leaky_relu(x)
            x = self.bn2(self.deconv2(x))
            return F.leaky_relu(x + input)

class BasicBlock2dDownsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
    ):
        super().__init__()
        self.need_downsample = in_channels != out_channels
        if self.need_downsample:
            self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
            self.conv2 = nn.Conv2d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_downsample:
            x = self.bn1(self.conv1(input))
            y = self.bn2(self.conv2(input))
            y = F.leaky_relu(y)
            y = self.bn3(self.conv3(y))
            return F.leaky_relu(x + y)
        else:
            x = self.bn1(self.conv1(input))
            x = F.leaky_relu(x)
            x = self.bn2(self.conv2(x))
            return F.leaky_relu(x + input)

class BasicBlock2dUpsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
    ):
        super().__init__()
        self.need_upsample = in_channels != out_channels
        if self.need_upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose2d(in_channels,  out_channels, kernel_size = 2, stride = 2, padding = 0, output_padding = 0, bias = False)
            self.deconv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.deconv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_upsample:
            x = self.bn1(self.deconv1(input))
            y = self.bn2(self.deconv2(input))
            y = F.leaky_relu(y)
            y = self.bn3(self.deconv3(y))
            return F.leaky_relu(x + y)
        else:
            x = self.bn1(self.deconv1(input))
            x = F.leaky_relu(x)
            x = self.bn2(self.deconv2(x))
            return F.leaky_relu(x + input)

class ACE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock1dDownsample( 64,  64)
        self.layer2 = BasicBlock1dDownsample( 64, 128)
        self.layer3 = BasicBlock1dDownsample(128, 128)
        self.layer4 = BasicBlock1dDownsample(128, 256)
        self.layer5 = BasicBlock1dDownsample(256, 256)
        self.layer6 = BasicBlock1dDownsample(256, 512)
        self.layer7 = BasicBlock1dDownsample(512, 512)
        self.downsample1 = nn.Conv1d(  1,  64, kernel_size = 5, stride = 5, padding = 0, bias = False)
        self.downsample2 = nn.Conv1d( 64,  64, kernel_size = 5, stride = 5, padding = 0, bias = False)
        self.downsample3 = nn.Conv1d(512, 512, kernel_size = 4, stride = 4, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm1d( 64)
        self.bn2 = nn.BatchNorm1d( 64)
        self.bn3 = nn.BatchNorm1d(512)
        self.ffn = nn.Sequential(
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.downsample1(input)))
        x = F.leaky_relu(self.bn2(self.downsample2(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(self.bn3(self.downsample3(x)))
        x = x.view(x.size(0), 1, -1)
        x = self.ffn(x)
        return x
    
class ACD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer7 = BasicBlock1dUpsample(512, 512)
        self.layer6 = BasicBlock1dUpsample(512, 256)
        self.layer5 = BasicBlock1dUpsample(256, 256)
        self.layer4 = BasicBlock1dUpsample(256, 128)
        self.layer3 = BasicBlock1dUpsample(128, 128)
        self.layer2 = BasicBlock1dUpsample(128,  64)
        self.layer1 = BasicBlock1dUpsample( 64,  64)
        self.upsample3 = nn.ConvTranspose1d(512, 512, kernel_size = 4, stride = 4, padding = 0, output_padding = 0, bias = False)
        self.upsample2 = nn.ConvTranspose1d( 64,  64, kernel_size = 5, stride = 5, padding = 0, output_padding = 0, bias = False)
        self.upsample1 = nn.ConvTranspose1d( 64,   1, kernel_size = 5, stride = 5, padding = 0, output_padding = 0, bias = False)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d( 64)
        self.ffn = nn.Sequential(
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ffn(input)
        x = F.leaky_relu(self.bn3(self.upsample3(x.view(x.size(0), 512, -1))))
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.leaky_relu(self.bn2(self.upsample2(x)))
        x = self.upsample1(x)
        return x

class VCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock2dDownsample( 64,  64)
        self.layer2 = BasicBlock2dDownsample( 64, 128)
        self.layer3 = BasicBlock2dDownsample(128, 128)
        self.layer4 = BasicBlock2dDownsample(128, 256)
        self.layer5 = BasicBlock2dDownsample(256, 256)
        self.layer6 = BasicBlock2dDownsample(256, 512)
        self.layer7 = BasicBlock2dDownsample(512, 512)
        self.downsample1 = nn.Conv2d(  3,   64, kernel_size = (5, 5), stride = (5, 5), padding = 0, bias = False)
        self.downsample2 = nn.Conv2d( 64,   64, kernel_size = (2, 2), stride = (2, 2), padding = 0, bias = False)
        self.downsample3 = nn.Conv2d(512,  512, kernel_size = (3, 4), stride = (3, 4), padding = 0, bias = False)
        self.downsample4 = nn.Conv2d(512, 1024, kernel_size = (2, 2), stride = (2, 2), padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(  64)
        self.bn2 = nn.BatchNorm2d(  64)
        self.bn3 = nn.BatchNorm2d( 512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.ffn = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
        )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.downsample1(input)))
        x = F.leaky_relu(self.bn2(self.downsample2(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(self.bn3(self.downsample3(x)))
        x = F.leaky_relu(self.bn4(self.downsample4(x)))
        x = x.view(x.size(0), 1, -1)
        x = self.ffn(x)
        return x
    
class VCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer7 = BasicBlock2dUpsample(512, 512)
        self.layer6 = BasicBlock2dUpsample(512, 256)
        self.layer5 = BasicBlock2dUpsample(256, 256)
        self.layer4 = BasicBlock2dUpsample(256, 128)
        self.layer3 = BasicBlock2dUpsample(128, 128)
        self.layer2 = BasicBlock2dUpsample(128,  64)
        self.layer1 = BasicBlock2dUpsample( 64,  64)
        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size = (2, 2), stride = (2, 2), padding = 0, output_padding = 0, bias = False)
        self.upsample3 = nn.ConvTranspose2d( 512, 512, kernel_size = (3, 4), stride = (3, 4), padding = 0, output_padding = 0, bias = False)
        self.upsample2 = nn.ConvTranspose2d(  64,  64, kernel_size = (2, 2), stride = (2, 2), padding = 0, output_padding = 0, bias = False)
        self.upsample1 = nn.ConvTranspose2d(  64,   3, kernel_size = (5, 5), stride = (5, 5), padding = 0, output_padding = 0, bias = False)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d( 64)
        self.ffn = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
        )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ffn(input)
        x = F.leaky_relu(self.bn4(self.upsample4(x.view(x.size(0), 1024, 1, 1))))
        x = F.leaky_relu(self.bn3(self.upsample3(x)))
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.leaky_relu(self.bn2(self.upsample2(x)))
        x = self.upsample1(x)
        return x

class Memory(nn.Module):
    def __init__(
        self,
        audio_dim: int = 512,
        video_dim: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.audio_mha = MHA(audio_dim, audio_dim, audio_dim, audio_dim, audio_dim * 2, num_heads)
        self.video_mha = MHA(video_dim, video_dim, video_dim, video_dim, video_dim * 2, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_memory: torch.Tensor,
        video_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_o = self.audio_mha(audio_memory, audio, audio)
        video_o = self.video_mha(video_memory, video, video)
        return (audio_o, video_o)

class Mixer(nn.Module):
    def __init__(
        self,
        audio_dim: int = 512,
        video_dim: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()
        self.audio_mha = MHA(audio_dim, video_dim, video_dim, audio_dim, audio_dim * 2, num_heads)
        self.video_mha = MHA(video_dim, audio_dim, audio_dim, video_dim, video_dim * 2, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_o = self.audio_mha(audio, video, video)
        video_o = self.video_mha(video, audio, audio)
        return (audio_o, video_o)

class Muxer(nn.Module):
    def __init__(
        self,
        audio_dim: int = 512,
        video_dim: int = 1024,
        num_heads: int = 8,
        num_mixer: int = 3,
    ):
        super().__init__()
        self.mixers = nn.ModuleList([Mixer() for _ in range(num_mixer)])
        self.audio_mha = MHA(audio_dim, audio_dim, audio_dim, audio_dim, audio_dim * 2, num_heads)
        self.video_mha = MHA(video_dim, video_dim, video_dim, video_dim, video_dim * 2, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_memory: torch.Tensor,
        video_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_m = audio_memory
        video_m = video_memory
        for mixer in self.mixers:
            audio_x, video_x = mixer(audio_m, video_m)
            audio_m = audio_x
            video_m = video_x
        audio_o = self.audio_mha(audio, audio_m, audio_m)
        video_o = self.video_mha(video, video_m, video_m)
        return (audio_o, video_o)

class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.ace = ACE()
        self.acd = ACD()
        self.vce = VCE()
        self.vcd = VCD()
        self.memory = Memory()
        self.muxer  = Muxer ()

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_memory: torch.Tensor,
        video_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_encode = self.ace(audio)
        video_encode = self.vce(video)
        audio_feature_memory, video_feature_memory = self.memory(audio_encode, video_encode, audio_memory, video_memory)
        audio_feature_encode, video_feature_encode = self.muxer(audio_encode, video_encode, audio_feature_memory, video_feature_memory)
        audio_feature_decode = self.acd(audio_feature_encode)
        video_feature_decode = self.vcd(video_feature_encode)
        return (audio_feature_decode, video_feature_decode, audio_feature_memory, video_feature_memory)

def initialize_weights(module: nn.Module) -> None:
    for layer in module.children():
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
        elif isinstance(layer, nn.Conv1d):
            layer.reset_parameters()
        elif isinstance(layer, nn.Conv2d):
            layer.reset_parameters()
        elif isinstance(layer, nn.ConvTranspose1d):
            layer.reset_parameters()
        elif isinstance(layer, nn.ConvTranspose2d):
            layer.reset_parameters()
        elif isinstance(layer, nn.LayerNorm):
            layer.reset_parameters()
        elif isinstance(layer, nn.BatchNorm1d):
            layer.reset_parameters()
        elif isinstance(layer, nn.BatchNorm2d):
            layer.reset_parameters()
        elif isinstance(layer, nn.MultiheadAttention):
            layer._reset_parameters()
        elif isinstance(layer, nn.ModuleList):
            initialize_weights(layer)
        elif isinstance(layer, nn.Sequential):
            initialize_weights(layer)
        elif isinstance(layer, nn.SiLU):
            pass
        elif isinstance(layer, nn.LeakyReLU):
            pass
        elif isinstance(layer, nn.AdaptiveAvgPool1d):
            pass
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(layer, FFN):
            initialize_weights(layer)
        elif isinstance(layer, MHA):
            initialize_weights(layer)
        elif isinstance(layer, BasicBlock1dDownsample):
            initialize_weights(layer)
        elif isinstance(layer, BasicBlock1dUpsample):
            initialize_weights(layer)
        elif isinstance(layer, BasicBlock2dDownsample):
            initialize_weights(layer)
        elif isinstance(layer, BasicBlock2dUpsample):
            initialize_weights(layer)
        elif isinstance(layer, ACE):
            initialize_weights(layer)
        elif isinstance(layer, ACD):
            initialize_weights(layer)
        elif isinstance(layer, VCE):
            initialize_weights(layer)
        elif isinstance(layer, VCD):
            initialize_weights(layer)
        elif isinstance(layer, Memory):
            initialize_weights(layer)
        elif isinstance(layer, Mixer):
            initialize_weights(layer)
        elif isinstance(layer, Muxer):
            initialize_weights(layer)
        elif isinstance(layer, Chobits):
            initialize_weights(layer)
        else:
            print(f"不支持初始化的层: {layer.__class__.__name__}")
