import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module: nn.Module):
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            layer.reset_parameters()
        elif isinstance(layer, nn.BatchNorm2d):
            layer.reset_parameters()

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

class BasicBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.need_downsample = in_channels != out_channels
        if self.need_downsample:
            self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size = 1, stride = 2, padding = 0, bias = False)
            # self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
            self.conv2 = nn.Conv1d(in_channels,  out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1   = nn.BatchNorm1d(out_channels)
            self.bn2   = nn.BatchNorm1d(out_channels)
            self.bn3   = nn.BatchNorm1d(out_channels)
        else:
            self.conv1 = nn.Conv1d(out_channels,  out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1   = nn.BatchNorm1d(out_channels)
            self.bn2   = nn.BatchNorm1d(out_channels)

    def forward(self, input):
        if self.need_downsample:
            x = F.relu(self.bn1(self.conv1(input)))
            y = F.relu(self.bn2(self.conv2(input)))
            y = F.relu(self.bn3(self.conv3(y)))
            out = F.relu(x + y)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(input)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(out + input)
            return out

class BasicBlock1dInverse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.need_upsample = in_channels != out_channels
        if self.need_upsample:
            self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False)
            self.deconv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.bn3 = nn.BatchNorm1d(out_channels)
        else:
            self.deconv1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.deconv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, input):
        if self.need_upsample:
            x = F.relu(self.bn1(self.deconv1(input)))
            y = F.relu(self.bn2(self.deconv2(input)))
            y = F.relu(self.bn3(self.conv3(y)))
            out = F.relu(x + y)
            return out
        else:
            out = F.relu(self.bn1(self.deconv1(input)))
            out = F.relu(self.bn2(self.deconv2(out)))
            out = F.relu(out + input)
            return out

class BasicBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.need_downsample = in_channels != out_channels
        if self.need_downsample:
            self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size = 1, stride = 2, padding = 0, bias = False)
            # self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
            self.conv2 = nn.Conv2d(in_channels,  out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1   = nn.BatchNorm2d(out_channels)
            self.bn2   = nn.BatchNorm2d(out_channels)
            self.bn3   = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(out_channels,  out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn1   = nn.BatchNorm2d(out_channels)
            self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        if self.need_downsample:
            x = F.relu(self.bn1(self.conv1(input)))
            y = F.relu(self.bn2(self.conv2(input)))
            y = F.relu(self.bn3(self.conv3(y)))
            out = F.relu(x + y)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(input)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(out + input)
            return out

class BasicBlock2dInverse(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.need_upsample = in_channels != out_channels
        if self.need_upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False)
            self.deconv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.deconv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        if self.need_upsample:
            x = F.relu(self.bn1(self.deconv1(input)))
            y = F.relu(self.bn2(self.deconv2(input)))
            y = F.relu(self.bn3(self.conv3(y)))
            out = F.relu(x + y)
            return out
        else:
            out = F.relu(self.bn1(self.deconv1(input)))
            out = F.relu(self.bn2(self.deconv2(out)))
            out = F.relu(out + input)
            return out

class ACE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv1d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1     = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = BasicBlock1d( 64,  64)
        self.layer2 = BasicBlock1d( 64,  64)
        self.layer3 = BasicBlock1d( 64, 128)
        self.layer4 = BasicBlock1d(128, 128)
        self.layer5 = BasicBlock1d(128, 256)
        self.layer6 = BasicBlock1d(256, 256)
        self.layer7 = BasicBlock1d(256, 512)
        self.layer8 = BasicBlock1d(512, 512)
        self.avgpool = nn.AdaptiveAvgPool1d((2))
        initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 1, -1)
        return x
    
class ACD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer8_inv = BasicBlock1dInverse(512, 512)
        self.layer7_inv = BasicBlock1dInverse(512, 256)
        self.layer6_inv = BasicBlock1dInverse(256, 256)
        self.layer5_inv = BasicBlock1dInverse(256, 128)
        self.layer4_inv = BasicBlock1dInverse(128, 128)
        self.layer3_inv = BasicBlock1dInverse(128, 64)
        self.layer2_inv = BasicBlock1dInverse(64, 64)
        self.layer1_inv = BasicBlock1dInverse(64, 64)
        self.upsample = nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn_up    = nn.BatchNorm1d(64)
        self.deconv1  = nn.ConvTranspose1d(64, 1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)

    def forward(self, input):
        x = F.interpolate(input.view(input.size(0), 512, -1), size=25, mode='linear')
        x = self.layer8_inv(x)
        x = self.layer7_inv(x)
        x = self.layer6_inv(x)
        x = self.layer5_inv(x)
        x = self.layer4_inv(x)
        x = self.layer3_inv(x)
        x = self.layer2_inv(x)
        x = self.layer1_inv(x)
        x = F.relu(self.bn_up(self.upsample(x)))
        x = self.deconv1(x)
        return x

class VCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1     = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = BasicBlock2d( 64,  64)
        self.layer2 = BasicBlock2d( 64,  64)
        self.layer3 = BasicBlock2d( 64, 128)
        self.layer4 = BasicBlock2d(128, 128)
        self.layer5 = BasicBlock2d(128, 256)
        self.layer6 = BasicBlock2d(256, 256)
        self.layer7 = BasicBlock2d(256, 512)
        self.layer8 = BasicBlock2d(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 2))
        initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 1, -1)
        return x
    
class VCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer8_inv = BasicBlock2dInverse(512, 512)
        self.layer7_inv = BasicBlock2dInverse(512, 256)
        self.layer6_inv = BasicBlock2dInverse(256, 256)
        self.layer5_inv = BasicBlock2dInverse(256, 128)
        self.layer4_inv = BasicBlock2dInverse(128, 128)
        self.layer3_inv = BasicBlock2dInverse(128, 64)
        self.layer2_inv = BasicBlock2dInverse(64, 64)
        self.layer1_inv = BasicBlock2dInverse(64, 64)
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn_up    = nn.BatchNorm2d(64)
        self.deconv1  = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.bn1_inv  = nn.BatchNorm2d(3)

    def forward(self, input):
        x = F.interpolate(input.view(input.size(0), 512, 1, 2), size=(15, 20), mode='bilinear')
        x = self.layer8_inv(x)
        x = self.layer7_inv(x)
        x = self.layer6_inv(x)
        x = self.layer5_inv(x)
        x = self.layer4_inv(x)
        x = self.layer3_inv(x)
        x = self.layer2_inv(x)
        x = self.layer1_inv(x)
        x = F.relu(self.bn_up(self.upsample(x)))
        x = self.deconv1(x)
        return x

class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MHA(1024, 1024, 1024, 1024, 1024 * 2, 8)

    def forward(self, input, memory):
        return self.mha(memory, input, input)

class Mixer(nn.Module):
    def __init__(
        self,
        audio_dim: int = 1024,
        video_dim: int = 1024,
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
        self.ace = ACE()
        self.acd = ACD()
        self.vce = VCE()
        self.vcd = VCD()
        self.audio_memory = Memory()
        self.video_memory = Memory()
        self.mixers = nn.ModuleList([Mixer() for _ in range(3)])
        self.audio_mha = Mixer()
        self.video_mha = Mixer()

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_memory: torch.Tensor,
        video_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_o = self.ace(audio)
        video_o = self.vce(video)
        audio_c = audio_o
        audio_m = self.audio_memory(audio_o, audio_memory)
        video_m = self.video_memory(video_o, video_memory)
        audio_o = audio_m
        video_o = video_m
        for mixer in self.mixers:
            audio_x, video_x = mixer(audio_o, video_o)
            audio_o = audio_x
            video_o = video_x
        audio_c, video_c = self.audio_mha(audio_c, audio_o)
        audio_c, video_c = self.video_mha(audio_c, video_o)
        audio_c = self.acd(audio_c)
        return audio_c, audio_m, video_m
