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
        self.norm = nn.RMSNorm(embed_dim)

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
        self.norm = nn.RMSNorm(o_dim)
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
        o = self.norm(self.proj(o) + query)
        return self.ffn(o)

class BasicBlock1dDownsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int = None,
        num_groups  : int = 32,
    ):
        super().__init__()
        self.need_downsample = kernel_size is not None
        if self.need_downsample:
            self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, bias = False)
            self.conv2 = nn.Conv1d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, bias = False)
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
            self.norm3 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_downsample:
            x = self.norm1(self.conv1(input))
            y = self.norm2(self.conv2(input))
            y = F.tanh(y)
            y = self.norm3(self.conv3(y))
            return F.tanh(x + y)
        else:
            x = self.norm1(self.conv1(input))
            x = F.tanh(x)
            x = self.norm2(self.conv2(x))
            return F.tanh(x + input)

class BasicBlock1dUpsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int = None,
        num_groups  : int = 32,
    ):
        super().__init__()
        self.need_upsample = kernel_size is not None
        if self.need_upsample:
            self.deconv1 = nn.ConvTranspose1d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose1d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, output_padding = 0, bias = False)
            self.deconv3 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
            self.norm3 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.deconv1 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_upsample:
            x = self.norm1(self.deconv1(input))
            y = self.norm2(self.deconv2(input))
            y = F.tanh(y)
            y = self.norm3(self.deconv3(y))
            return F.tanh(x + y)
        else:
            x = self.norm1(self.deconv1(input))
            x = F.tanh(x)
            x = self.norm2(self.deconv2(x))
            return F.tanh(x + input)

class ACE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock1dDownsample( 64,  64)
        self.layer2 = BasicBlock1dDownsample( 64, 128, 2)
        self.layer3 = BasicBlock1dDownsample(128, 128)
        self.layer4 = BasicBlock1dDownsample(128, 256, 2)
        self.layer5 = BasicBlock1dDownsample(256, 256)
        self.layer6 = BasicBlock1dDownsample(256, 512, 2)
        self.layer7 = BasicBlock1dDownsample(512, 512, 2)
        self.layer8 = BasicBlock1dDownsample(512, 512, 2)
        self.downsample1 = nn.Conv1d( 1, 64, kernel_size = 5, stride = 5, padding = 0, bias = False)
        self.downsample2 = nn.Conv1d(64, 64, kernel_size = 5, stride = 5, padding = 0, bias = False)
        self.norm1 = nn.GroupNorm(32, 64)
        self.norm2 = nn.GroupNorm(32, 64)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.norm1(self.downsample1(input)))
        x = F.tanh(self.norm2(self.downsample2(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(x.size(0), 1, -1)
        return x
    
class ACD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer8 = BasicBlock1dUpsample(512, 512, 2)
        self.layer7 = BasicBlock1dUpsample(512, 512, 2)
        self.layer6 = BasicBlock1dUpsample(512, 256, 2)
        self.layer5 = BasicBlock1dUpsample(256, 256)
        self.layer4 = BasicBlock1dUpsample(256, 128, 2)
        self.layer3 = BasicBlock1dUpsample(128, 128)
        self.layer2 = BasicBlock1dUpsample(128,  64, 2)
        self.layer1 = BasicBlock1dUpsample( 64,  64)
        self.upsample2 = nn.ConvTranspose1d(64, 64, kernel_size = 5, stride = 5, padding = 0, output_padding = 0, bias = False)
        self.upsample1 = nn.ConvTranspose1d(64,  1, kernel_size = 5, stride = 5, padding = 0, output_padding = 0, bias = False)
        self.norm2 = nn.GroupNorm(32, 64)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.view(input.size(0), 512, -1)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.tanh(self.norm2(self.upsample2(x)))
        x = self.upsample1(x)
        return x
    
class BasicBlock2dDownsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int | tuple[int, int] = None,
        num_groups  : int = 32,
    ):
        super().__init__()
        self.need_downsample = kernel_size is not None
        if self.need_downsample:
            self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, bias = False)
            self.conv2 = nn.Conv2d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, bias = False)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
            self.norm3 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_downsample:
            x = self.norm1(self.conv1(input))
            y = self.norm2(self.conv2(input))
            y = F.leaky_relu(y)
            y = self.norm3(self.conv3(y))
            return F.leaky_relu(x + y)
        else:
            x = self.norm1(self.conv1(input))
            x = F.leaky_relu(x)
            x = self.norm2(self.conv2(x))
            return F.leaky_relu(x + input)

class BasicBlock2dUpsample(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int | tuple[int, int] = None,
        num_groups  : int = 32,
    ):
        super().__init__()
        self.need_upsample = kernel_size is not None
        if self.need_upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose2d(in_channels,  out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, output_padding = 0, bias = False)
            self.deconv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
            self.norm3 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.deconv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False)
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.need_upsample:
            x = self.norm1(self.deconv1(input))
            y = self.norm2(self.deconv2(input))
            y = F.leaky_relu(y)
            y = self.norm3(self.deconv3(y))
            return F.leaky_relu(x + y)
        else:
            x = self.norm1(self.deconv1(input))
            x = F.leaky_relu(x)
            x = self.norm2(self.deconv2(x))
            return F.leaky_relu(x + input)

class VCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock2dDownsample( 64,   64)
        self.layer2 = BasicBlock2dDownsample( 64,  128, (2, 2))
        self.layer3 = BasicBlock2dDownsample(128,  128)
        self.layer4 = BasicBlock2dDownsample(128,  256, (2, 2))
        self.layer5 = BasicBlock2dDownsample(256,  256)
        self.layer6 = BasicBlock2dDownsample(256,  512, (2, 2))
        self.layer7 = BasicBlock2dDownsample(512,  512, (2, 2))
        self.layer8 = BasicBlock2dDownsample(512, 1024, (2, 2))
        self.downsample1 = nn.Conv2d( 3, 64, kernel_size = (5, 5), stride = (5, 5), padding = 0, bias = False)
        self.downsample2 = nn.Conv2d(64, 64, kernel_size = (3, 4), stride = (3, 4), padding = 0, bias = False)
        self.norm1 = nn.GroupNorm(32, 64)
        self.norm2 = nn.GroupNorm(32, 64)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.norm1(self.downsample1(input)))
        x = F.leaky_relu(self.norm2(self.downsample2(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(x.size(0), 1, -1)
        return x
    
class VCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer8 = BasicBlock2dUpsample(1024, 512, (2, 2))
        self.layer7 = BasicBlock2dUpsample( 512, 512, (2, 2))
        self.layer6 = BasicBlock2dUpsample( 512, 256, (2, 2))
        self.layer5 = BasicBlock2dUpsample( 256, 256)
        self.layer4 = BasicBlock2dUpsample( 256, 128, (2, 2))
        self.layer3 = BasicBlock2dUpsample( 128, 128)
        self.layer2 = BasicBlock2dUpsample( 128,  64, (2, 2))
        self.layer1 = BasicBlock2dUpsample(  64,  64)
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size = (3, 4), stride = (3, 4), padding = 0, output_padding = 0, bias = False)
        self.upsample1 = nn.ConvTranspose2d(64,  3, kernel_size = (5, 5), stride = (5, 5), padding = 0, output_padding = 0, bias = False)
        self.norm2 = nn.GroupNorm(32, 64)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.view(input.size(0), 1024, 1, 1)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.leaky_relu(self.norm2(self.upsample2(x)))
        x = self.upsample1(x)
        return x

class MemoryLayer(nn.Module):
    def __init__(
        self,
        memory_dim: int = 1024,
        num_heads : int = 8,
    ):
        super().__init__()
        self.excite_mha = MHA(memory_dim, memory_dim, memory_dim, memory_dim, memory_dim * 2, num_heads)
#       self.memory_mha = MHA(memory_dim, memory_dim, memory_dim, memory_dim, memory_dim * 2, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        memory: torch.Tensor,
        excite: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.excite_mha(memory, excite, excite)
#       memory = self.memory_mha(memory, memory, memory)
        return memory

class Memory(nn.Module):
    def __init__(
        self,
        audio_dim : int = 512,
        video_dim : int = 1024,
        memory_dim: int = 1024,
        num_heads : int = 8,
        num_layer : int = 3,
    ):
        super().__init__()
        # 分区：音频、视频、其他感知
        self.audio_zone = nn.Parameter(torch.randn(1, 1, 1))
        self.video_zone = nn.Parameter(torch.randn(1, 1, 1))
#       self.audio_zone = nn.Parameter(torch.randn(1, 1, memory_dim))
#       self.video_zone = nn.Parameter(torch.randn(1, 1, memory_dim))
        # 投影：音频、视频、其他感知
        self.audio_proj = nn.Linear(audio_dim, memory_dim)
        self.video_proj = nn.Linear(video_dim, memory_dim)
        # 记忆力层
        self.layers = nn.ModuleList([MemoryLayer(memory_dim, num_heads) for _ in range(num_layer)])

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio : torch.Tensor,
        video : torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        audio_proj = self.audio_proj(audio) + self.audio_zone
        video_proj = self.video_proj(video) + self.video_zone
        sense_proj = torch.cat((audio_proj, video_proj), dim = 1)
        for layer in self.layers:
            memory = layer(memory, sense_proj)
        return memory

class RecallLayer(nn.Module):
    def __init__(
        self,
        recall_dim: int,
        memory_dim: int,
        num_heads : int,
    ):
        super().__init__()
        self.memory_mha = MHA(recall_dim, memory_dim, memory_dim, recall_dim, memory_dim * 2, num_heads)
#       self.recall_mha = MHA(recall_dim, recall_dim, recall_dim, recall_dim, recall_dim * 2, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        recall: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        recall = self.memory_mha(recall, memory, memory)
#       recall = self.recall_mha(recall, recall, recall)
        return recall

class Recall(nn.Module):
    def __init__(
        self,
        audio_dim : int = 512,
        video_dim : int = 1024,
        memory_dim: int = 1024,
        num_heads : int = 8,
        num_layer : int = 3,
    ):
        super().__init__()
        self.audio_layers = nn.ModuleList([RecallLayer(audio_dim, memory_dim, num_heads) for _ in range(num_layer)])
        self.video_layers = nn.ModuleList([RecallLayer(video_dim, memory_dim, num_heads) for _ in range(num_layer)])

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio : torch.Tensor,
        video : torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.audio_layers:
            audio = layer(audio, memory)
        for layer in self.video_layers:
            video = layer(video, memory)
        return (audio, video)

class Chobits(nn.Module):
    def __init__(self):
        super().__init__()
        self.ace = ACE()
        self.acd = ACD()
        self.vce = VCE()
        self.vcd = VCD()
        self.memory = Memory()
        self.recall = Recall()

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        audio : torch.Tensor,
        video : torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_encode = self.ace(audio)
        video_encode = self.vce(video)
        memory = self.memory(audio_encode, video_encode, memory)
        audio_encode, video_encode = self.recall(audio_encode, video_encode, memory)
        audio_decode = self.acd(audio_encode)
        video_decode = self.vcd(video_encode)
        return (audio_decode, video_decode, memory)

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
        elif isinstance(layer, nn.RMSNorm):
            layer.reset_parameters()
        elif isinstance(layer, nn.GroupNorm):
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
        elif isinstance(layer, MemoryLayer):
            initialize_weights(layer)
        elif isinstance(layer, Recall):
            initialize_weights(layer)
        elif isinstance(layer, RecallLayer):
            initialize_weights(layer)
        elif isinstance(layer, Chobits):
            initialize_weights(layer)
        else:
            print(f"不支持初始化的层: {layer.__class__.__name__}")
