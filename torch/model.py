import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.0001,
    ):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        mu     : torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = -1)
        return kl.mean() * self.beta

class STFTLoss(nn.Module):
    def __init__(
        self,
        n_fft     : int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def db(
        self,
        stft: torch.Tensor
    ) -> torch.Tensor:
        spec = torch.abs(stft)
        spec_db = 20 * torch.log10(torch.clamp(spec, 1e-8))
        spec_db = torch.maximum(spec_db, spec_db.max() - 80)
        return spec_db

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        # 原始损失
        raw_loss = F.l1_loss(pred, true)
        # 计算STFT
        pred_spec = torch.stft(pred.squeeze(1), self.n_fft, self.hop_length, self.win_length, self.window, return_complex = True)
        true_spec = torch.stft(true.squeeze(1), self.n_fft, self.hop_length, self.win_length, self.window, return_complex = True)
        # 计算幅度
        pred_mag = torch.abs(pred_spec)
        true_mag = torch.abs(true_spec)
        # 计算角度
#       pred_pha = torch.angle(pred_spec)
#       true_pha = torch.angle(true_spec)
        # 计算对数语谱图（人耳感知）
        pred_db = self.db(pred_mag)
        true_db = self.db(true_mag)
        # 计算对数语谱图损失
        db_loss = F.l1_loss(pred_db, true_db)
#       db_loss = F.mse_loss(pred_db, true_db)
        # 计算损失：原始损失 + 对数语谱图损失
        loss = 0.6 * raw_loss + 0.4 * db_loss
        return loss

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

class ACE(nn.Module):
    def __init__(
        self,
        n_fft     : int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.downsample1 = nn.Conv2d(1, 64, kernel_size = 2, stride = 1, padding = 0, bias = False)
        self.norm1 = nn.GroupNorm(32, 64)
        self.layer1 = BasicBlock2dDownsample( 64,  64)
        self.layer2 = BasicBlock2dDownsample( 64, 128, [ 1, 2 ])
        self.layer3 = BasicBlock2dDownsample(128, 128)
        self.layer4 = BasicBlock2dDownsample(128, 256, [ 1, 2 ])
        self.layer5 = BasicBlock2dDownsample(256, 256)
        self.layer6 = BasicBlock2dDownsample(256, 256, [ 1, 2 ])
        self.layer7 = BasicBlock2dDownsample(256, 256)
        self.layer8 = BasicBlock2dDownsample(256, 256, [ 3, 4 ])
        self.fc_mu      = nn.Linear(256, 256)
        self.fc_log_var = nn.Linear(256, 256)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def reparameterize(
        self,
        mu     : torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stft = torch.stft(
            input.squeeze(1),
            n_fft          = self.n_fft,
            hop_length     = self.hop_length,
            win_length     = self.win_length,
            window         = self.window,
            return_complex = True,
        )
        spec = torch.abs(stft)
        spec_db = 20 * torch.log10(spec + 1e-8)
        spec_db = spec_db.transpose(1, 2).unsqueeze(1)
        x = F.leaky_relu(self.norm1(self.downsample1(spec_db)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.flatten(start_dim = 2)
        x = x.transpose(1, 2)
        mu      = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
class ACD(nn.Module):
    def __init__(
        self,
        n_fft     : int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        length    : int = 800,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length     = length
        self.register_buffer("window", torch.hann_window(win_length))
        self.layer8 = BasicBlock2dUpsample(256, 256, [ 3, 4 ])
        self.layer7 = BasicBlock2dUpsample(256, 256)
        self.layer6 = BasicBlock2dUpsample(256, 256, [ 1, 2 ])
        self.layer5 = BasicBlock2dUpsample(256, 256)
        self.layer4 = BasicBlock2dUpsample(256, 128, [ 1, 2])
        self.layer3 = BasicBlock2dUpsample(128, 128)
        self.layer2 = BasicBlock2dUpsample(128,  64, [ 1, 2 ])
        self.layer1 = BasicBlock2dUpsample( 64,  64)
        self.upsample1 = nn.ConvTranspose2d(64, 1, kernel_size = 2, stride = 1, padding = 0, output_padding = 0, bias = False)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.transpose(1, 2)
        x = x.view(x.size(0), 256, 2, 8)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.leaky_relu(self.upsample1(x))
        mag = (10 ** (x / 20)) - 1e-8
        mag = mag.squeeze(1).transpose(1, 2)
        wav = torch.istft(
            torch.polar(mag, torch.zeros_like(mag)),
            n_fft      = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window     = self.window,
            length     = self.length,
        )
        wav = wav.unsqueeze(1)
        return torch.tanh(wav)

class VCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample1 = nn.Conv2d(3, 64, kernel_size = (2, 2), stride = (2, 2), padding = 0, bias = False)
        self.norm1 = nn.GroupNorm(32, 64)
        self.layer1 = BasicBlock2dDownsample( 64,  64)
        self.layer2 = BasicBlock2dDownsample( 64, 128, (2, 2))
        self.layer3 = BasicBlock2dDownsample(128, 128)
        self.layer4 = BasicBlock2dDownsample(128, 256, (2, 2))
        self.layer5 = BasicBlock2dDownsample(256, 256)
        self.layer6 = BasicBlock2dDownsample(256, 512, (3, 4))
        self.layer7 = BasicBlock2dDownsample(512, 512)
        self.layer8 = BasicBlock2dDownsample(512, 512, (5, 5))
        self.fc_mu      = nn.Linear(512, 512)
        self.fc_log_var = nn.Linear(512, 512)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def reparameterize(
        self,
        mu     : torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(self.norm1(self.downsample1(input)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.flatten(start_dim = 2)
        x = x.transpose(1, 2)
        mu      = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
class VCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer8 = BasicBlock2dUpsample(512, 512, (5, 5))
        self.layer7 = BasicBlock2dUpsample(512, 512)
        self.layer6 = BasicBlock2dUpsample(512, 256, (3, 4))
        self.layer5 = BasicBlock2dUpsample(256, 256)
        self.layer4 = BasicBlock2dUpsample(256, 128, (2, 2))
        self.layer3 = BasicBlock2dUpsample(128, 128)
        self.layer2 = BasicBlock2dUpsample(128,  64, (2, 2))
        self.layer1 = BasicBlock2dUpsample( 64,  64)
        self.upsample1 = nn.ConvTranspose2d(64, 3, kernel_size = (2, 2), stride = (2, 2), padding = 0, output_padding = 0, bias = False)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.transpose(1, 2)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.tanh(self.upsample1(x))
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
        audio_dim : int = 256,
        video_dim : int = 512,
        memory_dim: int = 1024,
        num_heads : int = 8,
        num_layer : int = 3,
    ):
        super().__init__()
        # 验证：Memory or MemoryLayer
        # 分区：音频、视频、其他感知
        self.audio_embedding  = nn.Parameter(torch.randn(1, 1, 1))
        self.video_embedding  = nn.Parameter(torch.randn(1, 1, 1))
        self.memory_embedding = nn.Parameter(torch.randn(1, 1, 1))
#       self.audio_embedding  = nn.Parameter(torch.randn(1, 1, memory_dim))
#       self.video_embedding  = nn.Parameter(torch.randn(1, 1, memory_dim))
#       self.memory_embedding = nn.Parameter(torch.randn(1, 1, memory_dim))
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
        audio_proj = self.audio_proj(audio) + self.audio_embedding
        video_proj = self.video_proj(video) + self.video_embedding
        sense_proj = torch.cat((audio_proj, video_proj), dim = 1)
        memory_embedding = memory + self.memory_embedding
        for layer in self.layers:
            memory_embedding = layer(memory_embedding, sense_proj)
        return memory + memory_embedding

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
        audio_dim : int = 256,
        video_dim : int = 512,
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
        audio_encode, _ = self.ace(audio)
        video_encode, _ = self.vce(video)
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
