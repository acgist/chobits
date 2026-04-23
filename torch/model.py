import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.0001,
    ) -> None:
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
    ) -> None:
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def db(
        self,
        raw: torch.Tensor
    ) -> torch.Tensor:
        spec = torch.stft(raw.squeeze(1), self.n_fft, self.hop_length, self.win_length, self.window, return_complex = True)
        mag  = torch.abs(spec)       # 幅度
#       pha  = torch.angle(spec)     # 角度
#       stft = torch.polar(mag, pha) # 恢复
        db = 20.0 * torch.log10(mag + 1e-8)
        db = torch.clamp(db, min = -60)
        return db

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        # 原始损失
        raw_loss = F.l1_loss(pred, true)
        # 对数语谱图（人耳感知）
        pred_db = self.db(pred)
        true_db = self.db(true)
        # 对数语谱图损失
        db_loss = F.l1_loss(pred_db, true_db)
#       db_loss = F.mse_loss(pred_db, true_db)
        # 总损失：原始损失 + 对数语谱图损失
        loss = 0.6 * raw_loss + 0.4 * db_loss
        return loss

class FFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        scale    : int = 2,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * scale, bias = False),
            nn.SiLU(),
            nn.Linear(embed_dim * scale, embed_dim, bias = False),
        )
        self.norm = nn.RMSNorm(embed_dim)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-Norm
        return self.fc(self.norm(input)) + input

class MHA(nn.Module):
    def __init__(
        self,
        q_dim : int,
        kv_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.norm_q  = nn.RMSNorm(q_dim)
        self.norm_kv = nn.RMSNorm(kv_dim)
        self.attn = nn.MultiheadAttention(q_dim, num_heads, kdim = kv_dim, vdim = kv_dim, bias = False, batch_first = True)
        self.proj = nn.Linear(q_dim, q_dim, bias = False)
        self.ffn  = FFN(q_dim)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        q : torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-Norm
        residual = q
        q = self.norm_q(q)
        k = v = self.norm_kv(kv)
        o, _ = self.attn(q, k, v)
        o = self.proj(o) + residual
        return self.ffn(o)

class BasicBlock2d(nn.Module):
    """
    输入input必须能被kernel_size整除
    """
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int | tuple[int, int] = None,
        upsample    : bool = False,
        num_groups  : int = 32,
    ) -> None:
        super().__init__()
        num_groups = min(num_groups, out_channels)
        if kernel_size is None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.GroupNorm(num_groups, out_channels),
            )
            self.shortcut= nn.Identity() if in_channels == out_channels else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
#               nn.GroupNorm(num_groups, out_channels),
            )
        else:
            if upsample:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, output_padding = 0, bias = False),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.LeakyReLU(inplace = True),
                    nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, output_padding = 0, bias = False),
                    nn.GroupNorm(num_groups, out_channels),
                )
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, output_padding = 0, bias = False),
                    nn.GroupNorm(num_groups, out_channels),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, bias = False),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.LeakyReLU(inplace = True),
                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                    nn.GroupNorm(num_groups, out_channels),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = kernel_size, padding = 0, bias = False),
                    nn.GroupNorm(num_groups, out_channels),
                )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input)
        y = self.shortcut(input)
        return F.leaky_relu(x + y, inplace = True)

class ACE(nn.Module):
    def __init__(
        self,
        n_fft     : int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ) -> None:
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(inplace = True),
        )
        self.layer1 = BasicBlock2d( 64,  64)
        self.layer2 = BasicBlock2d( 64, 128, [ 1, 2 ])
        self.layer3 = BasicBlock2d(128, 128)
        self.layer4 = BasicBlock2d(128, 256, [ 1, 2 ])
        self.layer5 = BasicBlock2d(256, 256)
        self.layer6 = BasicBlock2d(256, 256, [ 1, 2 ])
        self.layer7 = BasicBlock2d(256, 256)
        self.layer8 = BasicBlock2d(256, 256, [ 3, 4 ])
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

    def forward(
        self,
        input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            input.squeeze(1),
            n_fft          = self.n_fft,
            hop_length     = self.hop_length,
            win_length     = self.win_length,
            window         = self.window,
            return_complex = True,
        )
        mag = torch.abs(spec)
        db  = 20.0 * torch.log10(mag + 1e-8)
        db  = torch.clamp(db, min = -60)
        db  = db / 60.0
        db  = db.transpose(1, 2).unsqueeze(1)
        x = self.layer0(db)
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
    ) -> None:
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length     = length
        self.register_buffer("window", torch.hann_window(win_length))
        self.layer8 = BasicBlock2d(256, 256, [ 3, 4 ], upsample = True)
        self.layer7 = BasicBlock2d(256, 256,           upsample = True)
        self.layer6 = BasicBlock2d(256, 256, [ 1, 2 ], upsample = True)
        self.layer5 = BasicBlock2d(256, 256,           upsample = True)
        self.layer4 = BasicBlock2d(256, 128, [ 1, 2 ], upsample = True)
        self.layer3 = BasicBlock2d(128, 128,           upsample = True)
        self.layer2 = BasicBlock2d(128,  64, [ 1, 2 ], upsample = True)
        self.layer1 = BasicBlock2d( 64,  64,           upsample = True)
        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size = 2, stride = 1, padding = 0, output_padding = 0, bias = False),
            nn.GroupNorm(1, 1),
            nn.Tanh(),
#           nn.LeakyReLU(inplace = True),
        )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
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
        x = self.layer0(x)
        db  = x * 60.0
        mag = torch.pow(10, db / 20.0) - 1e-8
        mag = torch.clamp(mag, min = 0.0)
        mag = mag.squeeze(1).transpose(1, 2)
        pha = torch.zeros_like(mag)
        wav = torch.istft(
            torch.polar(mag, pha),
            n_fft      = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window     = self.window,
            length     = self.length,
        )
        return torch.tanh(wav).unsqueeze(1)

class VCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = (2, 2), stride = (2, 2), padding = 0, bias = False),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(inplace = True),
        )
        self.layer1 = BasicBlock2d( 64,  64)
        self.layer2 = BasicBlock2d( 64, 128, (2, 2))
        self.layer3 = BasicBlock2d(128, 128)
        self.layer4 = BasicBlock2d(128, 256, (2, 2))
        self.layer5 = BasicBlock2d(256, 256)
        self.layer6 = BasicBlock2d(256, 512, (3, 4))
        self.layer7 = BasicBlock2d(512, 512)
        self.layer8 = BasicBlock2d(512, 512, (5, 5))
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

    def forward(
        self,
        input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layer0(input)
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
    def __init__(self) -> None:
        super().__init__()
        self.layer8 = BasicBlock2d(512, 512, (5, 5), upsample = True)
        self.layer7 = BasicBlock2d(512, 512,         upsample = True)
        self.layer6 = BasicBlock2d(512, 256, (3, 4), upsample = True)
        self.layer5 = BasicBlock2d(256, 256,         upsample = True)
        self.layer4 = BasicBlock2d(256, 128, (2, 2), upsample = True)
        self.layer3 = BasicBlock2d(128, 128,         upsample = True)
        self.layer2 = BasicBlock2d(128,  64, (2, 2), upsample = True)
        self.layer1 = BasicBlock2d( 64,  64,         upsample = True)
        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size = (2, 2), stride = (2, 2), padding = 0, output_padding = 0, bias = False),
            nn.GroupNorm(1, 3),
            nn.LeakyReLU(inplace = True),
        )

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
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
        x = self.layer0(x)
        return torch.tanh(x)

class MemoryLayer(nn.Module):
    def __init__(
        self,
        memory_dim: int = 1024,
        excite_dim: int = 1024,
        num_heads : int = 8,
    ) -> None:
        super().__init__()
        self.excite_mha = MHA(memory_dim, excite_dim, num_heads)
        self.memory_mha = MHA(memory_dim, memory_dim, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        memory: torch.Tensor,
        excite: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.excite_mha(memory, excite)
        memory = self.memory_mha(memory, memory)
        return memory

class Memory(nn.Module):
    def __init__(
        self,
        audio_dim : int = 256,
        video_dim : int = 512,
        memory_dim: int = 1024,
        excite_dim: int = 1024,
        num_heads : int = 8,
        num_layer : int = 3,
    ) -> None:
        super().__init__()
        # 验证：Memory or MemoryLayer
        # 分区：音频、视频、其他感知
        self.audio_embedding  = nn.Parameter(torch.randn(1, 1, 1))
        self.video_embedding  = nn.Parameter(torch.randn(1, 1, 1))
        self.memory_embedding = nn.Parameter(torch.randn(1, 1, 1))
#       self.audio_embedding  = nn.Parameter(torch.randn(1, 1, excite_dim))
#       self.video_embedding  = nn.Parameter(torch.randn(1, 1, excite_dim))
#       self.memory_embedding = nn.Parameter(torch.randn(1, 1, memory_dim))
        # 投影：音频、视频、其他感知
        self.audio_proj = nn.Linear(audio_dim, excite_dim)
        self.video_proj = nn.Linear(video_dim, excite_dim)
        # 记忆力层
        self.layers = nn.ModuleList([MemoryLayer(memory_dim, excite_dim, num_heads) for _ in range(num_layer)])

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
    ) -> None:
        super().__init__()
        self.memory_mha = MHA(recall_dim, memory_dim, num_heads)
        self.recall_mha = MHA(recall_dim, recall_dim, num_heads)

    def reset_parameters(self) -> None:
        initialize_weights(self)

    def forward(
        self,
        recall: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        recall = self.memory_mha(recall, memory)
        recall = self.recall_mha(recall, recall)
        return recall

class Recall(nn.Module):
    def __init__(
        self,
        audio_dim : int = 256,
        video_dim : int = 512,
        memory_dim: int = 1024,
        num_heads : int = 8,
        num_layer : int = 3,
    ) -> None:
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
    def __init__(self) -> None:
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
        elif isinstance(layer, nn.Identity):
            pass
        elif isinstance(layer, nn.ConvTranspose1d):
            layer.reset_parameters()
        elif isinstance(layer, nn.ConvTranspose2d):
            layer.reset_parameters()
        elif isinstance(layer, nn.MultiheadAttention):
            layer._reset_parameters()
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
        elif isinstance(layer, nn.ModuleList):
            initialize_weights(layer)
        elif isinstance(layer, nn.Sequential):
            initialize_weights(layer)
        elif isinstance(layer, nn.SiLU):
            pass
        elif isinstance(layer, nn.Tanh):
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
        elif isinstance(layer, BasicBlock2d):
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
