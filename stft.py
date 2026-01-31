
import torch
import torchaudio
import matplotlib.pyplot as plt

class STFTAudio:
    def __init__(self, n_fft = 400, hop_length = 80, win_length = 400):
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window     = torch.hann_window(win_length)
    
    def load_audio(self, audio_path):
        """加载文件"""
        return torchaudio.load(audio_path)
    
    def compute_stft(self, waveform):
        """计算STFT"""
        return torch.stft(
            waveform,
            n_fft          = self.n_fft,
            hop_length     = self.hop_length,
            win_length     = self.win_length,
            window         = self.window,
            center         = True,
            return_complex = True,
        )
    
    def compute_istft(self, stft):
        """恢复音频"""
        return torch.istft(
            stft,
            n_fft      = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window     = self.window,
            center     = True,
        )
    
    def mag_pha_decomposition(self, stft):
        """分解幅度和相位"""
        mag = torch.abs(stft)
        pha = torch.angle(stft)
        return mag, pha
    
    def mag_pha_composition(self, mag, pha):
        """合成幅度和相位"""
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        return torch.complex(real, imag)
    
    def compute_mask(self, stft_mix):
        """掩码计算"""
        mag_mix, pha_mix = self.mag_pha_decomposition(stft_mix)
        print(f"mag：{mag_mix.shape} {mag_mix.max()} {mag_mix.min()}")
        print(f"pha：{pha_mix.shape} {pha_mix.max()} {pha_mix.min()}")
        pha_mix.fill_(1.0)
        # return torch.polar(mag_mix, pha_mix)
        return self.mag_pha_composition(mag_mix, pha_mix)
    
    def visualize(self, stft, title = "Spectrogram"):
        """可视化频谱图"""
        mag = torch.abs(stft).squeeze()
#       pha = torch.angle(stft).squeeze()
        plt.figure(figsize = (12, 6))
        plt.imshow(
            (20 * torch.log10(mag + 1e-8)).numpy()[0],
            aspect = "auto",
            origin = "lower",
        )
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label = "Magnitude (dB)")
        plt.tight_layout()
        plt.show()

def main():
    audio = STFTAudio(n_fft = 400, hop_length = 80, win_length = 400)
    waveform, sample_rate = audio.load_audio("D:/tmp/dzht.wav")
    waveform = waveform[:, :80000]
    # waveform = waveform * 32768.0
    print(f"音频数据：{waveform.shape} {sample_rate} {waveform.max()} {waveform.min()}")
    stft_mix = audio.compute_stft(waveform)
    audio.visualize(stft_mix)
    stft_mix = audio.compute_mask(stft_mix)
    audio.visualize(stft_mix)
    waveform = audio.compute_istft(stft_mix)
    print(f"音频数据：{waveform.shape} {sample_rate} {waveform.max()} {waveform.min()}")
    torchaudio.save("D:/tmp/dzht_.wav", waveform, sample_rate)
    print("处理完成")

if __name__ == "__main__":
    main()
    