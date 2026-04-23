import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

wav, sr = librosa.load("D:/download/audio.wav", sr=8000, duration=10)
# wav = wav[1600:3200]
duration = librosa.get_duration(y=wav, sr=sr)
time = np.linspace(0, duration, len(wav))

n_fft = 512
hop_length = 128
win_length = 512

# 语谱图
stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
spec = np.abs(stft)
spec_db = librosa.amplitude_to_db(spec, ref=np.max, amin=1e-8, top_db=80)

# 梅尔频谱
mel_spec = librosa.feature.melspectrogram(
    y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-8, top_db=80)

# PyTorch 语谱图
torch_stft = torch.stft(
    torch.from_numpy(wav).float(),
    n_fft,
    hop_length,
    win_length,
    window=torch.hann_window(win_length),
    return_complex=True,
)
torch_spec = torch.abs(torch_stft)
torch_spec_db = 20 * torch.log10(torch.clamp(torch_spec, 1e-8))
# torch_spec_db -= torch_spec_db.max()
# # 高质量语音低噪声
# torch_spec_db = torch.clamp(torch_spec_db, min=-60)
# # 干净语音
# torch_spec_db = torch.clamp(torch_spec_db, min=-80)
# # 语音 音频 音乐
# torch_spec_db = torch.clamp(torch_spec_db, min=-100)
# 模拟librosa
torch_spec_db = torch.maximum(torch_spec_db, torch_spec_db.max() - 80)
torch_spec_db = torch_spec_db.numpy()

# 绘图
plt.figure(figsize=(15, 12))
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 时域波形图
plt.subplot(5, 1, 1)
plt.plot(time, wav, color="#1F77B4")
plt.title("时域波形图", fontsize=12)
plt.xlim(0, duration)
plt.ylabel("振幅")

# 线性语谱图
plt.subplot(5, 1, 2)
librosa.display.specshow(spec_db, sr=sr, x_axis="time", y_axis="linear", cmap="plasma")
plt.colorbar(format="%+2.0f dB")
plt.title("线性语谱图", fontsize=12)

# 对数语谱图（人耳感知）
plt.subplot(5, 1, 3)
librosa.display.specshow(spec_db, sr=sr, x_axis="time", y_axis="log", cmap="plasma")
plt.colorbar(format="%+2.0f dB")
plt.title("对数语谱图", fontsize=12)

# Pytorch 对数语谱图
plt.subplot(5, 1, 4)
plt.imshow(
    torch_spec_db,
    cmap="plasma",
    origin="lower",
    aspect="auto",
    extent=[0, duration, 0, sr // 2],
)
plt.colorbar(format="%+2.0f dB")
plt.title("PyTorch 对数语谱图", fontsize=12)
plt.yscale("log")
plt.ylim(20, sr // 2)
plt.xlabel("Time")
plt.ylabel("Hz")

# 梅尔频谱图
plt.subplot(5, 1, 5)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="plasma")
plt.colorbar(format="%+2.0f dB")
plt.title("梅尔频谱图", fontsize=12)

# 显示图谱
plt.tight_layout()
plt.show()

# import soundfile as sf

# 还原时域波形
# mag = 10 ** (torch_spec_db / 20)
# mag = torch.from_numpy(mag)
# wav = torch.istft(
#     torch.polar(mag, torch.zeros_like(mag)),
#     # torch.polar(mag, torch.angle(torch_stft)),
#     n_fft=n_fft,
#     hop_length=hop_length,
#     win_length=win_length,
#     window=torch.hann_window(win_length),
# )
# sf.write("D:/download/audio_.wav", wav.numpy(), sr)
