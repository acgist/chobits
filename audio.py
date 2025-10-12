
import math
import torch
import torchaudio
import matplotlib.pyplot as plt

def save_audio(mag, pha, sample_rate, output_path, n_fft, hop_length):
    window = torch.hann_window(n_fft)
    data = torch.polar(mag, pha)
    data = torch.istft(data, n_fft, hop_length = hop_length,  window = window)
    torchaudio.save(output_path, data.unsqueeze(0), sample_rate)

def generate_spectrogram(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    print(waveform.shape)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim = 0)
    else:
        waveform = waveform[0]
    waveform = waveform[30 * sample_rate:40 * sample_rate]
    torchaudio.save("D:/tmp/chobits-1.wav", waveform.unsqueeze(0), sample_rate)
    n_fft      = 400
    hop_length = 80
    window = torch.hann_window(n_fft)
    spec   = torch.stft(waveform, n_fft, hop_length = hop_length,  window = window, return_complex = True)
    mag = torch.abs(spec)
    pha = torch.angle(spec)
    print(f"{mag.shape} {pha.shape}")
    print(f"{mag.max()} {mag.min()} {pha.max()} {pha.min()}")
    print(torch.histogram(mag, bins = torch.arange(-50.0, 250.0, 50)))
    save_audio(mag, pha, sample_rate, "D:/tmp/chobits-2.wav", n_fft, hop_length)
    mag = torch.log10(mag + 1e-4) / 4
    pha = pha / math.pi
    print(f"{mag.max()} {mag.min()} {pha.max()} {pha.min()}")
    print(torch.histogram(mag, bins = torch.arange(-1.5, 2.0, 0.5)))
    save_audio(torch.pow(10, mag * 4) - 1e-4, pha * math.pi, sample_rate, "D:/tmp/chobits-3.wav", n_fft, hop_length)
    plt.subplot(2, 1, 1)
    plt.imshow(mag, aspect = "auto", origin = "lower")
    plt.title("mag")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label = "mag")
    plt.subplot(2, 1, 2)
    plt.imshow(pha, aspect = "auto", origin = "lower")
    plt.title("pha")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label = "pha")
    plt.show()

if __name__ == "__main__":
    generate_spectrogram("D:/tmp/chobits.wav")
