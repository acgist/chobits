import os
import torch

from tqdm import tqdm
from model import Chobits
from dataset import VideoReader
from torchcodec.encoders import AudioEncoder, VideoEncoder

video_reader = VideoReader("video.mp4", 200, 1)

model = Chobits()
if os.path.exists("chobits.ckpt"):
    model.load_state_dict(torch.load("chobits.ckpt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

audio_samples = []
video_frames  = []

with torch.no_grad():
    for index in tqdm(range(video_reader.sample)):
        success, audio, video = video_reader.read(index)
        if not success:
            continue
        audio = audio.to(device)
        video = video.to(device).float().sub(128.0).div(128.0)
        # 编码解码
        audio = model.acd(model.ace(audio))
        video = model.vcd(model.vce(video))
        audio_samples.append(audio)
        video_frames .append(video)
        # 模型推理
        # memory = torch.rand(1, 10, 1024).to(device)
        # audio, video, memory = model(audio, video, memory)
        # audio_samples.append(audio)
        # video_frames .append(video)

audio_encoder = AudioEncoder(samples = torch.cat(audio_samples, dim = 1).view(1, -1).cpu(), sample_rate = 8000)
audio_encoder.to_file("./output.wav", num_channels = 1, sample_rate = 8000)
print(f"保存音频成功 ./output.wav")

video_encoder = VideoEncoder(frames = torch.cat(video_frames, dim = 0).mul(128.0).add(128.0).to(torch.uint8).cpu(), frame_rate = 24)
video_encoder.to_file("./output.mp4", codec = "h264", pixel_format = "yuv420p")
print(f"保存视频成功 ./output.mp4")
