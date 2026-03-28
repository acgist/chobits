import torch

# from model import *
from dataset import VideoReader
from torchcodec.encoders import AudioEncoder

video_reader = VideoReader("D:/tmp/video.mp4", 1)

# JIT
# model = torch.jit.load("model.pt")
# if torch.cuda.is_available():
#     model.cuda()
# else:
#     model.cpu()
# model.eval()

# 原始模型
# model = torch.load("model.pth", weights_only = False)
# if torch.cuda.is_available():
#     model.cuda()
# else:
#     model.cpu()
# model.eval()

index  = 0
sample = []

while True:
    index += 1
    success, audio, video = video_reader.read(index)
    if not success:
        break
    # pred = model(audio, video)
    # sample.append(pred)
    sample.append(audio)

encoder = AudioEncoder(samples = torch.cat(sample, dim = 1), sample_rate = 8000)
encoder.to_file("./output.mp3", num_channels = 1, sample_rate = 8000)
