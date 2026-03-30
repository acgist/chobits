from model import *

import cv2
import numpy as np
import sounddevice as sd

from dataset import VideoReader, loadDataset

def test_ffn():
    model = FFN(512)
    model.eval()
    input = torch.randn(10, 512)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt") 

def test_mha():
    model = MHA(512, 512, 512, 512)
    model.eval()
    input = (
        torch.randn(10, 256, 512),
        torch.randn(10, 256, 512),
        torch.randn(10, 256, 512),
    )
    print(model(*input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_ace():
    model = ACE()
    model.eval()
    input = torch.randn(10, 1, 800)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_acd():
    model = ACD()
    model.eval()
    input = torch.randn(10, 1, 1024)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_vce():
    model = VCE()
    model.eval()
    input = torch.randn(10, 3, 480, 640)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_vcd():
    model = VCD()
    model.eval()
    input = torch.randn(10, 1, 1024)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_memory():
    model = Memory()
    model.eval()
    input = (
        torch.randn(10,  1, 1024),
        torch.randn(10, 10, 1024),
    )
    print(model(*input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_mixer():
    model = Mixer()
    model.eval()
    input = (
        torch.randn(10, 128, 1024),
        torch.randn(10, 256, 1024),
    )
    audio, video = model(*input)
    print(audio.shape)
    print(video.shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_chobits():
    model = Chobits()
    model.eval()
    input = (
        torch.randn(10, 1, 800),
        torch.randn(10, 3, 480, 640),
        torch.randn(10, 10, 1024),
        torch.randn(10, 10, 1024),
    )
    audio_c, audio_m, video_m = model(*input)
    print(audio_c.shape)
    print(audio_m.shape)
    print(video_m.shape)
    # torch.save(model, "chobits.pth")
    # torch.save(model.state_dict(), "chobits.pth")
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_reader():
    index = 0
    video_reader = VideoReader("D://tmp/video.mp4", 20, 1, 1000)
    cv2.namedWindow("chobits", cv2.WINDOW_AUTOSIZE)
    while True:
        index += 1
        success, audio_tensor, video_tensor = video_reader.read(index)
        if not success:
            break
        print(f"音频Tensor形状: {audio_tensor.shape} {audio_tensor.dtype}")
        print(f"视频Tensor形状: {video_tensor.shape} {video_tensor.dtype}")
        audio_np = audio_tensor.view(-1).numpy()
        sd.play(audio_np, samplerate = 8000)
        sd.wait()
        video_np = video_tensor.permute(0, 2, 3, 1).numpy()
        for frame in video_np:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("chobits", frame)
            if cv2.waitKey(20) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

def test_loader():
    loader = loadDataset("D://tmp", 1, 1, 1000)
    cv2.namedWindow("chobits", cv2.WINDOW_AUTOSIZE)
    for success, audio_tensor, video_tensor in loader:
        if not success.all():
            break
        print(f"音频Tensor形状: {audio_tensor.shape} {audio_tensor.dtype}")
        print(f"视频Tensor形状: {video_tensor.shape} {video_tensor.dtype}")
        audio_tensor = audio_tensor[0]
        video_tensor = video_tensor[0]
        audio_np = audio_tensor.view(-1).numpy()
        sd.play(audio_np, samplerate = 8000)
        sd.wait()
        video_np = video_tensor.permute(0, 2, 3, 1).numpy()
        for frame in video_np:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("chobits", frame)
            if cv2.waitKey(20) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # test_reader()
    # test_loader()
    # test_ffn()
    # test_mha()
    # test_ace()
    # test_acd()
    # test_vce()
    # test_vcd()
    # test_memory()
    # test_mixer()
    test_chobits()
