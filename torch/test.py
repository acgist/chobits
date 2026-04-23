import cv2
import torch
import sounddevice as sd

from tqdm import tqdm
from model import *
from dataset import VideoReader, loadDataset

def test_kl_loss():
    kl_loss = KLLoss()
    mu      = torch.randn(10, 1, 800)
    log_var = torch.randn(10, 1, 800)
    print(kl_loss(mu, log_var))

def test_stft_loss():
    stft_loss = STFTLoss()
    pred = torch.randn(10, 1, 800)
    true = torch.randn(10, 1, 800)
    print(stft_loss(pred, true))

def test_ffn():
    model = FFN(1024)
    model.eval()
    model.reset_parameters()
    input = torch.randn(10, 1024)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt") 

def test_mha():
    model = MHA(256, 512)
    model.eval()
    model.reset_parameters()
    input = (
        torch.randn(10, 10, 256),
        torch.randn(10, 10, 512),
    )
    print(model(*input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_basic_block2d():
    model = BasicBlock2d(64,  64)
    # model = BasicBlock2d(64, 128)
    # model = BasicBlock2d(64,  64, (2, 2))
    # model = BasicBlock2d(64, 128, (2, 2))
    # model = BasicBlock2d(64,  64, upsample = True)
    # model = BasicBlock2d(64, 128, upsample = True)
    # model = BasicBlock2d(64,  64, (2, 2), upsample = True)
    # model = BasicBlock2d(64, 128, (2, 2), upsample = True)
    model.eval()
    model.reset_parameters()
    input = torch.randn(10, 64, 128, 128)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_ace():
    model = ACE()
    model.eval()
    model.reset_parameters()
    input = torch.randn(10, 1, 800)
    # input = torch.range(0, 800, 1).float() / 800.0
    # input = input.unsqueeze(0).unsqueeze(0)
    mu, log_var = model(input)
    print(mu.shape)
    print(log_var.shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_acd():
    model = ACD()
    model.eval()
    model.reset_parameters()
    input = torch.randn(10, 16, 256)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_vce():
    model = VCE()
    model.eval()
    model.reset_parameters()
    input = torch.randn(10, 3, 480, 640)
    mu, log_var = model(input)
    print(mu.shape)
    print(log_var.shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_vcd():
    model = VCD()
    model.eval()
    model.reset_parameters()
    input = torch.randn(10, 16, 512)
    print(model(input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_memory():
    model = Memory()
    model.eval()
    model.reset_parameters()
    input = (
        torch.randn(10, 16,  256),
        torch.randn(10, 16,  512),
        torch.randn(10, 10, 1024),
    )
    print(model(*input).shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_recall():
    model = Recall()
    model.eval()
    model.reset_parameters()
    input = (
        torch.randn(10, 16,  256),
        torch.randn(10, 16,  512),
        torch.randn(10, 10, 1024),
    )
    audio, video = model(*input)
    print(audio.shape)
    print(video.shape)
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_chobits():
    model = Chobits()
    model.eval()
    model.reset_parameters()
    input = (
        torch.randn(10,  1, 800),
        torch.randn(10,  3, 480, 640),
        torch.randn(10, 10, 1024),
    )
    audio, video, memory = model(*input)
    print(audio.shape)
    print(video.shape)
    print(memory.shape)
    # torch.save(model.state_dict(), "chobits.ckpt")
    torch.jit.save(torch.jit.trace(model, input), "chobits.pt")

def test_trainer():
    model = Chobits()
    model.eval()
    model.reset_parameters()
    audio = torch.randn(10, 10, 1, 800)
    video = torch.randn(10, 10, 3, 480, 640)
    audio_encode, _ = model.ace(audio.reshape(-1, audio.size(2), audio.size(3)))
    audio_encode = audio_encode.view(audio.size(0), audio.size(1), -1, audio_encode.size(-1))
    video_encode, _ = model.vce(video.reshape(-1, video.size(2), video.size(3), video.size(4)))
    video_encode = video_encode.view(video.size(0), video.size(1), -1, video_encode.size(-1))
    print(audio_encode.shape)
    print(video_encode.shape)

def test_reader():
    video_reader = VideoReader("D://tmp/video.mp4", 20, 1, 1000)
    cv2.namedWindow("chobits", cv2.WINDOW_AUTOSIZE)
    for index in tqdm(range(video_reader.sample)):
        success, audio_tensor, video_tensor = video_reader.read(index)
        if not success:
            break
        # print(f"audio: {audio_tensor.shape} {audio_tensor.dtype}")
        # print(f"video: {video_tensor.shape} {video_tensor.dtype}")
        audio_data = audio_tensor.view(-1).numpy()
        sd.play(audio_data, samplerate = 8000)
        sd.wait()
        video_data = video_tensor.permute(0, 2, 3, 1).numpy()
        for frame in video_data:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("chobits", frame)
            if cv2.waitKey(20) == 27:
                return
    cv2.destroyAllWindows()

def test_loader():
    loader = loadDataset("D://tmp", 1, 1, 1000)
    cv2.namedWindow("chobits", cv2.WINDOW_AUTOSIZE)
    for audio_tensor, video_tensor in loader:
        # print(f"audio: {audio_tensor.shape} {audio_tensor.dtype}")
        # print(f"video: {video_tensor.shape} {video_tensor.dtype}")
        audio_tensor = audio_tensor[0]
        video_tensor = video_tensor[0]
        audio_data = audio_tensor.view(-1).numpy()
        sd.play(audio_data, samplerate = 8000)
        sd.wait()
        video_data = video_tensor.permute(0, 2, 3, 1).numpy()
        for frame in video_data:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("chobits", frame)
            if cv2.waitKey(20) == 27:
                return
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with torch.no_grad():
        # test_kl_loss()
        # test_stft_loss()
        # test_ffn()
        # test_mha()
        # test_basic_block2d()
        # test_ace()
        # test_acd()
        # test_vce()
        # test_vcd()
        # test_memory()
        # test_recall()
        test_chobits()
        # test_trainer()
        # test_reader()
        # test_loader()
