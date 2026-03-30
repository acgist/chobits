from model import *

model = ACE()
model.eval()
input = torch.randn(10, 1, 800)
print(model(input).shape)

# model = ACD()
# model.eval()
# input = torch.randn(10, 1024)
# print(model(input).shape)

# model = VCE()
# model.eval()
# input = torch.randn(10, 3, 480, 640)
# print(model(input).shape)

# model = VCD()
# model.eval()
# input = torch.randn(10, 1024)
# print(model(input).shape)

# model = MHA(512, 512, 512, 512)
# model.eval()
# input = (
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
#     torch.randn(10, 256, 512),
# )
# print(model(*input).shape)

# model = Mixer()
# model.eval()
# input = (
#     torch.randn(10, 128, 256),
#     torch.randn(10, 256, 512),
# )
# audio, video = model(*input)
# print(audio.shape)
# print(video.shape)

# model = Chobits()
# model.eval()
# input = (
#     torch.randn(10, 1, 800),
#     torch.randn(10, 3, 480, 640),
#     torch.randn(10, 10, 1024),
#     torch.randn(10, 10, 1024),
# )
# audio_c, audio_m, video_m = model(*input)
# print(audio_c.shape)
# print(audio_m.shape)
# print(video_m.shape)

# torch.save(model, "D:/download/chobits.pth")
# torch.save(model.state_dict(), "D:/download/chobits.pth")
torch.jit.save(torch.jit.trace(model, input), "D:/download/chobits.pt")
