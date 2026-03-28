import torch

from model import *

model = torch.load("chobits.pth", weights_only = False)
model.cpu()
model.eval()

model = torch.jit.trace(model, (
    torch.rand(1, 1, 800),
    torch.rand(1, 3, 180, 320)
))
model.save("chobits.pt")
