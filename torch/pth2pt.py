import os
import torch

from model import Chobits

model = Chobits()
if os.path.exists("chobits.ckpt"):
    model.load_state_dict(torch.load("chobits.ckpt"))
model.cpu()
model.eval()

model = torch.jit.trace(model, (
    torch.rand(1,  1, 800),
    torch.rand(1,  3, 480, 640),
    torch.rand(1, 10, 1024),
))
torch.jit.save(model, "chobits.pt")
