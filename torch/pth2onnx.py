import torch

from model import *

model = torch.load("chobits.pth", weights_only = False)
model.cpu()
model.eval()

torch.onnx.export(
    model,
    (
        torch.rand(1, 1, 800),
        torch.rand(1, 3, 180, 320)
    ),
    "chobits.onnx",
    dynamo         = True,
    opset_version  = 18,
    input_names  = [ "audio" ],
    output_names = [ "video" ],
)
