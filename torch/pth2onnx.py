import torch

from model import *

model = Chobits()
model.load_state_dict(torch.load("chobits.pth"))
model.cpu()
model.eval()

batch = torch.export.Dim("batch", min = 1)
torch.onnx.export(
    model,
    (
        torch.rand(1, 1, 800),
        torch.rand(1, 3, 480, 640),
        torch.rand(1, 10, 1024),
        torch.rand(1, 10, 1024),
    ),
    "chobits.onnx",
    dynamo         = True,
    opset_version  = 18,
    input_names    = [ "audio", "video", "audio_memory", "video_memory" ],
    output_names   = [ "output", "audio_memory", "video_memory" ],
    dynamic_shapes = (
        { 0: batch },
        { 0: batch },
        { 0: batch },
        { 0: batch },
    )
)