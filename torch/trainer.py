import os
import torch
import random
import torch.nn as nn

from tqdm import tqdm
from dataset import loadDataset

print("""

众鸟高飞尽，孤云独去闲。
相看两不厌，只有敬亭山。

""")

num_epochs  = 128 # 训练总的轮次
batch_size  =  32 # 训练批次数量
train_ratio = .80 # 训练集总占比

loader = loadDataset("/data/chobits/dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Chobits(label_size)
if os.path.exists("model.pth"):
    model = torch.load("model.pth", weights_only = False)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0003) # eps = 1e-8, weight_decay = 1e-2
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9999)

for epoch in range(num_epochs):
    step       = 0
    loss_sum   = 0.0
    loss_count = 0
    per_op_epoch = 10
    per_ck_epoch = 10000
    model.train()
    process = tqdm(loader, total = len(loader), desc = f"Epoch [{epoch + 1} / { num_epochs }]")
    for features, labels in process:
        step += 1
        features = features.to(device)
        labels   = labels  .to(device)
        pred = model(features)
        loss = criterion(pred, labels)
        # loss = loss / per_op_epoch
        loss.backward()
        loss_sum   += loss.item()
        loss_count += 1
        if step % per_op_epoch == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            process.set_postfix(loss = "{:.2f}".format(loss_sum / loss_count))
            loss_sum   = 0.0
            loss_count = 0
        if step % per_ck_epoch == 0:
            model.eval()
            torch.save(model, f"chobits.{epoch // per_ck_epoch % 10}.ckpt")
            # torch.save(model.state_dict(), f"chobits_last.ckpt")
            # torch.save.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss.item(),
            # }, save_path)
            model.train()

model.eval()
model = model.cpu()
torch.save(model, "model.pth")

print("""

贵逼人来不自由，龙骧凤翥势难收。
满堂花醉三千客，一剑霜寒十四州。
鼓角揭天嘉气冷，风涛动地海山秋。
东南永作金天柱，谁羡当时万户侯。

""")
