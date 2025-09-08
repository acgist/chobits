import os
import torch
import torch.nn as nn

from tqdm import tqdm
from chunk_dataset import load_embedding, loadTextDataset

class Chunk(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Chunk, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, input, hidden = None):
        if hidden is not None:
            h_0 = hidden
        else:
            h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        out, h_0 = self.gru(input, h_0)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out), h_0

if os.name == "nt":
    load_embedding("D:/tmp/hf/shibing624/text2vec-base-chinese")
    num_epochs    = 16
    train_dataset = loadTextDataset("chunk/dataset")
    val_dataset   = loadTextDataset("chunk/dataset")
else:
    load_embedding("/data/chunk/shibing624/text2vec-base-chinese")
    num_epochs    = 128
    train_dataset = loadTextDataset("chunk/train")
    val_dataset   = loadTextDataset("chunk/val")

# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ train_size, val_size ])

batch_size = 50

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = batch_size, shuffle = False, num_workers = 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Chunk(768, 20, 2, 1)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001) # eps = 1e-8, weight_decay = 1e-2
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9)

for epoch in range(num_epochs):
    loss_sum   = 0.0
    loss_count = 0
    accu_sum   = 0
    accu_count = 0
    # 训练集
    model.train()
    process = tqdm(train_loader, total = len(train_loader), desc = f"Epoch [{epoch + 1} / { num_epochs }]")
    for features, labels in process:
        features = features.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()
        pred, _ = model(features)
        loss    = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_sum   = loss_sum   + loss.item()
            loss_count = loss_count + 1
            accu_sum   = accu_sum   + ((pred > 0.5) == (labels == 1.0)).sum().item()
            accu_count = accu_count + labels.size(0)
            process.set_postfix(loss = "{:.2f}".format(loss_sum / loss_count), accu = "{:.2f}%".format(100 * accu_sum / accu_count))
    scheduler.step()
    # 验证集
    model.eval()
    if epoch % 20 == 0:
        torch.save(model, "model/model_cpk_{}.pth".format(epoch))
    with torch.no_grad():
        true_sum   = 0
        zero_sum   = 0
        accu_sum   = 0
        true_count = 0
        zero_count = 0
        accu_count = 0
        for features, labels in val_loader:
            features  = features.to(device)
            labels    = labels.to(device)
            pred, _   = model(features)
            accu_sum   = accu_sum   + ((pred > 0.5) == (labels == 1.0)).sum().item()
            accu_count = accu_count + labels.size(0)
            true_sum   = true_sum   + ((pred > 0.5) & (labels == 1.0)).sum().item()
            true_count = true_count + (labels == 1.0).sum().item()
            zero_sum   = zero_sum   + ((pred <= 0.5) & (labels == 0.0)).sum().item()
            zero_count = zero_count + (labels == 0.0).sum().item()
        print("Accuracy: {} / {} = {:.2f}% | 1 : {} / {} | 0 : {} / {}".format(accu_sum, accu_count, 100 * accu_sum / accu_count, true_sum, true_count, zero_sum, zero_count))

model = model.cpu()
torch.save(model, "model/model.pth")
model = torch.jit.trace(model, (torch.rand(1, 4, 768)))
model.save("model/model.pt")
