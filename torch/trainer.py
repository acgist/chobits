import os
import sys
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import Chobits
from dataset import loadDataset

class Triner:
    def train_audio_encode_decode(self, audio, model) -> float:
        audio_f = audio[:, 0, :, :]
        audio_o = model.ace(audio_f)
        audio_x = model.acd(audio_o)
        loss = F.mse_loss(audio_x, audio_f)
        # loss = loss / per_op_epoch
        loss.backward()
        return loss.item()
        # return loss.item() * per_op_epoch
    
    def train_video_encode_decode(self, video, model) -> float:
        video_f = video[:, 0, :, :, :]
        video_o = model.vce(video_f)
        video_x = model.vcd(video_o)
        loss = F.mse_loss(video_x, video_f)
        # loss = loss / per_op_epoch
        loss.backward()
        return loss.item()
        # return loss.item() * per_op_epoch
    
    def train_audio_memory(self, audio, model) -> float:
        audio_f = audio[:, 11, :, :]
        audio_l = audio[:, 0:10, :, :]
        audio_p = audio[:, 1:11, :, :]
        with torch.no_grad():
            audio_f_ = model.ace(audio_f)
            audio_l_ = model.ace(audio_l.reshape(-1, audio_l.size(2), audio_l.size(3)))
            audio_p_ = model.ace(audio_p.reshape(-1, audio_p.size(2), audio_p.size(3)))
        audio_o = model.audio_memory(audio_f_, audio_l_.view(audio.size(0), -1, audio_l_.size(-1)))
        loss = F.mse_loss(audio_o, audio_p_.view(audio.size(0), -1, audio_p_.size(-1)))
        # loss = loss / per_op_epoch
        loss.backward()
        return loss.item()
        # return loss.item() * per_op_epoch
    
    def train_video_memory(self, video, model) -> float:
        video_f = video[:, 11, :, :, :]
        video_l = video[:, 0:10, :, :, :]
        video_p = video[:, 1:11, :, :, :]
        with torch.no_grad():
            video_f_ = model.vce(video_f)
            video_l_ = model.vce(video_l.reshape(-1, video_l.size(2), video_l.size(3), video_l.size(4)))
            video_p_ = model.vce(video_p.reshape(-1, video_p.size(2), video_p.size(3), video_l.size(4)))
        video_o = model.video_memory(video_f_, video_l_.view(video.size(0), -1, video_l_.size(-1)))
        loss = F.mse_loss(video_o, video_p_.view(video.size(0), -1, video_p_.size(-1)))
        # loss = loss / per_op_epoch
        loss.backward()
        return loss.item()
        # return loss.item() * per_op_epoch
    
    def train_model(self, audio, video, model) -> float:
        audio_f = audio[:, 11, :, :]
        audio_p = audio[:, 12, :, :]
        video_f = video[:, 11, :, :, :]
        audio_l = audio[:, 0:10, :, :]
        video_l = video[:, 0:10, :, :, :]
        with torch.no_grad():
            audio_l_ = model.ace(audio_l.reshape(-1, audio_l.size(2), audio_l.size(3)))
            video_l_ = model.vce(video_l.reshape(-1, video_l.size(2), video_l.size(3), video_l.size(4)))
        pred, audio_, video_ = model(
            audio_f,
            video_f,
            audio_l_.view(audio.size(0), -1, audio_l_.size(-1)),
            video_l_.view(video.size(0), -1, video_l_.size(-1))
        )
        loss = F.mse_loss(pred, audio_p)
        # loss = loss / per_op_epoch
        loss.backward()
        return loss.item()
        # return loss.item() * per_op_epoch

    def train(self, dataset_path):
        num_epochs = 128 # 训练总的轮次
        batch_size =   4 # 训练批次数量

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = Chobits()
        if os.path.exists("chobits.pth"):
            model.load_state_dict(torch.load("chobits.pth"))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0003)
        if os.path.exists("optimizer.pth"):
            optimizer.load_state_dict(torch.load("optimizer.pth"))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9999)

        self.model = model
        self.optimizer = optimizer

        for epoch in range(num_epochs):
            loader = loadDataset(dataset_path, batch_size)
            loss_count = 0
            loss_sum   = 0.0
            audio_memory_loss_sum = 0.0
            video_memory_loss_sum = 0.0
            audio_encode_decode_loss_sum = 0.0
            video_encode_decode_loss_sum = 0.0
            per_op_epoch = 10
            per_ck_epoch = 10000
            model.train()
            optimizer.zero_grad()
            process = tqdm(loader, total = len(loader), desc = f"Epoch [{epoch + 1} / { num_epochs }]")
            for step, (success, audio, video) in enumerate(process):
                if not success.all():
                    continue
                audio = audio.to(device)
                video = video.to(device)
                audio = audio.float() / 65536.0
                video = video.float() / 255.0
                audio_encode_decode_loss_sum += self.train_audio_encode_decode(audio, model)
                video_encode_decode_loss_sum += self.train_video_encode_decode(video, model)
                audio_memory_loss_sum += self.train_audio_memory(audio, model)
                video_memory_loss_sum += self.train_video_memory(video, model)
                loss_sum += self.train_model(audio, video, model)
                loss_count += 1
                if (step + 1) % per_op_epoch == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    process.set_postfix(
                        loss = "{:.2f}".format(loss_sum / loss_count),
                        aedl = "{:.2f}".format(audio_encode_decode_loss_sum / loss_count),
                        vedl = "{:.2f}".format(video_encode_decode_loss_sum / loss_count),
                        aml  = "{:.2f}".format(audio_memory_loss_sum / loss_count),
                        vml  = "{:.2f}".format(video_memory_loss_sum / loss_count),
                    )
                    audio_encode_decode_loss_sum = 0.0
                    video_encode_decode_loss_sum = 0.0
                    audio_memory_loss_sum = 0.0
                    video_memory_loss_sum = 0.0
                    loss_sum   = 0.0
                    loss_count = 0
                if (step + 1) % per_ck_epoch == 0:
                    model.eval()
                    torch.save(model.state_dict(), f"chobits.ckpt")
                    torch.save(optimizer.state_dict(), f"optimizer.ckpt")
                    model.train()
        model.eval()
        model = model.cpu()
        torch.save(model.state_dict(), f"chobits.ckpt")

    def save(self):
        self.model.eval()
        torch.save(self.model.state_dict(), f"chobits.ckpt")
        torch.save(self.optimizer.state_dict(), f"optimizer.ckpt")

if __name__ == '__main__':
    print("""

    众鸟高飞尽，孤云独去闲。
    相看两不厌，只有敬亭山。

    """)
    trainer = Triner()
    try:
        trainer.train("D:/tmp")
        # trainer.train("/data/chobits/dataset")
    except KeyboardInterrupt:
        print("训练中断")
        trainer.save()
