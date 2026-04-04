import os
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import Chobits
from dataset import loadDataset
from torch.utils.tensorboard import SummaryWriter

class Triner:
    def train(
        self,
        dataset_path: str,
    ) -> None:
        self.running = True
        num_epochs = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = Chobits()
        if os.path.exists("chobits.ckpt"):
            print("加载模型权重：chobits.ckpt")
            model.load_state_dict(torch.load("chobits.ckpt"))
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0003)
        if os.path.exists("optimizer.ckpt"):
            print("加载优化器权重：optimizer.ckpt")
            optimizer.load_state_dict(torch.load("optimizer.ckpt"))
        else:
            model.reset_parameters()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9999)
        if os.path.exists("scheduler.ckpt"):
            print("加载调度器权重：scheduler.ckpt")
            scheduler.load_state_dict(torch.load("scheduler.ckpt"))
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        per_op_epoch = 10
        per_ck_epoch = 10000
        timeline = 20
        timeline_next = timeline + 1
        loss_count = 0
        audio_loss_sum               = 0.0
        video_loss_sum               = 0.0
        audio_memory_loss_sum        = 0.0
        video_memory_loss_sum        = 0.0
        audio_encode_decode_loss_sum = 0.0
        video_encode_decode_loss_sum = 0.0
        writer = SummaryWriter("runs/chobits_loss")
        loader = loadDataset(dataset_path, batch_size = 4, length = timeline)
        if not self.running:
            print("训练结束")
            return
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            process = tqdm(loader, total = len(loader), desc = f"Epoch [{epoch + 1} / { num_epochs }]")
            for step, (audio, video) in enumerate(process):
                if not self.running:
                    print("训练结束")
                    self.save()
                    return
                audio = audio.to(device)
                video = video.to(device).float().sub(128.0).div(128.0)
                # 训练音频编码解码
                audio_frame = audio[:, 0, :, :]
                audio_encode = model.ace(audio_frame)
                audio_decode = model.acd(audio_encode)
                # 损失函数
                audio_encode_decode_loss = F.mse_loss(audio_decode, audio_frame)
                audio_encode_decode_loss = audio_encode_decode_loss / per_op_epoch
                audio_encode_decode_loss.backward()
                audio_encode_decode_loss_sum += audio_encode_decode_loss.item() * per_op_epoch
                # 训练视频编码解码
                video_frame = video[:, 0, :, :, :]
                video_encode = model.vce(video_frame)
                video_decode = model.vcd(video_encode)
                # 损失函数
                video_encode_decode_loss = F.mse_loss(video_decode, video_frame)
                video_encode_decode_loss = video_encode_decode_loss / per_op_epoch
                video_encode_decode_loss.backward()
                video_encode_decode_loss_sum += video_encode_decode_loss.item() * per_op_epoch
                # 训练媒体记忆
                audio_memory_frame = audio[:, timeline, :, :]
                video_memory_frame = video[:, timeline, :, :, :]
                audio_memory_feature = audio[:, 0:timeline, :, :]
                video_memory_feature = video[:, 0:timeline, :, :, :]
                audio_memory_label = audio[:, 1:timeline_next, :, :]
                video_memory_label = video[:, 1:timeline_next, :, :, :]
                with torch.no_grad():
                    audio_memory_frame_encode = model.ace(audio_memory_frame)
                    video_memory_frame_encode = model.vce(video_memory_frame)
                    audio_memory_feature_encode = model.ace(audio_memory_feature.reshape(-1, audio_memory_feature.size(2), audio_memory_feature.size(3)))
                    audio_memory_feature_encode = audio_memory_feature_encode.view(audio.size(0), -1, audio_memory_feature_encode.size(-1))
                    video_memory_feature_encode = model.vce(video_memory_feature.reshape(-1, video_memory_feature.size(2), video_memory_feature.size(3), video_memory_feature.size(4)))
                    video_memory_feature_encode = video_memory_feature_encode.view(video.size(0), -1, video_memory_feature_encode.size(-1))
                    audio_memory_label_encode = model.ace(audio_memory_label.reshape(-1, audio_memory_label.size(2), audio_memory_label.size(3)))
                    audio_memory_label_encode = audio_memory_label_encode.view(audio.size(0), -1, audio_memory_label_encode.size(-1))
                    video_memory_label_encode = model.vce(video_memory_label.reshape(-1, video_memory_label.size(2), video_memory_label.size(3), video_memory_feature.size(4)))
                    video_memory_label_encode = video_memory_label_encode.view(video.size(0), -1, video_memory_label_encode.size(-1))
                audio_memory_pred_encode, video_memory_pred_encode = model.memory(
                    audio_memory_frame_encode,
                    video_memory_frame_encode,
                    audio_memory_feature_encode,
                    video_memory_feature_encode,
                )
                # 音频内存损失
                audio_memory_loss = F.mse_loss(audio_memory_pred_encode, audio_memory_label_encode)
                audio_memory_loss = audio_memory_loss / per_op_epoch
                audio_memory_loss.backward()
                audio_memory_loss_sum += audio_memory_loss.item() * per_op_epoch
                # 视频内存损失
                video_memory_loss = F.mse_loss(video_memory_pred_encode, video_memory_label_encode)
                video_memory_loss = video_memory_loss / per_op_epoch
                video_memory_loss.backward()
                video_memory_loss_sum += video_memory_loss.item() * per_op_epoch
                # 训练媒体混合
                audio_label = audio[:, timeline_next, :, :]
                video_label = video[:, timeline_next, :, :, :]
                with torch.no_grad():
                    audio_label_encode = model.ace(audio_label)
                    video_label_encode = model.vce(video_label)
                audio_pred_encode, video_pred_encode = model.muxer(
                    audio_memory_frame_encode,
                    video_memory_frame_encode,
                    audio_memory_label_encode,
                    video_memory_label_encode,
                    # audio_memory_pred_encode.detach(),
                    # video_memory_pred_encode.detach(),
                )
                # 音频损失
                # audio_loss = F.mse_loss(audio_pred_encode, audio_label_encode)
                # audio_loss = audio_loss / per_op_epoch
                # audio_loss.backward()
                # audio_loss_sum += audio_loss.item() * per_op_epoch
                # 视频损失
                # video_loss = F.mse_loss(video_pred_encode, video_label_encode)
                # video_loss = video_loss / per_op_epoch
                # video_loss.backward()
                # video_loss_sum += video_loss.item() * per_op_epoch
                # 联合损失
                audio_loss = F.mse_loss(audio_pred_encode, audio_label_encode)
                video_loss = F.mse_loss(video_pred_encode, video_label_encode)
                audio_loss = audio_loss / per_op_epoch
                video_loss = video_loss / per_op_epoch
                loss = audio_loss * 0.4 + video_loss * 0.6
                loss.backward()
                audio_loss_sum += audio_loss.item() * per_op_epoch
                video_loss_sum += video_loss.item() * per_op_epoch
                # 统计信息
                loss_count += 1
                if (step + 1) % per_op_epoch == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    process.set_postfix(
                        AL   = "{:.6f}".format(audio_loss_sum               / loss_count),
                        VL   = "{:.6f}".format(video_loss_sum               / loss_count),
                        AML  = "{:.6f}".format(audio_memory_loss_sum        / loss_count),
                        VML  = "{:.6f}".format(video_memory_loss_sum        / loss_count),
                        AEDL = "{:.6f}".format(audio_encode_decode_loss_sum / loss_count),
                        VEDL = "{:.6f}".format(video_encode_decode_loss_sum / loss_count),
                    )
                    writer.add_scalars("Loss", {
                        "AL"  : audio_loss_sum               / loss_count,
                        "VL"  : video_loss_sum               / loss_count,
                        "AML" : audio_memory_loss_sum        / loss_count,
                        "VML" : video_memory_loss_sum        / loss_count,
                        "AEDL": audio_encode_decode_loss_sum / loss_count,
                        "VEDL": video_encode_decode_loss_sum / loss_count,
                    }, step)
                    loss_count = 0
                    audio_loss_sum               = 0.0
                    video_loss_sum               = 0.0
                    audio_memory_loss_sum        = 0.0
                    video_memory_loss_sum        = 0.0
                    audio_encode_decode_loss_sum = 0.0
                    video_encode_decode_loss_sum = 0.0
                if (step + 1) % per_ck_epoch == 0:
                    self.save()
                    model.train()
        print("训练完成")
        self.save(True, True)

    def stop(self):
        print("训练停止")
        self.running = False

    def save(
        self,
        cpu: bool = False,
        pth: bool = False,
    ):
        print("保存模型")
        self.model.eval()
        if cpu:
            self.model.cpu()
        if pth:
            torch.save(self.model, f"chobits.pth")
        else:
            torch.save(self.model.state_dict(), f"chobits.ckpt_")
            torch.save(self.optimizer.state_dict(), f"optimizer.ckpt_")
            torch.save(self.scheduler.state_dict(), f"scheduler.ckpt_")
            os.rename(f"chobits.ckpt_", f"chobits.ckpt")
            os.rename(f"optimizer.ckpt_", f"optimizer.ckpt")
            os.rename(f"scheduler.ckpt_", f"scheduler.ckpt")

if __name__ == "__main__":
    print("""

    众鸟高飞尽，孤云独去闲。
    相看两不厌，只有敬亭山。

    """)
    trainer = Triner()
    def signal_handler(sig, frame):
        trainer.stop()
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        trainer.train("/data/chobits/video")
    except Exception as e:
        print(f"训练异常: {e}")
        trainer.stop()
