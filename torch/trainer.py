import os
import signal
import traceback
import threading
import torch
import torch.nn as nn
import torch.functional as F

from tqdm import tqdm
from model import Chobits, KLLoss, AudioLoss, VideoLoss
from dataset import loadDataset
from torch.utils.tensorboard import SummaryWriter

class Triner:
    def __init__(
        self,
        train_audio        : bool = True,
        train_video        : bool = True,
        train_memory_recall: bool = True,
    ) -> None:
        self.lock = threading.RLock()
        self.train_audio = train_audio
        self.train_video = train_video
        self.train_memory_recall = train_memory_recall

    def train(
        self,
        dataset_path: str,
    ) -> None:
        self.running = True
        num_epochs = 128
        batch_size = 4
        dtype  = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = Chobits()
        kl_loss_f = KLLoss()
        audio_loss_f = AudioLoss()
        video_loss_f = VideoLoss()
        if os.path.exists("chobits.ckpt"):
            print("加载模型权重：chobits.ckpt")
            model.load_state_dict(torch.load("chobits.ckpt"))
        else:
            model.reset_parameters()
        model.to(device = device, dtype = dtype)
        kl_loss_f.to(device = device, dtype = dtype)
        audio_loss_f.to(device = device, dtype = dtype)
        video_loss_f.to(device = device, dtype = dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
        if os.path.exists("optimizer.ckpt"):
            print("加载优化器权重：optimizer.ckpt")
            optimizer.load_state_dict(torch.load("optimizer.ckpt"))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9999)
        if os.path.exists("scheduler.ckpt"):
            print("加载调度器权重：scheduler.ckpt")
            scheduler.load_state_dict(torch.load("scheduler.ckpt"))
        if not self.running:
            print("训练结束")
            return
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        timeline     = 22
        per_op_epoch = 10
        per_ck_epoch = 10000
        loss_count     = 0
        loss_sum       = 0.0
        audio_loss_sum = 0.0
        video_loss_sum = 0.0
        audio_encode_decode_loss_sum = 0.0
        video_encode_decode_loss_sum = 0.0
        writer = SummaryWriter("runs/chobits")
        loader = loadDataset(dataset_path, batch_size = batch_size, length = timeline)
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
                audio = audio.to(device = device, dtype = dtype)
                video = video.to(device = device, dtype = dtype).sub(128.0).div(128.0)
                if self.train_audio:
                    # 训练音频编码解码
                    audio_frame  = audio[:, 0, :, :]
                    audio_mu, audio_log_var = model.ace(audio_frame)
                    audio_frame_encode = model.ace.reparameterize(audio_mu, audio_log_var)
                    audio_frame_decode = model.acd(audio_frame_encode)
                    # 损失函数
                    audio_encode_decode_loss = audio_loss_f(audio_frame_decode, audio_frame) + kl_loss_f(audio_mu, audio_log_var)
                    audio_encode_decode_loss = audio_encode_decode_loss / per_op_epoch
                    audio_encode_decode_loss.backward()
                    audio_encode_decode_loss_sum += audio_encode_decode_loss.item() * per_op_epoch
                if self.train_video:
                    # 训练视频编码解码
                    video_frame  = video[:, 0, :, :, :]
                    video_mu, video_log_var = model.vce(video_frame)
                    video_frame_encode = model.vce.reparameterize(video_mu, video_log_var)
                    video_frame_decode = model.vcd(video_frame_encode)
                    # 损失函数
                    video_encode_decode_loss = video_loss_f(video_frame_decode, video_frame) + kl_loss_f(video_mu, video_log_var)
                    video_encode_decode_loss = video_encode_decode_loss / per_op_epoch
                    video_encode_decode_loss.backward()
                    video_encode_decode_loss_sum += video_encode_decode_loss.item() * per_op_epoch
                if self.train_memory_recall:
                    # 计算记忆回忆
                    with torch.no_grad():
                        chaos = torch.randn(batch_size, 10, 1024, device = device, dtype = dtype)
                        audio_encode, _ = model.ace(audio.reshape(-1, audio.size(2), audio.size(3)))
                        audio_encode = audio_encode.view(audio.size(0), audio.size(1), -1, audio_encode.size(-1))
                        video_encode, _ = model.vce(video.reshape(-1, video.size(2), video.size(3), video.size(4)))
                        video_encode = video_encode.view(video.size(0), video.size(1), -1, video_encode.size(-1))
                        audio_label_encode = audio_encode[:, -1, :, :]
                        video_label_encode = video_encode[:, -1, :, :]
                        audio_feature_encode = audio_encode[:, -2, :, :]
                        video_feature_encode = video_encode[:, -2, :, :]
                    # 训练记忆
                    memory_feature = chaos
                    for i in range(timeline - 2):
                        memory_feature = model.memory(
                            audio_encode[:, i, :, :],
                            video_encode[:, i, :, :],
                            memory_feature,
                        )
                    # 训练回忆
                    audio_pred_encode, video_pred_encode = model.recall(
                        audio_feature_encode,
                        video_feature_encode,
                        memory_feature,
                    )
                    # 损失函数
                    audio_loss = F.l1_loss(audio_pred_encode, audio_label_encode)
                    video_loss = F.l1_loss(video_pred_encode, video_label_encode)
                    audio_loss = audio_loss / per_op_epoch
                    video_loss = video_loss / per_op_epoch
                    loss = audio_loss * 0.4 + video_loss * 0.6
                    loss.backward()
                    loss_sum       += loss.item()       * per_op_epoch
                    audio_loss_sum += audio_loss.item() * per_op_epoch
                    video_loss_sum += video_loss.item() * per_op_epoch
                # 统计信息
                loss_count += 1
                if (step + 1) % per_op_epoch == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    process.set_postfix({
                        "L"   : "{:.6f}".format(loss_sum        / loss_count),
                        "AL"  : "{:.6f}".format(audio_loss_sum  / loss_count),
                        "VL"  : "{:.6f}".format(video_loss_sum  / loss_count),
                        "AEDL": "{:.6f}".format(audio_encode_decode_loss_sum / loss_count),
                        "VEDL": "{:.6f}".format(video_encode_decode_loss_sum / loss_count),
                    })
                    writer.add_scalars("Loss", {
                        "L"   : loss_sum        / loss_count,
                        "AL"  : audio_loss_sum  / loss_count,
                        "VL"  : video_loss_sum  / loss_count,
                        "AEDL": audio_encode_decode_loss_sum / loss_count,
                        "VEDL": video_encode_decode_loss_sum / loss_count,
                    }, step)
                    loss_count     = 0
                    loss_sum       = 0.0
                    audio_loss_sum = 0.0
                    video_loss_sum = 0.0
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
        with self.lock:
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
    trainer = Triner(True, True, False)
    def signal_handler(sig, frame):
        trainer.stop()
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        trainer.train("/data/chobits/video")
    except Exception as e:
        traceback.print_exc()
        print(f"训练异常: {e}")
        trainer.stop()
