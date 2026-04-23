import os
import torch
import random
import threading

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchvision.transforms import v2

global_lock = threading.Lock()

class VideoReader:
    def __init__(
        self,
        path  : str,
        sample: int,        # 视频片段总数=视频时长/片段时间
        length: int = 12,   # 批次片段数量
        millis: int = 100,  # 片段时间
        ar    : int = 8000, # 音频采样
        ac    : int = 1,    # 音频通道
        vh    : int = 480,  # 视频高度
        vw    : int = 640,  # 视频宽度
    ) -> None:
        self.init   = False
        self.path   = path
        self.sample = sample
        self.length = length
        self.millis = millis
        self.ar     = ar
        self.ac     = ac
        self.al     = int(ar * millis / 1000)
        self.vc     = 3     
        self.vh     = vh
        self.vw     = vw

    def load(self) -> None:
        with global_lock:
            if self.init:
                return
            self.init = True
            self.lock = threading.Lock()
            self.audio_decoer = AudioDecoder(self.path, sample_rate = self.ar, num_channels = self.ac)
            self.video_decoer = VideoDecoder(self.path, seek_mode = "approximate", transforms = [v2.Resize((self.vh, self.vw))])

    def size(self) -> int:
        return self.sample - self.length + 1

    def read(self, index: int) -> tuple[bool, torch.Tensor, torch.Tensor]:
        if not self.init:
            self.load()
        with self.lock:
            try:
                # 使用AV_SAMPLE_FMT_FLTP采样
                audio_tensor = self.audio_decoer.get_samples_played_in_range(1. * self.millis * index / 1000., 1. * self.millis * (index + self.length) / 1000.).data
                video_tensor = self.video_decoer.get_frames_played_in_range (1. * self.millis * index / 1000., 1. * self.millis * (index + self.length) / 1000.).data
                audio_tensor = audio_tensor.view(self.length, 1, -1)
                video_tensor = video_tensor[torch.linspace(0, video_tensor.shape[0] - 1, self.length).long()]
                # 读取数据长度验证
                if (
                    audio_tensor.shape[0] != self.length or
                    video_tensor.shape[0] != self.length or
                    audio_tensor.shape[1] != self.ac     or
                    video_tensor.shape[1] != self.vc     or
                    audio_tensor.shape[2] != self.al     or
                    video_tensor.shape[2] != self.vh     or
                    video_tensor.shape[3] != self.vw
                ):
                    return False, None, None
                return True, audio_tensor, video_tensor
            except:
                return False, None, None
    
class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder: str,
        length: int = 12,
        millis: int = 100,
    ) -> None:
        files = []
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if not os.path.isfile(file_path):
                continue
            if file_path.endswith(".mp4") or file_path.endswith(".mkv"):
                files.append(file_path)
        index  = 0
        reader = []
        for file_path in tqdm(files, desc = "加载视频"):
            try:
                suport, sample = self.decode(file_path, millis)
                if not suport:
                    continue
                video_reader = VideoReader(file_path, sample, length, millis)
                reader.append((index, video_reader))
                index += video_reader.size()
            except Exception as e:
                print(f"加载视频异常：{file_path} - {e}")
        self.index  = index
        self.reader = reader
        print(f"视频文件数：{len(self.reader)}")
        print(f"视频总帧数：{self.index}")

    def __len__(self):
        return self.index
    
    def __getitem__(self, index):
        while True:
            for (jndex, reader) in reversed(self.reader):
                if index >= jndex:
                    success, audio_tensor, video_tensor = reader.read(index - jndex)
                    if success:
                        return audio_tensor, video_tensor
                    else:
                        # 读取失败随机选择一个视频
                        index = random.randint(0, self.index - 1)
                        continue
    
    def decode(
        self,
        path  : str,
        millis: int = 100,
    ) -> tuple[bool, int]:
        audio_decoer = AudioDecoder(path)
        video_decoer = VideoDecoder(path)
        suport_audio_codec = [ "aac",  "mp3"  ]
        suport_video_codec = [ "h264", "h265" ]
        if audio_decoer.metadata.codec in suport_audio_codec and video_decoer.metadata.codec in suport_video_codec:
            return True, int(min(audio_decoer.metadata.duration_seconds, video_decoer.metadata.duration_seconds) * 1000 / millis)
        else:
            return False, 0

def loadDataset(
    folder    : str,
    batch_size: int = 32,
    length    : int = 16,
    millis    : int = 100,
) -> DataLoader:
    return DataLoader(
        VideoDataset(folder, length, millis),
        shuffle            = True,
        drop_last          = True,
        batch_size         = batch_size, 
        num_workers        = 8,
        prefetch_factor    = 16,
        persistent_workers = True,
    )
