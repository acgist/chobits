import os
import torch
import threading
import torchvision

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder, VideoDecoder

global_lock = threading.Lock()

class VideoReader:
    def __init__(
        self,
        path  : str,
        sample: int,        # 视频片段总数
        length: int = 40,   # 批次片段数量
        millis: int = 100,  # 片段时间
        ar    : int = 8000, # 音频采样
        ac    : int = 1,    # 音频通道
        vh    : int = 480,  # 视频高度
        vw    : int = 640,  # 视频宽度
    ):
        self.init   = False
        self.path   = path
        self.sample = sample
        self.length = length
        self.millis = millis
        self.ar     = ar
        self.ac     = ac
        self.vh     = vh
        self.vw     = vw

    def load(self) -> None:
        with global_lock:
            if self.init:
                return
            self.init = True
            self.lock = threading.Lock()
            self.audio_decoer = AudioDecoder(self.path, sample_rate = self.ar, num_channels = self.ac)
            self.video_decoer = VideoDecoder(self.path)
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.vh, self.vw))
            ])

    def size(self) -> int:
        return self.sample - self.length + 1

    def read(self, index: int) -> tuple[bool, torch.Tensor, torch.Tensor]:
        if not self.init:
            self.load()
        with self.lock:
            try:
                audio_tensor = self.audio_decoer.get_samples_played_in_range(1. * self.millis * index / 1000., 1. * self.millis * (index + self.length) / 1000.).data
                video_tensor = self.video_decoer.get_frames_played_in_range (1. * self.millis * index / 1000., 1. * self.millis * (index + self.length) / 1000.).data
                audio_tensor = audio_tensor.view(self.length, 1, -1)
                video_tensor = video_tensor[torch.linspace(0, video_tensor.shape[0] - 1, self.length).long()]
                video_tensor = self.transform(video_tensor)
                return True, audio_tensor, video_tensor
            except Exception as e:
                print(f"读取视频异常：{self.path} - {index} - {repr(e)}")
        return False, None, None
    
class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder: str,
        length: int = 40,
        millis: int = 100,
    ):
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
                print(f"加载视频异常：{file_path} - {repr(e)}")
        self.index  = index
        self.reader = reader
        print(f"视频总数：{len(self.reader)}")
        print(f"视频总帧数：{self.index}")

    def __len__(self):
        return self.index
    
    def __getitem__(self, index):
        for (k, v) in reversed(self.reader):
            if index >= self.index:
                raise IndexError("索引错误")
            if index >= k:
                return v.read(index - k)
    
    def decode(
        self,
        path  : str,
        millis: int = 100,
    ) -> tuple[bool, int]:
        audio_decoer = AudioDecoder(path)
        video_decoer = VideoDecoder(path)
        # 出错格式：opus
        suport_audio_codec = [ "aac", "mp3", "pcm",  "flac" ]
        suport_video_codec = [ "vp8", "vp9", "h264", "h265" ]
        if audio_decoer.metadata.codec in suport_audio_codec and video_decoer.metadata.codec in suport_video_codec:
            return True, int(min(audio_decoer.metadata.duration_seconds, video_decoer.metadata.duration_seconds) * 1000 / millis)
        else:
            return False, 0

def loadDataset(
    folder    : str,
    batch_size: int = 32,
    length    : int = 40,
    millis    : int = 100,
) -> DataLoader:
    return DataLoader(
        VideoDataset(folder, length, millis),
        shuffle            = True,
        drop_last          = True,
        batch_size         = batch_size, 
        num_workers        = 4,
        prefetch_factor    = 4,
        persistent_workers = True,
    )
