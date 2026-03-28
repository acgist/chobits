
import cv2
import numpy as np
import sounddevice as sd

from dataset import VideoReader, loadDataset

# os.add_dll_directory("D:\\develop\\ffmpeg-n7.1.3-43-g5a1f107b4c-win64-gpl-shared-7.1\\bin")

def test_reader():
    index = 0
    video_reader = VideoReader("D:/tmp/video.mp4", 200, 1)
    cv2.namedWindow("chobits", cv2.WINDOW_AUTOSIZE)
    while True:
        index += 1
        success, audio_tensor, video_tensor = video_reader.read(index)
        if not success:
            break
        print(f"音频Tensor形状: {audio_tensor.shape} {audio_tensor.dtype}")
        print(f"视频Tensor形状: {video_tensor.shape} {video_tensor.dtype}")
        audio_np = audio_tensor.squeeze(0).numpy()
        sd.play(audio_np, samplerate = 8000)
        sd.wait()
        video_np = video_tensor.permute(0, 2, 3, 1).numpy()
        video_np = video_np.astype(np.uint8)
        for frame in video_np:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("chobits", frame)
            if cv2.waitKey(20) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

def test_loader():
    loader = loadDataset("D://tmp")
    cv2.namedWindow("chobits", cv2.WINDOW_AUTOSIZE)
    for batch in loader:
        print(batch)
        success, audio_tensor, video_tensor = batch
        success = success[0]
        audio_tensor = audio_tensor[0]
        video_tensor = video_tensor[0]
        if not success:
            break
        print(f"音频Tensor形状: {audio_tensor.shape} {audio_tensor.dtype}")
        print(f"视频Tensor形状: {video_tensor.shape} {video_tensor.dtype}")
        audio_np = audio_tensor.squeeze(0).numpy()
        sd.play(audio_np, samplerate = 8000)
        sd.wait()
        video_np = video_tensor.permute(0, 2, 3, 1).numpy()
        video_np = video_np.astype(np.uint8)
        for frame in video_np:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("chobits", frame)
            if cv2.waitKey(20) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_reader()
    test_loader()