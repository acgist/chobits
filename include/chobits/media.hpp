/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/chobits
 * github: https://github.com/acgist/chobits
 * 
 * 媒体
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef CHOBITS_MEDIA_HPP
#define CHOBITS_MEDIA_HPP

#include <tuple>
#include <vector>
#include <string>

namespace at {

    class Tensor;

}; // END OF at

namespace chobits::media {

extern bool open_media(int argc, char const *argv[]);
extern bool open_file(const std::string& file);
extern bool open_hardware();
extern void stop_all();

/**
 * @return [ audio, video, label ]
 */
extern std::tuple<at::Tensor, at::Tensor, at::Tensor> get_data();

/**
 * 短时傅里叶变换
 * 
 * 201 = win_size / 2 + 1
 * 480 = 7 | 4800 = 61 | 48000 = 601
 * [1, 201, 61, 2[实部, 虚部]]
 * 
 * @param pcm_data PCM数据
 * @param pcm_size PCM长度
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return 张量
 */
extern at::Tensor pcm_stft(
    short* pcm_data,
    int pcm_size,
    int n_fft    = 400,
    int hop_size = 80,
    int win_size = 400
);

/**
 * 短时傅里叶逆变换
 * 
 * @param tensor   张量
 * @param n_fft    傅里叶变换的大小
 * @param hop_size 相邻滑动窗口帧之间的距离
 * @param win_size 窗口帧和STFT滤波器的大小
 * 
 * @return PCM数据
 */
extern std::vector<short> pcm_istft(
    const at::Tensor& tensor,
    int n_fft    = 400,
    int hop_size = 80,
    int win_size = 400
);

} // END OF chobits::media

#endif // CHOBITS_MEDIA_HPP
