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
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>

namespace at {

    class Tensor;

}; // END OF at

namespace chobits::media {

extern bool open_media(int argc, char const *argv[]);
extern bool open_file(const std::string& file);
extern bool open_hardware();
extern void stop_all();

extern std::tuple<at::Tensor, at::Tensor, at::Tensor> get_data();

extern at::Tensor pcm_stft(short* pcm_data, int pcm_size, int n_fft = 400, int hop_size = 80, int win_size = 400);
extern std::vector<short> pcm_istft(const at::Tensor& tensor, int n_fft = 400, int hop_size = 80, int win_size = 400);

} // END OF chobits::media

#endif // CHOBITS_MEDIA_HPP
