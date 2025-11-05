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

namespace at {

class Tensor;

}; // END OF at

namespace chobits::media {

extern bool open_media();
extern bool open_file(const std::string& file);
extern bool open_device();
extern void stop_all();

extern std::tuple<bool, at::Tensor, at::Tensor, at::Tensor> get_data(bool train = true);
extern std::vector<short> set_data(const at::Tensor& tensor);

} // END OF chobits::media

#endif // CHOBITS_MEDIA_HPP
