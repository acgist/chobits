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

namespace at {

    class Tensor;

};

namespace chobits::media {

/**
 * 音视频文件输入
 * 
 * @param file 文件路径
 * 
 * @return 是否成功
 */
extern bool open_file(const std::string& file);

/**
 * 摄像头输入
 * 麦克风输入
 * 
 * @return 是否成功
 */
extern bool open_hardware();

/**
 * 扬声器输出
 * 显示器输出
 * 
 * @return 是否成功
 */
extern bool open_player();

/**
 * 播放音频
 * 
 * @param data 音频数据
 * @param len  数据长度
 * 
 * @return 是否成功
 */
extern bool play_audio(const void* data, int len);

/**
 * 播放视频
 * 
 * @param data 视频数据
 * @param len  数据长度
 * 
 * @return 是否成功
 */
extern bool play_video(const void* data, int len);

/**
 * 数据集
 * audio=100毫秒
 * video=20帧 / 10 = 2帧
 * 
 * @return 是否成功
 */
extern std::tuple<at::Tensor, at::Tensor> dataset();

/**
 * 关闭
 */
extern void stop_all();

} // END OF chobits::media

#endif // CHOBITS_MEDIA_HPP
