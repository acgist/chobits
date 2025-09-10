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

#include <string>

namespace chobits::media {

// 音视频文件输入
extern bool open_file(const std::string& file);
// 麦克风输入
// 摄像头输入
extern bool open_hardware();

// 数据集：audio=100毫秒 video=2帧
extern bool dataset();

// 扬声器输出
// 显示器输出
extern bool play();

// 关闭
extern bool stop_all();

} // END OF chobits::media

#endif // CHOBITS_MEDIA_HPP
