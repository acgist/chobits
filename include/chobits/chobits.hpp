/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/chobits
 * github: https://github.com/acgist/chobits
 * 
 * 人形电脑天使心
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef CHOBITS_HPP
#define CHOBITS_HPP

#include <string>
#include <cstdint>
#include <cstdlib>

namespace chobits {

extern bool running;   // 是否运行
extern bool mode_drop; // 丢弃模式
extern bool mode_eval; // 验证模式
extern bool mode_file; // 文件模式
extern bool mode_play; // 播放模式

extern int         batch_size;    // 训练批次大小
extern int         batch_length;  // 训练批次长度
extern int         train_epoch;   // 训练批次轮数
extern std::string train_dataset; // 训练数据集

extern int per_wind_second;   // 音频每秒窗口
extern int audio_sample_rate; // 音频采样率
extern int audio_nb_channels; // 音频通道

extern int video_width;  // 视频宽度
extern int video_height; // 视频高度

extern void stop_all();

} // END OF chobits

namespace angelbits = chobits;

#endif // CHOBITS_HPP
