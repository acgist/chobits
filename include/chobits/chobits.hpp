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

namespace chobits {

extern bool running;           // 是否运行
extern bool play_audio;        // 播放音频
extern int  batch_size;        // 训练批次
extern int  batch_wind;        // 每批窗口
extern int  video_skip;        // 视频跳帧
extern int  per_wind_second;   // 每秒窗口
extern int  audio_sample_rate; // 音频采样率
extern int  audio_nb_channels; // 音频通道
extern int  video_width;       // 视频宽度
extern int  video_height;      // 视频高度

extern void stop_all();

} // END OF chobits

namespace angelbits = chobits;

#endif // CHOBITS_HPP
