/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/chobits
 * github: https://github.com/acgist/chobits
 * 
 * 播放器
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef CHOBITS_PLAYER_HPP
#define CHOBITS_PLAYER_HPP

namespace chobits::player {

extern bool open_player();
extern void stop_player();

extern bool play_audio(const void* data, int len);
extern bool play_video(const void* data, int len);

} // END OF chobits::player

#endif // CHOBITS_PLAYER_HPP
