# 人形电脑天使心

通过视频观看和现实交互不断学习，模拟像人类一样的成长过程。

----

<p align="center">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/acgist/chobits?style=flat-square&label=Github%20stars&color=crimson" />
    <img alt="Gitee  stars" src="https://img.shields.io/badge/dynamic/json?style=flat-square&label=Gitee%20stars&color=crimson&url=https://gitee.com/api/v5/repos/acgist/chobits&query=$.stargazers_count&cacheSeconds=3600" />
    <br />
    <img alt="GitHub Workflow"  src="https://img.shields.io/github/actions/workflow/status/acgist/chobits/build.yml?style=flat-square&branch=master" />
    <img alt="GitHub release"   src="https://img.shields.io/github/v/release/acgist/chobits?style=flat-square&color=orange" />
    <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/acgist/chobits?style=flat-square&color=blue" />
    <img alt="GitHub license"   src="https://img.shields.io/github/license/acgist/chobits?style=flat-square&color=blue" />
</p>

## 视频

- 眼睛

## 音频

- 耳朵
- 嘴巴

## 动作（肌肉控制）

学习人类肌肉动作（暂不实现）

## 训练

### 视频训练

通过观看视频进行学习

### 生活训练

通过摄像头、麦克风、显示器和扬声器像人一样生活，通过现实沟通交流持续学习。

## 依赖

|名称|版本|官网|
|:--|:--|:--|
|ffmpeg|6.1.1|https://github.com/FFmpeg/FFmpeg|
|libtorch|2.7.0|https://github.com/pytorch/pytorch|

* `vcpkg install ffmpeg ffmpeg[sdl2]`

## 编译

```
mkdir build
cd build
cmake ..
make -j
```
