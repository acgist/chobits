# 人形电脑天使心

叽~

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

## 训练

- 输入：音频视频（耳朵眼睛）
- 输出：音频动作（嘴巴肌肉）

### 视频训练

观看视频进行学习

### 生活训练

现实互动持续学习

## 依赖

|名称|版本|官网|
|:--|:--|:--|
|SDL2|2.30.0|https://github.com/libsdl-org/SDL|
|ffmpeg|6.1.1|https://github.com/FFmpeg/FFmpeg|
|libtorch|2.8.0|https://github.com/pytorch/pytorch|

```
# linux

sudo  apt install ffmpeg -y
sudo  apt install libsdl2-dev -y
https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip

# windows

vcpkg install sdl2
vcpkg install ffmpeg
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.8.0%2Bcpu.zip
https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.8.0%2Bcu128.zip
https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.8.0%2Bcu128.zip
```

## 编译

```
# linux

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug|Release ..
make -j 8
# make install

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug|Release ..
cmake --build . -j 8
cmake --build . --parallel 8
# cmake --install .

# windows

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug|Release -T host=x64 -A x64 -G "Visual Studio 17 2022" ..
cmake --config Debug|Release --build . -j 8
cmake --config Debug|Release --build . --parallel 8
# cmake --install .
```
