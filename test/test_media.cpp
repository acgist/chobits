#include "chobits/media.hpp"

[[maybe_unused]] static void test_open_file() {
    chobits::media::open_file("D:/tmp/video.mp4");
}

[[maybe_unused]] static void test_open_hardware() {
    chobits::media::open_hardware();
}

int main() {
    // test_open_file();
    test_open_hardware();
    return 0;
}