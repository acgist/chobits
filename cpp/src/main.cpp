#include <cstdlib>
#include <cstdint>

int main(int argc, char const *argv[]) {
    #if _WIN32
    system("chcp 65001");
    #endif
    return 0;
}
