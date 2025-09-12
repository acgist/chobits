cmake_minimum_required(VERSION 3.22.0)

project(chobits VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(BUILD_TEST        "是否编译测试代码" ON)
option(BUILD_SHARED_LIBS "是否编译为共享库" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(CMAKE_HOST_WIN32)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd4804 /wd4819")
        
        add_compile_options("/utf-8")
        add_compile_options("/Zc:__cplusplus")
    else()
        message(FATAL_ERROR "需要安装MSVC")
    endif()
elseif(CMAKE_HOST_UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers -O0 -g")
    if(BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()

    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK    "${CMAKE_COMMAND} -E time")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
else()
    message(FATAL_ERROR "不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} deps/libtorch)

if(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        deps/sdl2
        deps/ffmpeg
    )
    
    find_package(FFMPEG CONFIG REQUIRED)
else()
    set(
        FFMPEG_LIBRARIES
        avutil
        avcodec
        swscale
        avdevice
        avformat
        swresample
    )
endif()

find_package(CUDA)

if(CUDA_FOUND)
    message(STATUS "使用GPU推理")
else()
    message(STATUS "使用CPU推理")
endif()

find_package(SDL2  CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)

aux_source_directory(src/chobits/ CHOBITS_FILES)

add_library(
    chobits-core
    ${CHOBITS_FILES}
)

target_include_directories(
    chobits-core PUBLIC
    include/
    ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(
    chobits-core
    ${TORCH_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

if(CMAKE_HOST_WIN32)
    add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        resources/win.rc
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRT")
        
        # target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:CONSOLE")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:MSVCRTD /entry:mainCRTStartup")

        # target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:WINDOWS")
    endif()
else()
    add_executable(
        ${PROJECT_NAME}
        src/main.cpp
    )
endif()

target_link_libraries(
    ${PROJECT_NAME}
    chobits-core
)

install(
    TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

if(CMAKE_HOST_WIN32)
    file(GLOB TORCH_DLLS "deps/libtorch/lib/*.dll")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        file(GLOB SDL2_DLLS   "deps/sdl2/debug/bin/*.dll")
        file(GLOB FFMPEG_DLLS "deps/ffmpeg/debug/bin/*.dll")
    else()
        file(GLOB SDL2_DLLS   "deps/sdl2/bin/*.dll")
        file(GLOB FFMPEG_DLLS "deps/ffmpeg/bin/*.dll")
    endif()
    
    install(FILES ${SDL2_DLLS}   DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${TORCH_DLLS}  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    install(FILES ${FFMPEG_DLLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
    
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SDL2_DLLS}   $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}  $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

if(BUILD_TEST)
    add_executable(
        test_media
        test/test_media.cpp
    )
    target_link_libraries(
        test_media
        chobits-core
    )
    add_test(NAME test_media COMMAND test_media)
    set_target_properties(test_media PROPERTIES FOLDER "test")
endif()
