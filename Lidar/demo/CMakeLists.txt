cmake_minimum_required(VERSION 3.4.1)
set(TARGET_NAME SDKTest)

# +++++++ setting +++++++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

# ###############
# ### opencv ####
# ###############
set(OpenCV440_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/opencv440/include")
set(OpenCV440_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdpart/opencv440/lib")
include_directories(${OpenCV440_INCLUDE_DIR})
link_directories(${OpenCV440_LIBS_DIR})

if(WIN32)
set(OpenCV440_LIST_LIBS
    opencv_core440
    opencv_imgproc440
    opencv_imgcodecs440
    opencv_highgui440
    opencv_videoio440
    )
    set (OpenCV440_LIBS "")
    foreach(x ${OpenCV440_LIST_LIBS})
        list (APPEND OpenCV440_LIBS debug ${x}d optimized ${x})
    endforeach()
elseif(UNIX)
    set(OpenCV440_LIBS
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_core
    opencv_videoio
    opencv_calib3d
    )
endif()

# ############
# ### SDK ####
# ############
set(SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(SDK_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
include_directories(${SDK_INCLUDE_DIR})
link_directories(${SDK_LIB_DIR})

if(WIN32)
    set(APP_PREFIX .exe)
    set(SDK_LIB SynexensSDK)
elseif(UNIX)
    set(APP_PREFIX)
    set(SDK_LIB csreconstruction2.0 SynexensSDK)
endif()

add_executable(SDKTest SDKTest.cpp)
if(WIN32)
target_link_libraries(SDKTest ${SDK_LIB} ${OpenCV440_LIBS})
elseif(UNIX)
target_link_libraries(SDKTest ${SDK_LIB} ${OpenCV440_LIBS}  udev dl)
endif()

add_executable(SDKTestNotOpencv SDKTestNotOpencv.cpp)
if(WIN32)
target_link_libraries(SDKTestNotOpencv ${SDK_LIB})
elseif(UNIX)
target_link_libraries(SDKTestNotOpencv ${SDK_LIB} udev dl)
endif()

add_executable(SDKTestYamlDemo SDKTestNotOpencv.cpp)
if(WIN32)
target_link_libraries(SDKTestYamlDemo ${SDK_LIB} ${OpenCV440_LIBS})
elseif(UNIX)
target_link_libraries(SDKTestYamlDemo ${SDK_LIB} ${OpenCV440_LIBS} udev dl)
endif()