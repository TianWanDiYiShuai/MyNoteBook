# 第七章：TX2 Inception V3部署

## 1、环境搭建与代码部署

### 1.1、环境说明

在刷机的时候，需要将，TensorRT，Deepstream CUDA CUDNN.opencv刷入TX2中，刷入之后会自动的配置开发环境，相应的软件安装路径为以下路径：

* CUDA10.0 ： /usr/local/cuda-10.0
* CUDNN7.5.0： CUDNN会自动将头文件和lib文件拷贝到CUDA的安装路径中
* Deepstream ４.０　：/opt/nvidia/deepstream
* Opencv3.3.1 : opencv会安装在linux系统默认安装路径中
* TensorRT5.1.6：linux系统默认安装路径

### 1.2、代码部署

* 将代码拷贝到TX2自己的工程目录

* 构建及编写cmake文件

  ```
  cmake_minimum_required(VERSION 3.8)

  project(InceptionV3 LANGUAGES CXX CUDA)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

  file(GLOB inception3_src
    common.cpp
    common.h
    inceptionV3.cpp
    inception_v3.h
    aiDefine.h
  )

  # 需要导入CUDA的连接库
  find_package(CUDA REQUIRED)

  if (NOT CUDA_FOUND)
       message(STATUS "CUDA not found. Project will not be built.")
  endif(NOT CUDA_FOUND)

  # 需要导入opencv的连接库
  find_package(OpenCV REQUIRED)
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

  if (NOT OpenCV)
      message(STATUS "count find opencv")
  endif(NOT OpenCV)

  include_directories(${OpenCV_INCLUDE_DIRS})

  include_directories(${CUDA_INCLUDE_DIRS})

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  # Declare the executable target built from your sources

  add_library(inception3 STATIC ${inception3_src})
  target_compile_features(inception3 PUBLIC cxx_std_11)

  add_executable(inception3_basic main.cpp)
  target_link_libraries(inception3_basic PRIVATE inception3 nvinfer nvparsers nvinfer_plugin cudnn cublas cudart_static nvToolsExt cudart rt dl pthread ${OpenCV_LIBS})
  ```

  **说明：Cmake版本，需要在3.8以上，TensorRT 和Deepstream为cmake 3.8以上版本编译，所以构建时版本要一致**

* 编译运行

  ```
  mkdir build
  cd build
  make
  make install
  # 在./build/bin目录下运行 ./inception3_basic
  ```

## 2、测试结果及分析

| batch | host\_to\_device\_time | inference\_time | divce\_to\_host\_time | count |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.524478 ms | 40.1982 ms | 0.108896 ms | 40.851876 ms |
| 2 | 0.995227 ms | 61.865 ms | 0.140671 ms | 63.000898 ms |
| 4 | 1.79077 ms | 118.407 ms | 0.248543 ms | 120.446313 ms |
| 6 | 2.67534 ms | 168.147 ms | 0.228703 ms | 171.051043 ms |
| 8 | 3.5579 ms | 222.661 ms | 0.227198 ms | 226.446098 ms |
| 10 | 4.32909 ms | 278.313 ms | 0.248606 ms | 282.890696‬ ms |
| 12 | 5.19157 ms | 332.284 ms | 0.250142 ms | 337.725712 ms |
| 14 | 6.02614 ms | 380.978 ms | 0.245341 ms | 387.249481 ms |
| 16 | 7.37462 ms | 436.376 ms | 0.246975 ms | 443.997595 ms |

### 2.1、分析统计结果

![](/Image/专业技能/TensorRT/inception统计折线图.png)

**在batch为2时有一个明显的上升趋势**







