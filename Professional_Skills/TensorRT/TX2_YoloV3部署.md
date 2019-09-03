# 第八章：TX2 Yolo V3部署

## 1、环境搭建与代码部署

### 1.1、环境说明

- CUDA10.0 ： /usr/local/cuda-10.0
- CUDNN7.5.0： CUDNN会自动将头文件和lib文件拷贝到CUDA的安装路径中
- Deepstream ４.０　：/opt/nvidia/deepstream
- Opencv3.3.1 : opencv会安装在linux系统默认安装路径中
- TensorRT5.1.6：linux系统默认安装路径

### 1.2、代码部署

- 编译yoloV3文件

  ```
  cd /opt/nvidia/deepstream/deepstream4.0/sources/objectDetector_Yolo
  export CUDA_VER=10.0 # 设置CUDA版本
  make -C nvdsinfer_custom_impl_Yolo #编译静态文件
  ```

  **说明：Cmake版本，需要在3.8以上，TensorRT 和Deepstream为cmake 3.8以上版本编译，所以构建时版本要一致**

- 设置deepstream和yoloV3的配置文件

  - deepstream_app_config_yoloV3.txt
  - config_infer_primary_yoloV3.txt

- 编译运行

  ```
  deepstream-app -c deepstream_app_config_yoloV3.txt
  ```

## 2、测试结果及分析

