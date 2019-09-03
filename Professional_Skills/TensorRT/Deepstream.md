# 第九章：Deepstream

## 1、介绍

&emsp;&emsp;DeepStream是基于NVIDIA运行的工具，它主要应用于视觉整个流程的解决方案，它跟其他视觉库（比如OpenCV）不一样的地方在于，它建立一个完整的端到端的支持方案，换句话说你的源无论是Camera、Video还是云服务器上的视频，从视频的编解码到后台的图像Inference再到展示出来的画面的完整pipeline上的各个细节它都能帮助大家，包括参数的设置。

- Deep Stream 实时视频流分析即结构化包含：实时视频解码和NN神经网络进行推理。
- 解码：由多个线程并行执行并将各种输入流输送到GPU HW硬件解码器;
- 推理：由一个主线程通过调用TensorRT推理引擎来处理所有批量推理任务。其中，插件系统允许用户将更复杂的工作流添加到流水线中。

![](/Image/专业技能/TensorRT/deepstream流水线.jpg)

### 1.1、解码推理流程

- DeepStream输入：本地视频文件（H.264、HEVC等）和 在线视频流，如30 路1080P

![](/Image/专业技能/TensorRT/deepstream流程.jpg)

- deepstream中运用到很多的模块，对于TensorRT中，需要运用的是其中的推理模块，它提供了针对TensorRT推理部署的统一接口，使用复杂工厂模式编程实现。

![](/Image/专业技能/TensorRT/deepstream_lib.jpg)