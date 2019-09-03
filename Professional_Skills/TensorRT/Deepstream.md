# 第九章：Deepstream

## 1、介绍

- Deep Stream 实时视频流分析即结构化包含：实时视频解码和NN神经网络进行推理。
- 解码：由多个线程并行执行并将各种输入流输送到GPU HW硬件解码器;
- 推理：由一个主线程通过调用TensorRT推理引擎来处理所有批量推理任务。其中，插件系统允许用户将更复杂的工作流添加到流水线中。

![](/Image/专业技能/TensorRT/deepstream流水线.jpg)

### 1.1、解码推理流程

- DeepStream输入：本地视频文件（H.264、HEVC等）和 在线视频流，如30 路1080P

![](/Image/专业技能/TensorRT/deepstream流程.jpg)