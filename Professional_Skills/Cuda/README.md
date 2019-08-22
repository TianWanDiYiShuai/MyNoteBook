# 第五模块：Cuda编程

## 1、CPU与GPU硬件架构介绍

CPU的结构主要包括`运算器`（ALU, Arithmetic and Logic Unit）、`控制单元`（CU, Control Unit）、`寄存器`（Register）、`高速缓存器`（Cache）和它们之间数据总线、控制总线和地址总线组成。

GPU结构与CPU类似，只是GPU的计算单元相对于CPU来说计算能力较弱但是数量更多。

![](/Image/专业技能/CUDA/CPU与GPU硬件结构对比.jpg)

为什么GPU特别擅长处理图像数据呢？这是因为图像上的每一个像素点都有被处理的需要，而且每个像素点处理的过程和方式都十分相似

## 2、显卡、显卡驱动、CUDA、CUDNN之间的关系

### 2.1、显卡

（GPU）主流是NVIDIA的GPU，深度学习本身需要大量计算。GPU的并行计算能力，在过去几年里恰当地满足了深度学习的需求。AMD的GPU基本没有什么支持，可以不用考虑。

### 2.2、显卡驱动

没有显卡驱动，就不能识别GPU硬件，不能调用其计算资源。但是呢，NVIDIA在Linux上的驱动安装特别麻烦。得屏蔽第三方显卡驱动。

### 2.3、**CUDA**

是NVIDIA推出的只能用于自家GPU的并行计算框架。只有安装这个框架才能够进行复杂的并行计算。主流的深度学习框架也都是基于CUDA进行GPU并行加速的。

![](/Image/专业技能/CUDA/cuda自动可扩展性.jpg)

* CUDA与OpenCL的区别

  CUDA与OpenCL的功能和架构相似，只是CUDA只针对NVIDIA的产品，而OpenCL是一种通用性框架，可以使用多种品牌的产品，所以CUDA的性能一般情况下要比OpenCL的性能要高10%~20%之间。

  ![](/Image/专业技能/CUDA/CUDA与OpenCL对比图.jpg)

### 2.4、CUDNN

NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。NVIDIA cuDNN可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow、加州大学伯克利分校的流行caffe软件。简单的**插入式设计**可以让开发人员专注于设计和实现神经网络模型，而不是简单调整性能，同时还可以在GPU上实现高性能现代并行计算。

## 2.5、cuda、cudnn、深度学习算法与深度学习框架之间的联系

* 深度学习计算流程图

![](/Image/专业技能/CUDA/CUDA编程结构.jpg)                                                                                                                              

