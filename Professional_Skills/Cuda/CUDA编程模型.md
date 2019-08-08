# 第二章：CUDA编程模型

## 2.1、CUDA编程结构

CUDA编程模型使用由C语言扩展生成的注释代码在异构计算系统中执行应用程序。

在一个异构环境中包含多个CPU和GPU，每个GPU和CPU的内存都由一条PCI-Express总线

分隔开。因此，需要注意区分以下内容。

* 主机：CPU及其内存（主机内存） 
* 设备：GPU及其内存（设备内存） 

为了清楚地指明不同的内存空间，主机内存中的变量名以h_或host_为 前缀，设备内存中的变量名以d_或device_为前缀。

一个典型的CUDA程序实现流程遵循以下模式。

1. 把数据从CPU内存拷贝到GPU内存。

2. 调用核函数对存储在GPU内存中的数据进行操作。

3. 将数据从GPU内存传送回到CPU内存。

![](/Image/专业技能/CUDA/CUDA应用程序.jpg)

## 2.2、内存管理

  在CUDA编程中，对于内存的管理分为对主机内存的管理，和对设备的内存的管理。

相关主机和设备的内存函数：

![](/Image/专业技能/CUDA/内存函数.jpg)

cudaMemcpy函数负责主机和设备之间的数据传输，其函数原型为：

```
cudaError_t cudaMemcpy (void* dst,const void* src,size_t count, cudaMemcpyKind kind)
```

此函数从src指向的源存储区复制一定数量的字节到dst指向的目标存储区。复制方向

由kind指定，其中的kind有以下几种。

* [ ] cudaMemcpyHostToHost

* [ ] cudaMemcpyHostTODevice

* [ ] cudaMemcpyDeviceToHost

* [ ] cudaMemcpyDeviceToDevice



