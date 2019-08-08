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

&emsp;&emsp;这个函数以同步方式执行，因为在cudaMemcpy函数返回以及传输操作完成之前主机 

应用程序是阻塞的。除了内核启动之外的CUDA调用都会返回一个错误的枚举类型cuda 

Error_t。如果GPU内存分配成功，函数返回： 

`cudaErrorMemoryAllocation`

否则返回：

`cudaErrorMemoryAlloction`

可以使用以下CUDA运行时函数将错误代码转化为可读的错误消息： 

`char* cudaGetErrorString(cudaError_t error)`

cudaGetErrorString函数和C语言中的strerror函数类似。 

---

CUDA编程模型从GPU架构中抽象出一个内存层次结构。图下所示的是一个简化的 

GPU内存结构，它主要包含两部分：全局内存和共享内存。

![](/Image/专业技能/CUDA/cuda抽象内存模型.jpg)

&emsp;&emsp;全局内存类似于CPU的系统内存，而共享内存类似于CPU的缓存。然而GPU的共享内存可以由CUDA 的内核直接控制。

**对比例子--矩阵相加：**

- CPU上计算

  ```
  #include <stdlib.h>
  #include <time.h>
  
  /*
   * This example demonstrates a simple vector sum on the host. sumArraysOnHost
   * sequentially iterates through vector elements on the host.
   */
  
  void sumArraysOnHost(float *A, float *B, float *C, const int N)
  {
      for (int idx = 0; idx < N; idx++)
      {
          C[idx] = A[idx] + B[idx];
      }
  
  }
  
  void initialData(float *ip, int size)
  {
      // generate different seed for random number
      time_t t;
      srand((unsigned) time(&t));
  
      for (int i = 0; i < size; i++)
      {
          ip[i] = (float)(rand() & 0xFF) / 10.0f;
      }
  
      return;
  }
  
  int main(int argc, char **argv)
  {
      int nElem = 1024;
      size_t nBytes = nElem * sizeof(float);
  
      float *h_A, *h_B, *h_C;
      h_A = (float *)malloc(nBytes);
      h_B = (float *)malloc(nBytes);
      h_C = (float *)malloc(nBytes);
  
      initialData(h_A, nElem);
      initialData(h_B, nElem);
  
      sumArraysOnHost(h_A, h_B, h_C, nElem);
  
      free(h_A);
      free(h_B);
      free(h_C);
  
      return(0);
  }
  
  ```

- GPU上计算

  ```
  #include "../common/common.h"
  #include <cuda_runtime.h>
  #include <stdio.h>
  
  /*
  这个例子在GPU和主机上演示了一个简单的向量和。
  * sumArraysOnGPU将向量和的工作分割到CUDA线程上
  * GPU。为了简单起见，在这个小示例中只使用了一个线程块。
  * sumArraysOnHost顺序遍历主机上的向量元素。
  这个版本的sumarray增加了主机定时器来测量GPU和CPU
  *性能
   */
  // 检查结果是否完成
  void checkResult(float *hostRef, float *gpuRef, const int N)
  {
      double epsilon = 1.0E-8;
      bool match = 1;
  
      for (int i = 0; i < N; i++)
      {
          if (abs(hostRef[i] - gpuRef[i]) > epsilon)
          {
              match = 0;
              printf("Arrays do not match!\n");
              printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                     gpuRef[i], i);
              break;
          }
      }
  
      if (match) printf("Arrays match.\n\n");
  
      return;
  }
  
  //初始化向量
  void initialData(float *ip, int size)
  {
      // generate different seed for random number
      time_t t;
      srand((unsigned) time(&t));
  
      for (int i = 0; i < size; i++)
      {
          ip[i] = (float)( rand() & 0xFF ) / 10.0f;
      }
  
      return;
  }
  // 在主机设备进行向量计算
  void sumArraysOnHost(float *A, float *B, float *C, const int N)
  {
      for (int idx = 0; idx < N; idx++)
      {
          C[idx] = A[idx] + B[idx];
      }
  }
  // 核函数，在GPU上进行向量计算
  __global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
  {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
  
      if (i < N) C[i] = A[i] + B[i];
  }
  
  int main(int argc, char **argv)
  {
      printf("%s Starting...\n", argv[0]);
  
      // 设置GPU的使用编号
      int dev = 0;
      // GPU设备的信息对象
      cudaDeviceProp deviceProp;
      // 获取指定编号的GPU信息
      CHECK(cudaGetDeviceProperties(&deviceProp, dev));
      printf("Using Device %d: %s\n", dev, deviceProp.name);
      //设置使用的GPU
      CHECK(cudaSetDevice(dev));
  
      // 设置vectors的大小
      int nElem = 1 << 24;
      printf("Vector size %d\n", nElem);
  
      // 主机内存分配大小
      size_t nBytes = nElem * sizeof(float);
  
      float *h_A, *h_B, *hostRef, *gpuRef;
      h_A     = (float *)malloc(nBytes);
      h_B     = (float *)malloc(nBytes);
      hostRef = (float *)malloc(nBytes);
      gpuRef  = (float *)malloc(nBytes);
  
      double iStart, iElaps;
  
      // 初始化主机端内存数据
      iStart = seconds();
      initialData(h_A, nElem);
      initialData(h_B, nElem);
      iElaps = seconds() - iStart;
      printf("initialData Time elapsed %f sec\n", iElaps);
      memset(hostRef, 0, nBytes);
      memset(gpuRef,  0, nBytes);
  
      // 在主机端进行向量的计算
      iStart = seconds();
      sumArraysOnHost(h_A, h_B, hostRef, nElem);
      iElaps = seconds() - iStart;
      printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
  
      // 分配GPU端内存
      float *d_A, *d_B, *d_C;
      CHECK(cudaMalloc((float**)&d_A, nBytes));
      CHECK(cudaMalloc((float**)&d_B, nBytes));
      CHECK(cudaMalloc((float**)&d_C, nBytes));
  
      // 将主机端的内存数据拷贝到GPU端的全局内存区
      CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
  
      // 在主机端调用核函数
      /*
      当数据被转移到GPU的全局内存后，主机端调用核函数在GPU上进行数组求和。一旦 内核被调用，控制权立刻被传回主机，这样的话，当核函数在GPU上运行时，主机可以执 行其他函数。因此，内核与主机是异步的。
      */
      int iLen = 512;
      dim3 block (iLen);
      dim3 grid  ((nElem + block.x - 1) / block.x);
  
      iStart = seconds();
      sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
      CHECK(cudaDeviceSynchronize());
      //主机得时间计算
      iElaps = seconds() - iStart;
      printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
             block.x, iElaps);
  
      // 检查核函数是否有错误
      CHECK(cudaGetLastError()) ;
  
      // 将GPU端的计算结构拷贝到主机端
      CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
  
      // 检查计算的结果
      checkResult(hostRef, gpuRef, nElem);
  
      // 释放设备端的内存
      CHECK(cudaFree(d_A));
      CHECK(cudaFree(d_B));
      CHECK(cudaFree(d_C));
  
      // 释放主机端的内存
      free(h_A);
      free(h_B);
      free(hostRef);
      free(gpuRef);
  
      return(0);
  }
  
  ```

  &emsp;&emsp;当数据被转移到GPU的全局内存后，主机端调用核函数在GPU上进行数组求和。一旦 内核被调用，控制权立刻被传回主机，这样的话，当核函数在GPU上运行时，主机可以执 行其他函数。因此，内核与主机是异步的。 

  &emsp;&emsp;当内核在GPU上完成了对所有数组元素的处理后，其结果将以数组d_C的形式存储在 

  GPU的全局内存中，然后用cudaMemcpy函数把结果从GPU复制回到主机的数组gpuRef中。

  `cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost)`

  &emsp;&emsp;cudaMemcpy的调用会导致主机运行阻塞。cudaMemcpyDeviceToHost的作用就是将存储在GPU上的数组d_c中的结果复制到gpuRef中。

  **注意：主机与设备之间的内存拷贝，一定要用cudaMemcpy函数。如果运用gpuRef = d_C则程序在运行时将会直接奔溃。**
