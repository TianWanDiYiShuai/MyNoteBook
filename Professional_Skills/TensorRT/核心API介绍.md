# 第三章：核心API介绍

## 1、TensorRT核心库关键接口介绍

- **1.1、网络定义（Network Definition）**

“网络定义”接口为应用程序提供了指定网络定义的方法。可以指定输入和输出张量，可以添加图层，还有用于配置每个支持的图层类型的界面。除了层类型（例如卷积层和循环层）以及插件层类型之外，应用程序还可以实现TensorRT本身不支持的功能。
- **1.2、优化构建（Builder）**

Builder接口允许从网络定义创建优化引擎。它允许应用程序指定最大批量和工作空间大小，最小可接受精度水平，自动调整的计时迭代计数，以及用于量化网络以8位精度运行的接口。
- **1.3、执行推理（Engine）**

Engine接口允许应用程序执行推理。它支持同步和异步执行，分析，枚举和查询引擎输入和输出的绑定。单个引擎可以具有多个执行上下文，允许使用单组训练参数来同时执行多个批次。
- **1.4、解析器（ Parser）**

    - **1.4.1、Caffe Parser**

    此解析器可用于解析在BVLC Caffe或NVCaffe 0.16中创建的Caffe网络。它还提供了为自定义图层注册插件工厂的功能。
    - **1.4.2、UFF Parser**

    此解析器可用于以UFF格式解析网络。它还提供了注册插件工厂和传递自定义图层的字段属性的功能。
    - **1.4.3、ONNX Parser**

    此解析器可用于解析ONNX模型。

## 2、TensorRT核心代码学习

[TensorRT API文档](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_network_definition.html)

### **2.1、网络定义**
&emsp;&emsp;网络定义的API中主要是包括对Tensor的定义，对Layer的定义。

#### **2.1.1、关于对Tensor的定义**
- 对张量的维度结构进行定义与实现
  ![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/E3C9D5608A524125B9CD8EA7D7800DEE/17634)
  类Dims为tensor维度操作的基类，实现二维，三维，和四维的子类的实现。

- Tensor的定义与操作

Tensor的定义与操作实现在类ITensor类中。 [link](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_tensor.html)
![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/90CF3DD9C4F64A71A34668C1923E6F35/17670)

#### **2.1.2、关于对Layer的定义**
&emsp;&emsp;网络层的基类为ILayer类，所有成的定义为这个基类的子类。   [link](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_layer.html)

![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/0649E24DE0BE4DC386A68D0D814245B1/17683)

- 用户实现自定义层的插件类

![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/70216FC079774A87A27B6572EE1315D5/17767)


### **2.2、优化构建**
优化构建的API主要在类class IBuilder中

### **2.3、执行推理**

### **2.4、解析器**

## **3、C++ API使用流程**

```
graph TD
A[实例化TensorRT对象]-->B[创建网络定义]
B --> E[构建推理引擎]
E --> F[序列化模型]
F --> G[执行推理]
G --> H[内存管理]
```

&emsp;&emsp;程序系统结构图：
![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/E2E74907C7024FF0A96449A4939E1AD4/18018)

### **3.1、构建阶段**


- 代码解析（TensorRT中的sampleMNIST例程）：

```C++
// 构建推理引擎的函数
bool SampleMNIST::build()
{
    // 构建推理引擎，需要传入ILogger 类的子类的对象
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
        return false;
    // 创建模型的网络对象
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
        return false;
    // 创建Caffe的解析器解析网络结构
    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
        return false;
    // 将构建器、网络对象、和解析器解析的网络结构进行关联，和网络结构填充。
    constructNetwork(builder, network, parser);
    // 设置网络结构定义的参数。
    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(16_MB);
    builder->allowGPUFallback(true);
    builder->setFp16Mode(mParams.fp16);
    builder->setInt8Mode(mParams.int8);
    builder->setStrictTypeConstraints(true);
    
    samplesCommon::enableDLA(builder.get(), mParams.dlaCore);
     // 执行推理和创建序列化文件
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter());

    if (!mEngine)
        return false;

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}
```

### **3.2、部署阶段**
- 代码解析（TensorRT中的sampleMNIST例程）

```
// 部署和应用TensorRT推理引擎
bool SampleMNIST::infer()
{
    // 创建一个缓冲区管理对象。
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // 随机选择一个图片来进行推理
    srand(time(NULL));
    const int digit = rand() % 10;

    //  将输入数据读入管理的缓冲区
    //  这个输入的数据为一个batchSize大小的Tensor数据
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, mParams.inputTensorNames[0], digit))
        return false;

    // 创建CUDA流对象执行推理
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 异步地将数据从主机输入缓冲区复制到（GPU）设备输入缓冲区
    buffers.copyInputToDeviceAsync(stream);

    // 异步的对排队的推理任务进行推理
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
        return false;

    // 异步地将数据从设备输出缓冲区复制到主机输出缓冲区
    buffers.copyOutputToHostAsync(stream);

    // 等待CUDA工作流完成
    cudaStreamSynchronize(stream);

    // 释放CUDA工作流
    cudaStreamDestroy(stream);

    // 检查并打印推理的输出
    // 应该只有一个输出张量
    assert(mParams.outputTensorNames.size() == 1);
    bool outputCorrect = verifyOutput(buffers, mParams.outputTensorNames[0], digit);

    return outputCorrect;
}
```
### **3.3、清理libprotobuf文件**
在解析器解析完模型文件后需要将，模型文件进行释放。

```
bool SampleMNIST::teardown()
{
    //注意已调用ShutdownProtobufLibrary()。之后使用协议缓冲区库的任何其他部分都是不安的
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}
```

最后结果：
![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/5AFEDC3E864845B1AEF310943FDCE747/18178)