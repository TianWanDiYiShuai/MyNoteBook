# 第一模块：TensorRT

## TensorRT简介：

  **TensorRT是NVIDIA针对神经网络inference（推理）阶段提供的加速器。TensorRT™的核心是一个C++库，可以促进NVIDIA图形处理单元（GPU）的高性能计算。它专注于在GPU上快速有效地运行已经训练过的网络，以产生结果（一个过程，在各个地方被称为评分，检测，回归或推断）。**  
![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/F2B7C4FA473549ED90E4D9C0865DBADE/16461)

  TensorRT通过组合层和优化内核选择来优化网络，从而提高延迟，吞吐量，功效和内存消耗。如果应用程序指定，它还将优化网络以低精度运行，进一步提高性能并降低内存需求。

  一般在深度学习项目中。在深度学习算法的训练阶段为了加快速度，会使用多GPU的分布式。但是在项目部署到应用的过程中，为了降低成本，往往是使用单个GPU机器甚至是嵌入式平台进行部署。部署端也需要要有与训练端一样的深度学习环境，如caffe，tensorflow等。由于训练的网络模型可能会很大（比如，inception，resnet等），参数很多，而且部署端的机器性能存在差异，就会导致推理速度慢，延迟高。这对于那些高实时性的应用场合是致命的，比如自动驾驶要求实时目标检测，目标追踪等。所以为了提高部署推理的速度，出现了很多轻量级神经网络，比如squeezenet，mobilenet，shufflenet等。基本做法都是基于现有的经典模型提出一种新的模型结构，然后用这些改造过的模型重新训练，再重新部署。

  **tensorRT则是对训练好的模型进行优化。tensorRT就只是推理优化器。当你的网络训练完之后，可以将训练模型文件直接丢进tensorRT中，而不再需要依赖深度学习框架（Caffe，TensorFlow等\)**  
  当在没有使用TensorRT时将深度学习项目移植到单个GPU机器甚至是嵌入式平台。需要被移植的平台也具备相应的深度学习框架的环境。![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/BD9800AD4A8A4E9685464B6E42E12CDB/16434)

  但是当应用TensorRT后可以将不同深度学习框架的模型进行推理优化这样不在需要在单个GPU机器甚至是嵌入式平台上配置相应的深度学习环境。  
![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/A7AC56C351CD4D62A9B4029999B0C8F6/16448)

  **可以将TensorRT看成一个前向传播的深度学习框架，这个框架可以将Caffe，TensorFlow的网络模型解析，然后与tensorRT中对应的层进行一一映射，把其他框架的模型统一全部 转换到tensorRT中，然后在tensorRT中可以针对NVIDIA自家GPU实施优化策略，并进行部署加速。**

[TensorRT官网](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)





