# 第一章：TensorRT加速原理

&emsp;&emsp;TensorRT的加速和优化主要体现在两个方面**一是TensorRT支持INT8和FP16的计算，通过在减少计算量和保持精度之间达到一个理想的平衡（trade-off），达到加速推断的目的。二是对于网络结构进行重构和优化**相对于训练过程，网络推断时模型结构及参数都已经固定，batchsize一般较小，对于精度的要求相较于训练过程也较低，这就给了很大的优化空间。

- **1、支持FP16 或者INT8的数据类型**

训练时由于梯度等对于计算精度要求较高，但是在inference阶段（只有前向网络阶段）可以利用精度较低的数据类型加速运算，降低模型的大小。**这里需要达到一个量化过程的平衡，当数据的精度降低时会对算法的精度有一定的影响，这里就需要在运行效率和精度之间进行一个平衡。但是总的来说在进行inference阶段时数据的精度对识别精度的影响相对于在训练阶段来说比较小。所以这里可以通过降低数据精度来进行优化。**

- **2、对网络结构的重构和优化**

    - **2.1、解析网络模型，消除无用的输出层**

        消除网络中的无用的输出层，减少计算
    - **2.2、对于网络结构的垂直整合**

        将目前主流的神经网络中的conv，BN，Relu三个层融合为一个层。如下图：
        ![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/EB8BF4C2378643CE8ADCC58F126CD7CB/16534)
    - **2.3、对于网络的水平组合**

        对于网络的水平组合，水平组合是指将输入为相同张量和执行相同操作的层融合一起
        ![image](https://note.youdao.com/yws/public/resource/a48e105e9dcf98f685bf69937a8ead17/xmlnote/88A957F9D2FB400BA2F9512FB123B5D0/16539)
    - **2.4、减少传输吞吐量**

        对于concat层，将contact层的输入直接送入下面的操作中，不用单独进行concat后在输入计算，相当于减少了一次传输吞吐。

- **3、此外，构建阶段还在虚拟数据上运行图层以从其内核目录中选择最快的图像，并在适当的情况下执行权重预格式化和内存优化。**

