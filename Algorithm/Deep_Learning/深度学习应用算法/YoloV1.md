# YoloV1

&emsp;&emsp;Yolo算法采用一个单独的CNN模型实现end-to-end的目标检测，整个系统如下图所示：首先将输入图片resize到448x448，然后送入CNN网络，最后处理网络预测结果得到检测的目标。相比R-CNN算法，其是一个统一的框架，其速度更快，而且Yolo的训练过程也是end-to-end的

![](/Image/算法/深度学习/深度学习应用算法/yolo1检测系统.jpg)

YoloV1将对象检测重新设计为单一回归问题，回归预测图像像素、边界框坐标、类概率。

## 1、思路流程

- 我们的系统将输入图像分为S*S的网格。如果对象的中心落入网格单元格中，则该网格单元格负责检测该对象。

![](/Image/算法/深度学习/深度学习应用算法/yolo1网格.jpg)

- 每个网格单元预测B个边界框的信息和置信度分数，共包含5个数据：

  - x：物体中心x的位置 (相对于网格边缘的坐标位置)

  - y：物体中心y的位置 (相对于网格边缘的坐标位置)

  - w：物体水平长度（相对于原始图片的预测位置）

  - h：物品垂直长度（相对于原始图片的预测位置）

  - conf：物品置信度，即有多大的概率这个框包含了物体，定义
    $$
    conf =P(object) \times IOU_{pred}^{truth}
    $$
    ，即该指标同时考虑物品的存在可能性和对应Bounding boxes与真实物体重叠的面积。如果该单元中不存在对象，则置信度分数应为零。否则置信度等于预测框与真实框之间的联合（IOU）交点。

- 每个网格单元还预测C个类别的可能概率，Pr(classi|Object)。这些概率取决于包含对象的网格单元。我们只预测每个网格单元的一组类别概率，而不管预测框的数量是多少。

- 在算法测试阶段，使用网格的类别概率乘以单个预测框的置信度来对每个预测框的特定类别的置信度评分。
  $$
  Pr(Class_i|Obiect)*Pr(Object)*IOUtruthpred=Pr(Class_i)*IOUtruthpred(1)
  $$
  系统将检测模型化为回归问题。它将图像划分为S*S的网格，并且每个网格单元预测B个边界框，对这些框的置信度以及C类概率。这些预测被编码为
  $$
  S\times S\times（B*5 +C）
  $$
  张量。

  ![](/Image/算法/深度学习/深度学习应用算法/yolo1算法流程.jpg)

## 2、网络设计

![](/Image/算法/深度学习/深度学习应用算法/yolo1网络结构.jpg)

检测网络有24个卷积层，其次是2个完全连接的层。交替的1*1卷积层减少了前一层的特征空间。在分辨率的一半（224*224输入图像）上对ImageNet分类任务的卷积层进行预处理，然后将分辨率加倍以进行检测。

## 3、网络训练

&emsp;&emsp;在训练之前，先在ImageNet上进行了预训练，其预训练的分类模型采用图8中前20个卷积层，然后添加一个average-pool层和全连接层。预训练之后，在预训练得到的20层卷积层之上加上随机初始化的4个卷积层和2个全连接层。由于检测任务一般需要更高清的图片，所以将网络的输入从224x224增加到了448x448。整个网络的流程如下图所示：

![](/Image/算法/深度学习/深度学习应用算法/yolo1训练.jpg)

&emsp;&emsp;Yolo算法将目标检测看成回归问题，所以采用的是均方差损失函数。但是对不同的部分采用了不同的权重值。首先区分定位误差和分类误差。对于定位误差，即边界框坐标预测误差，采用较大的权重。 

&emsp;&emsp;由于每个单元格预测多个边界框。但是其对应类别只有一个。那么在训练时，如果该单元格内确实存在目标，那么只选择与ground truth的IOU最大的那个边界框来负责预测该目标，而其它边界框认为不存在目标。这样设置的一个结果将会使一个单元格对应的边界框更加专业化，其可以分别适用不同大小，不同高宽比的目标，从而提升模型性能。大家可能会想如果一个单元格内存在多个目标怎么办，其实这时候Yolo算法就只能选择其中一个来训练，这也是Yolo算法的缺点之一。要注意的一点时，对于不存在对应目标的边界框，其误差项就是只有置信度，左标项误差是没法计算的。而只有当一个单元格内确实存在目标时，才计算分类误差项，否则该项也是无法计算的。 
综上讨论，最终的损失函数计算如下：

![](/Image/算法/深度学习/深度学习应用算法/yolo误差函数.jpg)

## 4、Yolo V1总结

-  优点
  - 是单管道策略，其训练与预测都是end-to-end，所以Yolo算法比较简洁且速度快
  - 由于Yolo是对整张图片做卷积，所以其在检测目标有更大的视野，它不容易对背景误判
  - Yolo的泛化能力强，在做迁移时，模型鲁棒性高。
- 缺点：
  - Yolo各个单元格仅仅预测两个边界框，而且属于一个类别
  - 对于小物体，Yolo的表现会不如人意
  - Yolo对于在物体的宽高比方面泛化率低，就是无法定位不寻常比例的物体。
  - Yolo的定位不准确
  - 单元格内存在多个目标时，这时候Yolo算法就只能选择其中一个来训练

