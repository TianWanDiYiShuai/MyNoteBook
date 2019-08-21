# Fast R-CNN

论文中的架构说明图：

![](/Image/算法/深度学习/深度学习应用算法/Fast-RCNN.jpg)

三维架构图：

![](/Image/算法/深度学习/深度学习应用算法/Fast-RCNN系统架构.png)

## 1、整体思路

### 1.1、训练

- 输入224×224的固定大小图片
- 候选框提取：与RCNN相同使用Selective Search提取候选框，
- 将提取出来的候选框归一化到固定大小，然后作用于CNN，CNN网络为5个卷积层+2个降采样层（分别跟在第一和第二个卷积层后面）
- 提取出的特征张量，假设其保留了原图片的空间位置信息，将候选框做对应变换后映射到特征张量上，提取出大小不同的候选区域的特征张量（region proposal个数大约为2000个）。
- 对于每个候选区域的特征张量，使用RoI pooling层将其大小归一化
- 经过全连接层提取固定长度的特征向量（两个output都为4096维的全连接层）
- 分别使用全连接层（output=21）+softmax和全连接层（output=84）+回归判断类别（回归是smoothL1）并计算原候选框的调整因子。

#### 1.1.1、ROIPooling

&emsp;&emsp;ROIs Pooling是Pooling层的一种，而且是针对RoIs(Region of Interest,特征图上的框)的Pooling，他的特点是**输入特征图**尺寸不固定，但是**输出特征图**尺寸固定；

- 在Fast RCNN中， RoI是指Selective Search完成后得到的“候选框”在特征图上的映射

![](/Image/算法/深度学习/深度学习应用算法/ROl.jpg)

- 在Faster RCNN中，候选框是经过RPN产生的，然后再把各个“候选框”映射到特征图上，得到RoIs。

**输入：**

  - 特征图：经过CNN网络得到的feature map
  -  rois：在Fast RCNN中，指的是Selective Search的输出；在Faster RCNN中指的是RPN的输出，一堆矩形候选框框，形状为N*5的矩阵，N表示ROI的数目（“5”：第一列表示图像index，其余四列表示其余的左上角和右下角坐标），其中值得注意的是：坐标的参考系不是针对feature map这张图的，而是针对原图的（神经网络最开始的输入）

**计算过程：**

- 根据输入image，将ROI映射到feature map对应位置；
- 将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；
- 对每个sections进行max pooling操作

&emsp;&emsp;先把roi中的坐标映射到feature map上，映射规则比较简单，就是把各个坐标除以“输入图片与feature map的大小的比值”，得到了feature map上的box坐标后，我们使用Pooling得到输出；由于输入的图片大小不一，所以这里我们使用的类似Spp Pooling，在Pooling的过程中需要计算Pooling后的结果对应到feature map上所占的范围，然后在那个范围中进行取max或者取average。Rol-Pool.jpg

![](/Image/算法/深度学习/深度学习应用算法/Rol-Pool.jpg)

**一个8\*8大小的feature map，一个ROI，以及输出大小为2*2.**

