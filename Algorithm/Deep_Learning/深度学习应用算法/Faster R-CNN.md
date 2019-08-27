# Faster R-CNN

## 1、整体框架

![](/Image/算法/深度学习/深度学习应用算法/Faster-RCNN整体框架.jpg)

* Conv layers提取特征图：

  作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取input image的feature maps,该feature maps会用于后续的RPN层和全连接层

* RPN\(Region Proposal Networks\):

  RPN网络主要用于生成region proposals，首先生成一堆Anchor box，对其进行裁剪过滤后通过softmax判断anchors属于前景\(foreground\)或者后景\(background\)，即是物体or不是物体，所以这是一个二分类；同时，另一分支bounding box regression修正anchor box，形成较精确的proposal（注：这里的较精确是相对于后面全连接层的再一次box regression而言）

* Roi Pooling：

  该层利用RPN生成的proposals和VGG16最后一层得到的feature map，得到固定大小的proposal feature map,进入到后面可利用全连接操作来进行目标识别和定位

* Classifier：

  会将Roi Pooling层形成固定大小的feature map进行全连接操作，利用Softmax进行具体类别的分类，同时，利用L1 Loss完成bounding box regression回归操作获得物体的精确位置.

**与Fast-RCNN和RCNN对比，Faster-RCNN的候选框不是采用Selective Search方法提取的，而是采用RPN网络回归得到候选框。从RCNN到fast RCNN，再到本文的faster RCNN，目标检测的四个基本步骤（候选区域生成，特征提取，分类，位置精修）faster RCNN把四个步骤统一到一个深度网络框架之内**

**aster RCNN可以简单地看做“区域生成网络+fast RCNN“的系统，用区域生成网络代替fast RCNN中的Selective Search方法**

## 2、网络结构![](/Image/算法/深度学习/深度学习应用算法/Faster-RCNN网络结构.jpg)



