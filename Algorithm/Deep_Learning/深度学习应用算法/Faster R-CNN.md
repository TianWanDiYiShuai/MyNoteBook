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

## 2、网络结构
![](/Image/算法/深度学习/深度学习应用算法/Faster-RCNN网络结构.jpg)

### 2.1、**Conv layers**

&emsp;&emsp;Faster RCNN首先是支持输入任意大小的图片的，比如上图中输入的P*Q，进入网络之前对图片进行了规整化尺度的设定，如可设定图像短边不超过600，图像长边不超过1000，我们可以假定M*N=1000*600（如果图片少于该尺寸，可以边缘补0，即图像会有黑色边缘）

-  13个conv层：kernel_size=3,pad=1,stride=1;conv层不会改变图片大小

- 13个relu层：激活函数，不改变图片大小

- 4个pooling层：kernel_size=2,stride=2;pooling层会让输出图片是输入图片的1/2

  ​       经过Conv layers，图片大小变成(M/16)*(N/16)

### 2.2、**RPN(Region Proposal Networks):**

**RPN网络把一个任意尺度的图片作为输入，输出一系列的矩形object proposals，每个object proposals都带一个objectness score，的网络**

#### 2.2.1、RPN的具体流程

![](/Image/算法/深度学习/深度学习应用算法/RPN网络.jpg)

- 使用一个小网络在最后卷积得到的特征图上进行滑动扫描，这个滑动网络每次与特征图上n*n（论文中n=3）的窗口全连接（图像的有效感受野很大，ZF是171像素，VGG是228像素）
- 然后映射到一个低维向量（256d for ZF / 512d for VGG），最后将这个低维向量送入到两个全连接层，即bbox回归层（reg）和box分类层（cls）
  - sliding window的处理方式保证reg-layer和cls-layer关联了conv5-3的全部特征空间。
  - reg层：预测proposal的anchor对应的proposal的（x,y,w,h）
  - cls层：判断该proposal是前景（object）还是背景（non-object）。
#### 2.2.2、RPN结构

&emsp;&emsp;**在上图中，3*3卷积核的中心点对应原图（re-scale，源代码设置re-scale为600*1000）上的位置（点），将该点作为anchor的中心点，在原图中框出多尺度、多种长宽比的anchors。所以，anchor不在conv特征图上，而在原图上。**

- 对于一个大小为H*W的特征层，它上面每一个像素点对应9个anchor,这里有一个重要的参数feat_stride = 16， 它表示特征层上移动一个点，对应原图移动16个像素点(16：因为经过了四次pooling，conv feature map为原图的1/16)

- 把这9个anchor的坐标进行平移操作，获得在原图上的坐标。之后根据ground truth label和这些anchor之间的关系生成rpn_lables 

- 让得到的卷积特征的每一个位置都负责原图中对应位置9种尺寸的框的检测，检测的目标就是判断框中是否存在一个物体。所以，共有51*39*9种框，这些框就是anchor。

  anchor的3种尺寸，每种尺度三个比例,它们的面积分别是128*128，256*256，512*512，每种面积又分为3种长宽比，分别是1：1，1：2，2：1。

![](/Image/算法/深度学习/深度学习应用算法/9种anchor.jpg)

**我们的方法的一个重要特性是是平移不变性,锚点本身和计算锚点的函数都是平移不变的。如果在图像中平移一个目标，那么proposal也会跟着平移，这时，同一个函数需要能够在任何位置都预测到这个proposal。我们的方法可以保证这种平移不变性。**

- 对anchor的后续处理

  - 去掉超出原图的边界的anchor box

  - 如果anchor box与ground truth的IoU值最大，标记为正样本，label=1

  - 如果anchor box与ground truth的IoU>0.7，标记为正样本，label=1

  - 如果anchor box与ground truth的IoU<0.3，标记为负样本，label=0

    ​     剩下的既不是正样本也不是负样本，不用于最终训练，label=-1

### 2.3、Roi Pooling

**后续与Fast-RCNN类似**

![](/Image/算法/深度学习/深度学习应用算法/Faster-RCNN后续.jpg)


