# R-CNN

![](/Image/算法/深度学习/深度学习应用算法/R-CNN算法流程.jpg)

如上图所示，R-CNN这个物体检查系统可以大致分为四步进行：

1. **获取输入图像**
2. **提取约2000个候选区域**
3. **将候选区域分别输入CNN网络（这里需要将候选图片进行缩放）**
4. **将CNN的输出输入SVM中进行类别的判定**

## 算法流程描述
### 1、候选框提取

&emsp;&emsp;即上图中的1~2，对于给定的输入图像，使用选择性搜索(selective search)的区域建议方法提取出大约2000个候选区域，即首先过滤掉那些大概率不包含物体的区域，通过这一阶段将原始图像中需要处理的区域大大减少；

- **selective search**：“[Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)”。selective search是一种“hierarchical bottom-up grouping”的方法，即层次、自底向上的分组方法，简单来说就是先找出图像中的相邻区域，然后定义一个领域相识度的计算规则，使用这些规则一直合并相似度较高的区域并且更新，最后知道覆盖至整张图像，完成处理的图像中不同的区域就是这些“region proposal”。以下是示意图：

  ![](/Image/算法/深度学习/深度学习应用算法/selective search算法示意图.jpg)

  原始的paper对算法的描述步骤如下：

  输入：一张图像（彩色） 输出：候选的box区域

  算法开始需要初始化一些区域的集合：$$R={r_1,r_2,...r_n}$$,文章中使用paper：“Efficient graph-based image segmentation”来做这件事；同时还初始化一个空的相似度区域的集合S=∅S=∅

  - 对于所有的相邻区域$$(r_i,r_j)$$，按照相似度规则计算相似度$$s(r_i,r_j)$$，并且计算$$S=S∪s(r_i,r_j)$$，即计算所有领域的相似度集合；

  - 假如S≠∅：
    - 获取S中相似度最高的一对区域$$s(r_i,s_j)=max(S)$$；
    - 对最相似的两个区域进行合并$$r_t=r_i∪r_j$$；
    - 从S集合中去除和$$r_i$$及$$r_j$$相邻的相似区域
    - 计算合并后的区域和其邻域的相似度集合$$S=S∪S_t及R=R∪R_t$$
    - 重复以上步骤直至S=∅

  - 从集合RR中提取出所有的boxes，即定位的矩形区域

    对于相似度的计算方法selective search的原始论文中定义为颜色、纹理、尺寸等几种不同度量方式的叠加之和：
    $$
    s(ri,rj)=a1scolour(ri,rj)+a2stexture(ri,rj)+a3ssize(ri,rj)+a4sfill(ri,rj)   a∈(0,1)
    $$





### 2、缩放候选区域

- 第一小步：需要对第一阶段中抽取得到的候选区域，经过一个叫做"warp"的过程，这个warp实际就是一个缩放的过程，因为第一步我们提取出的候选区域大小不一，但是后续接入的深度网络的输入是固定的，因此这些区域的大小需要适配CNN网络固定大小的输入；
- 第二小步：把第一小步中warp之后的候选区域接入到卷积神经网络，抽取一个相对低维的特征；

&emsp;&emsp;使用选择性搜索提取出大小不一的候选区域之后，然后经过卷积神经网络提取特征，文中提到的是使用12年的AlexNet，此网络的输入是固定大小的227×227227×227，输出是40964096维的特征向量，由于输出的尺寸固定，而RCNN第一步抽取的候选区域大小不一，因此给出了几种方法对候选区域的大小进行变换以适配网络输入：

- **各向异性（anisotropically）缩放**:即对候选区域缩放的高和宽的倍数不同，直接拉成CNN中输入大小的形式，这种变换在文中称为“warp”；文中进行的实验表明使用padding为16像素的“warp”方法效果最好
- **各向同性（isotropically）缩放**:即缩放的高和宽的比例倍数相同，这里又分为两种：一种对原始的图像按等比例缩放后的灰色区域进行填充，另一种不进行填充。

## 3、网络训练

### 3.1、**CNN网络训练**

&emsp;&emsp;训练分成“pre-training”+“fine-tuning”：首先选用caffe中已经训练好的模型在大的数据集上进行预训练（ILSVRC2012图像分类的数据集 ）得到一个初始的网络，然后在此网络的基础上细调：替换网络最后的输出层，因为ImageNet分类数据集的输出层包含1000个类，这里进行目标检测的类别不一致，比如VOC使用的是20个类，而ILSVRC2013 的目标检测使用的是200个类，把背景也当成一类的话，实际的网络输出层的个数应该是N+1N+1，NN为目标的类别。在细调的时候考虑与ground truth（也就是真实的包含物体的区域）的IoU超过0.5的为正类，其余的均为负类，迭代的时候使用SGD方法，选取0.001的学习率，使用包含32个正类样本和96个负类样本组成mini-bacth。

### 3.2、**SVM分类器的训练**

&emsp;&emsp;一旦CNN网络训练完成，去除网络中的最后的输出层，以全连接层的输出得到新的特征向量，以供训练SVM。这里文中提到了SVM训练阶段正负样本的划分，注意的是这里的正负类的选择和CNN网络训练时的不同，训练SVM时正类定义为ground truth，候选区域中与ground truth的IoU小于0.3的定义为负类，其余的候选区域忽略；

- **为什么RCNN使用SVM替代softmax层进行分类？**

**其一**，作者做过实验表明，在微调阶段尝试使用和SVM相同定义的正负类样本数，效果比较差，对此作出假设：怎样定义正负类样本数并不重要，重要的是造成这种结果的差异源于微调中数据的有限，当使用IoU0大于0.5的样本作为正类时（文中用“jittered”形容这些数据），样本增加了接近30倍，避免了过拟合；并且，这些jittered的数据导致了fine-tuning阶段的结果并不是最优的，因为现在选的正类样本并不是“精准定位”的（显然这是由于IoU阈值变高导致正类样本的门槛变“低”了）；

**其二**，根据第一点解释恰好引出了第二个点，即为什么使用SVM而不是直接使用softmax层进行分类，作者认为按照刚才的解释，微调阶段CNN并不能进行“精准定位”，并且softmax训练的时候只是随机采样负类样本，而SVM训练阶段可以考虑那些“hard negative”的样本。这里使用了“hard negative mining”的思想，以目标检测为例，第一轮训练的时候先按照预先定义好的正负类样本进行分类；第二轮对于那些本身是负类却被误判为正类的样本，即通常我们所说的“false positive”样本，计入负类样本重新进行训练，这个过程重复几次最终得到的效果会比较好。

基于以上两点考虑，作者认为有必要在CNN后面接上SVM进行分类。

### 3.3、 **非极大值抑制过滤重叠的bounding box**

&emsp;&emsp;在test阶段，候选区域经过CNN提取特征之后最后使用SVM进行分类评分，得到很多的bounding box，由于这些bounding box包含很多重叠的部分，如下：

![](/Image/算法/深度学习/深度学习应用算法/非极大值抑制.jpg)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;左图是预期的bounding box，右图是实际得到的bounding box

&emsp;&emsp;首先对输出的bounding box进行打分，找出评分最高的bounding box，计算其他的bounding box和它的IoU值，选取一个阈值，从输出的bounding box中移除那些IoU小于这个阈值的box，然后重复这个步骤，直至仅存在保留的bounding box。文中使用NMS基于每个单独的类。

### 3.4、**回归器修正box偏差**

&emsp;&emsp;候选区域经过SVM打分之后，需要经过回归器进行修正，修正的目的是得到一个新的bounding box，新的bounding box预测的偏差减小，文中使用的回归器修正启发于deformable part models（DPM），而且回归是基于每个特定类的。具体来说给定预测的bounding box和ground truth的集合(P_i,G_i)i=1,...,N，其中，P^i=(p^i_x,p^i_y,p^i_w,p^i_h)，G^i=(g^i_x,g^i_y,g^i_w,g^i_h)，x,y,w,h分别表示左上角的坐标以及box的宽和高，修正的目的是把PP变换到预测的ground truthG^，变换的的函数有四个分别是d_x(P),d_y(P),d_w(P),d_h(P)，变换的具体公式如下：
$$
\begin{equation}
\hat{G}_x=P_wd_x(P)+P_x \\
\hat{G}_y=P_hd_y(P)+P_y \\
\hat{G}_w=P_wexp(d_w(P)) \\
\hat{G}_h=P_hexp(d_h(P))
\end{equation}
$$
这里的四个$$d_x(P),d_y(P)d_w(P),d_h(P)$$由CNN最后一层的pooling层的特征经过线性变换得到：$$d∗(P)=w^T_∗∅(P)$$，因此这里我们需要学习的参数变成$$w_∗$$，此问题可以看成是一个标准的岭回归问题：

![](/Image/算法/深度学习/深度学习应用算法/IOU回归.jpg)

## 4、RCNN总结

-  RCNN的耗时体现在两个阶段，一个是区域候选需要一定时间，另一个是对于每一个区域候选框都需要重新丢进网络里才能抽取特征，特别是后者导致RCNN非常的慢；
- RCNN在网络的输入是使用的warp方法，这种缩放会不会对最终的精度有影响？
- RCNN显然并不是一个端到端的工作，通常情况下我们认为端到端的工作可以达到更好的效果，因为不同模块之间的衔接可能会存在一定的信息损失。