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

  算法开始需要初始化一些区域的集合：$$R={r_1,r_2,...r_n}R={r_1,r_2,...r_n}$$,文章中使用paper：“Efficient graph-based image segmentation”来做这件事；同时还初始化一个空的相似度区域的集合S=∅S=∅

  - 对于所有的相邻区域$$(r_i,r_j)(r_i,r_j)$$，按照相似度规则计算相似度$$s(r_i,r_j)s(r_i,r_j)$$，并且计算$$S=S∪s(r_i,r_j)S=S∪s(r_i,r_j)$$，即计算所有领域的相似度集合；
  - 假如S≠∅S≠∅：
    - 获取SS中相似度最高的一对区域$$s(r_i,s_j)=max(S)s(r_i,s_j)=max(S)$$；
    - 

- 

### 2、缩放候选区域

- 第一小步：需要对第一阶段中抽取得到的候选区域，经过一个叫做"warp"的过程，这个warp实际就是一个缩放的过程，因为第一步我们提取出的候选区域大小不一，但是后续接入的深度网络的输入是固定的，因此这些区域的大小需要适配CNN网络固定大小的输入；
- 第二小步：把第一小步中warp之后的候选区域接入到卷积神经网络，抽取一个相对低维的特征；











