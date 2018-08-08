DenseNet详解
============

# 一. DenseNet简介

> DenseNet是CVPR 2017最佳论文，论文中提出的DenseNet（Dense Convolutional Network）主要还是和ResNet及Inception网络做对比，思想上有借鉴，但却是全新的结构，网络结构并不复杂，却非常有效，在CIFAR指标上全面超越ResNet。DenseNet吸收了ResNet最精华的部分，并在此上做了更加创新的工作，使得网络性能进一步提升。作者是从feature入手，通过对feature的极致利用达到更好的效果和更少的参数。DenseNet的优点如下： 

> 1. 减轻了vanishing-gradient（梯度消失） 
> 2. 加强了feature的传递 
> 3. 更有效地利用了feature 
> 4. 一定程度上较少了参数数量

# 二. DenseNet结构

## (一) 整体结构
> 在深度学习网络中，随着网络深度的加深，梯度消失问题会愈加明显，目前很多论文都针对这个问题提出了解决方案，比如ResNet，Highway Networks，Stochastic depth，FractalNets等，尽管这些算法的网络结构有差别，但是核心都在于：create short paths from early layers to later layers。DenseNet延续这个思路，那就是在保证网络中层与层之间最大程度的信息传输的前提下，直接将所有层连接起来！

> DenseNet 是一种具有密集连接的卷积神经网络，在该网络中，任何两层之间都有直接的连接，也就是说，网络每一层的输入都是前面所有层输出的并集，而该层所学习的特征图也会被直接传给其后面所有层作为输入。在传统的卷积神经网络中，如果网络有L层，那么就会有L个连接，而Dense Block模块利用了该模块中前面所有层的信息，即每一个layer都和前面的layer有highway的稠密连接，那么highway稠密连接数目为L*(L+1)/2。下图是 DenseNet 的一个dense block示意图，一个block里面的结构如下，与ResNet中的BottleNeck基本一致：BN-ReLU-Conv

![image](https://github.com/ShaoQiBNU/DenseNet/blob/master/images/1.png)

> 图1是一个详细的dense block模块，其中层数为5即xl(l=0,1,2,3,4)，BN-ReLU-Conv为4即Hl(l=1,2,3,4)，网络增长率Growth Rate为4，即每一个layer输出的feature map的维度（channels）为4。传递过程如下：

```
x0是input层——height x width x 6，输进H1，经过BN-ReLU-Conv得到x1——height x width x 4

之后将x0和x1连接起来即concatenation处理，通道的合并，像Inception那样——height x width x 10，输进H2，经过BN-ReLU-Conv得到x2——height x width x 4

之后将x0、x1和x2连接起来——height x width x 14，输进H3，经过BN-ReLU-Conv得到x2——height x width x 4

之后将x0、x1、x2和x3连接起来——height x width x 18，输进H4，经过BN-ReLU-Conv得到x4——height x width x 4

之后将x0、x1、x2、x3和x4连接起来——height x width x 22，输进transition layers

```

> 一个DenseNet则由多个这种block组成，每个DenseBlock的之间层称为transition layers，由BN−>Conv−>averagePooling组成，如图所示：

![image](https://github.com/ShaoQiBNU/DenseNet/blob/master/images/2.png)

> 虽然都是shortcut连接，但是DenseNet与ResNet连接方式不同，ResNet的连接方式如下，其优点在于梯度可以直接通过恒等函数从后面的层流向早些时候层。然而，恒等函数和输出H的求和可能会妨碍信息在整个网络中的传播。

![image](https://github.com/ShaoQiBNU/DenseNet/blob/master/images/3.png)

> DenseNet的连接方式如下，可以有效地改善信息流动。

![image](https://github.com/ShaoQiBNU/DenseNet/blob/master/images/4.png)

## (二) 结构说明

### 1. Growth Rate 

> Growth Rate是Hl函数产生的feature map的数量，即每一个layer输出的feature map的维度。对于第L层layer，其输入的feature map维度为k0+k x (L-1)，k0为input layer的channels，k是Growth Rate。

### 2. Dense Block

> DenseNet由 Dense Block 和 Transition Layer 组成，DenseNet提出了三种应用版本分别为 DenseNet、DenseNet-B和DenseNet-BC，下面对这三种版本的两个模块分别进行详细说明。

#### (1) DenseNet

> DenseNet 的Dense Block的结构设计如下：

```
Batch Normalization ——> Relu ——> Conv (3 x 3) ——> dropout
```

#### (2) DenseNet-B和DenseNet-BC

> DenseNet-B和DenseNet-BC均采用了Bottleneck layers的设计结构，其目的在于减少feature map的数量，降低维度，减少计算量，又能融合各个通道的特征。以DenseNet-169的Dense Block（3）为例，包含32个1 x 1和3 x 3的卷积操作，也就是第32个子结构的输入是前面31层的输出结果，每层输出的channel是32（growth rate），那么如果不做bottleneck操作，第32层的3 x 3卷积操作的输入就是31 x 32 +（上一个Dense Block的输出channel），近1000了。而加上1 x 1的卷积，代码中的1 x 1卷积的channel是growth rate x 4，也就是128，然后再作为3 x 3卷积的输入。这就大大减少了计算量，这就是bottleneck。Dense Block的结构设计如下：

```
Batch Normalization ——> Relu ——> Conv (1 x 1),filters为 4 x k ——> dropout ——> Batch Normalization ——> Relu ——> Conv (3 x 3)
```

### 3. Transition Layer

#### (1) DenseNet和DenseNet-B

> Transition Layer的结构设计如下：

```
Batch Normalization ——> Relu ——> Conv (1 x 1),filters为 m ——> dropout ——> average pool (2 x 2, stride=2)
m为Dense Block输出的维度channels
```
#### (2) DenseNet-BC

> 因为每个Dense Block结束后的输出channel个数很多，可以设置参数θ来降维，0 < θ ≤ 1，当θ=1时，Transition Layer的结构同上；当0 < θ < 1，可将filters设置为θ·m，表示将这些输出缩小到原来的多少倍，默认是0.5，从而实现降维。Transition Layer的结构设计如下：

```
Batch Normalization ——> Relu ——> Conv (1 x 1),filters为 θ·m ——> dropout ——> average pool (2 x 2, stride=2)
m为Dense Block输出的维度channels   θ为compression factor
```

## (三) 网络设计

> 论文设计了几种DenseNet-BC网络结构，其中第一层卷积的卷积核设置为 2k，具体参数如图所示：

![image](https://github.com/ShaoQiBNU/DenseNet/blob/master/images/5.png)

> 以DenseNet-121为例对网络传递方式进行分析，过程如下：

> 初始化——卷积和pool

```
input: 输入 224 x 224 x 3   Growth Rate: k = 32  θ=0.5

conv1: 7 x 7 x 3, filters = 2k = 64, stride = 2   conv -->BN --> Relu   输出 112 x 112 x 64

max pool: 3 x 3, stride = 2   输出 56 x 56 x 64

```
> Dense Block

```
具体过程如图所示

Dense Block 1: 输出 56 x 56 x 256
Transition Layer 1: θ=0.5, m=256, 因此BN --> Relu --> conv 后变为 56 x 56 x 128, 之后average pool(s=2), 输出 28 x 28 x 128

Dense Block 2: 输出 28 x 28 x (128 + 32 x 12) = 28 x 28 x 512
Transition Layer 2: 输出 14 x 14 x 256

Dense Block 3: 输出 14 x 14 x 1024
Transition Layer 3: 输出 7 x 7 x 512

Dense Block 4: 输出 7 x 7 x 1024

```

![image](https://github.com/ShaoQiBNU/DenseNet/blob/master/images/6.png)

> Classification layer

```
global average pool: 输出 1 x 1 x 1024

fc: 1000 种类数，可根据数据集进行调整

softmax: 对fc进行激活函数处理

out: 输出

```

# 三. 代码

> 论文的代码地址为 https://github.com/liuzhuang13/DenseNet，参考https://github.com/flyyufelix/DenseNet-Keras，利用tensorflow构建DenseNet-121实现MNIST判别，代码如下：

```python

```

# 四.

# 五.
