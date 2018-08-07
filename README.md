DenseNet详解
============

# 一. DenseNet简介

> DenseNet是CVPR 2017最佳论文，论文中提出的DenseNet（Dense Convolutional Network）主要还是和ResNet及Inception网络做对比，思想上有借鉴，但却是全新的结构，网络结构并不复杂，却非常有效，在CIFAR指标上全面超越ResNet。DenseNet吸收了ResNet最精华的部分，并在此上做了更加创新的工作，使得网络性能进一步提升。作者是从feature入手，通过对feature的极致利用达到更好的效果和更少的参数。DenseNet的优点如下： 

> 1. 减轻了vanishing-gradient（梯度消失） 
> 2. 加强了feature的传递 
> 3. 更有效地利用了feature 
> 4. 一定程度上较少了参数数量

# 二. DenseNet详解

> 在深度学习网络中，随着网络深度的加深，梯度消失问题会愈加明显，目前很多论文都针对这个问题提出了解决方案，比如ResNet，Highway Networks，Stochastic depth，FractalNets等，尽管这些算法的网络结构有差别，但是核心都在于：create short paths from early layers to later layers。那么作者是怎么做呢？延续这个思路，那就是在保证网络中层与层之间最大程度的信息传输的前提下，直接将所有层连接起来！DenseNet 是一种具有密集连接的卷积神经网络。在该网络中，任何两层之间都有直接的连接，也就是说，网络每一层的输入都是前面所有层输出的并集，而该层所学习的特征图也会被直接传给其后面所有层作为输入。下图是 DenseNet 的一个dense block示意图，一个block里面的结构如下，与ResNet中的BottleNeck基本一致：BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) ，而一个DenseNet则由多个这种block组成。每个DenseBlock的之间层称为transition layers，由BN−>Conv(1×1)−>averagePooling(2×2)组成 

# 三.

# 四.

# 五.
