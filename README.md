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

> Growth Rate是Hl函数产生的feature map的数量，即每一个layer输出的feature map的维度。对于第L层layer（经过L-1层H函数作用），其输入的feature map维度为k0+k x (L-1)，k0为input layer的channels，k是Growth Rate。

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

#### 注意：每一个Dense Block输出影像的维度 = k0+k x nb_layers，k0为上一个Transition Layer输入的维度，k为Growth Rate，nb_layers为该Block的层数，如下图中DenseNet-121，Dense Block有6、12、24和16。 


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

> 论文的代码地址为 https://github.com/liuzhuang13/DenseNet. 参考https://github.com/flyyufelix/DenseNet-Keras. 利用tensorflow构建DenseNet-121实现MNIST判别，代码如下：

```python
########## load packages ##########
import tensorflow as tf


##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_sets",one_hot=True)


########## set net hyperparameters ##########
learning_rate=0.0001

epochs=20
batch_size_train=128
batch_size_test=100

display_step=20


########## set net parameters ##########
#### img shape:28*28 ####
n_input=784 

#### 0-9 digits ####
n_classes=10

#### dropout probability
dropout=0.75

#### growth_rate
growth_rate=32

#### theta
theta=0.5



# Handle Dimension Ordering for different backends
'''
img_input_shape=(224, 224, 3)
concat_axis = 3

img_input_shape=(3, 224, 224)
concat_axis=1
'''
global concat_axis

concat_axis=3


########## placeholder ##########
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])


##################### build net model ##########################

########## conv block ##########
def conv_block(x, growth_rate, dropout):
    '''
    conv_block: 卷积块，dense block的卷积运算——有 1 x 1 卷积 和 3 x 3 卷积

    x: 输入影像
    growth_rate: 即k，每一个layer输出的feature map的维度
    dropout: dropout probability

    return: x 卷积块输出
    '''

    ######## 1x1 Convolution (Bottleneck layer)，1 x 1卷积中 channel=growth rate x 4 ########
    inter_channel = growth_rate * 4  

    #### BN ####
    x=tf.layers.batch_normalization(x)

    #### relu ####
    x=tf.nn.relu(x)

    #### conv ####
    x=tf.layers.conv2d(x, filters=inter_channel, kernel_size=1, strides=1, padding='SAME')

    ## dropout ##
    if dropout:
        x=tf.nn.dropout(x, dropout)


    ######## 3x3 Convolution，3 x 3卷积中 channel=growth rate ########
    #### BN ####
    x=tf.layers.batch_normalization(x)

    #### relu ####
    x=tf.nn.relu(x)

    #### conv ####
    x=tf.layers.conv2d(x, filters=growth_rate, kernel_size=3, strides=1, padding='SAME')

    ## dropout ##
    if dropout:
        x=tf.nn.dropout(x, dropout)


    return x

########## dense block ##########
def dense_block(x, nb_layers, nb_filter, growth_rate, dropout):

    '''
    dense_block: dense block

    x: 输入影像
    nb_layers：dense block的层数
    nb_filter：输入影像维度
    growth_rate: 即k，每一个layer输出的feature map的维度
    dropout: dropout probability

    return: x dense_block输出
    '''

    #### 初始输入备份 ####
    concat_feat=x

    for i in range(nb_layers):

        #### conv 卷积运算 1 x 1 和 3 x 3 ####
        x=conv_block(concat_feat, growth_rate, dropout)

        #### 连接 ####
        concat_feat=tf.concat([concat_feat, x],concat_axis )

    #### dense block输出影像的维度 = nb_filter + nb_layers*growth_rate ####
    nb_filter+=nb_layers*growth_rate

    return concat_feat, nb_filter

########## transition block ##########
def transition_block(x, nb_filter, theta, dropout):

    '''
    transition_block: transition block

    x: 输入影像
    nb_filter：输入影像维度
    theta: compression factor 压缩因子
    dropout: dropout probability

    return: x dense_block输出
    '''

    #### BN ####
    x=tf.layers.batch_normalization(x)

    #### relu ####
    x=tf.nn.relu(x)

    #### conv ####
    x=tf.layers.conv2d(x, filters=int(nb_filter * theta), kernel_size=1, strides=1, padding='SAME')

    ## dropout ##
    if dropout:
        x=tf.nn.dropout(x, dropout)

    ## average pool ##
    x=tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    return x


########## DenseNet ##########
def DenseNet(x, nb_dense_block=4, growth_rate=32, nb_filter=64, theta=0.5, dropout=0.0, n_classes=1000):

    '''
    DenseNet: DenseNet

    x: 输入影像
    nb_dense_block: dense block个数，默认为4
    growth_rate: 即k，每一个layer输出的feature map的维度，默认为32
    nb_filter：第一层卷积核的维度，默认为2 x k=64
    theta: compression factor 压缩因子，默认为0.5
    dropout: dropout probability
    n_classes: 影像分类数，可根据数据集进行调整，ImageNet为1000，MNIST为10

    return: out DenseNet预测值输出
    '''

    ####### reshape input picture ########
    x=tf.reshape(x,shape=[-1,28,28,1])


    ####### dense block的层数，DenseNet-121各dense block的层数为[6,12,24,16] ########
    nb_layers=[6, 12, 24, 16]


    ####### first conv ########
    #### conv ####
    x=tf.layers.conv2d(x,filters=64,kernel_size=7,strides=2,padding='SAME')

    #### BN ####
    x=tf.layers.batch_normalization(x)

    #### relu ####
    x=tf.nn.relu(x)


    ####### max pool ########
    x=tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### three dense block ########
    for block_idx in range(nb_dense_block-1):

        x, nb_filter=dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout)

        x = transition_block(x, nb_filter, theta, dropout)

        nb_filter = int(nb_filter * theta)


    ####### last dense block ########
    x, nb_filter=dense_block(x, nb_layers[-1], nb_filter, growth_rate, dropout)


    #### BN ####
    x=tf.layers.batch_normalization(x)

    #### relu ####
    x=tf.nn.relu(x)


    ####### global average pool 全局平均池化 ########
    #x=tf.nn.avg_pool(x, ksize=[1,7,7,1],strides=[1,7,7,1],padding='VALID')


    ####### flatten 影像展平 ########
    flatten = tf.reshape(x, (-1, 1*1*1024))


    ####### out 输出，10类 可根据数据集进行调整 ########
    out=tf.layers.dense(flatten,n_classes)


    ####### softmax ########
    out=tf.nn.softmax(out)

    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred=DenseNet(x, growth_rate=growth_rate, theta=theta, dropout=dropout, n_classes=10)

#### loss 损失计算 ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size_test):
        batch_x,batch_y=mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))

```
