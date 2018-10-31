########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

########## set net hyperparameters ##########
learning_rate = 0.0001

epochs = 2
batch_size_train = 128
batch_size_test = 100

display_step = 20

########## set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

#### dropout probability
dropout = 0.75

#### growth_rate
growth_rate = 32

#### theta
theta = 0.5

# Handle Dimension Ordering for different backends
'''
img_input_shape=(224, 224, 3)
concat_axis = 3

img_input_shape=(3, 224, 224)
concat_axis=1
'''
global concat_axis

concat_axis = 3

########## placeholder ##########
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


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
    x = tf.layers.batch_normalization(x)

    #### relu ####
    x = tf.nn.relu(x)

    #### conv ####
    x = tf.layers.conv2d(x, filters=inter_channel, kernel_size=1, strides=1, padding='SAME')

    ## dropout ##
    if dropout:
        x = tf.nn.dropout(x, dropout)

    ######## 3x3 Convolution，3 x 3卷积中 channel=growth rate ########
    #### BN ####
    x = tf.layers.batch_normalization(x)

    #### relu ####
    x = tf.nn.relu(x)

    #### conv ####
    x = tf.layers.conv2d(x, filters=growth_rate, kernel_size=3, strides=1, padding='SAME')

    ## dropout ##
    if dropout:
        x = tf.nn.dropout(x, dropout)

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
    concat_feat = x

    for i in range(nb_layers):
        #### conv 卷积运算 1 x 1 和 3 x 3 ####
        x = conv_block(concat_feat, growth_rate, dropout)

        #### 连接 ####
        concat_feat = tf.concat([concat_feat, x], concat_axis)

    #### dense block输出影像的维度 = nb_filter + nb_layers*growth_rate ####
    nb_filter += nb_layers * growth_rate

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
    x = tf.layers.batch_normalization(x)

    #### relu ####
    x = tf.nn.relu(x)

    #### conv ####
    x = tf.layers.conv2d(x, filters=int(nb_filter * theta), kernel_size=1, strides=1, padding='SAME')

    ## dropout ##
    if dropout:
        x = tf.nn.dropout(x, dropout)

    ## average pool ##
    x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return x


########## DenseNet121 ##########
def DenseNet121(x, nb_dense_block=4, growth_rate=32, nb_filter=64, theta=0.5, dropout=0.0, n_classes=1000):
    '''
    DenseNet121: DenseNet121

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
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ####### dense block的层数，DenseNet121各dense block的层数为[6,12,24,16] ########
    nb_layers = [6, 12, 24, 16]

    ####### first conv ########
    #### conv ####
    x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME')

    #### BN ####
    x = tf.layers.batch_normalization(x)

    #### relu ####
    x = tf.nn.relu(x)

    ####### max pool ########
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### three dense block ########
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout)

        x = transition_block(x, nb_filter, theta, dropout)

        nb_filter = int(nb_filter * theta)

    ####### last dense block ########
    x, nb_filter = dense_block(x, nb_layers[-1], nb_filter, growth_rate, dropout)

    #### BN ####
    x = tf.layers.batch_normalization(x)

    #### relu ####
    x = tf.nn.relu(x)

    ####### global average pool 全局平均池化 ########
    # x=tf.nn.avg_pool(x, ksize=[1,7,7,1],strides=[1,7,7,1],padding='VALID')

    ####### flatten 影像展平 ########
    flatten = tf.reshape(x, (-1, 1 * 1 * 1024))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)

    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred = DenseNet121(x, growth_rate=growth_rate, theta=theta, dropout=dropout, n_classes=10)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##################### train and evaluate model ##########################

########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(mnist.train.num_examples // batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples // batch_size_test):
        batch_x, batch_y = mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
