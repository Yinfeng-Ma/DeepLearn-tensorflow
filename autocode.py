#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: mnist_cnn.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-21
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

IMAGE_SIZE=28*28
OUTPUT=28*28
L1=1024
L2=64
BATCH_SIZE=50
loop=np.int(mnist.train.images.shape[0]/BATCH_SIZE)
LEARNNING_RATE=1e-4
STEPS=1


def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.zeros(shape))


x=tf.placeholder(tf.float32,[None,IMAGE_SIZE])
y=tf.placeholder(tf.float32,[None,OUTPUT])

w1=get_weight([IMAGE_SIZE,L1])
b1=get_bias([L1])
y1=tf.nn.relu(tf.matmul(x,w1)+b1)

w2=get_weight([L1,L2])
b2=get_bias([L2])
y2=tf.nn.sigmoid(tf.matmul(y1,w2)+b2)


w3=get_weight([L2,L1])
b3=get_bias([L1])
y3=tf.nn.relu(tf.matmul(y2,w3)+b3)

w4=get_weight([L1,OUTPUT])
b4=get_bias([OUTPUT])
yy=tf.nn.sigmoid(tf.matmul(y3,w4)+b4)

loss=tf.reduce_sum(tf.square(y-yy))
train=tf.train.AdamOptimizer(LEARNNING_RATE).minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(STEPS):
        for i in range(loop):
            xs=mnist.train.next_batch(BATCH_SIZE)
            _,lossval=sess.run([train,loss],{x:xs[0],y:xs[0]})
            if i%100==0:
                print("Epoch %g, {%s/%s},loss is %g"%(step,i,loop,lossval))
    fig,ax=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=(8,4))
    y0=mnist.test.images[0:5]
    pre=sess.run(yy,{x:y0,y:y0})
    for image,row in zip([y0,pre],ax):
        for ima,ax in zip(image,row):
            ax.imshow(ima.reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    plt.show()



