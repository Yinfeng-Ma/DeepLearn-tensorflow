#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: mnist_cnn.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-21
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE=28
OUTPUT=10
LAYER=1024
CHANNEL=1
BATCH_SIZE=50
LEARNNING_RATE=1e-4
STEPS=10000

CONV1=5
CONV1_KERNEL=32

CONV2=5
CONV2_KERNEL=64

def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.zeros(shape))

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
def forward(x,keep_prob):
    x_ima=tf.reshape(x,[-1,IMAGE_SIZE,IMAGE_SIZE,CHANNEL])
    conv1_w=get_weight([CONV1,CONV1,CHANNEL,CONV1_KERNEL])
    conv1_b=get_bias([CONV1_KERNEL])
    conv1=conv2d(x_ima,conv1_w)
    h1=tf.nn.relu(conv1+conv1_b)

    pool1=max_pool2x2(h1)

    conv2_w=get_weight([CONV2,CONV2,CONV1_KERNEL,CONV2_KERNEL])
    conv2_b=get_bias([CONV2_KERNEL])
    conv2=conv2d(pool1,conv2_w)
    h2=tf.nn.relu(conv2+conv2_b)

    pool2=max_pool2x2(h2)
    pool2_shape=pool2.get_shape().as_list()
    node=pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
    pool2=tf.reshape(pool2,[-1,node])

    fc_w=get_weight([node,LAYER])
    fc_b=get_bias([LAYER])
    fc_h=tf.matmul(pool2,fc_w)+fc_b
    fc=tf.nn.relu(fc_h)

#    keep_prob=tf.placeholder(tf.float3)
    fc_dropout=tf.nn.dropout(fc,keep_prob)


    fo_w=get_weight([LAYER,OUTPUT])
    fo_b=get_bias([OUTPUT])
    fo_h=tf.matmul(fc_dropout,fo_w)+fo_b
    fo=tf.nn.softmax(fo_h)
    return fo
def backward(mnist):
    x=tf.placeholder(tf.float32,[None,IMAGE_SIZE*IMAGE_SIZE])
    y_=tf.placeholder(tf.float32,[None,OUTPUT])
    keep_prob=tf.placeholder(tf.float32)
    y=forward(x,keep_prob)

    ce=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step=tf.train.AdamOptimizer(LEARNNING_RATE).minimize(ce)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
        
            sess.run(train_step,feed_dict={x:xs,y_:ys,keep_prob:0.5})
            if i%200==0:
                acc=sess.run(accuracy,feed_dict={x:xs,y_:ys,keep_prob:1.0})
                print("After %d steps train, train accuracy is %g"%(i,acc))
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
        print("Test accuracy is %g")




def main():
    mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

    backward(mnist)

if __name__=="__main__":
    main()
