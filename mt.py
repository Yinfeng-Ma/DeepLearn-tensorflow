#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: mt.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-22


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

print(mnist.train.images.shape)


xs,ys=mnist.train.next_batch(50)
xs_ima=tf.reshape(xs,[-1,28,28,1])
w=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
xo=tf.nn.conv2d(xs_ima,w,strides=[1,1,1,1],padding='SAME')



with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(tf.shape(xo)))
