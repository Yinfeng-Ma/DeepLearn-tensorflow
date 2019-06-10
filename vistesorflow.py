#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: vistesorflow.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-21


import tensorflow as tf
import numpy as np

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+noise

def add_layer(inputs,in_size,out_size,n_layer,action_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            w=tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1),name='W')
            tf.summary.histogram(layer_name+'/weights',w)
        with tf.name_scope('biases'):
            b=tf.Variable(tf.zeros([out_size]),name='b')
            tf.summary.histogram(layer_name+'/biases',b)
    wx_plus_b=tf.matmul(inputs,w)+b
    if action_function is None:
        outputs=wx_plus_b
    else:
        outputs=action_function(wx_plus_b)
        tf.summary.histogram(layer_name+'/output',outputs )

    return outputs

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

l1=add_layer(xs,1,10,n_layer=1,action_function=tf.nn.relu)

prediction=add_layer(l1,10,1,n_layer=2,action_function=None)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter("log/",sess.graph)
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(result,i)




