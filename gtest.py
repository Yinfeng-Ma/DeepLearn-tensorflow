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

def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.zeros(shape))
G1=100
G2=128

L1=128
INPUT=28*28
OUTPUT=1
BATCH_SIZE=500

lr=1e-4
EPOCH=10000

nz=100

gw1=get_weight([G1,G2])
gb1=get_bias([G2])

gw2=get_weight([G2,INPUT])
gb2=get_bias([INPUT])

g_var=[gw1,gb1,gw2,gb2]

dw1=get_weight([INPUT,L1])
db1=get_bias([L1])

dw2=get_weight([L1,OUTPUT])
db2=get_bias([OUTPUT])

d_ver=[dw1,db1,dw2,db2]


def generator(z):
    g1=tf.nn.relu(tf.matmul(z,gw1)+gb1)
    g2=tf.nn.sigmoid(tf.matmul(g1,gw2)+gb2)
#    g3=tf.nn.sigmoid(tf.matmul(g2,gw3)+gb3)
    return g2
def discriminator(x):
    d1=tf.nn.relu(tf.matmul(x,dw1)+db1)
    d2=tf.nn.sigmoid(tf.matmul(d1,dw2)+db2)
    return d2

def sample_z(m,n):
    return np.random.uniform(-1.0,1.0,size=[m,n])


x=tf.placeholder(tf.float32,[None,INPUT])
z=tf.placeholder(tf.float32,[None,G1])

gz=generator(z)

dx=discriminator(x)
gx=discriminator(gz)

d_loss=-tf.reduce_mean(tf.log(dx))
g_loss=-tf.reduce_mean(tf.log(1.0-gx))

loss=d_loss+g_loss

gx_loss=-tf.reduce_mean(tf.log(gx))


dtrain=tf.train.AdamOptimizer(lr).minimize(loss,var_list=d_ver)
gtrain=tf.train.AdamOptimizer(lr).minimize(gx_loss,var_list=g_var)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(EPOCH):
        for i in range(mnist.train.images[0].shape[0]//BATCH_SIZE):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            zs=sample_z(nz,G1)
            _,dloss=sess.run([dtrain,loss],{x:xs,z:zs})
            _,gloss=sess.run([gtrain,gx_loss],{z:zs})
        if step%100==0:
            print("After %s step,d_loss is %g and g_loss is %g"%(step,dloss,gloss))

    gxx=sess.run(gz,{z:zs})

    fig=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=(4,4))
    

    for i,img in enumerate(gxx[:10]):
        ax=plt.subplot(2,5,i+1)
        plt.axis('off')
        plt.imshow(img.reshape(28,28))
    plt.show()
    plt.figure
    im=mnist.train.images[:10]
    fig=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=(4,4))
    for i,ima in enumerate(im):
        plt.subplot(2,5,i+1)
        plt.imshow(ima.reshape(28,28))
    plt.show()


