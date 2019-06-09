# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:21:07 2019

@author: T01
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


def get_weight(shape):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
#    if regularizer!=None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b


inputs=28*28
output=1
L1=128
L2=64
G1=100
G2=128
lr=1e-4
lr1=1e-4
epoch=600
batch_size=500
batch_loop=np.int(mnist.train.images.shape[0]/batch_size)


x=tf.placeholder(tf.float32,[None,inputs])
y=tf.placeholder(tf.float32,[None,G1])

dw1=get_weight([inputs,L1])
db1=get_bias([L1])
dw2=get_weight([L1,output])
db2=get_bias([output])

d_var=[dw1,dw2,db1,db2]

gw1=get_weight([G1,G2])
gb1=get_bias([G2])
gw2=get_weight([G2,inputs])
gb2=get_bias([inputs])

g_var=[gw1,gw2,gb1,gb2]

def discriminator(x):
   
    y1=tf.nn.relu(tf.matmul(x,dw1)+db1) 
    y21=tf.nn.sigmoid(tf.matmul(y1,dw2)+db2)
    return y21

def generator(z):
    
    y1=tf.nn.relu(tf.matmul(z,gw1)+gb1)

    y2=tf.nn.sigmoid(tf.matmul(y1,gw2)+gb2)

    
    return y2
def sample_z(m,n):
    return np.random.uniform(-1.0,1.0,size=[m,n])

gx=generator(y)

xd=discriminator(x)
xg=discriminator(gx)

d1=-tf.reduce_mean(tf.log(xd))
d2=-tf.reduce_mean(tf.log(1.0-xg))

d_loss=d1+d2

g_loss=-tf.reduce_mean(tf.log(xg))


d_train=tf.train.AdamOptimizer(lr).minimize(d_loss,var_list=d_var)
g_train=tf.train.AdamOptimizer(lr1).minimize(g_loss,var_list=g_var)
init=tf.global_variables_initializer()


#n1=3
#n2=10
#fig, ax = plt.subplots(nrows=n1,ncols=n2,sharex=True,sharey=True,figsize=(6,2) )

#ax = ax.flatten()
#for i in range(n1*n2):
 #   img = mnist.train.images[i].reshape(28, 28)
#    ax[i].imshow(img, cmap='Greys')
 #   ax[0].set_xticks([])
  #  ax[0].set_yticks([])
   # plt.tight_layout(pad=0.1)
    #plt.show()

with tf.Session() as sess:
    sess.run(init)
    for step in range(epoch):
        for i in range(batch_loop):
            xs=mnist.train.next_batch(batch_size)
            zs=sample_z(10,G1)
            _,lossval=sess.run([d_train,d_loss],{x:xs[0],y:zs})
            _,lossval1=sess.run([g_train,g_loss],{y:zs})
       
        if step%100==0:
            print("Epoch %s,loss of d is %g and loss of g is %g"%(step,lossval,lossval1))
                
    gz=sess.run(gx,{y:zs})
    fig,ax=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=(6,2) )
    for i,img in enumerate(gz):
        plt.subplot(2,5,i+1)
        plt.imshow(img.reshape(28,28))
    plt.show()
   







