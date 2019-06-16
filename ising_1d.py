# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:39:01 2019

@author: T01
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(12)

L=40
states=np.random.choice([-1,1],size=(10000,L))

#plt.quiver(states,1)
#plt.show()


def ising_1d_energies(states):
    J=np.zeros([L,L],)
    for i in range(L):
        J[i,(i+1)%L]=-1.0
    E=np.einsum('...i,ij,...j->...',states,J,states)
    return E
energies=ising_1d_energies(states)
energies=energies.reshape(-1,1)
states=np.einsum('...i,...j->...ij',states,states)

shape=states.shape
states=states.reshape(shape[0],shape[1]*shape[2])
Data=[states,energies]

n_samples=400
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples]
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2]



def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.zeros(shape))

INPUT=L*L
OUTPUT=1
lr=1e-4
epoch=5000


x=tf.placeholder(tf.float32,[None,INPUT])
y=tf.placeholder(tf.float32,[None,OUTPUT])
l1=tf.placeholder(tf.float32)

w=get_weight([INPUT,OUTPUT])
b=get_bias([OUTPUT])
y_pred=tf.matmul(x,w)+b
yme=tf.reduce_mean(y_pred)
y1=tf.reduce_sum(tf.square(y-yme))

loss=tf.reduce_sum(tf.square(y-y_pred))
loss=1-loss/y1
train=tf.train.AdamOptimizer(lr).minimize(loss)

init=tf.global_variables_initializer()

with tf.Session () as sess:
    sess.run(init)
    for step in range(epoch):
        sess.run(train,{x:X_train,y:Y_train})
        w0,b0=sess.run([w,b],{x:X_train,y:Y_train})
        lossval=sess.run(loss,{x:X_train,y:Y_train})
        if (step+1)%10==0:
            print("After %g steps, loss is %g"%(step,lossval))
    
    cmap_args=dict(vmin=-1,vmax=1.0,cmap='seismic')
    plt.imshow(w0.reshape(L,L),**cmap_args)
    plt.show()




