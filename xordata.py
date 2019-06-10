#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: xordata.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-06-04
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate(sample_num,mean,cov,diff,regression):
    num_class=2
    sample_per_class=int(sample_num/num_class)
    X0=np.random.multivariate_normal(mean,cov,sample_per_class)
    Y0=np.zeros(sample_per_class)
    for ci,d in enumerate(diff):
        X1=np.random.multivariate_normal(mean+d,cov,sample_per_class)
        Y1=(ci+1)*np.ones(sample_per_class)
    X0=np.concatenate((X0,X1))
    Y0=np.concatenate((Y0,Y1))
    if regression==False:
        class_ind=[[Y0[j][0]==i for i in range(num_class)]for j in range(len(Y0))]
        Y0=np.asarray(class_ind,dtype=np.float32)
    index=[i for i in range(len(Y0))]
    np.random.shuffle(index)
    X,Y=X0[index],Y0[index]
    return X,Y

np.random.seed(10)
num_class=2
mean=np.random.randn(num_class)
cov=np.eye(num_class)
X,Y=generate(1000,mean,cov,[3.0],True)
col=['r' if i==0 else 'b' for i in Y[:]]
plt.scatter(X[:,0],X[:,1],c=col)
plt.xlabel("Age (y)")
plt.ylabel("Size (cm)")
plt.show()

xx,yy=generate(100,mean,cov,[[3.0,0],[3.0,3.0],[0,3.0]],True)
plt.scatter(xx[:,0],xx[:,1])
plt.show()


lab_dim=1
inputs=2
lr=0.04
epoch=50
batch_size=25


x=tf.placeholder(tf.float32,[None,inputs])
y=tf.placeholder(tf.float32,[None,lab_dim])

w=tf.Variable(tf.random_normal([inputs,lab_dim]))
b=tf.Variable(tf.zeros([lab_dim]))

y0=tf.nn.sigmoid(tf.matmul(x,w)+b)
loss=-y*tf.log(y0)-(1-y)*tf.log(1-y0)
err=tf.square(y-y0)
loss=tf.reduce_mean(loss)
err=tf.reduce_mean(err)
train=tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epoch):
        sumerr=0
        for i in range(np.int32(len(Y)/batch_size)):
            x1=X[i*batch_size:(i+1)*batch_size]
            y1=Y[i*batch_size:(i+1)*batch_size]
            y1=np.reshape(y1,[-1,1])
            _,lossval,y0val,errval=sess.run([train,loss,y0,err],{x:x1,y:y1})
            sumerr=sumerr+errval

        print("After %s step, loss is %g and error is %g"%(step,lossval,sumerr/batch_size))

