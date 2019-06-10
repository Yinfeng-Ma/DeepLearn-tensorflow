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

xx,yy=generate(10,mean,cov,[[3.0,0],[3.0,3.0],[0,3.0]],True)
print(yy)
yy=yy%2

col=['r' if i==0 else 'b' for i in Y[:]]
plt.scatter(X[:,0],X[:,1],c=col)
plt.xlabel("Age (y)")
plt.ylabel("Size (cm)")
plt.show()

