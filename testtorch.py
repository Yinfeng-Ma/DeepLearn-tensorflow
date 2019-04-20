#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: testtorch.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-20


#import torch
import numpy as np
import matplotlib.pyplot as plt

def loss(w,b,points):
    err=0
    for i in range(len(points)):
        x=points[i][0]
        y=points[i][1]
        err+=(y-(w*x+b))**2
    return err/float(len(points))

def step_gradient(w_curr,b_curr,points,learning_rate):
    b_gradient=0
    w_gradient=0
    N=float(len(points))
    for i in range(len(points)):
        x=points[i][0]
        y=points[i][1]
        b_gradient+=-2/N*(y-(w_curr*x+b_curr))
        w_gradient+=-2/N*x*(y-(w_curr*x+b_curr))
    new_w=w_curr-learning_rate*w_gradient
    new_b=b_curr-learning_rate*b_gradient
    return [new_w,new_b]
x=np.linspace(0,1,101)
y=0.4*x+np.random.randn(len(x))/100.0
point=np.concatenate((x,y))

point=point.reshape(-1,len(x))
point=np.transpose(point)
[w,b]=np.random.randn(2)
print w,b
err=[]
for i in range(3000):
    [w,b]=step_gradient(w,b,point,0.01)
#    print loss(w,b,point)
    err.append(loss(w,b,point))
    if i%200==0:
        print w,b,loss(w,b,point)






print w,b
y1=w*x+b
plt.subplot(121)
plt.plot(point[:,0],point[:,1],'o')
plt.plot(x,y1,'+')
plt.subplot(122)
plt.plot(err)
plt.show()
