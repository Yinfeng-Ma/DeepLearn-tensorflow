#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: linear.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-05-01


import torch 
import matplotlib.pyplot as plt 

data=torch.ones(100,2)
x0=torch.normal(2*data,1)
y0=torch.zeros(100)

x1=torch.normal(-2*data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1), ).type(torch.LongTensor)

x,y=torch.autograd.Variable(x),torch.autograd.Variable(y)

#plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100, lw=0)
#plt.show()

class Net(torch.nn.Module):
    def __init__(self,INPUT,LAYER,OUTPUT):
        super(Net,self).__init__()
        self.layer=torch.nn.Linear(INPUT,LAYER)
        self.output=torch.nn.Linear(LAYER,OUTPUT)
    
    def forward(self,x):
        x=self.layer(x)
        x=torch.nn.functional.relu(x)
        x=self.output(x)
        return x

net=Net(2,10,2)

print(net)


plt.ion()
plt.show()

strain=torch.optim.SGD(net.parameters(),lr=0.005)
loss=torch.nn.CrossEntropyLoss()

for i in range(200):
    y_pre=net(x)
    loss_val=loss(y_pre,y)

    strain.zero_grad()
    loss_val.backward()
    strain.step()
    if i%10==0:
        plt.cla()
        pre_y=torch.max(torch.nn.functional.softmax(y_pre),1)[1]
        pre=pre_y.data.numpy().squeeze()
        y_tar=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pre,s=100,lw=0)
        acc=sum(pre==y_tar)/200.0
        plt.text(1.5,-4,'Accuracy=%.2f'%acc)
        plt.pause(1)
plt.ioff()
plt.show()
