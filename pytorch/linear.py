#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: linear.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-05-01


import torch 
import matplotlib.pyplot as plt 

x=torch.unsqueeze(torch.linspace(-1,1,201),dim=1)

y=x.pow(2)+0.2*torch.rand(x.size())
x,y=torch.autograd.Variable(x),torch.autograd.Variable(y)

#plt.scatter(x.data.numpy(),y.data.numpy())
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

net=Net(1,10,1)
print(net)
plt.ion()
plt.show()

strain=torch.optim.SGD(net.parameters(),lr=0.5)
loss=torch.nn.MSELoss()
for i in range(201):
    pre_y=net(x)
    loss_val=loss(pre_y,y)

    strain.zero_grad()
    loss_val.backward()
    strain.step()
    if i%10==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),pre_y.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%0.4f'%loss_val.data)
        plt.pause(0.1)
plt.ioff()
plt.show()
        

