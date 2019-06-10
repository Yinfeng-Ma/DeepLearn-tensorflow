#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: batch_trian.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-05-01
import torch 
import torch.utils.data as Data
import matplotlib.pyplot as plt
BATCH_SIZE=32
LR=0.01
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))

dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,)

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
net_SDG=Net(1,20,1)
net_Mon=Net(1,20,1)
net_RMSp=Net(1,20,1)
net_Adam=Net(1,20,1)
nets=[net_SDG,net_Mon,net_RMSp,net_Adam]

opt_SDG=torch.optim.SGD(net_SDG.parameters(),lr=LR)
opt_Mom=torch.optim.SGD(net_Mon.parameters(),lr=LR,momentum=0.8)
opt_RMSp=torch.optim.RMSprop(net_RMSp.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SDG,opt_Mom,opt_RMSp,opt_Adam]

loss=torch.nn.MSELoss()
loss_val=[[],[],[],[]]

for epoch in range(EPOCH):
    for step,(xs,ys) in enumerate(loader):
        xi,yi=torch.autograd.Variable(xs),torch.autograd.Variable(ys)
        for net,opt,loss_l in zip(nets,optimizers,loss_val):
            out=net(xi)
            lossval=loss(out,yi)
            opt.zero_grad()
            lossval.backward()
            opt.step()
            loss_l.append(lossval.data)

        
labels=["SDG","Momentum","RMSprop","Adam"]
for i ,loss_l in enumerate(loss_val):
    plt.plot(loss_l,label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim([0,0.2])
plt.show()
