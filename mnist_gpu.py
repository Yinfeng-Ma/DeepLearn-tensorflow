#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: mnist_torch.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-20

import torch
from  torchvision import datasets,transforms


INPUT=784
LAYER=200
OUTPUT=10
LEARNNING_RATE=0.01
epochs=10
batch_size=200


train_loader=torch.utils.data.DataLoader(
        datasets.MNIST('../data',train=True,download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
                ])),
            batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(
        datasets.MNIST('../data',train=False,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
            ])),
        batch_size=batch_size,shuffle=True)

class MFC(torch.nn.Module):
    def __init__(self):
        super(MFC,self).__init__()

        self.model=torch.nn.Sequential(
                torch.nn.Linear(INPUT,LAYER),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(LAYER,LAYER),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(LAYER,OUTPUT),
                torch.nn.ReLU(inplace=True),
                )
    def forward(self,x):
        x=self.model(x)
        return x

device=torch.device('cuda:0')
net=MFC().to(device)
optimizer=torch.optim.SGD(net.parameters(),lr=LEARNNING_RATE)
criteon=torch.nn.CrossEntropyLoss().to(device)
iter_count=0
total_step=len(train_loader)
for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.view(-1,28*28)
        data,target=data.to(device),target.to(device)

        logits=net(data)
        loss=criteon(logits,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_count+=1;
        if batch_idx%100==0:
            print("Train Epoch:{}[{}/{}({:.0f}%)]\t loss:{:.6f}".format(epoch,batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
    test_loss=0
    correct=0
    for data,target in test_loader:
        data=data.view(-1,28*28)
        data,target=data.to(device),target.to(device)
        logits=net(data)
        test_loss+=criteon(logits,target).item()

        pred=logits.data.max(1)[1]
        correct+=pred.eq(target.data).sum()
    test_loss/=len(test_loader.dataset)
    print("\nTest set :Average loss :{:.4f},Accuracy:{}/{}({:.0f}%)\n".format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
        
#    print(data.cpu().view(-1,1,28,28))

