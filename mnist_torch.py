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


w1,b1=torch.randn(INPUT,LAYER,requires_grad=True),torch.randn(LAYER,requires_grad=True)
w2,b2=torch.randn(LAYER,LAYER,requires_grad=True),torch.randn(LAYER,requires_grad=True)
w3,b3=torch.randn(LAYER,OUTPUT,requires_grad=True),torch.randn(OUTPUT,requires_grad=True)


torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    a1=torch.matmul(x,w1)+b1
    z1=torch.nn.functional.relu(a1)

    a2=torch.matmul(z1,w2)+b2
    z2=torch.nn.functional.relu(a2)

    a3=torch.matmul(z2,w3)+b3
    y=torch.nn.functional.relu(a3)
    return y
optimizer=torch.optim.SGD([w1,b1,w2,b2,w3,b3],lr=LEARNNING_RATE)
criteon=torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    for batch_idx,(data,targe) in enumerate(train_loader):
        data=data.view(-1,28*28)

        logits=forward(data)
        loss=criteon(logits,targe)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%100==0:
            print("Train Epoch:{}[{}/{}({:.0f}%)]\t loss:{:.6f}".format(epoch,batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
    test_loss=0
    correct=0
    for data,target in test_loader:
        data=data.view(-1,28*28)
        logits=forward(data)
        test_loss+=criteon(logits,target).item()

        pred=logits.data.max(1)[1]
        correct+=pred.eq(target.data).sum()
    test_loss/=len(test_loader.dataset)
    print("\nTest set :Average loss :{:.4f},Accuracy:{}/{}({:.0f}%)\n".format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))


