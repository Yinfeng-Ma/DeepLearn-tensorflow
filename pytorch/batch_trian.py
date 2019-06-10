#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: batch_trian.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-05-01
import torch 
import torch.utils.data as Data

BATCH_SIZE=5

x=torch.linspace(0,10,10)
y=torch.linspace(10,1,10)

dataset=Data.TensorDataset(x,y)

loader=Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,)


for epoch in range(3):
    for step,(xs,ys) in enumerate(loader):
        print("Epoch %g step:%g,xs:%s,ys:%s"%(epoch,step,xs.numpy(),ys.numpy()))

