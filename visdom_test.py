#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# File Name: visdom_test.py
# Author: Ma-yinfeng
# mail: yfma@iphy.ac.cn
# Created Time: 2019-04-20

import torch
from visdom import Visdom
viz=Visdom()

x=torch.arange(1,10,0.01)
y=torch.sin(x)

viz.line(X=x,Y=y,win='sinx',opts={'title':'y=sin(x)'})

for i in range(10):
    x=torch.Tensor([i])
    y=x
    
    viz.line(X=x,Y=y,win='poly',update='append' if i>0 else None)
viz.images(torch.randn(3,64,64).numpy(),win='random2')
viz.images(torch.randn(36,1,64,64).numpy(),nrow=6,win='random3')

