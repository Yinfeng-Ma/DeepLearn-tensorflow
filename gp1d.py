# -*- coding: utf-8 -*-
"""
Created on Sat May  4 22:28:39 2019

@author: T01
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def Simpson(f,h,n):
    #n=f.shape.as_list()[0]
    w=np.zeros([1,n])
    for i in range(1,(n+1)//2):
        w[0,2*i-1]=4.0
        w[0,2*i]=2.0
    w[0,0]=1.0
    w[0,n-1]=1.0
   
    w=tf.constant(w,dtype=tf.float32)
    s=h*(tf.reduce_sum(w*f,1))/3
        
   
    return s


def diffy(f,dx,n):
    w=np.zeros([n,n])
    for i in range(2,n-2):
        w[i-2,i]=1/(12*dx)
        w[i-1,i]=-8/(12*dx)
        w[i+1,i]=8/(12*dx)
        w[i+2,i]=-1/(12*dx)
    w[0,0]=-1/dx
    w[1,0]=1/dx
    w[0,1]=-0.5/dx
    w[2,1]=0.5/dx
    
    w[n-2,n-1]=-1/dx
    w[n-1,n-1]=1/dx
    w[n-3,n-2]=-0.5/dx
    w[n-1,n-2]=0.5/dx
    
    w=tf.constant(w,dtype=tf.float32)
    s=tf.matmul(f,w)
    return s


def ChenEn(f,x1,x2,n,g):
    x=np.linspace(x1,x2,n)
    dx=x[1]-x[0]
    t=0.5*diffy(f,dx,n)**2
    v=0.5*(x*f)**2
    vi=0.5*g*f**4
    den=t+v+vi
    dmu=den+vi
    en=Simpson(den,dx,n)
    mu=Simpson(dmu,dx,n)
    return en,mu


def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def get_bias(shape):
    return tf.Variable(tf.zeros(shape))


INPUT=401
L1=2048
L2=1024
OUTPUT=401
EPOCH=3000
lr=1e-4
x1=-10.0
x2=10.0
g=1.0

def forward(x):
    w1=get_weight([INPUT,L1])
    b1=get_bias([L1])
    a1=tf.matmul(x,w1)+b1
    y1=tf.nn.softplus(a1)
    
    
    
    w2=get_weight([L1,L2])
    b2=get_bias([L2])
    a2=tf.matmul(y1,w2)+b2
    y2=tf.nn.relu(a2)
    
    w3=get_weight([L2,OUTPUT])
    b3=get_bias([OUTPUT])
    #y=(tf.matmul(y2,w3)+b3)
    
    y=tf.nn.softplus(tf.matmul(y2,w3)+b3)
    w4=get_weight([OUTPUT,OUTPUT])
    b4=get_bias([OUTPUT])
    y=tf.matmul(y,w4)+b4
    
    
    return y

def backward(x):
    xx=tf.placeholder(tf.float32,[None,INPUT])

    y=forward(xx)
    dx=x[0,1]-x[0,0]
    n1=Simpson(y*y,dx,INPUT)
    psi=y/tf.sqrt(n1)
    en,mu=ChenEn(psi,x1,x2,INPUT,g)
    
    
    
    gs = 0
    gslist = [1,1,2,3,10,20,40,100,200,10000]
    ic = 0
    learnrate = tf.Variable(lr, trainable=False)

    updatelearnrate = tf.assign(learnrate,tf.multiply(learnrate,0.75))
    
    train_step=tf.train.AdamOptimizer(learnrate).minimize(en)
    #train_step=tf.train.GradientDescentOptimizer(learnrate).minimize(en)
    
    init=tf.global_variables_initializer()
    ener=[]
    with tf.Session() as sess:
        sess.run(init)
        print("STSRT")
        for i in range(EPOCH):
            if i % 150 == 0:
                if ic == gslist[gs]:
                    gs = gs + 1
                    ic = 1
                    sess.run(updatelearnrate)
                else:
                    ic = ic + 1
            _,psi0=sess.run([train_step,psi],{xx:x})
            en0,mu0=sess.run([en,mu],{xx:x})
            ener.append(en0)
            if i%100==0:
               print("After %d step trains, ground state energy is %g"%(i,en0))
        #plt.subplot(121)
        plt.plot(x[0,:],psi0[0,:],'o',label='psi(x)')
        plt.xlabel('x')
        plt.legend()
        #plt.subplot(122)
        plt.figure()
        plt.plot(ener[2000:])
        plt.show()
        print(psi0[0,195:205])
        print(sess.run(Simpson(psi0[0,:]**2,1.0/20,401)))
        print("En=",en0,"mu=",mu0)       
def main():
    x=np.linspace(x1,x2,INPUT)
    xx=x.reshape(-1,len(x))
#    xx=tf.constant(xx,tf.float32)
   # x=x.reshape(-1,len(x))
  
    backward(xx)
if __name__=="__main__":
    main()
            