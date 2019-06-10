# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:25:25 2019

@author: T01
"""

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
   
    #w=tf.constant(w.T,dtype=tf.float32)
    s=h*(tf.reduce_sum(w.T*f,0))/3
        
   
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
    
    w=tf.constant(w.T,dtype=tf.float32)
    s=tf.matmul(w,f)
    #s=np.dot(w.T,f)
    return s


def ChenEn(f,x1,x2,n,g):
    x=np.linspace(x1,x2,n)
    dx=x[1]-x[0]
    x=x.reshape(len(x),1)
    x=tf.constant(x,dtype=tf.float32)
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

nx=401
INPUT=1
L1=2048
L2=1024
OUTPUT=1
EPOCH=8000
lr=5e-5
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
    y=(tf.matmul(y2,w3)+b3)
    

    
    
    return y

def backward(x):
    xx=tf.placeholder(tf.float32,[nx,INPUT])

    y=forward(xx)
    dx=x[1,0]-x[0,0]
    n1=Simpson(y*y,dx,nx)
    psi=y/tf.sqrt(n1)
    en,mu=ChenEn(psi,x1,x2,nx,g)
    
    
    
    gs = 0
    gslist = [1,1,2,3,10,20,40,100,200,10000]
    ic = 0
    learnrate = tf.Variable(lr, trainable=False)

    updatelearnrate = tf.assign(learnrate,tf.multiply(learnrate,0.5))
    
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
        plt.plot(x[:,0],psi0[:,0],'o',label='psi(x)')
        plt.xlabel('x')
        plt.legend()
        #plt.subplot(122)
        plt.figure()
        plt.plot(ener[2000:])
        plt.show()
        print(psi0[0,195:205])
        print(sess.run(Simpson(psi0**2,1.0/20,401)))
        print("En=",en0,"mu=",mu0)       
def main():
    x=np.linspace(x1,x2,nx)
    xx=x.reshape(len(x),1)
    #y=np.exp(-xx*xx/2)
    #en,ch=ChenEn(y,-10,10,nx,1.0)
    #print(en)
    backward(xx)
if __name__=="__main__":
    main()
            
