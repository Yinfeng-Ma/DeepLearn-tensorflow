#!/usr/bin/env python

#-*-coding:UTF-8 -*-
"""
author=mayin
time=19-04-18
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

for i in range(len(x)):
    if y[i]==0:
        plt.plot(x[i][0],x[i][1],'r+')
    else:
        plt.plot(x[i][0],x[i][1],'ko')

plt.show()

INPUT=2
LAYER=5
OUTPUT=1
LEARNNING_RATE=0.01
STEPS=1000
def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def get_bais(shape):
    return tf.Variable(tf.zeros(shape))


def forward(x):
    w1=get_weight([INPUT,LAYER])
    b1=get_bais([LAYER])
    z1=tf.matmul(x,w1)+b1
    y1=tf.nn.relu(z1)
    
    w2=get_weight([LAYER,OUTPUT])
    b2=get_bais([OUTPUT])
    z2=tf.matmul(y1,w2)+b2
    y=tf.nn.relu(z2)
    return y

def backward(xdata,ylabes):
    x=tf.placeholder(tf.float32,[None,INPUT])
    y_=tf.placeholder(tf.float32,[None, OUTPUT])
    xs=tf.convert_to_tensor(xdata,tf.float32)
    y=forward(xs)
    loss=tf.reduce_mean(tf.square(y-y_))
#    train_step=tf.train.GradientDescentOptimizer(LEARNNING_RATE).minimize(loss)
    train_step=tf.train.AdamOptimizer(LEARNNING_RATE).minimize(loss)
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            sess.run(train_step,feed_dict={x:xdata,y_:ylabes})
            if i%100==0:
                loss_val=sess.run(loss,feed_dict={x:xdata,y_:ylabes})
                print("After %d step train, loss is %g"%(i,loss_val))
        print(sess.run(y,{x:xdata,y_:ylabes}))
def main():
    backward(x,y)
if __name__=="__main__":
    main()


