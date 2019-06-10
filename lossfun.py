#coding:UTF-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE=8;
seed=23455

cost=1
profit=9

rng=np.random.RandomState(seed)
X=rng.rand(32,2)
Y_=[[x1+x2+(rng.rand()/10.0-0.05)] for (x1,x2) in X]


x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
#w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#a=tf.matmul(x,w1)
y=tf.matmul(x,w1)

#loss_mse=tf.reduce_mean(tf.square(y_-y))
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),cost*(y-y_),profit*(y_-y)))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)



with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    step=20000
    for i in range(step):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%500==0:
            print "After %d train step, w1 is :" %(i)
            print sess.run(w1),"\n"

    print "Final w1 is :\n",sess.run(w1)
