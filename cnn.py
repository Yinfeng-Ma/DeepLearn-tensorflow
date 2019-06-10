#coding:UTF-8
#导入模板
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

seed=23455
BATCH_SIZE=32

rng=np.random.RandomState(seed)
X=rng.rand(32,2)
Y=[[int(x0+x1<1)] for (x0,x1) in X]


print "X:\n",X
print "Y:\n",Y
#前向传播

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))


w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))


a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
#反向传播
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#参数优化

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print "w1:\n",sess.run(w1)
    print "w2:\n",sess.run(w2)
    print "\n"

    step=3000
    t=[]
    for i in range(step):
        start=(i*BATCH_SIZE)%32
        end=BATCH_SIZE+start
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%500==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print "After %d steps, loss on all data is %f"%(i,total_loss)

        print "\n"
        print "w1:\n",sess.run(w1)
        print "w2:\n",sess.run(w2)

        total_loss=sess.run(loss,feed_dict={x:X,y_:Y})

        t.append(total_loss)



plt.plot(t)
plt.show()

