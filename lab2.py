import tensorflow as tf
import numpy as np

x = [4, 7, 10, 2, 11, 1, 3, 19, 5, 14]

y = [50, 61, 72, 29, 75, 35, 51, 97, 54, 88]

new_x = [2,4,6,8,10,12,14,16,18,20]


in_x = tf.placeholder(dtype=tf.float32, shape=[None])
in_y = tf.placeholder(dtype=tf.float32, shape=[None])
tf_x = tf.Variable(x, dtype=tf.float32)
tf_y = tf.Variable(y, dtype=tf.float32)
tf_z = tf.Variable(new_x, dtype=tf.float32)

w_layer1 = tf.Variable(tf.truncated_normal(shape=[1], dtype=tf.float32))
b_layer1 = tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32))
w_layer2 = tf.Variable(tf.truncated_normal(shape=[1], dtype=tf.float32))
b_layer2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32))
w_out = tf.Variable(tf.truncated_normal(shape=[1], dtype=tf.float32))
b_out = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32))

layer1 = in_x * w_layer1 + b_layer1
inty = tf.cast(layer1, dtype=tf.int32)
#layer2 = tf.nn.relu(layer1 * w_layer2 + b_layer2)
#out = tf.nn.relu(layer2 * w_out + b_out)

real_cost = tf.reduce_mean(tf.square(layer1 - in_y))

#cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=in_y, logits=layer1)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(real_cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, loss, whatout, whatint = sess.run([train, real_cost, layer1, inty], feed_dict={in_x: x, in_y: y})
        if(i % 100 == 0):
            print(loss)
            print(whatout)
            print(whatint)
    result = sess.run(layer1, feed_dict={in_x:new_x})
    print(result)

    arr = []
    arr.append(int(input("입력:")))
    result2 = sess.run(inty, feed_dict={in_x: arr})
    print(result2)