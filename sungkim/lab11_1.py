import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data # MNIST 불러오기

batch_size = 100
keep_prob = tf.placeholder(dtype=tf.float32)

X = tf.placeholder(tf.float32, shape=[None, 28*28])
Y = tf.placeholder(tf.float32, shape=[None, 10])

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_ = tf.reshape(X, shape=[-1, 28, 28, 1])

# convolutional weight는 xavier보다 일반적인 방식이 더 좋은가?
w_conv1 = tf.get_variable("w_conv1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
b_conv1 = tf.Variable(tf.random_normal(shape=[32], dtype=tf.float32))
w_conv2 = tf.get_variable("w_conv2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
b_conv2 = tf.Variable(tf.random_normal(shape=[64], dtype=tf.float32))

w_fc1 = tf.get_variable("w_fc1", shape=[7*7*64, 256], initializer=tf.contrib.layers.xavier_initializer())
b_fc1 = tf.Variable(tf.random_normal(shape=[256], dtype=tf.float32))
w_fc2 = tf.get_variable("w_fc2", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b_fc2 = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32))

y_conv1 = tf.nn.relu(tf.nn.conv2d(X_, w_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1)
y_conv1 = tf.nn.max_pool(y_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
y_conv1 = tf.nn.dropout(y_conv1, keep_prob=keep_prob)

y_conv2 = tf.nn.relu(tf.nn.conv2d(y_conv1, w_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)
y_conv2 = tf.nn.max_pool(y_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
y_conv2 = tf.nn.dropout(y_conv2, keep_prob=keep_prob)

y_flat = tf.reshape(y_conv2, [-1, 7*7*64])
y_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(y_flat, w_fc1) + b_fc1), keep_prob=keep_prob)
y_ = tf.matmul(y_fc1, w_fc2) + b_fc2


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))

# 두 트레이닝 기법의 차이가 꽤 크게 나타났다. test기준 56% vs 90%
#train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
train = tf.train.AdamOptimizer(0.001).minimize(cost)

predict = tf.argmax(y_, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y, 1)), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss, result, acc = sess.run([train, cost, predict, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            avg_cost += loss
            if i == total_batch-1:
                print("%d번째 =====" % epoch)
                print("loss : ", avg_cost/total_batch)
                print("result : ", result)
                print("accuracy = ", acc)

    testAcc = sess.run([accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
    print("!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@")
    print("test accuracy = ", testAcc)