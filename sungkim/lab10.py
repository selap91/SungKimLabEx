import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data # MNIST 불러오기

batch_size = 100
keep_prob = tf.placeholder(dtype=tf.float32)

X = tf.placeholder(tf.float32, shape=[None, 28*28])
Y = tf.placeholder(tf.float32, shape=[None, 10])

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#w_1 = tf.Variable(tf.random_normal(shape=[784, 256], dtype=tf.float32))
w_1 = tf.get_variable("W", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer()) # Xavier initializer !
b_1 = tf.Variable(tf.random_normal(shape=[512], dtype=tf.float32))
#w_2 = tf.Variable(tf.random_normal(shape=[256, 256], dtype=tf.float32))
w_2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b_2 = tf.Variable(tf.random_normal(shape=[512], dtype=tf.float32))
#w_3 = tf.Variable(tf.random_normal(shape=[256, 10], dtype=tf.float32))
w_3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b_3 = tf.Variable(tf.random_normal(shape=[512], dtype=tf.float32))
w_4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b_4 = tf.Variable(tf.random_normal(shape=[512], dtype=tf.float32))
w_5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b_5 = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32))

#y_1 = tf.nn.relu(tf.matmul(X, w_1) + b_1)
y_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, w_1) + b_1), keep_prob=keep_prob)
#y_2 = tf.nn.relu(tf.matmul(y_1, w_2) + b_2)
y_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(y_1, w_2) + b_2), keep_prob=keep_prob)
y_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(y_2, w_3) + b_3), keep_prob=keep_prob)
y_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(y_3, w_4) + b_4), keep_prob=keep_prob)
#y_ = tf.matmul(y_4, w_5) + b_5
y_ = tf.matmul(y_4, w_5) + b_5

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