import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data # MNIST 불러오기

X = tf.placeholder(tf.float32, shape=[None, 28*28])
Y = tf.placeholder(tf.float32, shape=[None, 10])

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

w_1 = tf.Variable(tf.random_normal(shape=[784, 10], dtype=tf.float32))
b_1 = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32))

y_ = tf.matmul(X, w_1) + b_1

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))

# 두 트레이닝 기법의 차이가 꽤 크게 나타났다. test기준 56% vs 90%
#train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
train = tf.train.AdamOptimizer(0.01).minimize(cost)

predict = tf.argmax(y_, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y, 1)), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, loss, result, acc = sess.run([train, cost, predict, accuracy], feed_dict={X: batch_x, Y: batch_y})
        if i % 200 == 0:
            print("%d번째 =====" % i)
            print("loss : ", loss)
            print("result : ", result)
            print("accuracy = ", acc)

    testAcc = sess.run([accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("test accuracy = ", testAcc)