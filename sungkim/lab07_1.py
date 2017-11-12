import tensorflow as tf

x_data = [[1,2,1], [1,3,2], [1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]]
y_data = [2, 2, 2, 1, 1, 1, 0, 0]

x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [2, 2, 2]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.int32, shape=[None])

Y_one = tf.one_hot(Y, 3, on_value=1.0, off_value=0.0)

w_1 = tf.Variable(tf.random_normal(shape=[3, 3], dtype=tf.float32))
b_1 = tf.Variable(tf.random_normal(shape=[3], dtype=tf.float32))

y_ = tf.matmul(X, w_1) + b_1

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one, logits=y_))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

predict = tf.argmax(y_, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(Y_one, 1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        _, loss, result, acc = sess.run([train, cost, predict, accuracy], feed_dict={X: x_data, Y: y_data})
        if i % 200 == 0:
            print("%d 번째 ======" % i)
            print("loss : ", loss)
            print("result : ", result)
            print("accuracy = ", acc)

    loss2, result2, acc2 = sess.run([cost, predict, accuracy], feed_dict={X: x_test, Y: y_test})
    print("testing======")
    print("loss : ", loss2)
    print("result : ", result2)
    print("accuracy = ", acc2)
