import tensorflow as tf

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None])


with tf.name_scope("layer1") as scope:
    w_1 = tf.Variable(tf.random_normal(shape=[2, 2], dtype=tf.float32), name='w_1')
    b_1 = tf.Variable(tf.zeros(shape=[2], dtype=tf.float32), name='b_1')
    tf.summary.histogram('w_1', w_1)
    tf.summary.histogram('b_1', b_1)
    y_1 = tf.nn.sigmoid(tf.matmul(X, w_1) + b_1)


with tf.name_scope("layer2") as scope:
    w_2 = tf.Variable(tf.random_normal(shape=[2, 1], dtype=tf.float32), name='w_2')
    b_2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32), name='b_2')
    tf.summary.histogram('w_2', w_2)
    tf.summary.histogram('b_2', b_2)
    y_2 = tf.reshape((tf.matmul(y_1, w_2) + b_2), shape=[-1])

with tf.name_scope("train_center") as scope:
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_2, labels=Y), name='loss')
    tf.summary.scalar('Loss', cost)
    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    predicted = tf.cast(y_2 > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, predicted), dtype=tf.float32), name='accuracy')
    tf.summary.scalar('Accuracy', accuracy)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./lab9_files/', sess.graph)

    for i in range(6001):
        if i % 300 == 0:
            _, loss, result, acc, s = sess.run([train, cost, predicted, accuracy, summary], feed_dict={X: x_data, Y: y_data})
            print("%d번째 ===============" % i)
            print("     loss : ", loss)
            print("     result : ", result)
            print("     accuracy = ", acc)
            writer.add_summary(s, i)
        else:
            _, s = sess.run([train, summary], feed_dict={X: x_data, Y: y_data})