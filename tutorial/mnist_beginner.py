from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

# we are going to learn W and b, just initialize them to zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# implement the softmax regression model
y = tf.nn.softmax(tf.matmul(x, W) + b)   # tf.matmul(x, W) = Wx

########## Training ##########
# define what it means for the model to be good\bad. We use cross-entropy here

# correct answer
y_ = tf.placeholder(tf.float32, [None, 10])

# cross=entropy
cross_entropy = -tf.reduce_sum( y_ * tf.log(y) )

# tensorflow will automatically use backpropagation algo. to determine how
# your varialbes affect the cost you ask it minimize.

# we ask tensorflow to minimize cross_entropy using the gradient descent algo.
# with learning rate of 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# init variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# train for 1000 times
for i in range(1000):
	# get a "batch" of one hundred random data points from our training set
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


########## Evaluating ##########

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# This should be about 91%


