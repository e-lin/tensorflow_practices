#
# Linear regression with one variable by using Tensorflow.
# Author: e-lin
#

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('display_step', 50, 'Display logs per step.')


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# testing data
# test_X = np.asarray([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
# test_Y = np.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
test_X = np.asarray([6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764])
test_Y = np.asarray([17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483])


def run_training(train_X, train_Y):
    m = train_X.shape[0] # number of examples

    # weights
    W = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    # W = tf.Variable(np.random.randn(), name="weight")
    # b = tf.Variable(np.random.randn(), name="bias")

    # linear model
    activation = tf.add(tf.mul(X, W), b)
    cost = tf.reduce_sum(tf.square(activation - Y)) / (2*m)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(FLAGS.max_steps):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

                if step % FLAGS.display_step == 0:
                    print "Step:", "%04d" % (step+1), "Cost=", "{:.9f}".format(sess.run(cost, \
                        feed_dict={X: train_X, Y:train_Y})), "W=", sess.run(W), "b=", sess.run(b)

        print "Optimization Finished!"
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print "Training Cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

        print "Testing.... (L2 loss Comparison)"
        testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
        print "Testing Cost=", testing_cost
        print "Absolute L2 liss difference:", abs(training_cost - testing_cost)

        # Graphic display
        plt.plot(train_X, train_Y, 'ro', label = "Original data")
        plt.plot(test_X, test_Y, 'bo', label = "Testing data")
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label = "Fitted line")
        plt.legend()
        plt.show()


def read_data(filename, read_from_file = True):
    if read_from_file:
        with open(filename) as fd:
            data_list = fd.read().splitlines()

            m = len(data_list) # number of examples
            train_X = np.empty(m) * np.nan
            train_Y = np.empty(m) * np.nan
            # Best way to initialize and fill an numpy array?
            # http://stackoverflow.com/questions/22414152/best-way-to-initialize-and-fill-an-numpy-array

            for i in range(m):
                x, y = data_list[i].split(",")
                train_X[i] = float(x)
                train_Y[i] = float(y)
    else:
        # default training data
        train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
        train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

    return train_X, train_Y


import sys

def main(argv):
    if not argv:
        print "Enter data filename."
        sys.exit()

    filename = argv[1]

    train_X, train_Y = read_data(filename)
    run_training(train_X, train_Y)

if __name__ == '__main__':
    tf.app.run()
