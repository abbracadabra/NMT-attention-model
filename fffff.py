import tensorflow as tf
import numpy as np

aa = tf.losses.softmax_cross_entropy(np.zeros([9,1,2]),np.ones([9,1,2]))

sess = tf.Session()
print(sess.run(aa))

