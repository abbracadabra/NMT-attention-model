
import numpy as np

import tensorflow as tf



aa = tf.placeholder(dtype=tf.float32,shape=[1])
bb=aa+1
cc=bb+1

sess = tf.Session()
print(sess.run(cc,feed_dict={bb:[3]}))



