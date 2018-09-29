import numpy as np
import tensorflow as tf

if __name__ == '__main__':
	a = np.reshape(np.arange(24), (3, 4, 2))
	print("a:")
	print(a)
	with tf.Session() as sess:
		a_t = tf.constant(a)
		nonzero_indices = tf.where(tf.not_equal(a_t, 0))  # Find indices where the tensor is not zero
		# Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
		nonzero_values = tf.gather_nd(a_t, nonzero_indices)
		shape = a_t.get_shape()
		sparse = tf.SparseTensor(nonzero_indices, nonzero_values, shape)
		dense = tf.sparse_tensor_to_dense(sparse)
		b = sess.run(dense)
		print("nonzero_indices: {}".format(sess.run(nonzero_indices)))
		print("nonzero_values: {}".format(sess.run(nonzero_values)))
		print("shape: {}".format(shape))
	result = np.all(a == b)
	print("result == {}".format(result))
