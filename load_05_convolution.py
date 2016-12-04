import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

test_size = 256


#Step 1 - Get Input Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# X = tf.placeholder("float", [None, 28, 28, 1], name='X')
# graph_def = tf.GraphDef()

# with open("bottleneck/train.pb", "rb") as f:
#     graph_def.ParseFromString(f.read())

# Unpersists graph from file
with tf.gfile.FastGFile('bottleneck/train.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predict_op = sess.graph.get_tensor_by_name('predict_op:0')
    # new_saver = tf.train.import_meta_graph('tich_chap.meta')
    # new_saver.restore(sess, "tich_chap")
    # all_vars = tf.trainable_variables()
    # for v in all_vars:
    #     print(sess.run(v))

    # print(tf.get_collection_ref('predict_op'))

    #predict_op = tf.get_collection("predict_op")[0]

    # graph_def = tf.GraphDef()
    # X, p_keep_conv, p_keep_hidden = (
    #           tf.import_graph_def(graph_def, name='', return_elements=[
    #               'X:0', 'p_keep_conv:0',
    #               'p_keep_hidden:0']))

    for i in range(2):
        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={"X:0": teX[test_indices],
                                                         "p_keep_conv:0": 1.0,
                                                         "p_keep_hidden:0": 1.0})))

    