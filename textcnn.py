# refer to: https://blog.csdn.net/u012762419/article/details/79561441
import numpy as np
import tensorflow as tf


class Setting:
    """
    all parameters are defined and set.
    """
    def __init__(self, vocab_size=100000, embedding_size=128):
        self.model_name = 'TextCNN'
        self.embedding_size = embedding_size
        self.n_filter_kernal = 128
        self.filter_size = [2, 3, 4, 5]
        self.fc_hidden_size = 1024
        self.n_class = 2
        self.vocab_size = vocab_size
        self.max_sentence_length = 20


class TextCNN(Setting):
    def __init__(self, per_trained_word_vector=None):
        super(Setting, self).__init__()
        # Setting.__init__(self)
        self.n_total_conv_filter = self.n_filter_kernal * len(self.filter_size)

        """
        define the structure of model.
        """
        with tf.name_scope('inputs'):
            self._input_x = tf.placeholder(tf.int16, shape=[None, self.max_sentence_length], name='input_x')
            self._input_y = tf.placeholder(tf.float16, shape=[None, self.n_class], name='input_y')
            self._keep_drop_prob = tf.placeholder(tf.float16, name='drop_out_prob')

        with tf.variable_scope('Embedding'):
            if isinstance(per_trained_word_vector, np.ndarray):
                assert per_trained_word_vector.shape[1] == self.embedding_size, 'per_trained_word_vector should match the dimension of embedding.'
                self.embedding = tf.get_variable(name='embedding',
                                                 shape=per_trained_word_vector.shape,
                                                 initializer=tf.initializers(per_trained_word_vector),
                                                 trainable=True)
            else:
                self.embedding = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))

        # convolution and pool
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_x)
        inputs = tf.expand_dims(inputs, -1)  # why expand the dimension of inputs

        for n, current_filter in enumerate(self.filter_size):
            with tf.variable_scope('filter_size_%d' % current_filter):
                # convolution layout
                # the former two dimensions are the size of filter
                # the third dimension is the number of current channels
                # the last dimension is the depth of conv_filter
                filter_shape = [current_filter, self.embedding_size, 1, self.n_filter_kernal]
                weight = tf.Variable(initial_value=tf.truncated_normal_initializer(shape=filter_shape, stddev=0.1), name='w_'+current_filter)
                bias = tf.Variable(initial_value=tf.constant(0.1, shape=self.n_filter_kernal), name='b_'+current_filter)
                conv = tf.nn.conv2d(inputs, weight, strides=[1,1,1,1], padding='VALID')

                # activation
                h = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
                # max pooling
                pooled = tf.nn.max_pool(h, ksize=[1,])




