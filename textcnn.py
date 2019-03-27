# refer to: https://blog.csdn.net/u012762419/article/details/79561441
import numpy as np
import tensorflow as tf
import os
from datetime import datetime


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
        self.learning_rate = 0.001


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
        # why expand the dimension of inputs? Does it aim to add the channel dimension?
        inputs = tf.expand_dims(inputs, -1)
        pooled_result = []

        for n, current_filter in enumerate(self.filter_size):
            with tf.variable_scope('filter_size_%d' % current_filter):
                # convolution layout
                # the former two dimensions are the size of filter
                # the third dimension is the number of current channels
                # the last dimension is the depth of conv_filter
                filter_shape = [current_filter, self.embedding_size, 1, self.n_filter_kernal]
                weight = tf.Variable(initial_value=tf.truncated_normal_initializer(shape=filter_shape, stddev=0.1), name='w_'+current_filter)
                bias = tf.Variable(initial_value=tf.constant(0.1, shape=self.n_filter_kernal), name='b_'+current_filter)
                conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='VALID')

                # activation
                h = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
                # max pooling

                pooled = tf.nn.max_pool(h, ksize=[1, self.max_sentence_length-current_filter+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='max_pool')
                pooled_result.append(pooled)

        pooled_concat = tf.concat(pooled_result, axis=3)  # concat on 4th dimension
        self.pooled_flatten = tf.reshape(pooled_concat, shape=[-1, self.n_total_conv_filter], name='pooled_flatten')

        # add dropout
        with tf.name_scope('dropout'):
            # self.dropout = tf.nn.dropout(self.pooled_flatten, 0.5, name='dropout')
            self.dropout = tf.nn.dropout(self.pooled_flatten, self._keep_drop_prob, name='dropout')

        # output layout
        with tf.name_scope('output'):
            w_full = tf.Variable(initial_value=tf.truncated_normal_initializer(shape=[self.n_total_conv_filter, self.n_class], stddev=0.1), name='full_connection_weight')
            b_full = tf.Variable(initial_value=tf.constant(0.1, shape=[self.n_class]), name='full_connection_bias')
            self.scores = tf.nn.xw_plus_b(self.dropout, w_full, b_full, name='scores')
            print("self.scores : ", self.scores.get_shape())
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")  # predict label , the output
            print("self.predictions : ", self.predictions.get_shape())

    def train(self, x, y):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self._input_y, name='loss')
            loss_mean = tf.reduce_mean(loss)

        with tf.name_scope('accuracy'):
            predict = self.predictions
            label = tf.argmax(self._input_y, axis=1, name='label')
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

        timestamp = 'TextCNN' + datetime.now().strftime( '%Y-%m-%d %H:%M:%S')
        output_dir = os.path.abspath(os.path.join(os.path.curdir, 'model', timestamp))
        print('Writing to {}\n'.format(output_dir))

        global_step = tf.Variable(0, trainable=False)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize((loss_mean))

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            STEP = 5000
            for i in range(STEP):
                pass







