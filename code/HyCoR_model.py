import tensorflow as tf
from tensorflow.contrib import rnn

class HyCoR(object):
    def __init__(self, n_steps, n_input, n_classes, n_hidden, vocab_size, embedding_size, filter_sizes, num_feature_maps, out_window_size):
        
        self.input_x = tf.placeholder(tf.int32, [None, n_steps, n_input], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # for future model evolution to avoid padding
        self.seqlen = tf.placeholder(tf.float32 , name='sentences_lengths')
        
        # keeping track of L2 regularization loss (optional)
        l2_loss = tf.constant(0.001)
         
        # define weights for bi-lstm seqs
        self.weights=[]
        for i in range(n_steps):
            self.weights.append(tf.Variable(tf.random_normal([2*n_hidden, n_classes])))
        
        self.biases =[]
        for i in range(n_steps):
            self.biases.append(tf.Variable(tf.random_normal([n_classes])))
        
        def cnn_step(conv_input_x, vocab_size, embedding_size, filter_sizes, num_feature_maps):
            
            # reshape for input to convolution - shape [batch_size, sentence_length]
            conv_input_x = tf.reshape(conv_input_x,shape=[-1, int(conv_input_x.shape[2])])
            
            # embedding layer   
            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                self.W = tf.Variable(tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),name='emb',)
                    
                self.embedded_chars = tf.nn.embedding_lookup(self.W, conv_input_x)
                
                embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    
                    # convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_feature_maps]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_b_k')
                    b = tf.Variable(tf.constant(0.1, shape=[num_feature_maps]), name='b_k')
                    conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1,1,1,1],padding='SAME',name='conv')
                    
                    # max pool over all sentence embeddings
                    pool_max =tf.reduce_max(conv,1,True)
                    
                    # apply nonlinearity
                    c = tf.nn.tanh(tf.nn.bias_add(pool_max, b), name='c_k')
                    
                    pooled_outputs.append(c)
            
            # combine all partial sentences embeddings
            num_feature_maps_total = num_feature_maps *len(pooled_outputs)*embedding_size
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_feature_maps_total])
            h_pool_flat = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            
            return h_pool_flat  
        
        # create global sentences embeddings     
        def cnn_steps(batch_x):
            _cnn_stack = []
            # get batch_x shapes
            sent_length = int(batch_x.shape[1]) 
            op_length = int(batch_x.shape[2])
            # loop through batch_x sentences
            for i in range(sent_length):
                # feed each cnn with a sentence
                slice_x = (tf.slice(batch_x, [0, i, 0], [-1, 1, op_length]))
                _cnn_stack.append(cnn_step(slice_x, vocab_size, embedding_size, filter_sizes, num_feature_maps))

            return tf.tuple(_cnn_stack)
        
        # ceate Bi-LSTM network
        def bi_lstm(x, weights, biases,n_steps,_window):

            # define lstm cells
            # forward direction cell
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            # backward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            # get lstm cell output
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
            except Exception: # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
            # provide rnn training information
            tf.summary.histogram('outputs', outputs)   
            
            # output window size
            window = 1 if (int(n_steps*_window)==0) else int(n_steps*_window)
            
            # get forward path of output window size
            forward_outputs = [tf.matmul(tf.slice(outputs[-1-i], [0, n_hidden], [-1, n_hidden]),tf.slice(weights[-1-i], [n_hidden, 0], [n_hidden, n_classes]))+ biases[-1-i] for i in range(window)] 
            # get backward path of output window size
            backward_outputs = [tf.matmul(tf.slice(outputs[i], [0, n_hidden], [-1, n_hidden]),tf.slice(weights[i], [n_hidden, 0], [n_hidden, n_classes]))+ biases[i] for i in range(window)]
            
            # add L2 regularization term at the output weights
            regularizer = tf.nn.l2_loss(weights)
            
            _x = tf.concat(forward_outputs + backward_outputs,1)
            self.h2 = tf.Variable(tf.random_normal([int(_x.shape[1]), n_classes]))
            self.b2 = tf.Variable(tf.random_normal([n_classes]))
            
            # classical layer
            X = tf.add(tf.matmul(_x,self.h2), self.b2,name='classical_layer')
           
            return X, regularizer
  
        # calculate prediction over current batch
        with tf.name_scope('prediction'):
            self.pred, regularizer = bi_lstm(cnn_steps(self.input_x), self.weights, self.biases,n_steps,out_window_size)
            self.logits = tf.argmax(tf.nn.softmax(self.pred),1)
        
        # calculate loss over batch (cross-entropy)
        with tf.name_scope('loss'):
            # define loss 
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels= self.input_y))
            self.loss =loss = tf.reduce_mean(self.loss + l2_loss * regularizer)
        
        # calculate acuracy over batch 
        with tf.name_scope('accuracy'):
            # evaluate model
            _pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(_pred, tf.float32), name='num_correct')
        
        # provide accuracy information
        tf.summary.scalar('accuracy', self.accuracy)
