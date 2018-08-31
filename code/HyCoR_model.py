import tensorflow as tf
from tensorflow.contrib import rnn


class HyCoR(object):
    def __init__(self, n_steps, n_input, n_classes, n_hidden, vocab_size, embedding_size, filter_sizes, num_feature_maps,rnn_out_window):
        
        self.input_x = tf.placeholder(tf.int32, [None, n_steps, n_input], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # for future model evolution to avoid padding
        self.seqlen = tf.placeholder(tf.float32 , name='sentences_lengths')
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.001)
         
        # define weights for seq to seq
        self.weights=[]
        for i in range(n_steps):
            self.weights.append(tf.Variable(tf.random_normal([2*n_hidden, n_classes])))
        
        self.biases =[]
        for i in range(n_steps):
            self.biases.append(tf.Variable(tf.random_normal([n_classes])))
            
       
        def cnn_step(conv_input_x, vocab_size, embedding_size, filter_sizes, num_feature_maps):
            
            # Reshape for input to convolution - shape [batch_size, sentence_length]
            conv_input_x = tf.reshape(conv_input_x,shape=[-1, int(conv_input_x.shape[2])])
            
            # Embedding layer   
            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                self.W = tf.Variable(tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),name='W',)
                    
                self.embedded_chars = tf.nn.embedding_lookup(self.W, conv_input_x)
                
                embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_feature_maps]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_feature_maps]), name='b')
                    conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1,1,1,1],padding='SAME',name='conv')
                    # max pool over all sentence embeddings
                    pool_max =tf.reduce_max(conv,1,True)
                    
                    # Apply nonlinearity
                    h = tf.nn.tanh(tf.nn.bias_add(pool_max, b), name='tanh')
                    
                    pooled_outputs.append(h)
            
            # Combine all pooled features
            num_feature_maps_total = num_feature_maps *len(pooled_outputs)*embedding_size
            
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_feature_maps_total])
            h_pool_flat = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            
            return h_pool_flat  
        
        # Create Convolution Sentence Embeddings     
        def cnn_steps(batch_x):
            _cnn_stack = []
            # get batch_x shapes
            sentences_length = int(batch_x.shape[1]) 
            op_length = int(batch_x.shape[2])
            # loop through batch_x sentences
            for i in range(sentences_length):
                # feed cnn with batch_x sentence
                slice_x = (tf.slice(batch_x, [0, i, 0], [-1, 1, op_length]))
                _cnn_stack.append(cnn_step(slice_x, vocab_size, embedding_size, filter_sizes, num_feature_maps))

            return tf.tuple(_cnn_stack)
        
        # Create Bi-direction LSTM network
        def bi_rnn(x, weights, biases,n_steps,_window):

            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            # Backward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            # Get lstm cell output
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
            except Exception: # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
            # provide rnn training information
            tf.summary.histogram('outputs', outputs)   
            
            # rnn window
            z = 1 if (int(n_steps*_window)==0) else int(n_steps*_window)
            
            window =z 
            
            # get forward path rnn window outputs
            forward_outputs = [tf.matmul(tf.slice(outputs[-1-i], [0, n_hidden], [-1, n_hidden]),tf.slice(weights[-1-i], [n_hidden, 0], [n_hidden, n_classes]))+ biases[-1-i] for i in range(window)] 
            # get backward path rnn window outputs
            backward_outputs = [tf.matmul(tf.slice(outputs[i], [0, n_hidden], [-1, n_hidden]),tf.slice(weights[i], [n_hidden, 0], [n_hidden, n_classes]))+ biases[i] for i in range(window)]
            
            regularizer = tf.nn.l2_loss(weights)
            
            _x = tf.concat(forward_outputs + backward_outputs,1)
            self.h2 = tf.Variable(tf.random_normal([int(_x.shape[1]), n_classes]))
            self.b2 = tf.Variable(tf.random_normal([n_classes]))
            
            # classical layer
            X = tf.add(tf.matmul(_x,self.h2), self.b2,name='classical_layer')
           
            return X, regularizer
  
  
        # Calculate prediction over current batch
        with tf.name_scope('prediction'):
            self.pred, regularizer = bi_rnn(cnn_steps(self.input_x), self.weights, self.biases,n_steps,rnn_out_window)
            self.logits = tf.argmax(tf.nn.softmax(self.pred),1)
        
        
        with tf.name_scope('loss'):
            # Define loss 
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels= self.input_y))
            self.loss =loss = tf.reduce_mean(self.loss + l2_loss * regularizer)
        
        with tf.name_scope('accuracy'):
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='num_correct')
        
        # provide accuracy information
        tf.summary.scalar('accuracy', self.accuracy)
        
        

    
