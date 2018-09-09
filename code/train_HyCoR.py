import tensorflow as tf 
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import json
os.chdir(os.environ['USERPROFILE'] +'\\Downloads\\HyCoR-master\\code')

from learn_metrics import calcMetric
from HyCoR_model import HyCoR
import data_helper
from sklearn.model_selection import train_test_split

# load the model's parameters
training_config = 'training_config.json'
params = json.loads(open(training_config).read())

dyn_dropout, _ = calcMetric.calcDropout(0,0,params['dropout_keep_prob'],0)

# Set the filename to train the model 
filename = 'PG-[123]b.xlsx' 

print('train file:', (filename[0:len(filename)-5]))

# set the number of classes to train
n_classes= 2

# remove stop words from dataset
rmv_stop_wrds = False
 
# set max or avg value (sets how the dataset will be exloited) 
input_base ='avg'
 
# preprocess dataset to calculate avg/max values
_oplen,_seqlen=data_helper._run_sentence_document_mode_pre_stage(filename, rmv_stop_wrds,n_classes,input_base)

print(input_base + ' num of Sentences/Opinion | num of Words/Sentence: ' + str(_oplen) + ' | ' + str(_seqlen))

# preprocess the dataset
x_,y,sentence_size,opinion_size,seqlengths,vocab_size = data_helper._run_sentence_document_mode(filename,_seqlen,_oplen,rmv_stop_wrds,n_classes) 

# monitor test accuracy for every experiment iteration
metric_list=[]

# set the number of experiments 
n_experiments = 5

for i in range(n_experiments):     
    
    #  convert labels to one-hot vector
    y_ = np.eye(int(np.max(y) + 1))[np.int32(y)]
    
    print('creating train/test datasets...')
    # set train - test datasets
    x_train,x_test,y_train,y_test,seqlen_train,seqlen_test = train_test_split(x_,y_,seqlengths,test_size=0.2)
    
    # split train to train/dev
    x_train, x_dev,y_train,y_dev,seqlen_train,seqlen_dev =train_test_split(x_train,y_train,seqlen_train,test_size=0.1)
    
    # transform to numpy arrays
    x_train,x_dev,x_test, seqlen_train = np.asarray(x_train),np.asarray(x_dev), np.asarray(x_test), np.array(seqlen_train)
    
    y_train,y_dev, y_test, seqlen_test = np.asarray(y_train),np.asarray(y_dev), np.asarray(y_test), np.array(seqlen_test)
    
    print('dataset: ' + str(len(x_))  + ' train/dev/test ' + str(len(x_train)) + '/' +str(len(x_dev)) +'/' + str(len(x_test)))
    
    # load batch size  
    batch_size = params['batch_size']
    
    # set this value between [0.05,1] to change the iteration value (i.e. for small datasets ~ 0.1:0.4 for big datasets ~ 0.5:0.7)
    iter_norm_factor = 0.5
    # calculate training iterations
    training_iters = int(params['n_epochs']*(1/iter_norm_factor) * (int(len(x_train))/params['batch_size']))
    
    print()
    print('Model Parameters')
    print('-------------------')
    print('training classes: ' + str(n_classes))
    print('n_hidden: ' + str(params['n_hidden']))
    print('embedding_size: ' + str(params['embedding_size']))
    print('base_dropout: ' + str(params['dropout_keep_prob']))
    print('filter_sizes: ' + str(params['filter_sizes']))
    print('num_feature_maps: ' + str(params['num_feature_maps']))
    print('rnn_out_window: ' + str(params['rnn_out_window']))
    print('n_epochs: ' + str(params['n_epochs']))
    print('batch_size: ' + str(params['batch_size']))
    print('-------------------')
    print()
    print('training iterations: ' + str(training_iters))
    print('training the HyCoR model...')
   
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            hycor = HyCoR(
                n_steps=x_train.shape[1],
                n_input=x_train.shape[2],
                n_classes = y_train.shape[1],
                n_hidden=params['n_hidden'],
                vocab_size=vocab_size,
                embedding_size=params['embedding_size'],
                filter_sizes=[int(i) for i in params['filter_sizes'].split(',')],
                num_feature_maps = params['num_feature_maps'],
                rnn_out_window=params['rnn_out_window'])
            
            # set model's optimizer
            optimizer = tf.train.AdamOptimizer( learning_rate=params['learning_rate']).minimize(hycor.loss)
            # run and train the model
            sess.run(tf.global_variables_initializer())
            step = 1
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/train', graph=tf.get_default_graph())
            dev_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/dev',
                                            graph=tf.get_default_graph())
            
            # initiate train/dev accuracies
            acc=0
            _acc=0
            _count=0
            
            # Keep training until reach max iterations
            while step <= training_iters:
               
                # get train batch
                batch_x, batch_y, batch_lengths =  data_helper.next_batch(batch_size, x_train,y_train, seqlen_train, True)
                # monitor training accuracy information
                summary,_ = sess.run([merged,optimizer], feed_dict={hycor.input_x: batch_x, hycor.input_y: batch_y,hycor.seqlen: batch_lengths, hycor.dropout_keep_prob: dyn_dropout})
                
                # Add to summaries
                train_writer.add_summary(summary, step)
    
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={hycor.input_x: batch_x, hycor.input_y: batch_y, hycor.seqlen: batch_lengths,  hycor.dropout_keep_prob: dyn_dropout})
                
                # monitor train accuracy information in python window
                if step % params['display_step'] == 0:
                    
                    # Calculate batch accuracy and print
                    acc = sess.run(hycor.accuracy, feed_dict={hycor.input_x: batch_x, hycor.input_y: batch_y ,hycor.seqlen: batch_lengths,  hycor.dropout_keep_prob: 1.0})
                    
                    # Calculate batch loss
                    loss = sess.run(hycor.loss, feed_dict={hycor.input_x: batch_x, hycor.input_y: batch_y ,hycor.seqlen: batch_lengths, hycor.dropout_keep_prob: dyn_dropout})
    
                    print("Iter " + str(step) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Accuracy= " + \
                        "{:.5f}".format(acc) + ", Dropout= " +\
                        "{:.2f}".format(dyn_dropout))  
                
                # monitor dev accuracy information
                if step % 5 == 0:
                    # get dev batch
                    batch_x, batch_y, batch_lengths  =  data_helper.next_batch((batch_size, int(x_dev.shape[0]) )[batch_size > int(x_dev.shape[0])], x_dev, y_dev, seqlen_dev, True)
                    
                    summary, _acc = sess.run([merged,hycor.accuracy], feed_dict={hycor.input_x: batch_x, hycor.input_y: batch_y,hycor.seqlen: batch_lengths,  hycor.dropout_keep_prob: 1.0})
                    
                    #calculate new dropout
                    dyn_dropout, _count = calcMetric.calcDropout(acc,_acc,params['dropout_keep_prob'],_count)
                    
                    # monitor for early stop
                    if _count > 500: 
                        print('Overfit Identidied')
                        step = training_iters
                    #print('dropout: ' + "{:.2f}".format(dyn_dropout))
                    
                    dev_writer.add_summary(summary, step )
        
                step += 1
                
            print("Optimization Finished!")
            # Calculate accuracy for test dataset
            test_len = int(x_test.shape[0])
            test_data = x_test[:test_len]
            test_label = y_test[:test_len]
            test_seqs = seqlen_test[:test_len]
            
            print("Overall Testing Accuracy:", sess.run(hycor.accuracy, feed_dict={hycor.input_x: test_data, hycor.input_y: test_label,hycor.seqlen: test_seqs, hycor.dropout_keep_prob: 1.0}))
            
            # get actual labels
            actual = np.array([np.where(r==1)[0][0] for r in test_label])
            predicted = hycor.logits.eval(feed_dict={hycor.input_x: test_data, hycor.dropout_keep_prob: 1.0})
            print('Confusion Matrix: (H:labels, V:Predictions)')
            cm = tf.confusion_matrix(actual,predicted,num_classes=y_train.shape[1])
            # get confusion matrix values
            var_cm = sess.run(cm)
            print(var_cm)
            accuracy = np.sum([var_cm[i,i] for i in range(var_cm.shape[1])])/np.sum(var_cm)
            # normalize confusion matrixA
            print('Precision | Recall | Fscore')
            if(y_train.shape[1]==2):
                print(calcMetric.pre_rec_fs2(var_cm))
            elif (y_train.shape[1]==3):
                print(calcMetric.pre_rec_fs3(var_cm))
            elif (y_train.shape[1]==4):
                print(calcMetric.pre_rec_fs4(var_cm))
            elif (y_train.shape[1]==5):
                print(calcMetric.pre_rec_fs5(var_cm))
            elif (y_train.shape[1]==6):
                print(calcMetric.pre_rec_fs6(var_cm))
                 
            metric_list.append(accuracy)

print('the acuracies per experiment')
print(metric_list)
            
            # open cmd window and type the script to run and monitor training in tensorboard   
            # tensorboard --logdir=/tmp/tensorflowlogs   


    
    
    
