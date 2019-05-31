import tensorflow as tf
import numpy as np
import data_reader
import time
import math

#def lstm(x):
   
def rnn(x):
    n_input = 300
    n_steps=  97
    n_hidden = 128
    n_classes = 19
    
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)
    
    W_fw = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1))
    U_fw = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
    b_fw = tf.Variable(tf.truncated_normal([n_hidden], stddev=0.1))

    W_out = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
    b_out = tf.Variable(tf.truncated_normal([n_classes], stddev=0.1))

    h0_fw = tf.Variable(tf.truncated_normal([1, n_hidden], stddev=0.1))
    
    for i in range(n_steps):
        if i == 0:
            h_fw = tf.tanh(tf.matmul(x[0], W_fw) + tf.matmul(h0_fw, U_fw) + b_fw)
        else:
            h_fw = tf.tanh(tf.matmul(x[i], W_fw) + tf.matmul(h_fw, U_fw) + b_fw)
    
    pred = tf.nn.softmax(tf.matmul(h_fw, W_out) + b_out)   
    return pred
    
    
    
def train(SemData, num_epochs):
    batch_size = 50
    display_step = 100
    n_input = 300
    n_steps=  97
    n_hidden = 128
    n_classes = 19
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    step = tf.placeholder(tf.int32)
    
    Y_pred = rnn(X)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    learning_rate = 1e-4 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    saver = tf.train.Saver(max_to_keep=1)
    
    test_sentences, test_labels = (SemData.test.sentences, SemData.test.labels)
    test_sentences = test_sentences.reshape([2717, n_steps, n_input])
    with tf.Session() as sess:
        start = time.clock()
        model_file=tf.train.latest_checkpoint('model')
        if not model_file == None:
            saver.restore(sess, model_file)
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            batch_sentences, batch_labels = SemData.train.next_batch(batch_size)
            batch_sentences = batch_sentences.reshape([batch_size, n_steps, n_input])
            if (i+1) % display_step == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={X:batch_sentences, Y:batch_labels})
                test_accuracy = accuracy.eval(session=sess, feed_dict={X:test_sentences, Y:test_labels})
                print("step %5d, train_accuracy = %.4g %% test_accuracy = %.4g %%" % (i+1, train_accuracy*100, test_accuracy*100))
                saver.save(sess, "Model/rnn_model", global_step=i+1)
            train_step.run(session=sess, feed_dict={X:batch_sentences, Y:batch_labels, step:i})
        print("------------------------------------")
        print("training time: ", time.clock()-start, " s")
        test_accuracy = accuracy.eval(session=sess, feed_dict={X:test_sentences, Y:test_labels})
        print("final accuracy = %.4g %%" % (test_accuracy*100))
        
        
    
    
