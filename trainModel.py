import tensorflow as tf
import glob
import cv2
import random
import numpy as np
import os
from time import time
import pickle
import sys

def recup(folder, num):
  X_train = np.load('./'+ folder + '/Xtrain_'+str(num)+'.npy')
  X_test = np.load('./'+ folder + '/Xtest_'+str(num)+'.npy')
  y_test = np.load('./'+ folder + '/Ytest_'+str(num)+'.npy')
  y_train = np.load('./'+ folder + '/Ytrain_'+str(num)+'.npy')
  X_testClass = np.load('./'+ folder + '/XtestClass_'+str(num)+'.npy')
  y_testClass = np.load('./'+ folder + '/YtestClass_'+str(num)+'.npy')
  return X_train, y_train, X_test, y_test, X_testClass, y_testClass

def recupTest(folder, num):
  X_test = np.load('./'+ folder + '/Xtest_'+str(num)+'.npy')
  y_test = np.load('./'+ folder + '/Ytest_'+str(num)+'.npy')
  X_testClass = np.load('./'+ folder + '/XtestClass_'+str(num)+'.npy')
  y_testClass = np.load('./'+ folder + '/YtestClass_'+str(num)+'.npy')
  return X_test, y_test, X_testClass, y_testClass

def recupTrain(folder, num):
  X_train = np.load('./'+ folder + '/Xtrain_'+str(num)+'.npy')
  y_train = np.load('./'+ folder + '/Ytrain_'+str(num)+'.npy')
  return X_train, y_train

def new_weights_conv(name,shape):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer_conv2d())

def new_weights_fc(name,shape):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer())
       
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32)

def new_conv_layer(name,input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,    # Width and height of each filter.
                   num_filters,    # Number of filters.
                   dropout,    # Dropout rate
                   use_pooling=True): # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights_conv(name,shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    layer_drop = tf.nn.dropout(layer, dropout, name=name+'_dropout')
    return layer_drop, weights
  
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(name,input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs, use_nonlinear):
    weights = new_weights_fc(name,[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_nonlinear:
      layer = tf.nn.relu(layer)

    return layer, weights


X_test, y_test, X_testClass, y_testClass = recupTest('dataTrain2',0)
# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 64
num_filters2 = 96
num_filters3 = 128

n_images = 10
n_classes = 27
batch_size = 64
imgSize = 64

keep_prob = tf.placeholder(tf.float32, shape=[], name='dropRate')
x = tf.placeholder(tf.float32, [None, imgSize, imgSize, n_images], name='input_x')
x_image = tf.reshape(x, [-1, imgSize, imgSize, n_images], name='input_x_image')
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

layer_conv1a, weights_conv1a = \
    new_conv_layer("conv1a",input=x_image,
                   num_input_channels=n_images,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1a1, weights_conv1a1 = \
    new_conv_layer("conv1a1",input=layer_conv1a,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1a2, weights_conv1a2 = \
    new_conv_layer("conv1a2",input=layer_conv1a1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   dropout=keep_prob,
                   use_pooling=True)

layer_conv1b, weights_conv1b = \
    new_conv_layer("conv1b",input=layer_conv1a2,
                   num_input_channels=num_filters1,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1b1, weights_conv1b1 = \
    new_conv_layer("conv1b1",input=layer_conv1b,
                   num_input_channels=num_filters2,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1b2, weights_conv1b2 = \
    new_conv_layer("conv1b2",input=layer_conv1b1,
                   num_input_channels=num_filters2,
                   filter_size=filter_size1,
                   num_filters=num_filters2,
                   dropout=keep_prob,
                   use_pooling=True)

layer_conv1c, weights_conv1c = \
    new_conv_layer("conv1c",input=layer_conv1b2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size1,
                   num_filters=num_filters3,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1c1, weights_conv1c1 = \
    new_conv_layer("conv1c1",input=layer_conv1c,
                   num_input_channels=num_filters3,
                   filter_size=filter_size1,
                   num_filters=num_filters3,
                   dropout=keep_prob,
                   use_pooling=False)

layer_conv1c2, weights_conv1c2 = \
    new_conv_layer("conv1c2",input=layer_conv1c1,
                   num_input_channels=num_filters3,
                   filter_size=filter_size1,
                   num_filters=num_filters3,
                   dropout=keep_prob,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv1c2)

layer_f, weights_f = new_fc_layer("fc",input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=n_classes,
                         use_nonlinear=False)

y_pred = tf.nn.softmax(layer_f, name='output_class')
y_pred_cls = tf.argmax(y_pred, dimension=1, name='output_pred')
get_test = tf.argmax(y_test,dimension=1)

print(layer_conv1a)
print(layer_flat)
print(layer_f)

rate = tf.placeholder(tf.float32, shape=[])
l_rate = 0.00003#5e-4
drop_rate = 0.8
beta = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_f,labels=y)) \
     + beta * (tf.nn.l2_loss(weights_f))

optimizer = tf.train.AdamOptimizer(rate).minimize(cost)

correct = tf.equal(tf.argmax(layer_f, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

saver = tf.train.Saver()
save_dir = 'final_model-10/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')

hm_epochs = 150
t = time()
compteur = 0
prec = 10e100

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess=sess, save_path=save_path)
  res, epoch = [0 for x in range(n_classes)], 0
  while epoch < hm_epochs and sum(res)/len(res) < 0.99:
    epoch_loss = 0
    epoch += 1
    for name in [0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000,105000,110000,115000]:
      X_train, y_train = recupTrain('dataTrain2', name)
      for g in range(0,len(X_train),batch_size):
        _, c = sess.run([optimizer, cost], feed_dict={rate: l_rate, x: X_train[g:g+batch_size], y: y_train[g:g+batch_size], keep_prob: drop_rate})
        
        sys.stdout.write('\r' + str(g) + '/' + str(len(X_train)))
        sys.stdout.flush()
        epoch_loss += c

      tempsEcoule = time() - t

      sys.stdout.write('\rEpoch : ' + str(epoch) + '.' + str(name) + ' Loss : ' + str(epoch_loss) + ' Batch size : ' + str(batch_size) \
         + ' LRate : ' + str(l_rate) + ' DropRate : ' + str(drop_rate) + ' Time : ' + str(tempsEcoule))
      res2 = accuracy.eval({x:X_train[:batch_size], y:y_train[:batch_size], keep_prob: 1})
      res3 = accuracy.eval({x:X_test[:batch_size], y:y_test[:batch_size], keep_prob: 1})
      
      for no in range(n_classes):
        res[no] = accuracy.eval({x:X_testClass[no][:batch_size], y:y_testClass[no][:batch_size], keep_prob: 1})
      sys.stdout.write('\nTrain : ' + str(res2) + ' Test : ' + str(res3))
      for no in range(n_classes):
        sys.stdout.write(' Test class' + str(no) + ' : ' + str(res[no]))
      sys.stdout.write('\n')
      t = time()

    if epoch_loss > prec:
      compteur += 1
    else:
      if compteur > 0:
        compteur -= 1
      prec = epoch_loss
      saver.save(sess=sess, save_path=save_path)
    if compteur >= 2:
      compteur = 0
      l_rate /= 1.5
      drop_rate -= 0.05
    #batch_size = int(batch_size*1.5)

  res2, res = 0, 0
  for g in range(0,len(X_train),batch_size):
      res2 += accuracy.eval({x:X_train[g:g+batch_size], y:y_train[g:g+batch_size], keep_prob: 1})
  res2 /= (g/batch_size) + 1
  for g in range(0,len(X_test),batch_size):
      res += accuracy.eval({x:X_test[g:g+batch_size], y:y_test[g:g+batch_size], keep_prob: 1})
  res /= (g/batch_size) + 1
print('Epoch', epoch,'loss :',epoch_loss,'train :',res2,'test :', res)

