import tensorflow as tf
from time import time
import numpy as np
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

def new_weights_fc(name,shape):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
           initializer=tf.contrib.layers.xavier_initializer())
       
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32)

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

X_test, y_test, X_testClass, y_testClass = recupTest('dataTrain',0)

imgSize = 64
n_classes = 15
batch_size = 256


hm_epochs = 150
t = time()
compteur = 0
prec = 10e100

with tf.Session() as sess:
  
  new_saver = tf.train.import_meta_graph('./final_model_128_2/best_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./final_model_128_2/'))

  graph = tf.get_default_graph()
  """for op in tf.get_default_graph().get_operations():
    print(str(op.name))
  df()"""
  layer_conv1c1 = graph.get_tensor_by_name("conv1c1_dropout/mul:0")
  x = graph.get_tensor_by_name("input_x:0")
  y = graph.get_tensor_by_name("Placeholder:0")
  keep_prob = graph.get_tensor_by_name("Placeholder_1:0")
  layer_conv1c1 = tf.stop_gradient(layer_conv1c1) # It's an identity function
  layer_flat, num_features = flatten_layer(layer_conv1c1)
  print(layer_flat, num_features)
  layer_f, weights_f = new_fc_layer("fc",input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=n_classes,
                         use_nonlinear=False)
  pred = tf.nn.softmax(layer_f)
  print(layer_f)
  rate = tf.placeholder(tf.float32, shape=[])
  l_rate = 0.0001#5e-4
  drop_rate = 0.75
  beta = 0.001
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_f,labels=y)) \
     + beta * (tf.nn.l2_loss(weights_f))

  optimizer = tf.train.AdamOptimizer(rate).minimize(cost)
  correct = tf.equal(tf.argmax(layer_f, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
  sess.run(tf.global_variables_initializer())
# Now, you run this with fine-tuning data in sess.run()
  res, epoch = [0 for x in range(n_classes)], 0
  while epoch < hm_epochs and sum(res)/len(res) < 0.99:
    epoch_loss = 0
    epoch += 1
    for name in [0]:#,15000,40000,60000,80000,100000]:
      X_train, y_train = recupTrain('dataTrain', name)
      for g in range(0,len(X_train),batch_size):
        _, c = sess.run([optimizer, cost], feed_dict={keep_prob: 1, rate: l_rate, x: X_train[g:g+batch_size], y: y_train[g:g+batch_size], keep_prob: drop_rate})
        
        sys.stdout.write('\r' + str(g) + '/' + str(len(X_train)))
        sys.stdout.flush()
        epoch_loss += c

    tempsEcoule = time() - t

    sys.stdout.write('\rEpoch : ' + str(epoch) + ' Loss : ' + str(epoch_loss) + ' Batch size : ' + str(batch_size) \
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
      #batch_size = int(batch_size*1.5)

  res2, res = 0, 0
  for g in range(0,len(X_train),batch_size):
      res2 += accuracy.eval({x:X_train[g:g+batch_size], y:y_train[g:g+batch_size], keep_prob: 1})
  res2 /= (g/batch_size) + 1
  for g in range(0,len(X_test),batch_size):
      res += accuracy.eval({x:X_test[g:g+batch_size], y:y_test[g:g+batch_size], keep_prob: 1})
  res /= (g/batch_size) + 1
print('Epoch', epoch,'loss :',epoch_loss,'train :',res2,'test :', res)