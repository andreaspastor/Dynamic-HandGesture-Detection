import tensorflow as tf
import cv2
import time
import numpy as np


imgSize = 64
cap = cv2.VideoCapture(0)
t = time.time()

gestures = ['Swuping Left', 'Swiping Right', 'Swiping Down', \
            'Swiping Up', 'Doing other things']

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./final_model_128_2/best_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./final_model_128_2/'))

  graph = tf.get_default_graph()
  """for op in tf.get_default_graph().get_operations():
    print(str(op.name))
  df()"""
  y_pred = graph.get_tensor_by_name("fc:0")
  x = graph.get_tensor_by_name("input_x:0")
  keep_prob = graph.get_tensor_by_name("dropRate:0")
  imgs = [np.zeros((imgSize,imgSize)) for x in range(5)]
  while True:
    ret, image_np = cap.read()

    cv2.imshow('object detection', cv2.resize(image_np, (400,300)))
    gray_image = cv2.cvtColor(cv2.resize(image_np, (imgSize,imgSize)), cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    imgs.append(gray_image)
    imgs = imgs[-5:]
    t2 = time.time()
    result = y_pred.eval({x:[np.dstack(imgs)], keep_prob: 1})

    if np.max(result) > 0.7:
      print(gestures[np.argmax(result)], 1/(time.time() - t), 1/(time.time() - t2))
    else:
      print(1/(time.time() - t), 1/(time.time() - t2))
    
    waitTime = int((0.3 - time.time() + t)*1000)
    waitTime = 1 if waitTime < 0 else waitTime
    t = time.time()
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break