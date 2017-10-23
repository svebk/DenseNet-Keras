"""Test ImageNet pretrained DenseNet"""

import os
import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K


# We only test DenseNet-121 in this script for demo purpose
from densenet121 import DenseNet

def get_layer_output_fn(model, layer_in, layer_out):
  return K.function([model.get_layer(layer_in).input, K.learning_phase()], [model.get_layer(layer_out).output])

def get_layer_output(lofn, input, phase=0):
  return lofn([input, phase])[0]

def preprocess_img(img_path):
  im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)

  # Subtract mean pixel and multiple by scaling constant
  # Reference: https://github.com/shicai/DenseNet-Caffe
  im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
  im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
  im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017

  if K.image_dim_ordering() == 'th':
    # Transpose image dimensions (Theano uses the channels as the 1st dimension)
    im = im.transpose((2,0,1))

  return im

def print_prediction(out):
  print 'Prediction: '+str(classes[np.argmax(out)])


if K.image_dim_ordering() == 'th':
  # Use pre-trained weights for Theano backend
  weights_path = 'imagenet_models/densenet121_weights_th.h5'
else:
  # Use pre-trained weights for Tensorflow backend
  weights_path = 'imagenet_models/densenet121_weights_tf.h5'

# Load ImageNet classes file
classes = []
with open('resources/classes.txt', 'r') as list_:
    for line in list_:
        classes.append(line.rstrip('\n'))


if __name__ == "__main__":
  # Test pretrained model
  model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
  sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  lofn = get_layer_output_fn(model, 'data', 'pool5')

  # Insert a new dimension for the batch_size
  for img_name in ['shark.jpg', 'cat.jpg']:
    im = preprocess_img(os.path.join('resources', img_name))
    im = np.expand_dims(im, axis=0)
    out = model.predict(im)
    print_prediction(out)
    pool = get_layer_output(lofn, im)
    print pool.shape, pool
