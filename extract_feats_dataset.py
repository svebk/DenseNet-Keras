"""Test ImageNet pretrained DenseNet"""

import os
import cv2
import glob
import json
import base64
import numpy as np
from argparse import ArgumentParser

from keras.optimizers import SGD
import keras.backend as K

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
  parser = ArgumentParser()
  parser.add_argument("-d", "--dataset_path", dest="dataset_path", required=True)
  parser.add_argument("-o", "--output_path", dest="output_path", default='./feats.jl')
  opts = parser.parse_args()

  # Test pretrained model
  model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
  sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  lofn = get_layer_output_fn(model, 'data', 'pool5')

  with open(opts.output_path, 'wt') as outf:
    class_id = 0
    for class_dir in glob.glob(os.path.join(opts.dataset_path, '*')):
      class_id += 1
      print "Processing class #{} in {}".format(class_id, class_dir)
      for img_name in glob.glob(os.path.join(class_dir, '*')):
        try:
          # We could try to batch this process
          im = preprocess_img(img_name)
          im = np.expand_dims(im, axis=0)
          #out = model.predict(im)
          #print_prediction(out)
          pool = np.squeeze(get_layer_output(lofn, im))
          #print pool.shape, pool
          out_dict = {"img_name": img_name.split(opts.dataset_path)[-1], "feat_densnet121": base64.b64encode(pool),
                      "feat_densnet121_dtype": str(pool.dtype), "class_id": class_id}
          outf.write(json.dumps(out_dict)+'\n')
        except Exception as inst:
          print "Could not process image: {}. {} {}".format(img_name, type(inst), inst)
