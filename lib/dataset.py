import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math
import os
import sys
import csv

data_dir = '/home/cheer/Project/VideoCaptioning/data'
IMG_SIZE = 224
batch_size = 32
_STRIDE = 8

def decode_img(img):
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.cast(img, tf.float32)
  return tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

def resample(keep_list):
  if len(keep_list) in range(int(_STRIDE / 2) + 1, _STRIDE):
    sample_list = keep_list + random.sample(keep_list, _STRIDE - len(keep_list))
    sample_list.sort()
  else:  
    keep_list = keep_list * (math.ceil(_STRIDE/len(keep_list)) + 1)
    sample_list = random.sample(keep_list, _STRIDE)
    sample_list.sort()
  return ','.join(['{:05d}.png'.format(x) for x in sample_list])

def process_label(file_path):
  img_list = []
  for i in range(_STRIDE):
    img_path = tf.strings.join([data_dir, 'Images', tf.strings.split(file_path, sep = ',')[0], tf.strings.split(file_path, sep = ',')[i + 1]], separator = '/')
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    img_list.append(img)
  return tf.stack(img_list), int(tf.strings.split(file_path, sep = ',')[-1])

def configure_for_performance(ds, name):
  ds = ds.cache('cache/' + name)
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

def get_list():
  label_files = os.listdir(os.path.join(data_dir, 'Labels', 'captions'))
  label_list = []
  with open(os.path.join(data_dir, 'Labels', 'verbs.txt'), 'r') as verb_file:
    verb_list = [x.strip() for x in verb_file.readlines()]
  with open(os.path.join(data_dir, 'Labels', 'nouns.txt'), 'r') as noun_file:
    noun_list = [x.strip() for x in noun_file.readlines()]
  for csv_file in label_files:
    csv_file = os.path.join(data_dir, 'Labels', 'captions', csv_file)
    with open(csv_file, 'r') as csvfile:
      spamreader = csv.reader(csvfile)
      next(spamreader, None)
      for row in spamreader:
        stride = resample(list(np.arange(int(row[0]), int(row[1]) + 1)))
        verb_class = str(verb_list.index(row[2].replace(' ', '').lower()))
        noun_class = str(noun_list.index(row[3].replace(' ', '').replace('others', 'other').lower()))
        label_list.append(csv_file.split('.')[0].split('/')[-1] + ',' + stride + ',' + verb_class + ',' + noun_class)
  random.shuffle(label_list)
  return label_list
    
def get_dataset():
  label_list = get_list()
  image_count = len(label_list)
  val_size = int(image_count * 0.2)
  train_ds = tf.data.Dataset.from_tensor_slices(label_list[val_size:])
  val_ds = tf.data.Dataset.from_tensor_slices(label_list[:val_size])
  train_ds = train_ds.map(process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_ds = val_ds.map(process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = configure_for_performance(train_ds, 'train')
  val_ds = configure_for_performance(val_ds, 'val')
  #image_batch, label_batch = next(iter(train_ds))
  #print (label_batch)
  return train_ds, val_ds

if __name__ == '__main__':
  get_dataset()

