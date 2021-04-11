import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Softmax
from tensorflow.keras import Model
from tensorflow.keras import layers

num_class = 11

class Encoder(Model):
  def __init__(self, layer_name):
    super(Encoder, self).__init__()
    self.rescale = layers.experimental.preprocessing.Rescaling(1./255)
    self.conv1 = Conv3D(16, 3, padding = 'same', activation='relu', input_shape = (8, 224, 224, 3), name = layer_name + 'conv1')
    self.conv2 = Conv3D(32, 3, padding = 'same', activation='relu', name = layer_name + 'conv2')
    self.conv3 = Conv3D(64, 3, padding = 'same', activation='relu', name = layer_name + 'conv2')
    self.maxpooling1 = MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), padding = 'valid', name = layer_name + 'pool1')
    self.maxpooling2 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = layer_name + 'pool2')
    self.dropout = Dropout(0.5, name = layer_name + 'dropout')
    self.flatten = Flatten(name = layer_name + 'flatten')
    self.d1 = Dense(128, activation='relu', name = layer_name + 'dense1')

  def call(self, x):
    x = self.rescale(x)
    x = self.conv1(x)
    x = self.maxpooling1(x)
    x = self.conv2(x)
    x = self.maxpooling2(x)
    x = self.conv3(x)
    x = self.maxpooling2(x)
    x = self.flatten(x)
    x = self.dropout(x)
    return self.d1(x)

class VCModel(Model):
  def __init__(self):
    super(VCModel, self).__init__()
    self.noun_encoder = Encoder(layer_name = 'noun_')
    self.verb_encoder = Encoder(layer_name = 'verb_')
    self.d_verb = Dense(12, name = 'verb_output')
    self.d_noun = Dense(17, name = 'noun_output')

  def call(self, x):
    noun = self.noun_encoder(x)
    verb = self.verb_encoder(x)
    return self.d_noun(noun)
