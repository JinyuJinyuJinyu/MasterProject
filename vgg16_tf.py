import tensorflow as tf
import numpy as np
import json
import time

model = tf.keras.applications.VGG16()

class Block1():
    def __init__(self):
        super(Block1, self).__init__()
        self.con2a = tf.keras.layers.Conv2D()
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')


        self.con2b = tf.keras.layers.Conv2D()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.relub = tf.keras.layers.Activation('relu')

        self.pool = tf.keras.layers.MaxPool2D()


class Block2():
    def __init__(self):
        super(Block2,self).__init__()
        self.con2a = tf.keras.layers.Conv2D()
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')

        self.con2a = tf.keras.layers.Conv2D()
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')

        self.con2b = tf.keras.layers.Conv2D()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.relub = tf.keras.layers.Activation('relu')

        self.pool = tf.keras.layers.MaxPool2D()

class VGG(tf.keras.models.Model):

    def __init__(self):
        super(VGG,self).__init__()
        self.conv2a = tf.keras.layers.Conv2D()


    def _make_layers(self):
        print()

    def call(self):
        print()