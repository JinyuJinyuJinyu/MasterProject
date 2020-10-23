import tensorflow as tf
import numpy as np
import json
import time


class Block1(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size=(3,3), stride=1):
        super(Block1, self).__init__()
        self.con2a = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')


        self.con2b = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.relub = tf.keras.layers.Activation('relu')

        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')
    def call(self, input_tensor, training=None):
        x = self.con2a(input_tensor)
        x = self.bn2a(x,training=training)
        x = self.relua(x)

        x = self.con2b(x)
        x = self.bn2b(x,training=training)
        x = self.relub(x)

        x = self.pool(x)
        return x
class Block2(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size=(3,3), stride=1):
        super(Block2,self).__init__()
        self.con2a = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')

        self.con2a = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')

        self.con2b = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.relub = tf.keras.layers.Activation('relu')

        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')
    def call(self, input_tensor, training=None):
        x = self.con2a(input_tensor)
        x = self.bn2a(x,training=training)
        x = self.relua(x)

        x = self.con2b(x)
        x = self.bn2b(x,training=training)
        x = self.relub(x)

        x = self.con2b(x)
        x = self.bn2b(x,training=training)
        x = self.relub(x)

        x = self.pool(x)
        return x

class VGG16(tf.keras.models.Model):

    def __init__(self,num_classes):
        super(VGG16,self).__init__()

        self.block2xa = self._make_layers(Block1, 3,[64,128,256],stride=1)

        self.block3xb = self._make_layers(Block2,2,[512,512], stride=1)


        self.densea = tf.keras.layers.Dense(4096,activation='relu')

        self.denseb = tf.keras.layers.Dense(4096,activation='relu')

        self.densec = tf.keras.layers.Dense(units=num_classes,activation='softmax')

    def _make_layers(self,block, num_blocks,filters,stride=1):
        convlayers = tf.keras.Sequential()

        for _ in range(num_blocks):
            convlayers.add(block(filters[_],stride))

        return convlayers
    def call(self,inputs, training=None):

        x = self.block2xa(inputs,training=training)
        x = self.block3xb(x,training=training)


        x = self.densea(x)
        x = self.denseb(x)
        x = self.densec(x)

        return x


batch_size = 64


def main():
    print()

if __name__ == '__main__':
    main()
