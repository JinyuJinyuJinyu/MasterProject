import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import json
import time

tf.keras.backend.set_floatx('float32')
tf.random.set_seed(0)
class Identity(tf.keras.layers.Layer):

    def __init__(self,filters,kernel_size=(3,3), stride=1):
        super(Identity, self).__init__()

        self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size,strides=stride,padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        self.conv2b = tf.keras.layers.Conv2D(filters, kernel_size,strides=1,padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        if stride == 1:
            self.downsample = lambda x: x
        else:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filters,kernel_size=(1, 1),strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, input_tensor, training=None):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x,training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x,training=training)

        x = tf.keras.layers.add([x,self.downsample(input_tensor)])

        return tf.nn.relu(x)


class Resnet_s(tf.keras.models.Model):

    def __init__(self, net_dim, kernel_size=(3,3), classes=10):
        super(Resnet_s,self).__init__()

        self.conv2a = tf.keras.layers.Conv2D(64,kernel_size,strides=(1,1),padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')
        self.pool2a = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')

        self.bocks1 = self.build_blks(64,net_dim[0])
        self.bocks2 = self.build_blks(128,net_dim[1],stride=2)
        self.bocks3 = self.build_blks(256,net_dim[2],stride=2)
        self.bocks4 = self.build_blks(512,net_dim[3],stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None):
        x = self.conv2a(inputs)
        x = self.bn2a(x,training=training)
        x = self.relua(x)
        x = self.pool2a(x)

        x = self.bocks1(x,training=training)
        x = self.bocks2(x,training=training)
        x = self.bocks3(x,training=training)
        x = self.bocks4(x,training=training)

        x = self.avgpool(x)

        output = self.fc(x)

        return output
    def build_blks(self, filter_num, blocks, stride=1):
        res_blk = tf.keras.Sequential()
        res_blk.add(Identity(filter_num, stride=stride))

        for _ in range(1,blocks):
            res_blk.add(Identity(filter_num,stride=stride))

        return res_blk

def preprocess(x_batch,y_batch):
    x_batch = tf.cast(x_batch, dtype=tf.float32) /255. - 0.5
    y_batch = tf.cast(y_batch, dtype=tf.int32)
    return x_batch, y_batch



batch_size = 64

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = tf.squeeze(y_train,axis=1)

_,x_val,_,y_val = train_test_split(x_test,y_test,test_size=0.2,shuffle=True)
y_val = tf.squeeze(y_val, axis=1)


train_set = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_set = train_set.shuffle(1024).map(preprocess).batch(batch_size)

val_set = tf.data.Dataset.from_tensor_slices((x_test,y_test))
val_set = val_set.map(preprocess).batch(batch_size)


optis = [tf.keras.optimizers.Adam(lr=1e-3)]
f_name = ['testtest.json']



def main(optimizer,fname):

    resnet18 = Resnet_s([2,2,2,2])
    resnet18.build(input_shape=(None,32,32,3))
    # resnet34 = Resnet_s([3,4,6,3])
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-3,name='GradientDescent')
    # optimizer = tf.compat.v1.train.MomentumOptimizer(1e-3, momentum=0.9, use_locking=False, name='Momentum', use_nesterov=False)

    f = open(fname, "w", encoding='utf-8')
    outfile = []


    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = resnet18(x_batch, training=True)
            # default dtype float32
            y_onehot = tf.one_hot(y_batch, depth=10)
            loss_value = tf.keras.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            grads = tape.gradient(loss_value, resnet18.trainable_variables)
            optimizer.apply_gradients(zip(grads, resnet18.trainable_variables))
        return loss_value


    def val_step(x_batch_val, y_batch_val):

        val_logits = resnet18(x_batch_val, training=False)
        y_onehot_val = tf.one_hot(y_batch_val, depth=10)
        loss = tf.keras.losses.categorical_crossentropy(y_onehot_val, val_logits, from_logits=True)
        loss = tf.reduce_sum(loss)

        prob = tf.nn.softmax(val_logits,axis=1)

        preds = tf.argmax(prob, axis=1)
        preds = tf.cast(preds, dtype=tf.int32)
        mtx = tf.math.confusion_matrix(y_batch_val, preds, num_classes=10)

        # print(np.array(mtx))
        return loss, mtx


    val_time = 0
    print('start training TensorFlow')
    start_time = time.time()
    for epoch in range(1,101):
        print('epoch: ', epoch)
        # train
        # print('training epoch: ',epoch)
        for step, (x_batch, y_batch) in enumerate(train_set):
            train_step(x_batch, y_batch)
            # for x_batch_val, y_batch_val in val_set:
            #     loss, confusion_mtx = val_step(x_batch_val, y_batch_val)
            #     exit()

        if True:
            val_start_time = time.time()
            val_info = {}
            # print('validating')
            lossess = 0
            confusion_matrix = np.zeros((10,10))

            for x_batch_val, y_batch_val in val_set:
                loss, confusion_mtx = val_step(x_batch_val, y_batch_val)

                confusion_matrix = np.add(confusion_matrix, confusion_mtx)
                lossess += loss.numpy()

            val_info['epoch: '] = epoch
            val_info['test loss'] = lossess / 10000
            val_info['test acc'] = (tf.linalg.trace(confusion_matrix).numpy() / 10000) * 100
            val_info['confusion matrix'] = confusion_matrix.tolist()
            outfile.append(val_info)
            val_time += (time.time() - val_start_time)

    ttl_time = {}
    ttl_time['training time'] = (time.time() - start_time - val_time)
    ttl_time['total time'] = (time.time() - start_time)
    ttl_time['val time'] = val_time
    outfile.append(ttl_time)
    json.dump(outfile, f, separators=(',', ':'), indent=4)
    f.close()

if __name__ == '__main__':
    for opti, f in zip(optis, f_name):
        main(opti, f)
