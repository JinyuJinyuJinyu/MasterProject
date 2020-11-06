import tensorflow as tf
import utils
import numpy as np
import json
import time

tf.random.set_seed(0)

class Block1(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size=(3,3), stride=1):
        super(Block1, self).__init__()
        self.con2a = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.relua = tf.keras.layers.Activation('relu')
        self.dropouta = tf.keras.layers.Dropout(0.4)


        self.con2b = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.relub = tf.keras.layers.Activation('relu')

        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')
    def call(self, input_tensor, training=None):
        x = self.con2a(input_tensor)
        x = self.bn2a(x,training=training)
        x = self.relua(x)
        x = self.dropouta(x)

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
        self.dropouta = tf.keras.layers.Dropout(0.5)

        self.con2b = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.relub = tf.keras.layers.Activation('relu')
        self.dropoutb = tf.keras.layers.Dropout(0.5)

        self.con2c = tf.keras.layers.Conv2D(filters,kernel_size,strides=stride,padding='same')
        self.bn2c = tf.keras.layers.BatchNormalization()
        self.reluc = tf.keras.layers.Activation('relu')

        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')
    def call(self, input_tensor, training=None):
        x = self.con2a(input_tensor)
        x = self.bn2a(x,training=training)
        x = self.relua(x)
        x = self.dropouta(x)

        x = self.con2b(x)
        x = self.bn2b(x,training=training)
        x = self.relub(x)
        x = self.dropoutb(x)

        x = self.con2c(x)
        x = self.bn2c(x,training=training)
        x = self.reluc(x)

        x = self.pool(x)
        return x

class vgg16(tf.keras.models.Model):

    def __init__(self,num_classes):
        super(vgg16,self).__init__()

        self.block2xa = self._make_layers(Block1, [64,128],stride=1)

        self.block3xb = self._make_layers(Block2, [256,512,512], stride=1)

        self.flattena = tf.keras.layers.Flatten(name='flatten')
        self.densea = tf.keras.layers.Dense(4096,activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        #4096
        self.denseb = tf.keras.layers.Dense(4096,activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(5e-4))

        self.densec = tf.keras.layers.Dense(units=num_classes,activation='softmax')

    def _make_layers(self,block,filters,stride=1):
        convlayers = tf.keras.Sequential()

        for flter in filters:
            convlayers.add(block(flter,stride))

        return convlayers
    def call(self,inputs, training=None):

        x = self.block2xa(inputs,training=training)
        x = self.block3xb(x,training=training)

        x = self.flattena(x)
        x = self.densea(x)
        x = self.denseb(x)
        x = self.densec(x)

        return x



def preprocess(x_batch,y_batch):
    x_batch = tf.cast(x_batch, dtype=tf.float32) /255. - 0.5
    y_batch = tf.cast(y_batch, dtype=tf.int32)
    return x_batch, y_batch



batch_size = 16
epoches = 1
# optis = [tf.keras.optimizers.SGD(learning_rate=1e-3)]
# f_name = ['SGD_VGG16_tf.json']
x_train, x_test, y_train, y_test = utils.load_dat()



# print(x_train.dtype,x_test.dtype,y_train.dtype,y_test.dtype)
# print(y_train.shape)


train_set = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_set = train_set.shuffle(1024).map(preprocess).batch(batch_size)


val_set = tf.data.Dataset.from_tensor_slices((x_test,y_test))
val_set = val_set.map(preprocess).batch(batch_size)

optis = [tf.keras.optimizers.Adam(lr=1e-3),tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=0.9),
         tf.compat.v1.train.GradientDescentOptimizer(1e-3,name='GradientDescent')]
f_name = ['adam_VGG16_tf.json','SGD_VGG16_tf.json','gradient_descent_VGG16_tf.json']




def main(optimizer,fname):

    VGG16 = vgg16(1000)
    VGG16.build(input_shape=(None,170,170,3))

    # optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-3,name='GradientDescent')
    # optimizer = tf.compat.v1.train.MomentumOptimizer(lr=1e-3, momentum=0.9, use_locking=False, name='Momentum', use_nesterov=False)

    f = open(fname, "w", encoding='utf-8')
    outfile = []

    

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = VGG16(x_batch, training=True)
            y_onehot = tf.one_hot(y_batch, depth=1000)
            loss_value = tf.keras.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            grads = tape.gradient(loss_value, VGG16.trainable_variables)
            optimizer.apply_gradients(zip(grads, VGG16.trainable_variables))
        return loss_value


    def val_step(x_batch_val, y_batch_val):

        val_logits = VGG16(x_batch_val, training=False)
        # print(val_logits.shape)
        # val_logits = tf.reshape(val_logits, (x_batch_val.shape[0], 10))
        y_onehot_val = tf.one_hot(y_batch_val, depth=1000)
        loss = tf.keras.losses.categorical_crossentropy(y_onehot_val, val_logits, from_logits=True)
        loss = tf.reduce_sum(loss)

        prob = tf.nn.softmax(val_logits,axis=1)
        preds = tf.argmax(prob, axis=1)
        preds = tf.cast(preds, dtype=tf.int32)
        mtx = tf.math.confusion_matrix(y_batch_val, preds, num_classes=1000)

        # print(np.array(mtx))
        return loss, mtx

    val_time = 0
    print('start training TensorFlow')
    start_time = init_time = time.time()
    for epoch in range(1, epoches + 1):
        # train
        for step, (x_batch, y_batch) in enumerate(train_set):
            train_step(x_batch, y_batch)
            # for x_batch_val, y_batch_val in val_set:
            #     loss, confusion_mtx = val_step(x_batch_val, y_batch_val)
            #
            #     confusion_matrix = np.add(confusion_matrix, confusion_mtx)
            #     lossess += loss.numpy()
            #     exit()

        if True:
            val_start_time = time.time()
            val_info = {}
            # print('validating')
            lossess = 0
            confusion_matrix = np.zeros((1000, 1000))

            for x_batch_val, y_batch_val in val_set:
                loss, confusion_mtx = val_step(x_batch_val, y_batch_val)

                confusion_matrix = np.add(confusion_matrix, confusion_mtx)
                lossess += loss.numpy()

            val_info['epoch: '] = epoch
            val_info['test loss'] = lossess / 10000
            val_info['test acc'] = (tf.linalg.trace(confusion_matrix).numpy() / 100)
            if epoch % epoches == 0:
                val_info['confusion matrix'] = confusion_matrix.tolist()
            print('training epoch: ', epoch,'  .accu: ', tf.linalg.trace(confusion_matrix).numpy() / 100)
            outfile.append(val_info)
            val_time += (time.time() - val_start_time)
        if epoch == 1:
            init_time = time.time() - init_time

    ttl_time = {}
    ttl_time['training time'] = (time.time() - start_time - val_time)
    ttl_time['total time'] = (time.time() - start_time)
    ttl_time['val time'] = val_time
    ttl_time['init time'] = init_time
    ttl_time['ave time'] = (time.time() - start_time - val_time - init_time) / epoches
    outfile.append(ttl_time)
    json.dump(outfile, f, separators=(',', ':'), indent=4)
    f.close()

if __name__ == '__main__':
    for opti,f in zip(optis,f_name):
        main(opti, f)
