
#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import numpy as np
import datetime
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
from tqdm import tqdm

from matplotlib import pyplot as plt
from IPython import display
from siam_data_loader import Image_Dataset
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
dataset = Image_Dataset()
#from tensorflow.keras.datasets import mnist
DATASET = "Eyes"
TRAIN_SIZE = 600
TEST_SIZE = 150
BUFFER_SIZE = 400
BATCH_SIZE = 20
ENHANCE = False
DROPOUT_RATE = 0.2
K_HOT = 100
N_STEP =int(TRAIN_SIZE/BATCH_SIZE)
if ENHANCE:
    INPUT_SHAPE = (128, 128, 3)
else:
    INPUT_SHAPE = (24, 24, 1)
MARGIN = 0.05
ALPHA = 0.05

train_ds = tf.data.Dataset.from_generator(dataset._train_generator ,args=[TRAIN_SIZE], output_types= (tf.float32,tf.float32,tf.float32))#, output_shapes =(tf.TensorShape([32,32,3]),tf.TensorShape([32,32,3]),tf.TensorShape([None])))
#train_ds = train_ds.shuffle(BUFFER_SIZE)
print("train dataset has been created!!!")
# train_ds = train_ds.batch(BATCH_SIZE)
# #%%
# print("test dataset has been created!!!")
# test_ds = tf.data.Dataset.from_generator(dataset._test_dataset_generator, args= [TEST_SIZE, ENHANCE, K_HOT], output_types= (tf.float32,tf.float32))#, output_shapes=(tf.TensorShape([10, 128,128,3]),tf.TensorShape([10,128,128,3])) )#, (tf.TensorShape([32,32,3]),tf.TensorShape([32,32,3]),tf.TensorShape([1])))
# #test_ds = test_ds.shuffle(BUFFER_SIZE)

def generate_images(images,  epoch, order):
  plt.figure(figsize=(15,15))

  display_list = images
  title = ['Left image', 'Predicted Imege']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i].numpy())
    plt.axis('off')
  plt.savefig('./results/epoch_images'+str(DATASET)+"_"+ str(epoch) +"_"+str(order) +'.png')
  plt.close()

def downsample(filters, size, apply_batchnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    bias_initializer = tf.random_normal_initializer(0.5, 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, 
        padding='same',
        kernel_initializer=initializer,
        # kernel_regularizer=regularizers.l2(0.001),
        bias_initializer = bias_initializer,
        use_bias=True))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    # result.add(tf.keras.layers.MaxPooling2D(64, 2,padding = 'same'))
    # result.add(tf.keras.layers.Dropout(DROPOUT_RATE))
    return result

def conv_net():
    input_shape = INPUT_SHAPE
    inputs = tf.keras.layers.Input(input_shape, name = "input_conv_net")
    x = downsample(32, 3, apply_batchnorm=False)(inputs)
    x = downsample(64, 3)(x)
    # x = downsample(128, 2)(x)
    # x = downsample(256, 2)(x)
    # x = tf.keras.layers.Dense(128,activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation="relu")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

conv_net_model = conv_net()
def siamese_net():
    input_shape = INPUT_SHAPE
    
    tf.keras.utils.plot_model(conv_net_model, 'conv_net_model.png', show_shapes=True)
    left_input = tf.keras.layers.Input(input_shape, name = "left_image")
    right_input = tf.keras.layers.Input(input_shape, name = "right_image" )
    
    encoded_l = conv_net_model(left_input)
    encoded_r = conv_net_model(right_input)
    
    L1_distance =tf.keras.layers.Lambda(lambda x: tf.abs(x[0]-x[1]))([encoded_l,encoded_r]) 
    L1_distance = L1_distance
    prediction = tf.keras.layers.Dense(1,activation='sigmoid',use_bias=False)(L1_distance)    
    return  tf.keras.Model(inputs=[left_input,right_input],outputs=[prediction,L1_distance])

#Summary writer
log_dir="logs/"
os.makedirs(log_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'batch_size={} dropout={} enhance={} '.format(BATCH_SIZE,DROPOUT_RATE, ENHANCE) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tf.summary.trace_on(graph=True, profiler=True)

#Model saver
siamese_optimizer = tf.keras.optimizers.Adam()#2e-4, beta_1=0.5
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
os.makedirs(checkpoint_prefix, exist_ok=True)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits =True) #tf.keras.losses.BinaryCrossentropy(from_logits =True)
accuracy_object = tf.keras.metrics.Accuracy()    
model_siamese = siamese_net()
model_siamese.summary()
tf.keras.utils.plot_model(model_siamese, 'siamese_net_model.png', show_shapes=True)
checkpoint = tf.train.Checkpoint(siamese_optimizer=siamese_optimizer,
    model = model_siamese)



@tf.function
def train_step(left, right, label):
    left, right, label = tf.cast(left, tf.float32),tf.cast(right, tf.float32),tf.cast(label, tf.float32)
    left, right = tf.expand_dims(left, axis=-1), tf.expand_dims(right, axis=-1)
    with tf.GradientTape() as tape:
        output,L1_distance = model_siamese([left, right],training=True)

        loss = loss_object(label, output)

        L1_loss= tf.math.reduce_sum(L1_distance, axis = [1])
        L1_loss = label*L1_loss+ (1-label)*tf.math.maximum(0.0, L1_loss-MARGIN)  
        total_loss =loss + ALPHA*L1_loss
        gradients = tape.gradient(total_loss, model_siamese.trainable_weights)
        siamese_optimizer.apply_gradients(zip(gradients, model_siamese.trainable_weights))
        return total_loss,L1_loss


@tf.function
def test_step(left, right, labels):
    left, right, label = tf.cast(left, tf.float32),tf.cast(right, tf.float32),tf.cast(labels, tf.float32)
    left, right = tf.expand_dims(left, axis=-1), tf.expand_dims(right, axis=-1)
    output,L1_distance = model_siamese([left, right], training=False)
    accuracy_object.update_state(label, tf.math.round(output))
    accuracy  = accuracy_object.result()
    return output, accuracy

def fit(epochs):
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        start = time.time()
        pbar = tqdm(total = N_STEP, desc = "train epoch: {}".format(epoch))
        for n, (left_image, right_image, label) in train_ds.enumerate():

            loss, L1_distance = train_step(left_image, right_image, label)           
            #Test part            
            step= epoch*N_STEP+ n.numpy()
            if step%(N_STEP/10)==0:
                total_acc = []
                for _, (left, right, label) in train_ds.shuffle(buffer_size=10).take(10).enumerate():
                    predictions,accuracy = test_step(left, right,label)
                    total_acc.append(np.mean(accuracy))
                with summary_writer.as_default():
                    tf.summary.scalar('loss', np.mean(loss), step=step)
                    tf.summary.scalar('accuracy', np.mean(total_acc), step=step)
                    tf.summary.histogram('label', label, step=step)
                    tf.summary.histogram('prediction', predictions, step=step)
                
            pbar.update(1)


        # images = []
        
        # for order, (left, right) in test_ds.take(5).enumerate():
        #     #print("order : ", order.numpy())
        #     predictions = test_step(left, right)
        #     images =[ left[0], right[np.argmax(predictions.numpy())]]
        #     generate_images(images, epoch, order.numpy())
        

        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        #print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    checkpoint.save(file_prefix = checkpoint_prefix)

fit(10)
model_siamese.save('./models/'+ 'Siam_01')