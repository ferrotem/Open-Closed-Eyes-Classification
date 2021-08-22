import os
import numpy as np
import sys
import json
from tqdm import tqdm
sys.path.append("..")
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


import argparse
from utils import read_image
parser = argparse.ArgumentParser(description="Inference file")
parser.add_argument('test_path', type=str, help='/путь/до/папки test')
import csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference file")
    parser.add_argument('test_path', type=str, help='/путь/до/папки test')

    args = parser.parse_args()
    test_path = args.test_path


    list_img_path = os.listdir(test_path)
    list_img_path.sort()
    test_data = [np.array(read_image(os.path.join(test_path, x))) for x in list_img_path]

    train_data = np.load('siam_data.npy')
    positive_class = train_data[-20:]

    MODEL_PATH = "./models/Siam_01"
    model_siamese = tf.keras.models.load_model(MODEL_PATH, compile=False)

    label = []
    for i in tqdm(range(len(test_data))):
        right = np.repeat(np.expand_dims(test_data[i], axis=0), 20, axis=0)
        
        # print("SHape ", right.shape)
        left = positive_class
        left, right = tf.cast(left, tf.float32),tf.cast(right, tf.float32)
        left, right = tf.expand_dims(left, axis=-1), tf.expand_dims(right, axis=-1)
        left, right = left/255, right/255 
        output,L1_distance = model_siamese([left, right])
        if np.mean(output)>0.5:
            label.append(["test/"+list_img_path[i],1])
        else:
            label.append(["test/"+list_img_path[i],1])
        
    with open('result.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerows(label)
        print("labels created") 


    
