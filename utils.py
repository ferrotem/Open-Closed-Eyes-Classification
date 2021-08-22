import os
import json
from PIL import Image
import numpy as np
import config as cfg

def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)


def file_writer(file, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(file)) 


def imm_resize(img, width=224, height=224):
    imgn = Image.fromarray(img.astype(np.uint8))
    imgx = imgn.resize((width, height), Image.ANTIALIAS)
    return np.array(imgx)/255

def read_image(img_path):
    return Image.open(img_path)