#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from os import path
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gzip
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data():
    files = [
        'train-labels-idx1-ubyte.gz', 
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join("data/fashion", fname))
    
    with gzip.open(paths[0], 'rb') as labelpath:
        y_train = np.frombuffer(labelpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as labelpath:
        y_test = np.frombuffer(labelpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
    return (x_train, y_train),(x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Wrap bitstring in JSON
json_data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print(json_data)

json_file = open('./predict.json', 'w')
json_file.write(json_data)
json_file.close()

headers = {
    "content-type": "application/json",
    "Host": "fashion-mnist-sample.default.example.com"
}

json_response = requests.post('http://47.95.32.195:80/v1/models/fashion-mnist-sample:predict', data=json_data, headers=headers)
print(json_response)
predictions = json.loads(json_response.text)['predictions']
print(predictions[0])
print('The model thought this was a {} (class {})'.format(class_names[np.argmax(predictions[0])], np.argmax(predictions[0])))
