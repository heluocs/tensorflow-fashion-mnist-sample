#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import requests
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

f = open("./predict.json",'rb+')
json_data = f.read()

headers = {
    "content-type": "application/json",
    "Host": "fashion-mnist-sample.default.example.com"
}

json_response = requests.post('http://47.95.32.195:80/v1/models/fashion-mnist-sample:predict', data=json_data, headers=headers)
print(json_response)
predictions = json.loads(json_response.text)['predictions']
print(predictions[0])
print('The model thought this was a {} (class {})'.format(class_names[np.argmax(predictions[0])], np.argmax(predictions[0])))
