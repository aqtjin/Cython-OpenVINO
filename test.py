import cv2
import openvino_inference_support
import numpy as np
import os

path_dir = "/home/aqtjin/image_test/"
images = os.listdir(path_dir)

data = cv2.imread(path_dir + images[0])
data = np.transpose(data, (2, 0, 1))
data = np.expand_dims(data, axis=0)
images.pop(0)
for image in images:
    img = cv2.imread(path_dir + image)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    data = np.concatenate((data, img), axis=0)

model = openvino_inference_support.Load_OpenVINO_Model("/home/aqtjin/resnet_v1_50/resnet_v1_50.xml", "/home/aqtjin/resnet_v1_50/resnet_v1_50.bin", 0, 4)

openvino_inference_support.openvino_predict(model, data)
