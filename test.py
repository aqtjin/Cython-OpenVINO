import cv2
import openvino_inference_support
import numpy as np
import os

xml_path = "/home/aqtjin/resnet_v1_50/resnet_v1_50.xml"
bin_path = "/home/aqtjin/resnet_v1_50/resnet_v1_50.bin"
device_type = 0
BatchSize = 4

img_dir = "/home/aqtjin/image_test/"
images = os.listdir(img_dir)

data = cv2.imread(img_dir + images[0])
data = np.transpose(data, (2, 0, 1))
data = np.expand_dims(data, axis=0)
images.pop(0)
for image in images:
    img = cv2.imread(img_dir + image)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    data = np.concatenate((data, img), axis=0)

model = openvino_inference_support.Load_OpenVINO_Model(xml_path, bin_path, device_type, BatchSize)

openvino_inference_support.openvino_predict(model, data)
