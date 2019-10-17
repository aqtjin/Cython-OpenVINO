import openvino_inference_support

model = openvino_inference_support.Load_OpenVINO_Model("/home/aqtjin/resnet_v1_50/resnet_v1_50.xml", "/home/aqtjin/resnet_v1_50/resnet_v1_50.bin", 0, 4)

openvino_inference_support.openvino_predict(model)
