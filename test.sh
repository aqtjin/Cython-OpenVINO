#!bin/bash




rm openvino_inference_support.cpython-37m-x86_64-linux-gnu.so
rm openvino_inference_support.cpp
python setup.py build_ext --inplace
python test.py