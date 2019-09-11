# openvino_inference_support.pyx
# distutils: language = c++

from numpy import ndarray
import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string

BATCH_SIZE = 4
arr = np.random.randint(0, 10, size = [32, 3, 224, 224])

ctypedef fused T:
    int
    float
    double
    long
    short

cdef extern from "CTensor.hpp" namespace "InferenceEngine":
    cdef cppclass CTensor:
        CTensor(ndarray _data, vector[size_t] _shape)
        CTensor(vector[size_t] _shape)
        void printCTensor()

cdef extern from "OpenVINOInferenceSupportive.hpp" namespace "InferenceEngine":
        cdef long* loadOpenVINOIR(const string modelFilePath, const string weightFilePath, const int deviceType, const int batchSize)
        cdef CTensor predict(long executable_network, CTensor datatensor)

cdef CTensor *input
cdef CTensor output
cdef long *model
for i in range(arr.shape[0] // BATCH_SIZE):
    input = new CTensor(arr[0:(i+1)*BATCH_SIZE].flatten(), arr[0:(i+1)*BATCH_SIZE].shape)
    model = loadOpenVINOIR("/home/aqtjin/resnet_v1_50.xml", "/home/aqtjin/resnet_v1_50.bin", 0, 4)
    output = predict(model, *input)
    output.printCTensor()
