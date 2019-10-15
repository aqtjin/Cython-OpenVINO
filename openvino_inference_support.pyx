# openvino_inference_support.pyx
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref

BATCH_SIZE = 4
arr = np.random.randint(0, 10, size = [32, 3, 224, 224])

ctypedef fused T:
    int
    float
    double
    long
    short

ctypedef T * t_point
ctypedef float * f_point


#cdef extern from "CTensor.hpp":
cdef extern from "OpenVINOInferenceSupportive.hpp":
    cdef cppclass CTensor[T]:
        CTensor(f_point _data, vector[size_t] _shape)
        CTensor(vector[size_t] _shape)
        printCTensor()


cdef extern from "inference_engine/ie_iexecutable_network.hpp":
    cdef cppclass ExecutableNetwork:
        ExecutableNetwork()

cdef extern from "OpenVINOInferenceSupportive.hpp":
    cdef cppclass OpenVINOInferenceSupportive:
        @staticmethod
        ExecutableNetwork* loadOpenVINOIR(const string modelFilePath, const string weightFilePath, const int deviceType, const int batchSize)
        @staticmethod
        CTensor[float] predict(ExecutableNetwork executable_network, CTensor[float] datatensor)

def openvino_predict():
    cdef CTensor[float] *input
    cdef CTensor[float] *output
    cdef ExecutableNetwork * model
    cdef int array_size = 1
    cdef vector[size_t] shape
    cdef float * array
    for i in range(arr.shape[0] // BATCH_SIZE):
        # From python array or ndarray to CTensor
        for s in arr[0:(i+1)*BATCH_SIZE].shape:
            array_size *= s
            shape.push_back(s)

        array = <float *> malloc(array_size)

        for j in range(len(arr[0:(i+1)*BATCH_SIZE].flatten())):
            array[j] = arr[0:(i+1)*BATCH_SIZE].flatten()[j]

        input = new CTensor[float](array, shape)
        model = OpenVINOInferenceSupportive.loadOpenVINOIR("/home/aqtjin/resnet_v1_50.xml", "/home/aqtjin/resnet_v1_50.bin", 0, 4)
        OpenVINOInferenceSupportive.predict(deref(model), deref(input))


