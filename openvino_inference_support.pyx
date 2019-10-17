# openvino_inference_support.pyx
# distutils: language = c++
# cython: language_level=3, boundscheck=False


import time
import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref

BATCH_SIZE = 4
arr = np.random.random_sample(size = [16, 3, 224, 224])

ctypedef fused T:
    int
    float
    double
    long
    short

ctypedef T * t_point
ctypedef float * f_point

cdef char* model_path = "/home/aqtjin/resnet_v1_50/resnet_v1_50.xml"
cdef char* model_bin = "/home/aqtjin/resnet_v1_50/resnet_v1_50.bin"

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
    cdef float [::1] array

    # Load Model
    model = OpenVINOInferenceSupportive.loadOpenVINOIR(model_path, model_bin, 0, 1)

    # Create CTensor Shape
    for s in arr[0:1*BATCH_SIZE].shape:
        shape.push_back(s)

    print("Begin to prepare data")
    for i in range(arr.shape[0] // BATCH_SIZE):
        # From python array or ndarray to CTensor
        time_s2 = time.time()
        array = np.ascontiguousarray(arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE].flatten(), dtype=np.single)
        # for j in range(len(arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE].flatten())):
        #     array[j] = arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE].flatten()[j]
        time_e2 = time.time()
        print("Time Slot:" + str(time_e2 - time_s2))

        input = new CTensor[float](&array[0], shape)
        # free(array)
        print("Begin here")
        OpenVINOInferenceSupportive.predict(deref(model), deref(input))
        print("Predict successful")
