# openvino_inference_support.pyx
# distutils: language = c++
# cython: language_level=3, boundscheck=False


import time
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref

BATCH_SIZE = 4

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
        void * getData()


cdef extern from "inference_engine/ie_iexecutable_network.hpp":
    cdef cppclass ExecutableNetwork:
        ExecutableNetwork()

cdef extern from "OpenVINOInferenceSupportive.hpp":
    cdef cppclass OpenVINOInferenceSupportive:
        @staticmethod
        ExecutableNetwork* loadOpenVINOIR(const string modelFilePath, const string weightFilePath, const int deviceType, const int batchSize)
        @staticmethod
        CTensor[float]* predictPTR(ExecutableNetwork executable_network, CTensor[float] datatensor)

cdef pointer_to_numpy_array(void * ptr, np.npy_intp size):
    '''Convert c pointer to numpy array.
    The memory will be freed as soon as the ndarray is deallocated.
    '''
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    cdef np.ndarray[np.float, ndim=1] arr = \
            np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT32, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

def openvino_predict(Loadedmodel, data):
    cdef CTensor[float] *input
    cdef CTensor[float] *output
    cdef ExecutableNetwork * model
    cdef int array_size = 1
    cdef vector[size_t] shape
    cdef float [::1] array
    cdef void * re

    cdef unsigned long ptr = Loadedmodel
    model = <ExecutableNetwork *> ptr
    # Create CTensor Shape
    for s in data[0:1*BATCH_SIZE].shape:
        shape.push_back(s)

    print("Begin to prepare data")
    for i in range(data.shape[0] // BATCH_SIZE):
        # From python array or ndarray to CTensor
        time_s2 = time.time()
        array = np.ascontiguousarray(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE].flatten(), dtype=np.single)
        # for j in range(len(arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE].flatten())):
        #     array[j] = arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE].flatten()[j]
        time_e2 = time.time()
        print("Time Slot:" + str(time_e2 - time_s2))

        input = new CTensor[float](&array[0], shape)
        print("Begin here")
        output = OpenVINOInferenceSupportive.predictPTR(deref(model), deref(input))
        # re = output.getData()
        arr = pointer_to_numpy_array(re, 4)
        print("Predict successful")

def Load_OpenVINO_Model(xml_path, bin_path, deviceType, batchSize):
    # cdef ExecutableNetwork * model
    cdef void* ptr = OpenVINOInferenceSupportive.loadOpenVINOIR(bytes(xml_path, encoding='utf8'), bytes(bin_path, encoding='utf8'), deviceType, batchSize)
    model = <unsigned long> ptr
    return model
