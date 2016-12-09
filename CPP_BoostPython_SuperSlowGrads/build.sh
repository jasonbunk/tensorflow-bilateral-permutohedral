#!/bin/bash

#rm ../pymytestgradslib.so
rm pymytestgradslib.so

# debug build
#g++ -std=c++0x -fPIC PythonCVMatConvert.cpp PythonUtils.cpp OnlineAffineTransf.cpp -g -std=c++11 -I/usr/include/python2.7 -lpython2.7 -lboost_python-py27 -lopencv_core -lopencv_imgproc -shared -o pymytestgradslib.so -O0

NUMPY_INCLUDE_DIR=/usr/local/lib/python2.7/dist-packages/numpy/core/include/

# "release" build

#g++ -E OnlineAffineTransf.cpp -std=c++11 -I$NUMPY_INCLUDE_DIR -I/usr/include/python2.7 > OnlineAffineTransf.preprocessor_out

g++ -fPIC PythonCVMatConvert.cpp PythonUtils.cpp OnlineAffineTransf.cpp -std=c++11 -I$NUMPY_INCLUDE_DIR -I/usr/include/python2.7 -lpython2.7 -lboost_python-py27 -lopencv_core -lopencv_imgproc -shared -o pymytestgradslib.so -O2 -fopenmp

#mv pymytestgradslib.so ..
