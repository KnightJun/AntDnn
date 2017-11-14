TEMPLATE = lib
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -fopenmp -mavx2
QMAKE_LFLAGS += -fopenmp

SOURCES += \
    ../../../modules/Activation.cpp \
    ../../../modules/Convolution.cpp \
    ../../../modules/Dense.cpp \
    ../../../modules/Merge.cpp \
    ../../../modules/Pooling.cpp \
    ../../../modules/RunTimeDetect.cpp \
    ../../../modules/Tensor.cpp \
    ../../../modules/TensorOperating.cpp
INCLUDEPATH += ../../../include/antdnn \
    ../../../3rdparty

DESTDIR += ../../../bin/mingw
