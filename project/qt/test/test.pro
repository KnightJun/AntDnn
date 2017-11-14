TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../../../unit_test/test.cpp

INCLUDEPATH += ../../../include
LIBS += -L../../../bin/mingw
LIBS += -lantdnn
