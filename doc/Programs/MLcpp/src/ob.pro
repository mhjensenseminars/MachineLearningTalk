TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    neuralquantumstate.cpp \
    hamiltonian.cpp \
    sampler/sampler.cpp \
    optimizer/optimizer.cpp \
    sampler/gibbs/gibbs.cpp \
    sampler/metropolis/metropolis.cpp \
    optimizer/sgd/sgd.cpp \
    optimizer/asgd/asgd.cpp

HEADERS += \
    neuralquantumstate.h \
    hamiltonian.h \
    sampler/sampler.h \
    optimizer/optimizer.h \
    sampler/gibbs/gibbs.h \
    sampler/metropolis/metropolis.h \
    optimizer/sgd/sgd.h \
    optimizer/asgd/asgd.h

INCLUDEPATH += /usr/local/include/eigen3/
