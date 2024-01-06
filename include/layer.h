#ifndef LAYER_H
#define LAYER_H

#include <string>
#include "matrix.h"

class Layer {
public:
    virtual ~Layer() {};

    virtual MatrixBuffer& forward(MatrixBuffer& A) = 0;
    virtual MatrixBuffer& backprop(MatrixBuffer& dZ, float learning_rate) = 0;
};

#endif
