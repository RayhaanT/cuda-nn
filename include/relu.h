#ifndef RELU_H
#define RELU_H

#include "layer.h"
#include "matrix.h"

class Relu : public Layer {
private:
    MatrixBuffer A; // Our output
    MatrixBuffer Z; // Input (from linear layer)
    MatrixBuffer dZ; // Derivative
    Shape shape;

public:
    Relu(Shape shape);
    ~Relu() {};

    MatrixBuffer& forward(MatrixBuffer& Z);
    MatrixBuffer& backprop(MatrixBuffer& dA, float learning_rate = 0.01);
};

#endif
