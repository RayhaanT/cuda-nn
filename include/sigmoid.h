#ifndef SIGMOID_H
#define SIGMOID_H

#include "layer.h"
#include "matrix.h"

class Sigmoid : public Layer {
private:
    MatrixBuffer A; // Our output
    MatrixBuffer Z; // Input (from linear layer)
    MatrixBuffer dZ; // Derivative
    Shape shape;

public:
    Sigmoid(Shape shape);
    ~Sigmoid() {};

    MatrixBuffer& forward(MatrixBuffer& Z);
    MatrixBuffer& backprop(MatrixBuffer& dA, float learning_rate = 0.01);
};

#endif
