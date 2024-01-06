#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
#include "matrix.h"

#define WEIGHTS_INIT 0.01

class Linear : public Layer {
private:
    MatrixBuffer W; // Weights
    MatrixBuffer b; // Biases

    MatrixBuffer Z; // Our output
    MatrixBuffer A; // Input (from regularization)
    MatrixBuffer dA;
                     
    void initBias();
    void initWeights();

    void updateWeights(MatrixBuffer& dZ, float learningRate);
    void updateBias(MatrixBuffer& dZ, float learningRate);

public:
    Linear(Shape wShape, Shape zShape);
    Linear(MatrixBuffer& w, MatrixBuffer &b);
    ~Linear() {};

    MatrixBuffer& forward(MatrixBuffer& Z);
    MatrixBuffer& backprop(MatrixBuffer& dA, float learningRate = 0.01);
};

#endif
