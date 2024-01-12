#ifndef NN_H
#define NN_H

#include "layer.h"
#include "bce.h"
#include <vector>

class NeuralNetwork {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    BCECost bce;

    MatrixBuffer Y;
    MatrixBuffer dY;
    float learningRate;

public:
    NeuralNetwork(float learningRate, Shape shape);

    MatrixBuffer inference(MatrixBuffer m);
    void backprop(MatrixBuffer predictions, MatrixBuffer target);

    void addLayer(std::shared_ptr<Layer> layer);
    void addLayer(Layer* layer);
    std::vector<Layer*> getLayers() const;
};

#endif
