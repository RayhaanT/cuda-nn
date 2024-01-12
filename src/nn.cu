#include "nn.h"
#include <memory>

NeuralNetwork::NeuralNetwork(float learningRate, Shape shape) :
	learningRate(learningRate), Y(shape), dY(shape)
{
    dY.allocate();
}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
	this->layers.push_back(layer);
}

void NeuralNetwork::addLayer(Layer* layer) {
    this->layers.emplace_back(layer);
}

MatrixBuffer NeuralNetwork::inference(MatrixBuffer X) {
	MatrixBuffer Z = X;
	for (auto layer : layers) {
		Z = layer->forward(Z);
	}
	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(MatrixBuffer predictions, MatrixBuffer target) {
	MatrixBuffer error = bce.dCost(predictions, target, dY);
	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learningRate);
	}
	cudaDeviceSynchronize();
}

std::vector<Layer*> NeuralNetwork::getLayers() const {
    std::vector<Layer*> ret;
    ret.reserve(layers.size());
    for (auto &l : layers) {
        ret.emplace_back(l.get());
    }
	return ret;
}
