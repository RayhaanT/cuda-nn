#include "linear.h"
#include "matrix.h"
#include "sigmoid.h"
#include "relu.h"
#include "nn.h"
#include "dataset.h"

#include <iostream>

#define EPOCHS 100

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[0][i] > 0.5 ? 1 : 0;
		if (prediction == targets[0][i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}

int main() {
	srand(time(NULL));

	Dataset dataset(2, 100, 1);
	BCECost bce_cost;

	NeuralNetwork nn(0.01, Shape(1, 1));
	nn.addLayer(new Linear(Shape(2, 30), Shape(30, 1)));
	nn.addLayer(new Relu(Shape(30, 1)));
	nn.addLayer(new Linear(Shape(30, 1), Shape(1, 1)));
	nn.addLayer(new Sigmoid(Shape(1, 1)));

	// Training
	for (int epoch = 0; epoch <= EPOCHS; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			MatrixBuffer Y = nn.inference(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets()[batch]);
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	// Final accuracy
    int last = dataset.getNumOfBatches() - 1;
	Matrix Y(nn.inference(dataset.getBatches()[last]));
    Y.print();
    std::cout << "---" << std::endl;
    dataset.getTargets()[last].print();
	float accuracy = computeAccuracy(
			Y, dataset.getTargets()[last]);
	std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return 0;
}
