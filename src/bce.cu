#include "bce.h"

__global__ void ComputeBCECost(float* predictions, float* target,
                                       int size, float* cost) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float partial = target[index] * logf(predictions[index])
                + (1.0f - target[index]) * logf(1.0f - predictions[index]);
        atomicAdd(cost, - partial / size);
    }
}

__global__ void dBCECost(float* predictions, float* target, float* dY,
								     	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		dY[index] = -(target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]));
	}
}

float BCECost::cost(MatrixBuffer predictions, MatrixBuffer target) {
	float* cost;
	cudaMallocManaged(&cost, sizeof(float));
	*cost = 0.0f;

	dim3 blockSize(256);
	dim3 numBlocks((predictions.shape.x + blockSize.x - 1) / blockSize.x);
	ComputeBCECost<<<numBlocks, blockSize>>>(predictions.get(), target.get(), predictions.shape.x, cost);
	cudaDeviceSynchronize();

	float cost_value = *cost;
	cudaFree(cost);

	return cost_value;
}

MatrixBuffer BCECost::dCost(MatrixBuffer predictions, MatrixBuffer target, MatrixBuffer dY) {
	dim3 blockSize(256);
	dim3 numBlocks((predictions.shape.x + blockSize.x - 1) / blockSize.x);
	dBCECost<<<numBlocks, blockSize>>>(predictions.get(), target.get(),
														   dY.get(), predictions.shape.x);
	return dY;
}
