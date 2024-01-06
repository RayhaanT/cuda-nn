#include "sigmoid.h"

__device__ float sigmoid(float x) {
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidForward(float* Z, float* A, int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Zx * Zy) {
        A[index] = sigmoid(Z[index]);
    }
}

__global__ void sigmoidBackprop(float* Z, float* dA, float* dZ,
                                          int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Zx * Zy) {
        dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
    }
}

Sigmoid::Sigmoid(Shape shape) :
    shape(shape), A(shape), Z(shape), dZ(shape)
{
    A.allocate();
    dZ.allocate();
}

MatrixBuffer& Sigmoid::forward(MatrixBuffer& Z) {
    this->Z = Z;
    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    sigmoidForward<<<num_of_blocks, block_size>>>(Z.get(), A.get(), Z.shape.x, Z.shape.y);

    return A;
}

MatrixBuffer& Sigmoid::backprop(MatrixBuffer& dA, float learning_rate) {
    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
    sigmoidBackprop<<<num_of_blocks, block_size>>>(
            Z.get(), dA.get(), dZ.get(),
            Z.shape.x, Z.shape.y);

    return dZ;
}
