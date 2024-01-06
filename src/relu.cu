#include "relu.h"

__global__ void reluForward(float* Z, float* A,
                                      int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Zx * Zy) {
        A[index] = fmaxf(Z[index], 0);
    }
}

__global__ void reluBackprop(float* Z, float* dA, float* dZ,
                                       int Zx, int Zy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Zx * Zy) {
        if (Z[index] > 0) {
            dZ[index] = dA[index];
        }
        else {
            dZ[index] = 0;
        }
    }
}

Relu::Relu(Shape shape) :
    shape(shape), A(shape), Z(shape), dZ(shape)
{
    A.allocate();
    dZ.allocate();
}

MatrixBuffer& Relu::forward(MatrixBuffer& Z) {
    this->Z = Z;
    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    reluForward<<<num_of_blocks, block_size>>>(Z.get(), A.get(), Z.shape.x, Z.shape.y);

    return A;
}

MatrixBuffer& Relu::backprop(MatrixBuffer& dA, float learning_rate) {
    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
    reluBackprop<<<num_of_blocks, block_size>>>(
            Z.get(), dA.get(), dZ.get(),
            Z.shape.x, Z.shape.y);

    return dZ;
}
