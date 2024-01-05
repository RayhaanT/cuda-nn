#include "matrix.h"

Matrix::Matrix(Shape shape) :
    deviceAllocated(false), hostAllocated(false), shape(shape) { }

void Matrix::allocateHostMem() {
    if (!hostAllocated) {
        dataHost = std::unique_ptr<float>(new float[shape.width*shape.height]);
        hostAllocated = true;
    }
}

void Matrix::allocateDeviceMem() {
    if (!deviceAllocated) {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.width * shape.width * sizeof(float));
        dataDevice = std::unique_ptr<float>(device_memory);
        deviceAllocated = true;
    }
}

void Matrix::writeThrough() {
    if (!deviceAllocated) {
        allocateDeviceMem();
    }
    cudaMemcpy(dataDevice.get(), dataHost.get(),
            shape.width * shape.height * sizeof(float), cudaMemcpyHostToDevice);
}

Matrix::Row::Row(int index, Matrix* parent) :
    index(index), parent(parent)
{ }

float& Matrix::Row::operator[](const int ind) {
    return parent->dataHost.get()[ind + parent->shape.width*index];
}

Matrix::ConstRow::ConstRow(int index, const Matrix* parent) :
    index(index), parent(parent)
{ }

const float& Matrix::ConstRow::operator[](const int ind) {
    return parent->dataHost.get()[ind + parent->shape.width*index];
}

Matrix::Row Matrix::operator[](const int index) {
    return Matrix::Row(index, this);
}

const Matrix::ConstRow Matrix::operator[](const int index) const {
    return Matrix::ConstRow(index, this);
}
