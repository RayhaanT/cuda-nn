#include "matrix.h"
#include <iostream>

MatrixBuffer::MatrixBuffer(Shape shape) :
    data(nullptr), allocated(false), shape(shape) { }

void MatrixBuffer::allocate() {
    if (!allocated) {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.x * sizeof(float));
        data = std::shared_ptr<float>(device_memory, [&](float* ptr){ cudaFree(ptr); });
        allocated = true;
    }
}

void MatrixBuffer::write(float* from) {
    cudaMemcpy(data.get(), from, shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
}

void MatrixBuffer::read(float* to) {
    cudaMemcpy(to, data.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
}

float* MatrixBuffer::read() {
    float* hostBuf = new float[shape.x*shape.y];
    read(hostBuf);
    return hostBuf;
}

Matrix::Matrix(Shape shape) :
    deviceAllocated(false), shape(shape), dataDevice(shape)
{
    dataHost = std::unique_ptr<float>(new float[shape.x*shape.y]);
}

Matrix::Matrix(MatrixBuffer buf) :
    deviceAllocated(buf.isAllocated()),
    dataDevice(buf),
    shape(buf.shape)
{
    dataHost = std::unique_ptr<float>(dataDevice.read());
}

void Matrix::writeThrough() {
    if (!deviceAllocated) {
        dataDevice.allocate();
    }
    dataDevice.write(dataHost.get());
}

void Matrix::print() {
    for (int y = 0; y < shape.y; y++) {
        for (int x = 0; x < shape.x; x++) {
            std::cout << (*this)[y][x] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix::Row::Row(int index, Matrix* parent) :
    index(index), parent(parent)
{ }

float& Matrix::Row::operator[](const int ind) {
    return parent->dataHost.get()[ind + parent->shape.x*index];
}

Matrix::ConstRow::ConstRow(int index, const Matrix* parent) :
    index(index), parent(parent)
{ }

const float& Matrix::ConstRow::operator[](const int ind) {
    return parent->dataHost.get()[ind + parent->shape.x*index];
}

Matrix::Row Matrix::operator[](const int index) {
    return Matrix::Row(index, this);
}

const Matrix::ConstRow Matrix::operator[](const int index) const {
    return Matrix::ConstRow(index, this);
}
