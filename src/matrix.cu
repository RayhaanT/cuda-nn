#include "matrix.h"
#include <utility>

MatrixBuffer::MatrixBuffer(Shape shape) :
    data(nullptr), allocated(false), shape(shape) { }

MatrixBuffer::MatrixBuffer(MatrixBuffer&& buf) :
    data(std::move(buf.data)), allocated(buf.isAllocated()), shape(buf.shape) {}

void MatrixBuffer::allocate() {
    if (!allocated) {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.x * sizeof(float));
        data = std::unique_ptr<float>(device_memory);
        allocated = true;
    }
}

void MatrixBuffer::write(float* from) {
    cudaMemcpy(data.get(), from, shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
}

void MatrixBuffer::read(float* to) {
    cudaMemcpy(to, data.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
}

Matrix::Matrix(Shape shape) :
    deviceAllocated(false), shape(shape), dataDevice(shape)
{
    dataHost = std::unique_ptr<float>(new float[shape.x*shape.y]);
}

Matrix::Matrix(MatrixBuffer &&buf) :
    deviceAllocated(buf.isAllocated()),
    dataDevice(std::move(buf)),
    shape(buf.shape)
{
    dataHost = std::unique_ptr<float>(new float[shape.x*shape.y]);
}

void Matrix::writeThrough() {
    if (!deviceAllocated) {
        dataDevice.allocate();
    }
    dataDevice.write(dataHost.get());
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
