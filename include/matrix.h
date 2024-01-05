#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

struct Shape {
    int x;
    int y;
    Shape(int x, int y) : x(x), y(y) {}
};

class MatrixBuffer {
private:
    std::unique_ptr<float> data;
    bool allocated;

public:
    Shape shape;

    MatrixBuffer(Shape shape);
    MatrixBuffer(MatrixBuffer &&buf);

    void allocate();
    void write(float* from);
    void read(float* to);
    bool isAllocated() { return allocated; }
};

class Matrix {
private:
    bool deviceAllocated;
    std::unique_ptr<float> dataHost;
    MatrixBuffer dataDevice;

public:
    Shape shape;

    Matrix(Shape shape);
    Matrix(MatrixBuffer &&buf);

    void writeThrough();

    struct Row {
    private:
        int index;
        Matrix* parent;
    public:
        Row(int index, Matrix* parent);
        float& operator[](const int index);
    };
    struct ConstRow {
    private:
        int index;
        const Matrix* parent;
    public:
        ConstRow(int index, const Matrix* parent);
        const float& operator[](const int index);
    };
    friend Row;

    Row operator[](const int index);
    const ConstRow operator[](const int index) const;
};

#endif
