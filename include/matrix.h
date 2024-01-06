#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

struct Shape {
    int x;
    int y;
    Shape(int x, int y) : x(x), y(y) {}
    bool operator==(Shape const& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
};

class MatrixBuffer {
private:
    std::shared_ptr<float> data;
    bool allocated;

public:
    Shape shape;

    MatrixBuffer(Shape shape);

    void allocate();
    void write(float* from);
    void read(float* to);
    float* read();
    bool isAllocated() { return allocated; }
    float* get() { return data.get(); }
};

class Matrix {
private:
    bool deviceAllocated;
    std::unique_ptr<float> dataHost;
    MatrixBuffer dataDevice;

public:
    Shape shape;

    Matrix(Shape shape);
    Matrix(MatrixBuffer buf);

    void writeThrough();
    void print();

    operator MatrixBuffer&() {
        writeThrough();
        return dataDevice;
    }

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

    Row operator[](const int index);
    const ConstRow operator[](const int index) const;
};

#endif
