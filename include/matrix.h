#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

struct Shape {
    int width;
    int height;
};



class Matrix {
private:
    bool deviceAllocated;
    bool hostAllocated;

    void allocateDeviceMem();
    void allocateHostMem();

    std::unique_ptr<float> dataDevice;
    std::unique_ptr<float> dataHost;

public:
    Shape shape;

    Matrix(Shape shape);

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
