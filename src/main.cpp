#include "matrix.h"
#include "sigmoid.h"

int main(void)
{
    Shape s(3, 3);
    Matrix a(s);
    a[0][0] = 1;
    a[0][1] = 2;
    a[0][2] = 3;
    a[1][0] = 3;
    a[1][1] = 2;
    a[1][2] = 1;
    a[2][0] = 2;
    a[2][1] = 1;
    a[2][2] = 3;
    a.print();
    // Matrix b(s);
    // b[0][0] = -5.0/12.0;
    // b[0][1] = 0.25;
    // b[0][2] = 1.0/3.0;
    // b[1][0] = 7.0/12.0;
    // b[1][1] = 0.25;
    // b[1][2] = -2.0/3.0;
    // b[2][0] = 1.0/12.0;
    // b[2][1] = -0.25;
    // b[2][2] = 1.0/3.0;
    // b.print();

    Sigmoid sig(s);
    auto r = sig.forward(a);
    Matrix(r).print();

    return 0;
}
