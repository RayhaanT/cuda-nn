#include "linear.h"
#include "matrix.h"
#include "sigmoid.h"
#include "relu.h"

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

    Matrix b(s);
    b[0][0] = -5.0/12.0;
    b[0][1] = 0.25;
    b[0][2] = 1.0/3.0;
    b[1][0] = 7.0/12.0;
    b[1][1] = 0.25;
    b[1][2] = -2.0/3.0;
    b[2][0] = 1.0/12.0;
    b[2][1] = -0.25;
    b[2][2] = 1.0/3.0;
    b.print();

    Matrix z(s);
    z[0][0] = 0;
    z[0][1] = 0;
    z[0][2] = 0;
    z[1][0] = 0;
    z[1][1] = 0;
    z[1][2] = 0;
    z[2][0] = 0;
    z[2][1] = 0;
    z[2][2] = 0;

    Linear lin(b, z);
    auto r = lin.forward(a);
    Matrix(r).print();

    return 0;
}
