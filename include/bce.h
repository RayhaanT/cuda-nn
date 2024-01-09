#include "matrix.h"

class BCECost {
public:
    float cost(MatrixBuffer predictions, MatrixBuffer target);
    MatrixBuffer dCost(MatrixBuffer predictions, MatrixBuffer target, MatrixBuffer dY);
};
