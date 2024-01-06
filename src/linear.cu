#include "linear.h"
#include "matrix.h"
#include <random>

__global__ void linearForward( float* W, float* A, float* Z, float* b,
									int Wx, int Wy,
									int Ax, int Ay) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Zx = Ax;
	int Zy = Wy;

	float val = 0;

	if (row < Zy && col < Zx) {
		for (int i = 0; i < Wx; i++) {
			val += W[row * Wx + i] * A[i * Ax + col];
		}
		Z[row * Zx + col] = val + b[row];
	}
}

__global__ void linearBackprop(float* W, float* dZ, float *dA,
									int Wx, int Wy, int dZx, int dZy) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int dAx = dZx;
	int dAy = Wx;

	float val = 0.0f;

	if (row < dAy && col < dAx) {
		for (int i = 0; i < Wy; i++) {
			val += W[i * Wx + row] * dZ[i * dZx + col];
		}
		dA[row * dAx + col] = val;
	}
}

__global__ void linearUpdateWeights(  float* dZ, float* A, float* W,
										   int dZx, int dZy, int Ax, int Ay,
										   float learningRate) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int Wx = Ay;
	int Wy = dZy;

	float val = 0.0f;

	if (row < Wy && col < Wx) {
		for (int i = 0; i < dZx; i++) {
			val += dZ[row * dZx + i] * A[col * Ax + i];
		}
		W[row * Wx + col] = W[row * Wx + col] - learningRate * (val / Ax);
	}
}

__global__ void linearUpdateBias(  float* dZ, float* b,
										int dZx, int dZy,
										int bx,
										float learningRate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZx * dZy) {
		int x = index % dZx;
		int y = index / dZx;
		atomicAdd(&b[y], - learningRate * (dZ[y * dZx + x] / dZx));
	}
}

Linear::Linear(Shape wShape, Shape zShape) :
	W(wShape), b(Shape(zShape.x, wShape.y)), Z(zShape),
    A(Shape(zShape.x, wShape.y)), dA(Shape(zShape.x, wShape.y))
{
	b.allocate();
	W.allocate();
    Z.allocate();
    dA.allocate();
	initBias();
	initWeights();
}

Linear::Linear(MatrixBuffer& w, MatrixBuffer& b) :
	W(w), b(b), Z(b.shape),
    A(Shape(b.shape.x, w.shape.x)), dA(Shape(b.shape.x, w.shape.y))
{
    Z.allocate();
    dA.allocate();
	initBias();
}

void Linear::initBias() {
    Matrix mb(b);

	for (int x = 0; x < b.shape.x; x++) {
		mb[0][x] = 0;
	}

	mb.writeThrough();
}

void Linear::initWeights() {
	std::default_random_engine generator;
	std::normal_distribution<float> normalDist(0.0, 1.0);
    Matrix mW(W);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			mW[y][x] = normalDist(generator) * WEIGHTS_INIT;
		}
	}

	mW.writeThrough();
}

MatrixBuffer& Linear::forward(MatrixBuffer& A) {
	this->A = A;
	dim3 blockSize(8, 8);
	dim3 blockNum(	(Z.shape.x + blockSize.x - 1) / blockSize.x,
						(Z.shape.y + blockSize.y - 1) / blockSize.y);
	linearForward<<<blockNum, blockSize>>>( W.get(), A.get(), Z.get(), b.get(),
												  W.shape.x, W.shape.y,
												  A.shape.x, A.shape.y);
	return Z;
}

MatrixBuffer& Linear::backprop(MatrixBuffer& dZ, float learningRate) {
	dim3 blockSize(8, 8);
	dim3 blockNum(	(A.shape.x + blockSize.x - 1) / blockSize.x,
						(A.shape.y + blockSize.y - 1) / blockSize.y);
	linearBackprop<<<blockNum, blockSize>>>( W.get(), dZ.get(), dA.get(),
												   W.shape.x, W.shape.y,
												   dZ.shape.x, dZ.shape.y);
	updateBias(dZ, learningRate);
	updateWeights(dZ, learningRate);

	return dA;
}

void Linear::updateWeights(MatrixBuffer& dZ, float learningRate) {
	dim3 blockSize(8, 8);
	dim3 blockNum(	(W.shape.x + blockSize.x - 1) / blockSize.x,
						(W.shape.y + blockSize.y - 1) / blockSize.y);
	linearUpdateWeights<<<blockNum, blockSize>>>(dZ.get(), A.get(), W.get(),
													   dZ.shape.x, dZ.shape.y,
													   A.shape.x, A.shape.y,
													   learningRate);
}

void Linear::updateBias(MatrixBuffer& dZ, float learningRate) {
	dim3 blockSize(256);
	dim3 blockNum( (dZ.shape.y * dZ.shape.x + blockSize.x - 1) / blockSize.x);
	linearUpdateBias<<<blockNum, blockSize>>>(dZ.get(), b.get(),
													dZ.shape.x, dZ.shape.y,
													b.shape.x, learningRate);
}
