#include "dataset.h"
#include <random>

Dataset::Dataset(int dim, int batchsz, int batchn) :
	dimension(dim), batchSize(batchsz), batchNum(batchn)
{
	for (int i = 0; i < batchNum; i++) {
		batches.push_back(Matrix(Shape(batchSize, dimension)));
		targets.push_back(Matrix(Shape(batchSize, 1)));

		for (int k = 0; k < batchSize; k++) {
            bool pos = true;
            for (int j = 0; j < dimension; j++) {
                float next = static_cast<float>(rand()) / RAND_MAX - 0.5;
                if (next < 0) {
                    pos = false;
                }
                batches[i][j][k] = next;
            }

			if (pos) {
				targets[i][0][k] = 1;
			}
			else {
				targets[i][0][k] = 0;
			}
		}

		batches[i].writeThrough();
		targets[i].writeThrough();
	}
}

int Dataset::getNumOfBatches() {
	return batchNum;
}

std::vector<Matrix>& Dataset::getBatches() {
	return batches;
}

std::vector<Matrix>& Dataset::getTargets() {
	return targets;
}
