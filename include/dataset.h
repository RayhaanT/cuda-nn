#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"
#include <vector>

class Dataset {
private:
    int dimension;
	int batchSize;
	int batchNum;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	Dataset(int dimension, int batchsz, int batchn);

    int getDimension();
	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();
};

#endif
