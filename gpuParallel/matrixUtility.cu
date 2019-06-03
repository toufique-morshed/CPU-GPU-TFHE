#include "matrixUtility.h"
#include <iostream>


EXPORT void testPrintMatrix() {
    std::cout << " testing matrix operations" << std::endl;
}

//considers the matrix is in GPU
EXPORT LweSample_16* matrixToVector(LweSample_16 ***matrix, int row, int col, int bitSize,
                                    const TFheGateBootstrappingCloudKeySet *bk){
    std::cout << "testing matrix to vector" << std::endl;
    const int n = bk->bk->in_out_params->n;
    int vLength = row * col;
    int totalBitSize = vLength * bitSize;
    LweSample_16 *vector = convertBitToNumberZero_GPU(totalBitSize, bk);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int index = i * col + j;
            int sI = index * bitSize;
            cudaMemcpy(vector->a + sI * n, matrix[i][j]->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(vector->b + sI, matrix[i][j]->b, bitSize * sizeof(int));
            memcpy(vector->current_variance + sI, matrix[i][j]->current_variance, bitSize * sizeof(double));
        }
    }
    return vector;
}
//considers the matrix is in GPU
EXPORT LweSample_16*** vectorToMatrix(LweSample_16 *vector, int row, int col, int bitSize,
                                      const TFheGateBootstrappingCloudKeySet *bk){
    std::cout << "testing vector to matrix" << std::endl;
    const int n = bk->bk->in_out_params->n;
    LweSample_16 ***matrix = new LweSample_16**[row];
    for (int i = 0; i < row; ++i) {
        matrix[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            matrix[i][j] = convertBitToNumberZero_GPU(bitSize, bk);
            int index = i * col + j;
            int sI = index * bitSize;
            cudaMemcpy(matrix[i][j]->a, vector->a + sI * n, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(matrix[i][j]->b, vector->b + sI, bitSize * sizeof(int));
            memcpy(matrix[i][j]->current_variance, vector->current_variance + sI, bitSize * sizeof(double));
        }
    }
    return  matrix;
}
//port the matrix to GPU
EXPORT LweSample_16*** matrixToDevice(LweSample_16 ***h_matrix, int row, int col, int bitSize,
                                      const TFheGateBootstrappingCloudKeySet *bk) {
    std::cout << "taking matrix to device" << std::endl;
    const int n = bk->bk->in_out_params->n;
    LweSample_16 ***d_matrix = new LweSample_16**[row];
    for (int i = 0; i < row; ++i) {
        d_matrix[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            d_matrix[i][j] = convertBitToNumberZero_GPU(bitSize, bk);
            cudaMemcpy(d_matrix[i][j]->a, h_matrix[i][j]->a, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
            memcpy(d_matrix[i][j]->b, h_matrix[i][j]->b, bitSize * sizeof(int));
            memcpy(d_matrix[i][j]->current_variance, h_matrix[i][j]->current_variance, bitSize * sizeof(double));
        }
    }
    return  d_matrix;
}
//prepare left matrix to multiply and create LweSample array (vector)
EXPORT LweSample_16** matMul_prepareLeftMat(LweSample_16 ***matrix, int row, int col, int nDuplication, int bitSize,
                                            const TFheGateBootstrappingCloudKeySet *bk) {
    std::cout << "Preparing left matrix to multiply" << std::endl;
    int vLen = row * col * nDuplication;
    LweSample_16 **vector = new LweSample_16*[vLen];
    for (int i = 0; i < row; ++i) {
        for (int k = 0; k < nDuplication; ++k) {
            for (int j = 0; j < col; ++j) {
                int index = i * nDuplication * col + k * col + j;
                vector[index] = matrix[i][j];
            }
        }
    }
    return vector;
}

//prepare right matrix to multiply and create LweSample array (vector)
EXPORT LweSample_16** matMul_prepareRightMat(LweSample_16 ***matrix, int row, int col, int nDuplication, int bitSize,
                                            const TFheGateBootstrappingCloudKeySet *bk) {
    std::cout << "Preparing right matrix to multiply" << std::endl;
    int vLen = row * col * nDuplication;
    LweSample_16 **vector = new LweSample_16*[vLen];
    for (int k = 0; k < nDuplication; ++k) {
        for (int i = 0; i < col; ++i) {
            for (int j = 0; j < row; ++j) {
                int index = k * col * row + i * row  + j;
                vector[index] = matrix[j][i];
            }
        }
    }
    return vector;
}
