//
// Created by morshed on 4/3/2019.
//

#ifndef MATRIXUTILITY_H
#define MATRIXUTILITY_H

#include "tfhe.h"
#include "tfhe_io.h"

EXPORT void testPrintMatrix();
EXPORT LweSample_16* matrixToVector(LweSample_16 ***matrix, int row, int col, int bitSize,
                                    const TFheGateBootstrappingCloudKeySet *bk);
EXPORT LweSample_16*** vectorToMatrix(LweSample_16 *vector, int row, int col, int bitSize,
                                      const TFheGateBootstrappingCloudKeySet *bk);
EXPORT LweSample_16*** matrixToDevice(LweSample_16 ***h_matrix, int row, int col, int bitSize,
                                      const TFheGateBootstrappingCloudKeySet *bk);
EXPORT LweSample_16** matMul_prepareLeftMat(LweSample_16 ***matrix, int row, int col, int nDuplication, int bitSize,
                                            const TFheGateBootstrappingCloudKeySet *bk);
EXPORT LweSample_16** matMul_prepareRightMat(LweSample_16 ***matrix, int row, int col, int nDuplication, int bitSize,
                                             const TFheGateBootstrappingCloudKeySet *bk);
#endif
