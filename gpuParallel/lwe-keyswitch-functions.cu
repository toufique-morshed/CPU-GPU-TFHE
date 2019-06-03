#ifndef TFHE_TEST_ENVIRONMENT

#include <iostream>
#include "lwe-functions.h"
#include "lwekeyswitch.h"
#include "numeric_functions.h"
#include <random>
#include <assert.h>

using namespace std;
#else
#undef EXPORT
#define EXPORT
#endif


/*
Renormalization of KS
 * compute the error of the KS that has been generated and translate the ks to recenter the gaussian in 0
*/
void renormalizeKSkey(LweKeySwitchKey *ks, const LweKey *out_key, const int *in_key) {
    const int n = ks->n;
    const int basebit = ks->basebit;
    const int t = ks->t;
    const int base = 1 << basebit;

    Torus32 phase;
    Torus32 temp_err;
    Torus32 error = 0;
    // double err_norm = 0; 

    // compute the average error
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < t; ++j) {
            for (int h = 1; h < base; ++h) { // pas le terme en 0
                // compute the phase 
                phase = lwePhase(&ks->ks[i][j][h], out_key);
                // compute the error 
                Torus32 x = (in_key[i] * h) * (1 << (32 - (j + 1) * basebit));
                temp_err = phase - x;
                // sum all errors 
                error += temp_err;
            }
        }
    }
    int nb = n * t * (base - 1);
    error = dtot32(t32tod(error) / nb);

    // relinearize
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < t; ++j) {
            for (int h = 1; h < base; ++h) { // pas le terme en 0
                ks->ks[i][j][h].b -= error;
            }
        }
    }

}


/**
 * fills the KeySwitching key array
 * @param result The (n x t x base) array of samples. 
 *        result[i][j][k] encodes k.s[i]/base^(j+1)
 * @param out_key The LWE key to encode all the output samples 
 * @param out_alpha The standard deviation of all output samples
 * @param in_key The (binary) input key
 * @param n The size of the input key
 * @param t The precision of the keyswitch (technically, 1/2.base^t)
 * @param basebit Log_2 of base
 */
void lweCreateKeySwitchKey_fromArray(LweSample ***result,
                                     const LweKey *out_key, const double out_alpha,
                                     const int *in_key, const int n, const int t, const int basebit) {
    const int base = 1 << basebit;       // base=2 in [CGGI16]

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < t; j++) {
            for (int k = 0; k < base; k++) {
                Torus32 x = (in_key[i] * k) * (1 << (32 - (j + 1) * basebit));
                lweSymEncrypt(&result[i][j][k], x, out_alpha, out_key);
                //printf("i,j,k,ki,x,phase=%d,%d,%d,%d,%d,%d\n",i,j,k,in_key->key[i],x,lwePhase(&result->ks[i][j][k],out_key));
            }
        }
    }
}


/**
 * translates the message of the result sample by -sum(a[i].s[i]) where s is the secret
 * embedded in ks.
 * @param result the LWE sample to translate by -sum(ai.si). 
 * @param ks The (n x t x base) key switching key 
 *        ks[i][j][k] encodes k.s[i]/base^(j+1)
 * @param params The common LWE parameters of ks and result
 * @param ai The input torus array
 * @param n The size of the input key
 * @param t The precision of the keyswitch (technically, 1/2.base^t)
 * @param basebit Log_2 of base
 */
void lweKeySwitchTranslate_fromArray(LweSample *result,
                                     const LweSample ***ks, const LweParams *params,
                                     const Torus32 *ai,
                                     const int n, const int t, const int basebit) {
    const int base = 1 << basebit;       // base=2 in [CGGI16]
    const int32_t prec_offset = 1 << (32 - (1 + basebit * t)); //precision
    const int mask = base - 1;

//    cout << "n: " << n << endl;//1024
//    cout << "t: " << t << endl;//8
//    cout << "basebit: " << basebit << endl;//2

    for (int i = 0; i < n; i++) {
        const uint32_t aibar = ai[i] + prec_offset;
        for (int j = 0; j < t; j++) {
            const uint32_t aij = (aibar >> (32 - (j + 1) * basebit)) & mask;
            if (aij != 0) {
                lweSubTo(result, &ks[i][j][aij], params);
            }
        }
    }
//    cout << "old: ";
//    for (int j = 0; j < 10; ++j) {
//        cout << result->a[j] << " ";
//    }
//    cout << endl;
}

//new
__global__ void lweKeySwitchVectorSubstraction_gpu(int *destination, int *source, int startIndex, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[startIndex + id] -= source[id];
    }
}

__global__ void addScalarToSelf(uint32_t *destination, int32_t scalarVal, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] += scalarVal;
    }
}

__global__ void calculateAijFromAibar(uint32_t *aij, uint32_t *aibar,
                                      int j, int basebit, int mask, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        aij[id] = (aibar[id] >> (32 - (j + 1) * basebit)) & mask;
    }
}

__global__ void getAibar(uint32_t *d_aibar, const Torus32 *ai, int32_t prec_offset, int i, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int startIndex = id * n;
        d_aibar[id] = ai[startIndex + i] + prec_offset;
    }
}

__global__ void lweKeySwitchVectorSubstraction_gpu_testing(int *destination, Torus32 **source, uint32_t *d_aij,
                                                           int n, int params_n, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / params_n;
        int aij = d_aij[bitIndex];
        if (aij != 0) {
            destination[id] -= source[aij][id];
        }
    }
}

__global__ void lweKeySwitchVectorSubstraction_gpu_testing_2(int *destination, Torus32 **source, uint32_t *d_aij,
                                                           int n, int params_n, int nOutputs, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / params_n;
        int mulBitSizeParamsn = bitSize * params_n;
        int aij = d_aij[bitIndex];
        if (aij != 0) {
            int index = id % mulBitSizeParamsn;
            destination[id] -= source[aij][index];
        }
    }
}

__global__ void lweKeySwitchVectorSubstraction_gpu_testing_2_vector(int *destination, Torus32 **source, uint32_t *d_aij,
                                                             int n, int params_n, int vLength, int nOutputs, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / params_n;
        int mulBitSizeParamsn = bitSize * params_n;
        int aij = d_aij[bitIndex];
        if (aij != 0) {
            int index = id % mulBitSizeParamsn;
            destination[id] -= source[aij][index];
        }
    }
}

__global__ void
lweKeySwitchVectorSubstraction_B_gpu_testing(int *destination, int *source, uint32_t *index_source, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int index = id % length;
        destination[id] -= source[index_source[index]];
    }
}

__global__ void
lweKeySwitchVectorSubstraction_B_gpu_testing_2(int *destination, int *source, uint32_t *index_source, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int index = index_source[id];
        destination[id] -= source[index];
    }
}

__global__ void
lweKeySwitchVectorSubstraction_B_gpu_testing_2_vector(int *destination, int *source, uint32_t *index_source, int vLength, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int index = index_source[id];
        destination[id] -= source[index];
    }
}

__global__ void
lweKeySwitchVectorAddition_cv_gpu_testing(double *destination, double *source, uint32_t *index_source, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int index = id % length;
        destination[id] += source[index_source[index]];
    }
}

__global__ void
lweKeySwitchVectorAddition_cv_gpu_testing_2(double *destination, double *source, uint32_t *index_source, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int index = index_source[id];
        destination[id] += source[index];
    }
}

__global__ void
lweKeySwitchVectorAddition_cv_gpu_testing_2_vector(double *destination, double *source, uint32_t *index_source,
                                                   int vLength, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int index = index_source[id];
        destination[id] += source[index];
    }
}



__global__ void getAibarCoalesce(uint32_t *d_aibar, const Torus32 *ai, int32_t prec_offset, int bitSize, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int i = id/bitSize;
        int tID = id % bitSize;
        int startIndex = tID * n;
        d_aibar[id] = ai[startIndex + i] + prec_offset;
    }
}

__global__ void calculateAijFromAibarCoalesce(uint32_t *aij, uint32_t *aibar, int bitSize, int t, int basebit, int mask, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int i = id /(bitSize * t);
        int j = (id / bitSize) % t;
        int tId = id % bitSize;
        aij[id] = (aibar[i * bitSize + tId] >> (32 - (j + 1) * basebit)) & mask;
    }
}



__global__ void lweKeySwitchVectorSubstraction_gpu_testing_coalesce(int *destinationA, Torus32 *sourceA, uint32_t *d_aij,
                                                                    int *destinationB, int *sourceB,
                                                                    double *destinationCV, double *sourceCV,
                                                                    int ks_n, int ks_t, int ks_base, int bitSize, int n,
                                                                    int params_n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int A = ks_n, B = ks_t, C = ks_base, D = params_n;
        int bitIndex = id/params_n;
        int tId = id % (bitSize * params_n);
        for (int i = 0; i < n; ++i) {
            int sI =  i * (ks_t * bitSize);
            for (int j = 0; j < ks_t; ++j) {
                int sI2 = sI + j * bitSize;
                int aij = d_aij[sI2 + bitIndex];
                if (aij != 0) {
                    destinationA[id] -= sourceA[i * B * C * D + j * C * D + aij * D + (id % D)];//sourceA[(i * B * C * D + j * C * D+ aij * params_n +  id)];//source[aij][id];
                }
                if(id < bitSize) {
                    int bi = d_aij[sI2 + id];
                    destinationB[id] -= sourceB[i * B * C + j * C + bi];
                    destinationCV[id] += sourceCV[i * B * C + j * C + bi];

                }
            }
        }
    }
}

__global__ void lweKeySwitchVectorSubstraction_gpu_testing_coalesce_2(int *destinationA, Torus32 *sourceA, uint32_t *d_aij,
                                                                    int *destinationB, int *sourceB,
                                                                    double *destinationCV, double *sourceCV,
                                                                    int ks_n, int ks_t, int ks_base, int nOutputs, int bitSize, int n,
                                                                    int params_n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int A = ks_n, B = ks_t, C = ks_base, D = params_n;
        int bitIndex = id/params_n;
//        int tId = id % (bitSize * params_n * nOutputs);
        int mulBitSizeParamsn = bitSize * params_n;
        for (int i = 0; i < n; ++i) {//n
            int sI = i * (ks_t * bitSize * nOutputs);
            for (int j = 0; j < ks_t; ++j) {
                int sI2 = sI + j * bitSize * nOutputs;
                int aij = d_aij[sI2 + bitIndex];
                if (aij != 0) {
                    int index = id % mulBitSizeParamsn;
                    destinationA[id] -= sourceA[i * B * C * D + j * C * D + aij * D + (id % D)];
                }
                if(id < bitSize * nOutputs) {
                    int bi = d_aij[sI2 + id];
                    destinationB[id] -= sourceB[i * B * C + j * C + bi];
//                    destinationCV[id] += sourceCV[i * B * C + j * C + bi];
                }
            }
        }
    }
}

__global__ void lweKeySwitchVectorSubstraction_gpu_testing_coalesce_2_vector(int *destinationA, Torus32 *sourceA, uint32_t *d_aij,
                                                                             int *destinationB, int *sourceB,
                                                                             double *destinationCV, double *sourceCV,
                                                                             int ks_n, int ks_t, int ks_base, int nOutputs, int vLength, int bitSize, int n,
                                                                             int params_n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int A = ks_n, B = ks_t, C = ks_base, D = params_n;
        int bitIndex = id/params_n;
        int mulBitSizeParamsn = bitSize * params_n;
        for (int i = 0; i < n; ++i) {//n
            int sI = i * (ks_t * bitSize * nOutputs * vLength);
            for (int j = 0; j < ks_t; ++j) {
                int sI2 = sI + j * bitSize * nOutputs * vLength;
                int aij = d_aij[sI2 + bitIndex];
                if (aij != 0) {
                    int index = id % mulBitSizeParamsn;
                    destinationA[id] -= sourceA[i * B * C * D + j * C * D + aij * D + (id % D)];
                }
                if(id < bitSize * nOutputs * vLength) {
                    int bi = d_aij[sI2 + id];
                    destinationB[id] -= sourceB[i * B * C + j * C + bi];
                    destinationCV[id] += sourceCV[i * B * C + j * C + bi];
                }
            }

        }


    }
}




void lweKeySwitchTranslate_fromArray_16(LweSample_16 *result,
                                        const LweSample ***ks,
                                        const LweParams *params,
                                        const Torus32 *ai,
                                        const int n, const int t,
                                        const int basebit, int bitSize, Torus32 ****ks_a_gpu,
                                        Torus32 ****ks_a_gpu_extended,
                                        int ***ks_b_gpu, double ***ks_cv_gpu,
                                        Torus32* ks_a_gpu_extendedPtr,
                                        Torus32 *ks_b_gpu_extendedPtr,
                                        double *ks_cv_gpu_extendedPtr) {
    const int base = 1 << basebit;       // base=2 in [CGGI16]
    const int32_t prec_offset = 1 << (32 - (1 + basebit * t)); //precision
    const int mask = base - 1;
    int BLOCKSIZE = params->n;
    int gridSize;
//    cout << "basebit: " << basebit << endl;//2
//    cout << "prec_offset: " << prec_offset << endl;//32768
//    cout << "mask: " << mask << endl;//3
//    cout << "base: " << base << endl;//4
//    cout << "t: " << t << endl;//8
    //increment start
    /*send b of key switch to cuda and compute over there*/
    //create b for result in cuda
    int *result_b_gpu;
    double *result_cv_gpu;
    cudaMalloc(&result_b_gpu, bitSize * sizeof(int));
    cudaMalloc(&result_cv_gpu, bitSize * sizeof(double));
    //copy result->b to result_b_cuda
    cudaMemcpy(result_b_gpu, result->b, bitSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cv_gpu, result->current_variance, bitSize * sizeof(double), cudaMemcpyHostToDevice);

    BLOCKSIZE = 1024;

   /*
    cout << "n: " << n << endl;//1024
    cout << "t: " << t << endl;//8
    int gridSize1 = (int) ceil((float) (bitSize * params->n) / BLOCKSIZE);
    for (int i = 0; i < n; ++i) {//n
        gridSize = (int) ceil((float) (bitSize) / BLOCKSIZE);
        getAibar << < gridSize, BLOCKSIZE >> > (d_aibar, ai, prec_offset, i, n, bitSize);
        //testing aibar start
        uint32_t *h_aibar = new uint32_t[bitSize];
        cudaMemcpy(h_aibar, d_aibar, bitSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int j = 0; j < bitSize; ++j) {
            cout << h_aibar[j] << " ";
        }
        cout << endl;
        //testing aibar end
        for (int j = 0; j < t; ++j) {
            calculateAijFromAibar <<<gridSize, BLOCKSIZE>>>(d_aij, d_aibar, j, basebit, mask, bitSize);
            //testing aij start
            cout << "i:" << i << ": ";
            uint32_t *h_aij = new uint32_t[bitSize];
            cudaMemcpy(h_aij, d_aij, bitSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            for (int j = 0; j < bitSize; ++j) {
                cout << h_aij[j] << " ";
            }
            cout << endl;
            //testing aibar end

//            lweKeySwitchVectorSubstraction_gpu_testing<<<gridSize1, BLOCKSIZE>>>
//                                                                       (result->a, ks_a_gpu_extended[i][j],
//                                                                               d_aij, n, params->n, bitSize, bitSize *
//                                                                                                             params->n);
//
//            lweKeySwitchVectorSubstraction_B_gpu_testing<<<1, bitSize>>>
//                                                                 (result_b_gpu, ks_b_gpu[i][j], d_aij, bitSize);
//            lweKeySwitchVectorAddition_cv_gpu_testing<<<1, bitSize>>>
//                                                              (result_cv_gpu, ks_cv_gpu[i][j], d_aij, bitSize);
        }
//        cout << endl;

    }


    double *h_ResCV = new double[bitSize];
    cudaMemcpy(h_ResCV, result_cv_gpu, bitSize * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * params->n;
//        for (int j = 0; j < 10; ++j) {
//            cout << h_ResA[sI + j] << " ";
//        }
//        cout << endl;
        cout << h_ResCV[i] << " ";
    }
    cout << endl;


//    int *h_ResA = new int[bitSize * params->n];
//    cudaMemcpy(h_ResA, result->a, bitSize * params->n * sizeof(Torus32), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * params->n;
//        for (int j = 0; j < 10; ++j) {
//            cout << h_ResA[sI + j] << " ";
//        }
//        cout << endl;
//    }
*/
    //outer loop parallelization
    int coal_d_aibarSize = bitSize * n;
//    cout << "n: " << n << endl;//1024
//    cout << "coal_d_aibarSize: " << coal_d_aibarSize << endl;
    //calculate aibar
    uint32_t *coal_d_aibar;
    cudaMalloc(&coal_d_aibar, coal_d_aibarSize * sizeof(uint32_t));

    gridSize = (int) ceil((float) (coal_d_aibarSize) / BLOCKSIZE);
//    cout << "gridSize: " << gridSize << endl;
    getAibarCoalesce<<<gridSize, BLOCKSIZE>>>(coal_d_aibar, ai, prec_offset, bitSize, n, coal_d_aibarSize);

    //testing aibar start
//    uint32_t *h_coal_d_aibar = new uint32_t[coal_d_aibarSize];
//    cudaMemcpy(h_coal_d_aibar, coal_d_aibar, coal_d_aibarSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < coal_d_aibarSize; ++i) {
//        cout << h_coal_d_aibar[i] << " ";
//    }
//    cout << endl;

/*    for (int i = 0; i < XX; ++i) {
        int sI = i * bitSize;
        for (int j = 0; j < bitSize; ++j) {
            cout << h_coal_d_aibar[sI + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    //testing aibar end
    */
    //calculate aij
    int coal_d_aijSize = n * t * bitSize;
//    cout << "coal_d_aijSize: " << coal_d_aijSize << endl;
    uint32_t  *coal_d_aij;
    cudaMalloc(&coal_d_aij, coal_d_aijSize * sizeof(uint32_t));
    gridSize = (int) ceil((float) (coal_d_aijSize) / BLOCKSIZE);
//    cout << "gridSize: " << gridSize << endl;
    calculateAijFromAibarCoalesce<<<gridSize, BLOCKSIZE>>>(coal_d_aij, coal_d_aibar, bitSize, t, basebit, mask, coal_d_aijSize);

    //testing aij start
//    uint32_t *h_coal_d_aij = new uint32_t[coal_d_aijSize];
//    cudaMemcpy(h_coal_d_aij, coal_d_aij, coal_d_aijSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < coal_d_aijSize; ++i) {
//        cout << h_coal_d_aij[i] << " ";
//    }
//    cout << endl;

/*    for (int i = 0; i < XX; ++i) {
        int sI = i * bitSize * t;
        for (int j = 0; j < t; ++j) {
            int sI2 = sI + j * bitSize;
            cout << "i:" << i << ": ";
            for (int k = 0; k < bitSize; ++k) {
                cout << h_coal_d_aij[sI2 + k] << " ";
            }
//            cout << h_coal_d_aij[sI + j] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    //testing aij end*/
    //calculate result->a
    int length = params->n * bitSize;//500*bitSize
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "gridSize: " << gridSize << endl;
    int ks_n = 1024, ks_t = 8, ks_base = 4;
//    int A = ks_n, B = ks_t, C = ks_base, D = bitSize * params->n;
//        cout << "A: " << A << " D: " << D << endl;

    lweKeySwitchVectorSubstraction_gpu_testing_coalesce<<<gridSize, BLOCKSIZE>>>(result->a, ks_a_gpu_extendedPtr, coal_d_aij,
                                                                                result_b_gpu, ks_b_gpu_extendedPtr,
                                                                                result_cv_gpu, ks_cv_gpu_extendedPtr,
                                                                                ks_n, ks_t, ks_base, bitSize, n,
                                                                                params->n, length);

    cudaMemcpy(result->b, result_b_gpu, bitSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->current_variance, result_cv_gpu, bitSize * sizeof(double), cudaMemcpyDeviceToHost);
    /*int *h_ResA = new int[bitSize * params->n];
    cudaMemcpy(h_ResA, result->a, bitSize * params->n * sizeof(Torus32), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize; ++i) {
        int sI = i * params->n;
        for (int j = 0; j < 10; ++j) {
            cout << h_ResA[sI + j] << " ";
        }
        cout << endl;
    }
//    h_ResCV = new double[bitSize];
//    cudaMemcpy(h_ResCV, result_cv_gpu, bitSize * sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
////        int sI = i * params->n;
////        for (int j = 0; j < 10; ++j) {
////            cout << h_ResA[sI + j] << " ";
////        }
////        cout << endl;
//        cout << h_ResCV[i] << " ";
//    }
//    cout << endl;
    */


    cudaFree(result_b_gpu);
    cudaFree(result_cv_gpu);
    cudaFree(coal_d_aibar);
    cudaFree(coal_d_aij);

    int error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Cuda error code testing here: %d\n", error);
        exit(1);
    }
    /***version 2 ends***/
}

void lweKeySwitchTranslate_fromArray_16_2(LweSample_16 *result,
                                          const LweSample ***ks,
                                          const LweParams *params,
                                          const Torus32 *ai,
                                          const int n, const int t,
                                          const int basebit, int nOutputs,
                                          int bitSize, Torus32 ****ks_a_gpu,
                                          Torus32 ****ks_a_gpu_extended,
                                          int ***ks_b_gpu, double ***ks_cv_gpu,
                                          Torus32* ks_a_gpu_extendedPtr,
                                          Torus32 *ks_b_gpu_extendedPtr,
                                          double *ks_cv_gpu_extendedPtr) {

    const int base = 1 << basebit;       // base=2 in [CGGI16]
    const int32_t prec_offset = 1 << (32 - (1 + basebit * t)); //precision
    const int mask = base - 1;

    int totalBits = nOutputs * bitSize;
    int BLOCKSIZE = 1024;

    //send b of key switch to cuda and compute over there
    int *result_b_gpu;
    double *result_cv_gpu;
    cudaMalloc(&result_b_gpu, totalBits * sizeof(int));
    cudaMalloc(&result_cv_gpu, totalBits * sizeof(double));

    //copy result->b to result_b_cuda
    cudaMemcpy(result_b_gpu, result->b, totalBits * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cv_gpu, result->current_variance, totalBits * sizeof(double), cudaMemcpyHostToDevice);

    /***version 2 starts***/
    /*for (int i = 0; i < n; ++i) {//n
        getAibar<<<gridSize, BLOCKSIZE>>>(d_aibar, ai, prec_offset, i, n, totalBits);
        //testing aibar start
//        uint32_t *h_aibar = new uint32_t[nOutputs * bitSize];
//        cudaMemcpy(h_aibar, d_aibar, nOutputs * bitSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < nOutputs * bitSize; ++j) {
//            cout << h_aibar[j] << " ";
//        }
//        cout << endl;
        //testing aibar end
//        for (int j = 0; j < t; ++j) {//t
//            calculateAijFromAibar<<<gridSize, BLOCKSIZE>>>(d_aij, d_aibar, j, basebit, mask, totalBits);
//            //testing aij start
//            cout << "i:" << i << ": ";
//            uint32_t *h_aij = new uint32_t[bitSize * nOutputs];
//            cudaMemcpy(h_aij, d_aij, nOutputs * bitSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//            for (int j = 0; j < nOutputs * bitSize; ++j) {
//                cout << h_aij[j] << " ";
//            }
//            cout << endl;
            //testing aibar end

//            lweKeySwitchVectorSubstraction_gpu_testing_2<<<gridSize1, BLOCKSIZE>>>
//                                                                      (result->a, ks_a_gpu_extended[i][j],
//                                                                      d_aij, n, params->n, nOutputs, bitSize, length);
//            lweKeySwitchVectorSubstraction_B_gpu_testing_2<<<gridSize, totalBits>>>
//                                                                       (result_b_gpu, ks_b_gpu[i][j],
//                                                                        d_aij, bitSize, totalBits);
//            lweKeySwitchVectorAddition_cv_gpu_testing_2<<<gridSize, totalBits>>>
//                                                                    (result_cv_gpu, ks_cv_gpu[i][j],
//                                                                            d_aij, bitSize, totalBits);
        }
    }

    //test result start
//    double *h_ResCV = new double[totalBits];
//    cudaMemcpy(h_ResCV, result_cv_gpu, totalBits * sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < totalBits; ++i) {
////        int sI = i * params->n;
////        for (int j = 0; j < 10; ++j) {
////            cout << h_ResA[sI + j] << " ";
////        }
////        cout << endl;
//        cout << h_ResCV[i] << " ";
//    }
//    cout << endl;
//    cout << endl;
//    cudaMemcpy(result_b_gpu, result->b, sizeof(int) * totalBits, cudaMemcpyHostToDevice);
//    cudaMemcpy(result_cv_gpu, result->current_variance, sizeof(double) * totalBits, cudaMemcpyHostToDevice);
    //test result->a end
    */

    //calculate aibar
    int coal_d_aibarSize = totalBits * n;//
    uint32_t *coal_d_aibar;
    cudaMalloc(&coal_d_aibar, coal_d_aibarSize * sizeof(uint32_t));
    int gridSize = (int) ceil((float) (coal_d_aibarSize) / BLOCKSIZE);

    getAibarCoalesce<<<gridSize, BLOCKSIZE>>>(coal_d_aibar, ai, prec_offset, totalBits, n, coal_d_aibarSize);
    /*//testing aibar start
    uint32_t *h_coal_d_aibar = new uint32_t[coal_d_aibarSize];
    cudaMemcpy(h_coal_d_aibar, coal_d_aibar, coal_d_aibarSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < XX; ++i) {
        int sI = i * bitSize * nOutputs;
        for (int j = 0; j < bitSize * nOutputs; ++j) {
            cout << h_coal_d_aibar[sI + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    //testing aibar end*/
    //calculate aij
    int coal_d_aijSize = n * t * totalBits;
    uint32_t *coal_d_aij;
    cudaMalloc(&coal_d_aij, coal_d_aijSize * sizeof(uint32_t));
    gridSize = (int) ceil((float) (coal_d_aijSize) / BLOCKSIZE);

    calculateAijFromAibarCoalesce<<<gridSize, BLOCKSIZE>>>(coal_d_aij, coal_d_aibar, totalBits, t, basebit, mask, coal_d_aijSize);
    /*//testing aij start
    uint32_t *h_coal_d_aij = new uint32_t[coal_d_aijSize];
    cudaMemcpy(h_coal_d_aij, coal_d_aij, coal_d_aijSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cout << endl;
    for (int i = 0; i < XX; ++i) {
        int sI = i * totalBits * t;
        for (int j = 0; j < t; ++j) {
            int sI2 = sI + j * totalBits;
            cout << "i:" << i << ": ";
            for (int k = 0; k < totalBits; ++k) {
                cout << h_coal_d_aij[sI2 + k] << " ";
            }
//            cout << h_coal_d_aij[sI + j] << " ";
            cout << endl;
        }
//        cout << endl;
    }
    cout << endl;
    //testing aij end*/

    //calculate result
    int length = params->n * totalBits;
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    int ks_n = n, ks_t = t, ks_base = 4;
    lweKeySwitchVectorSubstraction_gpu_testing_coalesce_2<<<gridSize, BLOCKSIZE>>>(result->a, ks_a_gpu_extendedPtr, coal_d_aij,
                                                          result_b_gpu, ks_b_gpu_extendedPtr,
                                                          result_cv_gpu, ks_cv_gpu_extendedPtr,
                                                          ks_n, ks_t, ks_base, nOutputs, bitSize, n,
                                                          params->n, length);

    cudaMemcpy(result->b, result_b_gpu, totalBits * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->current_variance, result_cv_gpu, totalBits * sizeof(double), cudaMemcpyDeviceToHost);

/*
    //test result start
//    h_ResCV = new double[totalBits];
//    cudaMemcpy(h_ResCV, result_cv_gpu, totalBits * sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < totalBits; ++i) {
////        int sI = i * params->n;
////        for (int j = 0; j < 10; ++j) {
////            cout << h_ResA[sI + j] << " ";
////        }
////        cout << endl;
//        cout << h_ResCV[i] << " ";
//    }
//    cout << endl;
    //test result->a end
*/
    cudaFree(result_b_gpu);
    cudaFree(result_cv_gpu);
    cudaFree(coal_d_aibar);
    cudaFree(coal_d_aij);

    int error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Cuda error code testing here: %d\n", error);
        exit(1);
    }
}
void lweKeySwitchTranslate_fromArray_16_2_vector(LweSample_16 *result,
                                                 const LweSample ***ks,
                                                 const LweParams *params,
                                                 const Torus32 *ai,
                                                 const int n, const int t,
                                                 const int basebit,
                                                 int vLength,
                                                 int nOutputs,
                                                 int bitSize, Torus32 ****ks_a_gpu,
                                                 Torus32 ****ks_a_gpu_extended,
                                                 int ***ks_b_gpu, double ***ks_cv_gpu,
                                                 Torus32 *ks_a_gpu_extendedPtr,
                                                 Torus32 *ks_b_gpu_extendedPtr,
                                                 double *ks_cv_gpu_extendedPtr) {
    const int base = 1 << basebit;       // base=2 in [CGGI16]
    const int32_t prec_offset = 1 << (32 - (1 + basebit * t)); //precision
    const int mask = base - 1;
    int totalBits = vLength * nOutputs * bitSize;
    /*send b of key switch to cuda and compute over there*/
    //create b for result in cuda
    int *result_b_gpu;
    double *result_cv_gpu;
    cudaMalloc(&result_b_gpu, totalBits * sizeof(int));
    cudaMalloc(&result_cv_gpu, totalBits * sizeof(double));
    //copy result->b to result_b_cuda
    cudaMemcpy(result_b_gpu, result->b, totalBits * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(result_cv_gpu, result->current_variance, totalBits * sizeof(double), cudaMemcpyHostToDevice);

/*    uint32_t *d_aibar, *d_aij;
    cudaMalloc(&d_aibar, totalBits * sizeof(uint32_t));
    cudaMalloc(&d_aij, totalBits * sizeof(uint32_t));

    BLOCKSIZE = params->n;
    int length = totalBits * params->n;
    int gridSize1 = (int) ceil((float) (length) / BLOCKSIZE);
    int gridSize = (int) ceil((float) (totalBits) / BLOCKSIZE);


    for (int i = 0; i < n; ++i) {//n
        getAibar<<<gridSize, BLOCKSIZE>>>(d_aibar, ai, prec_offset, i, n, totalBits);

        for (int j = 0; j < t; ++j) {
            calculateAijFromAibar<<<gridSize, BLOCKSIZE>>>(d_aij, d_aibar, j, basebit, mask, totalBits);

//            lweKeySwitchVectorSubstraction_gpu_testing_2_vector<<<gridSize1, BLOCKSIZE>>>
//                                                                      (result->a, ks_a_gpu_extended[i][j],
//                                                                              d_aij, n, params->n, vLength, nOutputs, bitSize, length);
            lweKeySwitchVectorSubstraction_B_gpu_testing_2_vector<<<gridSize, totalBits>>>
                                                                       (result_b_gpu, ks_b_gpu[i][j],
                                                                               d_aij, vLength, bitSize, totalBits);
            lweKeySwitchVectorAddition_cv_gpu_testing_2_vector<<<gridSize, totalBits>>>
                                                                    (result_cv_gpu, ks_cv_gpu[i][j],
                                                                            d_aij, vLength, bitSize, totalBits);
        }
    }


    cudaMemcpy(result->b, result_b_gpu, totalBits * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->current_variance, result_cv_gpu, totalBits * sizeof(double), cudaMemcpyDeviceToHost);


*/
    //outer loop parallelization
    //calculate aibar
    int BLOCKSIZE = 1024;
    int coal_d_aibarSize = totalBits * n;//n
    uint32_t *coal_d_aibar;
    cudaMalloc(&coal_d_aibar, coal_d_aibarSize * sizeof(uint32_t));
    int gridSize = (int) ceil((float) (coal_d_aibarSize) / BLOCKSIZE);
    getAibarCoalesce<<<gridSize, BLOCKSIZE>>>(coal_d_aibar, ai, prec_offset, totalBits, n, coal_d_aibarSize);

    //calculate d_aij
    int coal_d_aijSize = n * t * totalBits;
    uint32_t *coal_d_aij;
    cudaMalloc(&coal_d_aij, coal_d_aijSize * sizeof(uint32_t));
    gridSize = (int) ceil((float) (coal_d_aijSize) / BLOCKSIZE);
    calculateAijFromAibarCoalesce<<<gridSize, BLOCKSIZE>>>(coal_d_aij, coal_d_aibar, totalBits, t, basebit, mask, coal_d_aijSize);

    int length = params->n * totalBits;
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    int ks_n = n, ks_t = t, ks_base = 4;
    lweKeySwitchVectorSubstraction_gpu_testing_coalesce_2_vector<<<gridSize, BLOCKSIZE>>>
                                                                             (result->a, ks_a_gpu_extendedPtr, coal_d_aij,
                                                                             result_b_gpu, ks_b_gpu_extendedPtr,
                                                                             result_cv_gpu, ks_cv_gpu_extendedPtr,
                                                                             ks_n, ks_t, ks_base, nOutputs,
                                                                             vLength, bitSize, n,
                                                                             params->n, length);

    cudaMemcpy(result->b, result_b_gpu, totalBits * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result->current_variance, result_cv_gpu, totalBits * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(coal_d_aibar);
    cudaFree(coal_d_aij);
    cudaFree(result_b_gpu);
    cudaFree(result_cv_gpu);

    int error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Cuda error code testing here in key switching: %d\n", error);
        exit(1);
    }
}


EXPORT void lweCreateKeySwitchKey_old(LweKeySwitchKey *result, const LweKey *in_key, const LweKey *out_key) {
    const int n = result->n;
    const int basebit = result->basebit;
    const int t = result->t;

    //TODO check the parameters


    lweCreateKeySwitchKey_fromArray(result->ks,
                                    out_key, out_key->params->alpha_min,
                                    in_key->key, n, t, basebit);

    // renormalize
    renormalizeKSkey(result, out_key, in_key->key); // ILA: reverifier 
}









/*
Create the key switching key: normalize the error in the beginning
 * chose a random vector of gaussian noises (same size as ks) 
 * recenter the noises 
 * generate the ks by creating noiseless encryprions and then add the noise
*/
EXPORT void lweCreateKeySwitchKey(LweKeySwitchKey *result, const LweKey *in_key, const LweKey *out_key) {
    const int n = result->n;
    const int t = result->t;
    const int basebit = result->basebit;
    const int base = 1 << basebit;
    const double alpha = out_key->params->alpha_min;
    const int sizeks = n * t * (base - 1);
    //const int n_out = out_key->params->n;

    double err = 0;

    // chose a random vector of gaussian noises
    double *noise = new double[sizeks];
    for (int i = 0; i < sizeks; ++i) {
        normal_distribution<double> distribution(0., alpha);
        noise[i] = distribution(generator);
        err += noise[i];
    }
    // recenter the noises
    err = err / sizeks;
    for (int i = 0; i < sizeks; ++i) noise[i] -= err;


    // generate the ks
    int index = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < t; ++j) {

            // term h=0 as trivial encryption of 0 (it will not be used in the KeySwitching)
            lweNoiselessTrivial(&result->ks[i][j][0], 0, out_key->params);
            //lweSymEncrypt(&result->ks[i][j][0],0,alpha,out_key);

            for (int h = 1; h < base; ++h) { // pas le terme en 0
                /*
                // noiseless encryption
                result->ks[i][j][h].b = (in_key->key[i]*h)*(1<<(32-(j+1)*basebit));
                for (int p = 0; p < n_out; ++p) {
                    result->ks[i][j][h].a[p] = uniformTorus32_distrib(generator);
                    result->ks[i][j][h].b += result->ks[i][j][h].a[p] * out_key->key[p];
                }
                // add the noise 
                result->ks[i][j][h].b += dtot32(noise[index]);
                */
                Torus32 mess = (in_key->key[i] * h) * (1 << (32 - (j + 1) * basebit));
                lweSymEncryptWithExternalNoise(&result->ks[i][j][h], mess, noise[index], alpha, out_key);
                index += 1;
            }
        }
    }


    delete[] noise;
}











//sample=(a',b')
EXPORT void lweKeySwitch(LweSample *result, const LweKeySwitchKey *ks, const LweSample *sample) {
    const LweParams *params = ks->out_params;
    const int n = ks->n;
    const int basebit = ks->basebit;
    const int t = ks->t;
    //test morshed start
//    cout << "old ks->n: " << ks->n << endl;//1024
//    cout << "old ks->basebit: " << ks->basebit << endl;//2
//    cout << "old ks->t: " << ks->t << endl;//8
    //test morshed end

//    for (int i = 0; i < 10; ++i) {
//        cout << sample->a[i] << " ";
//    }
//    cout << endl;
    lweNoiselessTrivial(result, sample->b, params);
//    for (int i = 0; i < 10; ++i) {
//        cout << result->a[i] << " ";
//    }
//    cout << endl;
//    cout << sample->b << endl;

    lweKeySwitchTranslate_fromArray(result,
                                    (const LweSample ***) ks->ks, params,
                                    sample->a, n, t, basebit);

//    for (int i = 0; i < 10; ++i) {
//        cout << result->a[i] << " ";
//    }
//    cout << endl;
//    cout << "old b: " << result->b << endl;
//    cout << "old cv: " << result->current_variance << endl;
}

//new
__global__ void setVectorToConstVal(int *destination, int val, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = val;
    }
}

//sample=(a',b')
EXPORT void lweKeySwitch_16(LweSample_16 *result, const LweKeySwitchKey *ks, const LweSample_16 *sample,
                            int bitSize, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                            int ***ks_b_gpu, double ***ks_cv_gpu, Torus32* ks_a_gpu_extendedPtr,
                            Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {
    const LweParams *params = ks->out_params;
    const int n = ks->n;
    const int basebit = ks->basebit;
    const int t = ks->t;
//    cout << "new ks->n: " << ks->n << endl;//1024
//    cout << "new ks->basebit: " << ks->basebit << endl;//2
//    cout << "new ks->t: " << ks->t << endl;//8
//        cout << "ks->out_params->n: " << ks->out_params->n << endl;
    //assuming ks->out_params->n  is 500
    int BLOCKSIZE = 1024;//params->n;
//    cout << "BLOCKSIZE: " << BLOCKSIZE << endl;
    int gridSize = (int) ceil((float) (ks->out_params->n * bitSize) / BLOCKSIZE);
    setVectorToConstVal<<<gridSize, BLOCKSIZE>>>(result->a, 0, ks->out_params->n * bitSize);

    for (int i = 0; i < bitSize; ++i) {
        result->b[i] = sample->b[i];
        result->current_variance[i] = 0.;
    }

//    int number = 500;//1024;//
//    int * temp_a = new int[bitSize * number];
//    cudaMemcpy(temp_a, result->a, bitSize * number * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * number;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_a[sI + j] << " ";
//        }
//        cout << endl;
////        cout << "new b: " << result->b[i] << endl;
////        cout << "new cv: " << result->current_variance[i] << endl;
//    }
//    cout << endl;
    lweKeySwitchTranslate_fromArray_16(result,
                                       (const LweSample ***) ks->ks, params,
                                       sample->a, n, t, basebit, bitSize, ks_a_gpu, ks_a_gpu_extended,
                                       ks_b_gpu, ks_cv_gpu,
                                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    //test start
//    int number = params->n;
//    int * temp_a = new int[bitSize * number];
//    cudaMemcpy(temp_a, result->a, bitSize * number * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * number;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_a[sI + j] << " ";
//        }
//        cout << endl;
//        cout << "new b: " << result->b[i] << endl;
//        cout << "new cv: " << result->current_variance[i] << endl;
//    }
//    cout << endl;
    //test end
}

EXPORT void lweKeySwitch_16_2(LweSample_16 *result, const LweKeySwitchKey *ks, const LweSample_16 *sample,
                              int nOutputs, int bitSize, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                              int ***ks_b_gpu, double ***ks_cv_gpu,
                              Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    const LweParams *params = ks->out_params;
    const int n = ks->n;
    const int basebit = ks->basebit;
    const int t = ks->t;
//    cout << "new ks->n: " << ks->n << endl;//1024
//    cout << "new ks->basebit: " << ks->basebit << endl;//2
//    cout << "new ks->t: " << ks->t << endl;//8
//    cout << "ks->out_params->n: " << ks->out_params->n << endl;//500

    int BLOCKSIZE = params->n;
    int length = nOutputs * bitSize * params->n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);//32
    setVectorToConstVal<<<gridSize, BLOCKSIZE>>>(result->a, 0, length);

    for (int i = 0; i < nOutputs * bitSize; ++i) {
        result->b[i] = sample->b[i];
        result->current_variance[i] = 0.;
    }

    lweKeySwitchTranslate_fromArray_16_2(result,
                                         (const LweSample ***) ks->ks, params,
                                         sample->a, n, t, basebit, nOutputs, bitSize, ks_a_gpu, ks_a_gpu_extended,
                                         ks_b_gpu, ks_cv_gpu,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
//    //test start
////    int number = params->n;
////    int * temp_a = new int[bitSize * number];
////    cudaMemcpy(temp_a, result->a, bitSize * number * sizeof(int), cudaMemcpyDeviceToHost);
////    for (int i = 0; i < bitSize; ++i) {
////        int sI = i * number;
////        for (int j = 0; j < 10; ++j) {
////            cout << temp_a[sI + j] << " ";
////        }
////        cout << endl;
////        cout << "new b: " << result->b[i] << endl;
////        cout << "new cv: " << result->current_variance[i] << endl;
////    }
////    cout << endl;
//    //test end
}

EXPORT void lweKeySwitch_16_2_vector(LweSample_16 *result, const LweKeySwitchKey *ks, const LweSample_16 *sample,
                                     int vLength, int nOutputs, int bitSize, Torus32 ****ks_a_gpu,
                                     Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                                     Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                     double *ks_cv_gpu_extendedPtr) {

    const LweParams *params = ks->out_params;
    const int n = ks->n;
    const int basebit = ks->basebit;
    const int t = ks->t;
    int BLOCKSIZE = params->n;
    int length = vLength * nOutputs * bitSize * params->n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    setVectorToConstVal<<<gridSize, BLOCKSIZE>>>(result->a, 0, length);

    for (int i = 0; i < vLength * nOutputs * bitSize; ++i) {
        result->b[i] = sample->b[i];
        result->current_variance[i] = 0.;
    }
    lweKeySwitchTranslate_fromArray_16_2_vector(result,
                                         (const LweSample ***) ks->ks, params,
                                         sample->a, n, t, basebit, vLength, nOutputs, bitSize, ks_a_gpu, ks_a_gpu_extended,
                                         ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
}




/**
 * LweKeySwitchKey constructor function
 */
EXPORT void init_LweKeySwitchKey(LweKeySwitchKey *obj, int n, int t, int basebit, const LweParams *out_params) {
    const int base = 1 << basebit;
    LweSample *ks0_raw = new_LweSample_array(n * t * base, out_params);

    new(obj) LweKeySwitchKey(n, t, basebit, out_params, ks0_raw);
}

/**
 * LweKeySwitchKey destructor
 */
EXPORT void destroy_LweKeySwitchKey(LweKeySwitchKey *obj) {
    const int n = obj->n;
    const int t = obj->t;
    const int base = obj->base;
    delete_LweSample_array(n * t * base, obj->ks0_raw);

    obj->~LweKeySwitchKey();
}

//------------------------------------------------
//  autogenerated constructor/destructors/allocators
//------------------------------------------------

EXPORT LweKeySwitchKey *alloc_LweKeySwitchKey() {
    return (LweKeySwitchKey *) malloc(sizeof(LweKeySwitchKey));
}

EXPORT LweKeySwitchKey *alloc_LweKeySwitchKey_array(int nbelts) {
    return (LweKeySwitchKey *) malloc(nbelts * sizeof(LweKeySwitchKey));
}

//free memory space for a LweKey
EXPORT void free_LweKeySwitchKey(LweKeySwitchKey *ptr) {
    free(ptr);
}

EXPORT void free_LweKeySwitchKey_array(int nbelts, LweKeySwitchKey *ptr) {
    free(ptr);
}

//initialize the key structure
//(equivalent of the C++ constructor)
EXPORT void
init_LweKeySwitchKey_array(int nbelts, LweKeySwitchKey *obj, int n, int t, int basebit, const LweParams *out_params) {
    for (int i = 0; i < nbelts; i++) {
        init_LweKeySwitchKey(obj + i, n, t, basebit, out_params);
    }
}

//destroys the LweKeySwitchKey structure
//(equivalent of the C++ destructor)
EXPORT void destroy_LweKeySwitchKey_array(int nbelts, LweKeySwitchKey *obj) {
    for (int i = 0; i < nbelts; i++) {
        destroy_LweKeySwitchKey(obj + i);
    }
}

//allocates and initialize the LweKeySwitchKey structure
//(equivalent of the C++ new)
EXPORT LweKeySwitchKey *new_LweKeySwitchKey(int n, int t, int basebit, const LweParams *out_params) {
    LweKeySwitchKey *obj = alloc_LweKeySwitchKey();
    init_LweKeySwitchKey(obj, n, t, basebit, out_params);
    return obj;
}

EXPORT LweKeySwitchKey *new_LweKeySwitchKey_array(int nbelts, int n, int t, int basebit, const LweParams *out_params) {
    LweKeySwitchKey *obj = alloc_LweKeySwitchKey_array(nbelts);
    init_LweKeySwitchKey_array(nbelts, obj, n, t, basebit, out_params);
    return obj;
}

//destroys and frees the LweKeySwitchKey structure
//(equivalent of the C++ delete)
EXPORT void delete_LweKeySwitchKey(LweKeySwitchKey *obj) {
    destroy_LweKeySwitchKey(obj);
    free_LweKeySwitchKey(obj);
}

EXPORT void delete_LweKeySwitchKey_array(int nbelts, LweKeySwitchKey *obj) {
    destroy_LweKeySwitchKey_array(nbelts, obj);
    free_LweKeySwitchKey_array(nbelts, obj);
}

//__global__ void lweKeySwitchTranslate_fromArray_gpu_Helper(int *result, int *sample_a, int prec_offset,
//                                                           int N, int n, int bitSize, int bigLength, int smallLength) {
//    int id = blockIdx.x * blockDim.x + threadIdx.x;
//    if(id < bigLength) {
//        int aibar = sample_a[id] + prec_offset;
//    }
//
//}


