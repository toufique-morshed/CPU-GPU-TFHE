#include <stdio.h>
#include <iostream>
//#include "include/tfhe/tfhe.h"
//#include "include/tfhe/tfhe_io.h"
#include <stdlib.h>
#include<stdint.h>
#include<inttypes.h>

#include "Cipher.h"
#include <omp.h>
#include <assert.h>
#include "lagrangehalfc_impl.h"
#include <thread>
#include "cudaFFTTest.h"
#include "matrixUtility.h"
#include<cuda_profiler_api.h>
#include <string>
#include <cmath>

#define D2D cudaMemcpyDeviceToDevice
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice

#define nExp 5


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

void freeLweSample_16_gpu(LweSample_16 *toFree) {
    cudaFree(toFree->a);
    toFree->a = NULL;
    free(toFree->b);
    free(toFree->current_variance);
    delete toFree;
}


cufftDoubleComplex ****sendBootstrappingKeyToGPU(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    const TGswParams *bk_params = bk->bkFFT->bk_params;//bk->bk_params;
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const TLweParams *accum_params = bk->bkFFT->accum_params;
    const LweParams *in_params = bk->bkFFT->in_out_params;
    const int N = accum_params->N;
    const int Nx2 = 2 * N;
    const int n = in_params->n;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;
    /*//test variables
    cout << "n: " << n << endl;
    cout << "N: " << N << endl;
    cout << "Ns2: " << Ns2 << endl;
    cout << "bitSize: " << bitSize << endl;
    cout << "tlwe_params->k: " << tlwe_params->k << endl;//1
    cout << "bk_params->l: " << bk_params->l << endl;//2
    cout << "bk_params->kpl: " << bk_params->kpl << endl;//4*/
    cufftDoubleComplex ****cudaBkFFT = new cufftDoubleComplex ***[n];//n = 500
    for (int i = 0; i < n; ++i) {//500
        cudaBkFFT[i] = new cufftDoubleComplex **[kpl];
        for (int j = 0; j < kpl; ++j) {//4
            cudaBkFFT[i][j] = new cufftDoubleComplex *[k + 1];
//            cudaMallocManaged(&(cudaBkFFT[i][j]), sizeof(cufftDoubleComplex *) * (k + 1));
            for (int m = 0; m <= k; ++m) {//1
                cudaMalloc(&(cudaBkFFT[i][j][m]), bitSize * Ns2 * sizeof(cufftDoubleComplex));
                cufftDoubleComplex *temp_host_copy = new cufftDoubleComplex[Ns2 * bitSize];
                for (int n = 0; n < Ns2; ++n) {
                    temp_host_copy[n].x = ((LagrangeHalfCPolynomial_IMPL *) (
                            ((bk->bkFFT->bkFFT + i)->all_samples + j)->a + m))->coefsC[n].real();
                    temp_host_copy[n].y = ((LagrangeHalfCPolynomial_IMPL *) (
                            ((bk->bkFFT->bkFFT + i)->all_samples + j)->a + m))->coefsC[n].imag();
//                    if(i < 2 && n < 9){
//                        cout << "(" << temp_host_copy[n].x << "," << temp_host_copy[n].y << ") ";
//                    }
                }
//                if(i < 2){
//                    cout << endl;
//                }
                for (int x = 0; x < bitSize; ++x) {
                    int sI = x * Ns2;
                    memcpy(temp_host_copy + sI, temp_host_copy, Ns2 * sizeof(cufftDoubleComplex));
                }
                cudaMemcpy(cudaBkFFT[i][j][m], temp_host_copy, bitSize * Ns2 * sizeof(cufftDoubleComplex),
                           cudaMemcpyHostToDevice);
                delete[] temp_host_copy;
            }
        }
    }
    /****conversion ends****/
    return cudaBkFFT;
}

cufftDoubleComplex ***sendBootstrappingKeyToGPUCoalesce(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    const TGswParams *bk_params = bk->bkFFT->bk_params;//bk->bk_params;
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const TLweParams *accum_params = bk->bkFFT->accum_params;
    const LweParams *in_params = bk->bkFFT->in_out_params;
    const int N = accum_params->N;
    const int Nx2 = 2 * N;
    const int n = in_params->n;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;
    //test variables
//    cout << "n: " << n << endl;
//    cout << "N: " << N << endl;
//    cout << "Ns2: " << Ns2 << endl;
//    cout << "bitSize: " << bitSize << endl;
//    cout << "tlwe_params->k: " << tlwe_params->k << endl;//1
//    cout << "bk_params->l: " << bk_params->l << endl;//2
//    cout << "bk_params->kpl: " << bk_params->kpl << endl;//4
//    cout << "________________________________" << endl;
    cufftDoubleComplex ***cudaBkFFT = new cufftDoubleComplex **[n];//n = 500
    for (int i = 0; i < n; ++i) {//500
        cudaBkFFT[i] = new cufftDoubleComplex *[k + 1];
        for (int j = 0; j <= k; ++j) {
            cufftDoubleComplex *temp = new cufftDoubleComplex[kpl * Ns2 * bitSize];
            for (int m = 0; m < kpl; ++m) {
                int startIndex = m * (Ns2 * bitSize);
                for (int n = 0; n < Ns2; ++n) {
                    temp[startIndex + n].x = ((LagrangeHalfCPolynomial_IMPL *) (
                            ((bk->bkFFT->bkFFT + i)->all_samples + m)->a + j))->coefsC[n].real();
                    temp[startIndex + n].y = ((LagrangeHalfCPolynomial_IMPL *) (
                            ((bk->bkFFT->bkFFT + i)->all_samples + m)->a + j))->coefsC[n].imag();
//                    if (i < 100 && n < 9) {
//                        cout << "(" << temp[startIndex + n].x << "," << temp[startIndex + n].y << ") ";
//                    }
                }
//                if(i < 100) {
//                    cout << endl;
//                }
                for (int bIndex = 1; bIndex < bitSize; ++bIndex) {
                    memcpy(temp + startIndex + (bIndex * Ns2), temp + startIndex, Ns2 * sizeof(cufftDoubleComplex));
                    for (int n = 0; n < 9; ++n) {
//                        if (i < 2 && bIndex < 3) {
//                            cout << "(" << temp[startIndex + (bIndex * Ns2) + n].x << "," << temp[startIndex + (bIndex * Ns2) + n].y << ") ";
//                        }
                    }
//                    if(i < 2 && bIndex < 3) cout << endl;
                }
            }
            cudaMalloc(&(cudaBkFFT[i][j]), kpl * Ns2 * bitSize * sizeof(cufftDoubleComplex));
            cudaMemcpy(cudaBkFFT[i][j], temp, kpl * Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
            free(temp);
        }
    }
    /****conversion ends****/
    return cudaBkFFT;
}


cufftDoubleComplex *sendBootstrappingKeyToGPUCoalesceExt(const TFheGateBootstrappingCloudKeySet *bk) {
    const TGswParams *bk_params = bk->bkFFT->bk_params;//bk->bk_params;
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const TLweParams *accum_params = bk->bkFFT->accum_params;
    const LweParams *in_params = bk->bkFFT->in_out_params;
    const int N = accum_params->N;
    const int Nx2 = 2 * N;
    const int n = in_params->n;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;

    //test variables
//    cout << "________________________________" << endl;
//    cout << "n: " << n << endl;//500
//    cout << "N: " << N << endl;//1024
//    cout << "Ns2: " << Ns2 << endl;//512
//    cout << "tlwe_params->k: " << tlwe_params->k << endl;//1
//    cout << "bk_params->l: " << bk_params->l << endl;//2
//    cout << "bk_params->kpl: " << bk_params->kpl << endl;//4
//    cout << "________________________________" << endl;

    int totalElem = n * (k + 1) * kpl * Ns2;
    cufftDoubleComplex *cudaBkFFTCoalExt;
    cufftDoubleComplex *temp = new cufftDoubleComplex [totalElem];
    cudaMalloc(&cudaBkFFTCoalExt, totalElem * sizeof(cufftDoubleComplex));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < (k + 1); ++j) {
            for (int m = 0; m < kpl; ++m) {
                for (int p = 0; p < Ns2; ++p) {
                    int sI = i * (k + 1) * kpl * Ns2 + j * kpl * Ns2 + m * Ns2 + p;
                    temp[sI].x = ((LagrangeHalfCPolynomial_IMPL *) (
                            ((bk->bkFFT->bkFFT + i)->all_samples + m)->a + j))->coefsC[p].real();
                    temp[sI].y = ((LagrangeHalfCPolynomial_IMPL *) (
                            ((bk->bkFFT->bkFFT + i)->all_samples + m)->a + j))->coefsC[p].imag();
//                    if (i < 100 && p < 9) {
//                        cout << "(" << temp[sI].x << "," << temp[sI].y << ") ";
////                        cout << sI << endl;
//                    }
                }
//                if(i < 100) cout << endl;
            }

        }
    }
    cudaMemcpy(cudaBkFFTCoalExt, temp, totalElem * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    return cudaBkFFTCoalExt;
}

int ***sendKeySwitchBtoGPU(const TFheGateBootstrappingCloudKeySet *bk) {
    //send ks_b to gpu
    int ks_n = bk->bkFFT->ks->n;
    int ks_t = bk->bkFFT->ks->t;
    int ks_base = bk->bkFFT->ks->base;//base;
    int ***ks_b_gpu = new int **[ks_n];
    for (int i = 0; i < ks_n; ++i) {
        ks_b_gpu[i] = new int *[ks_t];
        for (int j = 0; j < ks_t; ++j) {
            cudaMalloc(&ks_b_gpu[i][j], sizeof(int) * ks_base);
            int *tempb = new int[ks_base];
            for (int k = 0; k < ks_base; ++k) {
                tempb[k] = bk->bkFFT->ks->ks[i][j][k].b;
            }
            cudaMemcpy(ks_b_gpu[i][j], tempb, sizeof(int) * ks_base, cudaMemcpyHostToDevice);
            free(tempb);
        }
    }
    return ks_b_gpu;
}

int *sendKeySwitchBtoGPUOnePtr(const TFheGateBootstrappingCloudKeySet *bk) {
    //send ks_b to gpu
    int ks_n = bk->bkFFT->ks->n;
    int ks_t = bk->bkFFT->ks->t;
    int ks_base = bk->bkFFT->ks->base;//base;
    int *ks_b_gpuOnePtr;
    cudaMalloc(&ks_b_gpuOnePtr, ks_n * ks_t * ks_base * sizeof(int));
    for (int i = 0; i < ks_n; ++i) {
        for (int j = 0; j < ks_t; ++j) {
            int *tempb = new int[ks_base];
            for (int k = 0; k < ks_base; ++k) {
                tempb[k] = bk->bkFFT->ks->ks[i][j][k].b;
            }
            cudaMemcpy(ks_b_gpuOnePtr + i * ks_t * ks_base + j * ks_base, tempb, sizeof(int) * ks_base, cudaMemcpyHostToDevice);
            free(tempb);
        }
    }
    return ks_b_gpuOnePtr;
}

double ***sendKeySwitchCVtoGPU(const TFheGateBootstrappingCloudKeySet *bk) {
    //send ks_cv to gpu
    int ks_n = bk->bkFFT->ks->n;
    int ks_t = bk->bkFFT->ks->t;
    int ks_base = bk->bkFFT->ks->base;//base;
    double ***ks_cv_gpu = new double **[ks_n];
    for (int i = 0; i < ks_n; ++i) {
        ks_cv_gpu[i] = new double *[ks_t];
        for (int j = 0; j < ks_t; ++j) {
            cudaMalloc(&ks_cv_gpu[i][j], sizeof(double) * ks_base);
            double *tempcv = new double[ks_base];
            for (int k = 0; k < ks_base; ++k) {
                tempcv[k] = bk->bkFFT->ks->ks[i][j][k].current_variance;
            }
            cudaMemcpy(ks_cv_gpu[i][j], tempcv, sizeof(double) * ks_base, cudaMemcpyHostToDevice);
            free(tempcv);
        }
    }
    return ks_cv_gpu;
}

double *sendKeySwitchCVtoGPUOnePtr(const TFheGateBootstrappingCloudKeySet *bk) {
    //send ks_cv to gpu
    int ks_n = bk->bkFFT->ks->n;
    int ks_t = bk->bkFFT->ks->t;
    int ks_base = bk->bkFFT->ks->base;//base;
    double *ks_cv_gpuOnePtr;
    cudaMalloc(&ks_cv_gpuOnePtr, ks_n * ks_t * ks_base * sizeof(double));
    for (int i = 0; i < ks_n; ++i) {
        for (int j = 0; j < ks_t; ++j) {
            double *tempcv = new double[ks_base];
            for (int k = 0; k < ks_base; ++k) {
                tempcv[k] = bk->bkFFT->ks->ks[i][j][k].current_variance;
            }
            cudaMemcpy(ks_cv_gpuOnePtr + i * ks_t * ks_base + j * ks_base, tempcv, sizeof(double) * ks_base, cudaMemcpyHostToDevice);
            free(tempcv);
        }
    }
    return ks_cv_gpuOnePtr;
}

Torus32 ****sendKeySwitchKeyToGPU(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    /****store ks->a in gpu starts****/
    //variables
    const int ks_n = bk->bkFFT->ks->n;//bk->ks->n;
    const int ks_t = bk->bkFFT->ks->t;
    const int ks_base = bk->bkFFT->ks->base;
    const LweParams *ks_params = bk->bkFFT->ks->out_params;
    LweSample ***ks = bk->bkFFT->ks->ks;
    //test
//    cout << "ks_n: " << ks_n << endl;
//    cout << "ks_t: " << ks_t << endl;
//    cout << "ks_base: " << ks_base << endl;
//    cout << "ks_params->n: " << ks_params->n << endl;
    Torus32 ****ks_a_gpu = new Torus32 ***[ks_n];
    for (int i = 0; i < ks_n; ++i) {
        ks_a_gpu[i] = new Torus32 **[ks_t];
        for (int j = 0; j < ks_t; ++j) {
            ks_a_gpu[i][j] = new Torus32 *[ks_base];
            for (int k = 0; k < ks_base; ++k) {
                cudaMalloc(&(ks_a_gpu[i][j][k]), ks_params->n * sizeof(Torus32));
                cudaMemcpy(ks_a_gpu[i][j][k], (&(ks[i][j][k]))->a, ks_params->n * sizeof(Torus32),
                           cudaMemcpyHostToDevice);
            }
        }
    }
    /****store ks->a in gpu ends****/
    return ks_a_gpu;
}

Torus32 ****sendKeySwitchKeyToGPU_extended(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    /***Key Transfer v2 starts***/
    //variables
    const int ks_n = bk->bkFFT->ks->n;
    const int ks_t = bk->bkFFT->ks->t;
    const int ks_base = bk->bkFFT->ks->base;
    const LweParams *ks_params = bk->bkFFT->ks->out_params;
    LweSample ***ks = bk->bkFFT->ks->ks;
    //test
//    cout << "ks_n: " << ks_n << endl;//1024
//    cout << "ks_t: " << ks_t << endl;//8
//    cout << "ks_base: " << ks_base << endl;//4
//    cout << "ks_params->n: " << ks_params->n << endl;//500
    //extend ks in GPU
    Torus32 ****ks_a_gpu_extended = new Torus32 ***[ks_n];
    for (int i = 0; i < ks_n; ++i) {
        ks_a_gpu_extended[i] = new Torus32 **[ks_t];
        for (int j = 0; j < ks_t; ++j) {
            cudaMallocManaged(&ks_a_gpu_extended[i][j], ks_base * sizeof(Torus32 *));
            for (int k = 0; k < ks_base; ++k) {
                cudaMalloc(&ks_a_gpu_extended[i][j][k], bitSize * ks_params->n * sizeof(Torus32));
                Torus32 *temp = new Torus32[bitSize * ks_params->n];
                memcpy(temp, (&(ks[i][j][k]))->a, sizeof(Torus32) * ks_params->n);
                //replicate the key upto bit size
                for (int l = 1; l < bitSize; ++l) {
                    int sI = l * ks_params->n;
                    memcpy(temp + sI, temp, ks_params->n * sizeof(Torus32));
                }
                //send to GPU
                cudaMemcpy(ks_a_gpu_extended[i][j][k], temp, ks_params->n * sizeof(Torus32) * bitSize,
                           cudaMemcpyHostToDevice);
                delete[] temp;
            }
        }
    }
    return ks_a_gpu_extended;
}

Torus32 *sendKeySwitchKeyToGPU_extendedOnePointer(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    /***Key Transfer v2 starts***/
    //variables
    const int ks_n = bk->bkFFT->ks->n;
    const int ks_t = bk->bkFFT->ks->t;
    const int ks_base = bk->bkFFT->ks->base;
    const LweParams *ks_params = bk->bkFFT->ks->out_params;
    LweSample ***ks = bk->bkFFT->ks->ks;
    //test
//    cout << "ks_n: " << ks_n << endl;//1024
//    cout << "ks_t: " << ks_t << endl;//8
//    cout << "ks_base: " << ks_base << endl;//4
//    cout << "ks_params->n: " << ks_params->n << endl;//500

    //version 2 for one array
    Torus32 *ks_a_gpu_extendedPtr;
    cudaMalloc(&ks_a_gpu_extendedPtr, ks_n * ks_t * ks_base * bitSize * ks_params->n * sizeof(Torus32));

    for (int i = 0; i < ks_n; ++i) {
        for (int j = 0; j < ks_t; ++j) {
            for (int k = 0; k < ks_base; ++k) {
                Torus32 *temp = new Torus32[bitSize * ks_params->n];
                memcpy(temp, (&(ks[i][j][k]))->a, sizeof(Torus32) * ks_params->n);
                //replicate the key upto bit size
                for (int l = 1; l < bitSize; ++l) {
                    int sI = l * ks_params->n;
                    memcpy(temp + sI, temp, ks_params->n * sizeof(Torus32));
                }
                int A = ks_n, B = ks_t, C = ks_base, D = bitSize * ks_params->n;
                cudaMemcpy(ks_a_gpu_extendedPtr + i * B * C * D + j * C * D + k * D,
                           temp,
                           bitSize * ks_params->n * sizeof(Torus32),
                           cudaMemcpyHostToDevice);
            }

        }
    }
//    cout << endl;
//    cout << "ks_n * ks_t * ks_base * bitSize * ks_params->n: " << ks_n * ks_t * ks_base * bitSize * ks_params->n
//         << endl;


    return ks_a_gpu_extendedPtr;
}



Torus32 ****sendKeySwitchKeyToGPU_extended_2(int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    /***Key Transfer v2 starts***/
    //variables
    const int ks_n = bk->bkFFT->ks->n;
    const int ks_t = bk->bkFFT->ks->t;
    const int ks_base = bk->bkFFT->ks->base;
    const LweParams *ks_params = bk->bkFFT->ks->out_params;
    LweSample ***ks = bk->bkFFT->ks->ks;
    int length = nOutputs * bitSize * ks_params->n;
    //test
//    cout << "ks_n: " << ks_n << endl;//1024
//    cout << "ks_t: " << ks_t << endl;//8
//    cout << "ks_base: " << ks_base << endl;//4
//    cout << "ks_params->n: " << ks_params->n << endl;//500
    //extend ks in GPU
    Torus32 ****ks_a_gpu_extended = new Torus32 ***[ks_n];
    for (int i = 0; i < ks_n; ++i) {
        ks_a_gpu_extended[i] = new Torus32 **[ks_t];
        for (int j = 0; j < ks_t; ++j) {
            cudaMallocManaged(&ks_a_gpu_extended[i][j], ks_base * sizeof(Torus32 *));
            for (int k = 0; k < ks_base; ++k) {
                cudaMalloc(&ks_a_gpu_extended[i][j][k], length * sizeof(Torus32));
                Torus32 *temp = new Torus32[length];
                memcpy(temp, (&(ks[i][j][k]))->a, sizeof(Torus32) * ks_params->n);
                //replicate the key upto bit size
                for (int l = 1; l < nOutputs * bitSize; ++l) {
                    int sI = l * ks_params->n;
                    memcpy(temp + sI, temp, ks_params->n * sizeof(Torus32));
                }
                //send to GPU
                cudaMemcpy(ks_a_gpu_extended[i][j][k], temp, length * sizeof(Torus32), cudaMemcpyHostToDevice);
                delete[] temp;
            }
        }
    }
    ////version 2
//    Torus32 ****ks_a_gpu_extended_p2 = new Torus32***[ks_n];
//    for (int i = 0; i < ks_n; ++i) {
//        ks_a_gpu_extended_p2[i] = new Torus32**[ks_t];
//        for (int j = 0; j < ks_t; ++j) {
//            cudaMalloc(&ks_a_gpu_extended[i][j], ks_base * sizeof(Torus32*));
//            for (int k = 0; k < ks_base; ++k) {
//                Torus32 *temp = new Torus32[bitSize * ks_params->n];
//                memcpy(temp, (&(ks[i][j][k]))->a, sizeof(Torus32) * ks_params->n);
//                //replicate the key upto bit size
//                for (int l = 1; l < bitSize; ++l) {
//                    int sI = l * ks_params->n;
//                    memcpy(temp + sI, temp, ks_params->n * sizeof(Torus32));
//                }
//                //send to GPU
//                int *d_temp;
//                cudaMalloc(&d_temp, ks_params->n * sizeof(Torus32) * bitSize);
//                cudaMemcpy(d_temp, temp, ks_params->n * sizeof(Torus32) * bitSize, cudaMemcpyHostToDevice);
////                cout << " I am here" << endl;
//                int BLOCKSIZE = ks_params->n;
//                int gridSize = (int) ceil((float) (ks_params->n * bitSize) / BLOCKSIZE);
//                cudaAllocationHelper<<<1, 1>>>(ks_a_gpu_extended_p2[i][j], d_temp, k, 1);
//                delete[] temp;
//            }
//        }
//    }
    /***Key Transfer v2 ends***/
//    cout << "________original__________" << endl;
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 2; ++j) {
//            for (int k = 0; k < 2; ++k) {
//                for (int l = 0; l < 15; ++l) {
//                    cout << (&(ks[i][j][k]))->a[l] << " ";
//                }
//                cout << endl;
//            }
//        }
//    }
//    cout << "_________________________" << endl;
//

    return ks_a_gpu_extended;
}


void testCipher(char *name, LweSample_16 *to_test, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                TFheGateBootstrappingSecretKeySet *key) {
    const LweParams *in_out_params = bk->params->in_out_params;
    //test start
    int *temp = new int[bitSize * in_out_params->n];
    cudaMemcpy(temp, to_test->a, sizeof(int) * bitSize * in_out_params->n, cudaMemcpyDeviceToHost);
    cudaFree(to_test->a);
    to_test->a = temp;
    LweSample *to_testOutput1 = convertNumberToBits(to_test, bitSize, bk);
    Cipher leftShiftCipher1(bitSize);
    leftShiftCipher1.data = to_testOutput1;
    cout << name << ": " << decryptCheck(leftShiftCipher1, key) << endl;

    cudaMalloc(&(to_test->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(to_test->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    //test end
}


__global__ void vecOneBitLeftShift(int *destnation, int *source, int n, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / n;
        if (bitIndex != 0) {
            destnation[id] = source[id - n];
        }
    }
}

__global__ void vecSetAPortionToZero(int *destination, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = 0;
    }
}

void leftShiftCuda_16(LweSample_16 *cudaResult, LweSample_16 *cudaA, int bitSize,
                      const TFheGateBootstrappingCloudKeySet *bk) {

    const LweParams *in_out_params = bk->params->in_out_params;

    int BLOCKSIZE = in_out_params->n;
    int gridSize = (int) ceil((float) (bitSize * in_out_params->n) / BLOCKSIZE);

    vecOneBitLeftShift << < gridSize, BLOCKSIZE >> >
                                      (cudaResult->a, cudaA->a, in_out_params->n, bitSize, bitSize * in_out_params->n);
    vecSetAPortionToZero << < 1, BLOCKSIZE >> > (cudaResult->a, 1 * in_out_params->n);
    for (int i = 1; i < bitSize; ++i) {
        cudaResult->b[i] = cudaA->b[i - 1];
        cudaResult->current_variance[i] = cudaA->current_variance[i - 1];
    }

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    int message = 0;
    cudaResult->b[0] = message ? MU : -MU;
    cudaResult->current_variance[0] = 0.;
}

__global__ void cudaLeftShift(int *destination, int *source, int bitSize, int n, int nBitShift, int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < len) {
        int bIndex = id / n;
        int newVal = bIndex < nBitShift ? 0 : source[id - n * nBitShift];
        destination[id] = newVal;
    }
}

void leftShiftCuda_16(LweSample_16 *cudaResult, LweSample_16 *cudaA, int bitSize, int nBitShift,
                      const TFheGateBootstrappingCloudKeySet *bk) {

//    cout << nInputBits << "\t" << nOutputBits << endl;
    const LweParams *in_out_params = bk->params->in_out_params;
    int BLOCKSIZE = in_out_params->n;
    int inputLength = bitSize * in_out_params->n;
    int gridSize = (int) ceil((float) (inputLength) / BLOCKSIZE);
    //shift result->a
    cudaLeftShift<<<gridSize, BLOCKSIZE>>>(cudaResult->a, cudaA->a, bitSize, in_out_params->n, nBitShift, inputLength);
    //shift result->b and result->cv
    for (int i = nBitShift; i < bitSize; ++i) {
        cudaResult->b[i] = cudaA->b[i - nBitShift];
        cudaResult->current_variance[i] = cudaA->current_variance[i - nBitShift];
    }
    //set shifted bits
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    int message = 0;
    for (int i = 0; i < nBitShift; ++i) {
        cudaResult->b[i] = message ? MU : -MU;
        cudaResult->current_variance[i] = 0.;
    }
}



__global__ void oneBitLeftShiftInTwoBitOutput(int *destination, int *source, int bitSize, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = 0;
        int index = id / n;
        if (index > 0) {
            destination[id] = source[id - n];
        }
    }
}

void oneBitLeftShiftFromTwoOutputs(LweSample_16 *output, LweSample_16 *input, int bitSize,
                                   const TFheGateBootstrappingCloudKeySet *bk) {
    //input has to be two bit output like andxor
    //position means whether the input is in the left part or the right part
    //for now the position is fixed it is int he first number
    const int n = bk->params->in_out_params->n;
    //change a
    int length = bitSize * n;
    int BLOCKSIZE = 1024;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    oneBitLeftShiftInTwoBitOutput<<<gridSize, BLOCKSIZE>>>(output->a, input->a, bitSize, n, bitSize * n);

    //change b and current variance
    for (int i = 1; i < bitSize; ++i) {
        output->b[i] = input->b[i - 1];
        output->current_variance[i] = input->current_variance[i - 1];
    }

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    output->b[0] = -MU;
    output->current_variance[0] = 0.;
}


void taskLevelParallelAdd(LweSample_16 *result, LweSample_16 *a, LweSample_16 *b, int nBits,
                          const TFheGateBootstrappingCloudKeySet *bk,
                          cufftDoubleComplex *cudaBkFFTCoalesceExt,
                          Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                          TFheGateBootstrappingSecretKeySet *key) {

    const int nOut = 2;
    static const int n = 500;
    LweSample_16 *taskResult = convertBitToNumberZero_GPU(nBits * nOut, bk);  //bitSize * nOut -> to accomodate intermediate compound gate
    LweSample_16 *tempB = convertBitToNumberZero_GPU(nBits, bk);

    //tempB = b
    cudaMemcpy(tempB->a, b->a, nBits * n * sizeof(int), D2D);
    memcpy(tempB->b, b->b, nBits * sizeof(int));
//    testCipher("tempB", tempB, nBits, bk, key);

    //result = a
    cudaMemcpy(result->a, a->a, nBits * n * sizeof(int), D2D);
    memcpy(result->b, a->b, nBits * sizeof(int));
//    testCipher("result", result, nBits, bk, key);

    for (int i = 0; i < nBits; ++i) {
        bootsANDXOR_fullGPU_n_Bit_vector(taskResult, result, tempB, 1, //1 -> vLength
                                         nBits, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
        oneBitLeftShiftFromTwoOutputs(tempB, taskResult, nBits, bk);

        cudaMemcpy(result->a, taskResult->a + (nBits * n), sizeof(int) * nBits * n, D2D);
        memcpy(result->b, taskResult->b + nBits, nBits * sizeof(int));
    }
//    testCipher("Addition Result", result, nBits, bk, key);
        freeLweSample_16_gpu(taskResult);
        freeLweSample_16_gpu(tempB);
}

__global__ void oneBitLeftShiftInTwoBitOutput_vector(int *destination, int *source, int vLength, int bitSize, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = 0;
        int index = (id / n) % bitSize;
        if (index > 0) {
            destination[id] = source[id - n];
        }
    }
}


void oneBitLeftShiftFromTwoOutputs_vector(LweSample_16 *output, LweSample_16 *input, int vLength, int bitSize,
                                          const TFheGateBootstrappingCloudKeySet *bk) {
    const int n = bk->params->in_out_params->n;
    int totalBitSize = vLength * bitSize;
    int length = totalBitSize * n;
    int BLOCKSIZE = n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    oneBitLeftShiftInTwoBitOutput_vector<<<gridSize, BLOCKSIZE>>>(output->a, input->a, vLength, bitSize, n, length);
    memcpy(output->b, input->b - 1, totalBitSize * sizeof(int));
    memcpy(output->current_variance, input->current_variance - 1, totalBitSize * sizeof(double));
//    for (int i = 1; i < bitSize; ++i) {
//        output->b[i] = input->b[i - 1];
//        output->current_variance[i] = input->current_variance[i - 1];
//    }
//    for (int i = bitSize; i < 2 * bitSize; ++i) {
//        output->b[i] = input->b[i - 1];
//        output->current_variance[i] = input->current_variance[i - 1];
//    }
//
//    for (int i = 2 * bitSize; i < 3 * bitSize; ++i) {
//        output->b[i] = input->b[i - 1];
//        output->current_variance[i] = input->current_variance[i - 1];
//    }


    static const Torus32 MU = modSwitchToTorus32(1, 8);
    for (int i = 0; i < vLength; ++i) {
        output->b[i * bitSize] = -MU;
        output->current_variance[i * bitSize] = 0.;
    }



}


void taskLevelParallelAdd_Vector(LweSample_16 *mainResult, LweSample_16 *a, LweSample_16 *b, int vLength, int bitSize,
                                 const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                                 cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu,
                                 Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                                 Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                 double *ks_cv_gpu_extendedPtr,
                                 TFheGateBootstrappingSecretKeySet *key) {
//    cout << "I am inside Large Addition Circuit" << endl;

    int nOutputs = 2;
    const LweParams *in_out_params = bk->params->in_out_params;
    const int n = in_out_params->n;
    LweSample_16 *taskResult = convertBitToNumberZero_GPU(vLength * bitSize * nOutputs, bk);
    LweSample_16 *tempB = convertBitToNumberZero_GPU(bitSize * vLength, bk);
//    LweSample_16 *tempxxx = convertBitToNumberZero_GPU(bitSize, bk);

    //tempB = b
    cudaMemcpy(tempB->a, b->a, vLength * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(tempB->b, b->b, vLength * bitSize * sizeof(int));
    memcpy(tempB->current_variance, b->current_variance, vLength * bitSize * sizeof(double));

    //result = a
    cudaMemcpy(mainResult->a, a->a, vLength * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(mainResult->b, a->b, vLength * bitSize * sizeof(int));
    memcpy(mainResult->current_variance, a->current_variance, vLength * bitSize * sizeof(double));

    for (int i = 0; i < bitSize; ++i) {
//        cout << "Iteration: ********" << i << endl;
        bootsANDXOR_16_vector(taskResult, mainResult, tempB, nOutputs, vLength, bitSize, bk, cudaBkFFT,
                              cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                              ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
        //mainResult = the xor part
        cudaMemcpy(mainResult->a, taskResult->a + vLength * bitSize * n, vLength * bitSize * n * sizeof(int),
                   cudaMemcpyDeviceToDevice);
        memcpy(mainResult->b, taskResult->b + vLength * bitSize, vLength * bitSize * sizeof(int));
        memcpy(mainResult->current_variance, taskResult->current_variance + vLength * bitSize,
               vLength * bitSize * sizeof(double));
        //tempB = the AND part
        oneBitLeftShiftFromTwoOutputs_vector(tempB, taskResult, vLength, bitSize, bk);
        /*
    cout << endl;
    cout << "----------" << endl;
    for (int i = 0; i < 4; ++i) {
        cudaMemcpy(tempxxx->a, a->a + i * bitSize * n, bitSize * n * sizeof(int), cudaMemcpyDeviceToHost);
        memcpy(tempxxx->b, a->b + i * bitSize, bitSize * sizeof(int));
        memcpy(tempxxx->current_variance, a->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("vD_1", tempxxx, bitSize, bk, key);

        cudaMemcpy(tempxxx->a, b->a + i * bitSize * n, bitSize * n * sizeof(int), cudaMemcpyDeviceToHost);
        memcpy(tempxxx->b, b->b + i * bitSize, bitSize * sizeof(int));
        memcpy(tempxxx->current_variance, b->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("vD_2", tempxxx, bitSize, bk, key);

        cudaMemcpy(tempxxx->a, tempB->a + i * bitSize * n, bitSize * n * sizeof(int), cudaMemcpyDeviceToHost);
        memcpy(tempxxx->b, tempB->b + i * bitSize, bitSize * sizeof(int));
        memcpy(tempxxx->current_variance, tempB->current_variance + i * bitSize, bitSize * sizeof(double));
        testCipher("vANDshifted", tempxxx, bitSize, bk, key);

        cudaMemcpy(tempxxx->a, mainResult->a + i * bitSize * n, bitSize * n * sizeof(int), cudaMemcpyDeviceToHost);
        memcpy(tempxxx->b, mainResult->b + i * bitSize, bitSize * sizeof(int));
        memcpy(tempxxx->current_variance, mainResult->current_variance + i * bitSize, bitSize * sizeof(double));
        testCipher("vXOR", tempxxx, bitSize, bk, key);
        cout << endl;
    }*/


    }
//    LweSample_16 *temp = convertBitToNumberZero_GPU(bitSize, bk);
//    for (int i = 0; i < vLength; ++i) {
//        cudaMemcpy(temp->a, a->a + (i * bitSize * n), bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, a->b + i * bitSize, bitSize * sizeof(int));
//        memcpy(temp->current_variance, a->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("a", temp, bitSize, bk, key);
//
//        cudaMemcpy(temp->a, b->a + (i * bitSize * n), bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, b->b + i * bitSize, bitSize * sizeof(int));
//        memcpy(temp->current_variance, b->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("b", temp, bitSize, bk, key);
//
//        cudaMemcpy(temp->a, mainResult->a + (i * bitSize * n), bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, mainResult->b + i * bitSize, bitSize * sizeof(int));
//        memcpy(temp->current_variance, mainResult->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("sum", temp, bitSize, bk, key);
//        cout << endl;
////        if(i < vLength) {
////            testCipher("AND", carry, bitSize, bk, key);
////        } else {
////            testCipher("XOR", carry, bitSize, bk, key);
////        }
//    }
//    //mainResult = the xor part
//    cudaMemcpy(mainResult->a, taskResult->a + vLength * bitSize * n, vLength * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(mainResult->b, taskResult->b + vLength * bitSize, vLength * bitSize * sizeof(int));
//    memcpy(mainResult->current_variance, taskResult->current_variance + vLength * bitSize, vLength * bitSize * sizeof(double));
//    cout << "After split" << endl;
//    for (int i = 0; i < vLength; ++i) {
//        cudaMemcpy(carry->a, mainResult->a + (i * bitSize * n), bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(carry->b, mainResult->b + i * bitSize, bitSize * sizeof(int));
//        memcpy(carry->current_variance, mainResult->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("XOR", carry, bitSize, bk, key);
//    }
    //tempB = the AND part
//    oneBitLeftShiftFromTwoOutputs_vector(tempB, taskResult, vLength, bitSize, bk);
//    cout << "After split and shift" << endl;
//    for (int i = 0; i < vLength; ++i) {
//        cudaMemcpy(carry->a, tempB->a + (i * bitSize * n), bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(carry->b, tempB->b + i * bitSize, bitSize * sizeof(int));
//        memcpy(carry->current_variance, tempB->current_variance + i * bitSize, bitSize * sizeof(double));
//        testCipher("AND * 2", carry, bitSize, bk, key);
//    }
//    cudaMemcpy(mainResult->a, taskResult->a + vLength * bitSize * n, vLength * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(mainResult->b, taskResult->b + vLength * bitSize, vLength * bitSize * sizeof(int));
//    memcpy(mainResult->current_variance, taskResult->current_variance + vLength * bitSize, vLength * bitSize * sizeof(double));


}


void taskLevelParallelAdd_bitwise(LweSample_16 *result, LweSample_16 *a, LweSample_16 *b,
                                  const int nBits, const TFheGateBootstrappingCloudKeySet *bk,
                                  cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                  Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                  TFheGateBootstrappingSecretKeySet *key) {
    const int nOut = 2, n = 500, nAdditionBits = 1;

    LweSample_16 *r = convertBitToNumberZero_GPU(2, bk);
    LweSample_16 *t = convertBitToNumberZero_GPU(2, bk);

    //pointers to positions
    LweSample_16 *r1 = new LweSample_16;
    LweSample_16 *t1 = new LweSample_16;
    LweSample_16 *ai = new LweSample_16;
    LweSample_16 *bi = new LweSample_16;

    t1->a = t->a + n;
    t1->b = t->b + nAdditionBits;
    t1->current_variance = t->current_variance + nAdditionBits;

    r1->a = r->a + n;
    r1->b = r->b + nAdditionBits;
    r1->current_variance = r->current_variance + nAdditionBits;

    bootsANDXOR_fullGPU_n_Bit_vector(r, a, b, 1,
                                     1, cudaBkFFTCoalesceExt,
                                     ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

    cudaMemcpy(result->a, r->a + n, n * sizeof(int), D2D);
    memcpy(result->b, r->b + nAdditionBits, sizeof(int));


    cudaMemcpy(r->a + nAdditionBits * n, r->a, n * sizeof(int), D2D);
    memcpy(r->b + nAdditionBits, r->b, sizeof(int));

    for (int bI = 1; bI < nBits; ++bI) {
        ai->a = a->a + n * bI;
        ai->b = a->b + bI;

        bi->a = b->a + n * bI;
        bi->b = b->b + bI;

        bootsXORXOR_fullGPU_n_Bit_vector(t,
                                         ai, r1,
                                         bi, r1,
                                         1, nAdditionBits, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

        bootsAND_fullGPU_n_Bit(t,
                               t, t1,
                               nAdditionBits, cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

        bootsXORXOR_fullGPU_n_Bit_vector(r,
                                         ai, t1,
                                         t, r1,
                                         1, nAdditionBits, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

        cudaMemcpy(result->a + bI * n, r->a, n * sizeof(int), D2D);
        memcpy(result->b + bI, r->b, sizeof(int));
    }

    freeLweSample_16_gpu(r);
    freeLweSample_16_gpu(t);
    delete r1;
    delete t1;
    delete ai;
    delete bi;
}


void test_AND_XOR_CompoundGate_Addition(LweSample *ca, LweSample *cb, int bitSize,
                                        const TFheGateBootstrappingCloudKeySet *bk,
                                        cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                        Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                        TFheGateBootstrappingSecretKeySet *key) {
    //get keys
    const int n = 500;
    const int nOut = 2;
    double sTime;
    //convert to number
    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);
    LweSample_16 *additionResult = convertBitToNumberZero_GPU(bitSize, bk);
    LweSample_16 *andResult = convertBitToNumberZero_GPU(bitSize, bk);
    LweSample_16 *xorResult = convertBitToNumberZero_GPU(bitSize, bk);
    LweSample_16 *andxorResult = convertBitToNumberZero_GPU(bitSize * nOut, bk);

    //Prepare data
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    cout << "...Starting Experiments for A and B of " << bitSize << " bits... " << endl;
    testCipher("A", a, bitSize, bk, key);
    testCipher("B", b, bitSize, bk, key);

    cout << "-----Starting AND operation-----" << endl;
    for (int i = 0; i < nExp; ++i) {
        cudaCheckErrors("starting AND operation");
        sTime = omp_get_wtime();
        bootsAND_fullGPU_n_Bit(andResult, a, b, bitSize, cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
//        bootsAND_fullGPU_1_Bit_Stream(andResult, a, b, bitSize, cudaBkFFTCoalesceExt, ks_a_gpu_extendedPtr,
//                                ks_b_gpu_extendedPtr);
        cout << "AND time taken:\t" << omp_get_wtime() - sTime << endl;
    }
    testCipher("A . B", andResult, bitSize, bk, key);

    cout << "-----Starting XOR operation-----" << endl;
    for (int i = 0; i < nExp; ++i) {
        cudaCheckErrors("starting XOR operation");
        sTime = omp_get_wtime();
        bootsXOR_fullGPU_n_Bit(xorResult, a, b, bitSize, cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
//        bootsAND_fullGPU_1_Bit_Stream(andResult, a, b, bitSize, cudaBkFFTCoalesceExt, ks_a_gpu_extendedPtr,
//                                ks_b_gpu_extendedPtr);
        cout << "XOR time taken:\t" << omp_get_wtime() - sTime << endl;
    }
    testCipher("A + B", xorResult, bitSize, bk, key);


    //ANDXOR Together
    cout << "-----Starting AND XOR operation-----" << endl;
    for (int i = 0; i < nExp; ++i) {
        int vLength = 1;
        cudaCheckErrors("starting AND XOR operation");
        sTime = omp_get_wtime();
        bootsANDXOR_fullGPU_n_Bit_vector(andxorResult, a, b, vLength, bitSize, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
        cout << "AND XOR time taken:\t" << omp_get_wtime() - sTime << endl;
    }
    //split the result
    cudaMemcpy(andResult->a, andxorResult->a, bitSize * n * sizeof(int), D2D);
    memcpy(andResult->b, andxorResult->b, bitSize * sizeof(int));
    testCipher("A (.) B v2", andResult, bitSize, bk, key);

    cudaMemcpy(xorResult->a, andxorResult->a + (bitSize * n), bitSize * n * sizeof(int), D2D);
    memcpy(xorResult->b, andxorResult->b + bitSize, bitSize * sizeof(int));
    testCipher("A (+) B v2", xorResult, bitSize, bk, key);

    //XORXOR Together
    cout << "-----Starting XOR XOR operation-----" << endl;
    for (int i = 0; i < nExp; ++i) {
        int vLength = 1;
        cudaCheckErrors("starting AND XOR operation");
        sTime = omp_get_wtime();
        bootsXORXOR_fullGPU_n_Bit_vector(andxorResult, a, b, a, b, vLength, bitSize, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
        cout << "XOR XOR time taken:\t" << omp_get_wtime() - sTime << endl;
    }

    //split the result
    cudaMemcpy(andResult->a, andxorResult->a, bitSize * n * sizeof(int), D2D);
    memcpy(andResult->b, andxorResult->b, bitSize * sizeof(int));
    testCipher("A (+) B v2", andResult, bitSize, bk, key);

    cudaMemcpy(xorResult->a, andxorResult->a + (bitSize * n), bitSize * n * sizeof(int), D2D);
    memcpy(xorResult->b, andxorResult->b + bitSize, bitSize * sizeof(int));
    testCipher("A (+) B v2", xorResult, bitSize, bk, key);

    //addition starting
    cout << "-----Starting Addition (1) operation-----" << endl;
    for (int i = 0; i < nExp; ++i) {
        sTime = omp_get_wtime();
        taskLevelParallelAdd_bitwise(additionResult, a, b, bitSize, bk, cudaBkFFTCoalesceExt,
                                     ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
        cout << "ADDITION (1) time taken:\t" << omp_get_wtime() - sTime << endl;
    }
    testCipher("A + B (1)", additionResult, bitSize, bk, key);

    cout << "-----Starting Addition (n) operation-----" << endl;
    for (int i = 0; i < nExp; ++i) {
        sTime = omp_get_wtime();
        taskLevelParallelAdd(additionResult, a, b, bitSize, bk, cudaBkFFTCoalesceExt,
                             ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
        cout << "ADDITION (n) time taken:\t" << omp_get_wtime() - sTime << endl;
    }
    testCipher("A + B (n)", additionResult, bitSize, bk, key);

    freeLweSample_16_gpu(a);
    freeLweSample_16_gpu(b);
    freeLweSample_16_gpu(additionResult);
    freeLweSample_16_gpu(andResult);
    freeLweSample_16_gpu(xorResult);
    freeLweSample_16_gpu(andxorResult);
}


/**
 *
 * @param output array of outputs
 * @param input array of inputs
 * @param vecSize n number of input and n/2 outputs
 * @param bitSize 16/32
 * @param bk bootstrapping key
 * @param cudaBkFFT bkFFT in cuda
 * @param cudaBkFFTCoalesce BkFFTCoalesced
 * @param ks_a_gpu key switching a
 * @param ks_a_gpu_extended key switching a bitsized
 * @param ks_b_gpu key switching b bitsized
 * @param ks_cv_gpu key switching cv bitsized
 * @param key combination of all keys
 */
void BOOTS_Add_vector(LweSample_16 *output, LweSample_16 **input, int vecSize, int bitSize,
                      const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                      cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                      int ***ks_b_gpu, double ***ks_cv_gpu, Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                      double *ks_cv_gpu_extendedPtr, TFheGateBootstrappingSecretKeySet *key) {

    int currentVecSize = vecSize / 2;
//    cout << "currentVecSize: " << currentVecSize << endl;
    int n = bk->params->in_out_params->n;
    //combine the numbers
    LweSample_16 *data1 = convertBitToNumberZero_GPU(bitSize * currentVecSize, bk);
    LweSample_16 *data2 = convertBitToNumberZero_GPU(bitSize * currentVecSize, bk);
    LweSample_16 *dataRes = convertBitToNumberZero_GPU(bitSize * currentVecSize, bk);

    //copy data
    for (int i = 0; i < currentVecSize; ++i) {
        cudaMemcpy(data1->a + (i * bitSize * n), input[i]->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data1->b + (i * bitSize), input[i]->b, bitSize * sizeof(int));
        memcpy(data1->current_variance + (i * bitSize), input[i]->current_variance, bitSize * sizeof(double));
//        testCipher("input1 ", input[i], bitSize, bk, key);

        int j = i + currentVecSize;
        cudaMemcpy(data2->a + (i * bitSize * n), input[j]->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data2->b + (i * bitSize), input[j]->b, bitSize * sizeof(int));
        memcpy(data2->current_variance + (i * bitSize), input[j]->current_variance, bitSize * sizeof(double));
//        testCipher("input2 ", input[j], bitSize, bk, key);
    }
/*
    for (int i = 0; i < 4; ++i) {
        testCipher("D_1: ", input[i], bitSize, bk, key);
        testCipher("D_2: ", input[i + currentVecSize], bitSize, bk, key);
        taskLevelParallelAdd(dataRes, input[i], input[i + currentVecSize], bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce,
                                ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                             ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        testCipher("output_seq: ", dataRes, bitSize, bk, key);
        int * temp = new int[n * bitSize];
        cudaMemcpy(temp, dataRes->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int j = 0; j < bitSize; ++j) {
            int sI = j * n;
            for (int k = 0; k < 10; ++k) {
                cout << temp[j * n + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
        cout << endl;
    }
*/


//    dataRes = convertBitToNumberZero_GPU(bitSize * currentVecSize, bk);
        double sTime = omp_get_wtime();
    taskLevelParallelAdd_Vector(dataRes, data1, data2, currentVecSize, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce,
                                ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);

    cout << "currentVecSize: " << currentVecSize << "time: " << omp_get_wtime() - sTime << endl;
//    LweSample_16 *testingData = convertBitToNumberZero_GPU(bitSize, bk);
//    for (int i = 0; i < 3; ++i) {
//        cudaMemcpy(testingData->a, dataRes->a + (i * bitSize * n), bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(testingData->b, dataRes->b + (i * bitSize), bitSize * sizeof(int));
//        memcpy(testingData->current_variance, dataRes->current_variance + (i * bitSize), bitSize * sizeof(double));
//        testCipher("out_vector", testingData, bitSize, bk, key);
//        int * temp = new int[n * bitSize];
//        cudaMemcpy(temp, testingData->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < bitSize; ++j) {
//            int sI = j * n;
//            for (int k = 0; k < 10; ++k) {
//                cout << temp[j * n + k] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//        cout << endl;
//    }
//         cout << "currentVecSize" << currentVecSize << endl;


    for (currentVecSize /= 2; currentVecSize >= 1 ; currentVecSize /= 2) {
        cudaMemcpy(data1->a, dataRes->a, currentVecSize * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data1->b, dataRes->b, currentVecSize * bitSize * sizeof(int));
        memcpy(data1->current_variance, dataRes->current_variance, currentVecSize * bitSize * sizeof(double));

        cudaMemcpy(data2->a, dataRes->a + currentVecSize * bitSize * n, currentVecSize * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data2->b, dataRes->b + currentVecSize * bitSize, currentVecSize * bitSize * sizeof(int));
        memcpy(data2->current_variance, dataRes->current_variance + currentVecSize * bitSize, currentVecSize * bitSize * sizeof(double));

        sTime = omp_get_wtime();
        taskLevelParallelAdd_Vector(dataRes, data1, data2, currentVecSize, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce,
                                    ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                                    ks_a_gpu_extendedPtr,
                                    ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        cout << "currentVecSize: " << currentVecSize << "time: " << omp_get_wtime() - sTime << endl;


    }
    cudaMemcpy(output->a, dataRes->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(output->b, dataRes->b, bitSize * sizeof(int));
    memcpy(output->current_variance, dataRes->current_variance, bitSize * sizeof(double));

    testCipher("Final res", output, bitSize, bk, key);


}

void taskLevelParallelAdd_bitwise_vector_coalInput(LweSample_16 *result, const LweSample_16 *a, const LweSample_16 *b,
                                                   const int vLength, const int nBits, const int nCoal,
                                                   const TFheGateBootstrappingCloudKeySet *bk,
                                                   cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                                   Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                                   TFheGateBootstrappingSecretKeySet *key) {
    const int nOut = 2, n = 500;

    LweSample_16 *r = convertBitToNumberZero_GPU(2 * nCoal * vLength, bk);
    LweSample_16 *t = convertBitToNumberZero_GPU(2 * nCoal * vLength, bk);
    LweSample_16 *ai = convertBitToNumberZero_GPU(1 * nCoal * vLength, bk);
    LweSample_16 *bi = convertBitToNumberZero_GPU(1 * nCoal * vLength, bk);
//    LweSample_16 *temp = convertBitToNumberZero_GPU(nBits, bk);
    //pointers
    LweSample_16 *r1 = new LweSample_16;
    LweSample_16 *t1 = new LweSample_16;

    t1->a = t->a + nCoal * vLength * n;
    t1->b = t->b + nCoal * vLength;

    r1->a = r->a + nCoal * vLength * n;
    r1->b = r->b + nCoal * vLength;

    int bI = 0;
    for (int i = 0; i < vLength * nCoal; ++i) {
        cudaMemcpy(ai->a + i * n, a->a + i * nBits * n, n * sizeof(Torus32), D2D);
        memcpy(ai->b + i, a->b + i * nBits, sizeof(int));
        /*cudaMemcpy(temp->a, ai->a + i * n, n * sizeof(int), D2D);
        memcpy(temp->b, ai->b + i, nBits * sizeof(int));
        testCipher("ai", temp, 1, bk, key);*/

        cudaMemcpy(bi->a + i * n, b->a + i * nBits * n, n * sizeof(Torus32), D2D);
        memcpy(bi->b + i, b->b + i * nBits, sizeof(int));
        /*cudaMemcpy(temp->a, bi->a + i * n, n * sizeof(int), D2D);
        memcpy(temp->b, bi->b + i, nBits * sizeof(int));
        testCipher("bi", temp, 1, bk, key);*/
    }
    bootsANDXOR_fullGPU_n_Bit_vector(r, ai, bi, nCoal * vLength, 1, cudaBkFFTCoalesceExt,
                                     ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

    //test copy r0 to main result
    for (int i = 0; i < nCoal * vLength; ++i) {
        cudaMemcpy(result->a + i * nBits * n, r1->a + i * n, n * sizeof(Torus32), D2D);
        memcpy(result->b + i * nBits, r1->b + i, sizeof(int));
        /*cudaMemcpy(temp->a, r->a + i * n, n * sizeof(int), D2D);
        memcpy(temp->b, r->b + i, sizeof(int));
        testCipher("r0", temp, 1, bk, key);

        cudaMemcpy(temp->a, result->a + i * bitSize * n, n * sizeof(int), D2D);
        memcpy(temp->b, result->b + i * bitSize, sizeof(int));
        testCipher("mainResult0", temp, 1, bk, key);*/
    }
    cudaMemcpy(r1->a, r->a, nCoal * vLength * n * sizeof(Torus32), D2D);
    memcpy(r1->b, r->b, nCoal * vLength * sizeof(int));

    /*for (int i = 0; i < vLength; ++i) {
        cudaMemcpy(temp->a, r1->a + i * n, n * sizeof(int), D2D);
        memcpy(temp->b, r1->b + i, sizeof(int));
        testCipher("r1", temp, 1, bk, key);
    }*/
    for (bI = 1; bI < nBits; ++bI) {
        //get ai and bi
        for (int i = 0; i < nCoal * vLength; ++i) {
            cudaMemcpy(ai->a + i * n, a->a + i * nBits * n + bI * n, n * sizeof(Torus32), D2D);
            memcpy(ai->b + i, a->b + i * nBits + bI, sizeof(int));

            cudaMemcpy(bi->a + i * n, b->a + i * nBits * n + bI * n, n * sizeof(Torus32), D2D);
            memcpy(bi->b + i, b->b + i * nBits + bI, sizeof(int));

            /*cout << "---1---" << endl;
            cudaMemcpy(temp->a, r1->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, r1->b + i, sizeof(int));
            testCipher("r1", temp, 1, bk, key);

            cudaMemcpy(temp->a, ai->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, ai->b + i, sizeof(int));
            testCipher("ai", temp, 1, bk, key);

            cudaMemcpy(temp->a, bi->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, bi->b + i, sizeof(int));
            testCipher("bi", temp, 1, bk, key);
            cout << endl;*/
        }

        bootsXORXOR_fullGPU_n_Bit_vector(t,
                                         ai, r1,
                                         bi, r1,
                                         nCoal * vLength, 1, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
        /*for (int i = 0; i < vLength; ++i) {
            cout << "ai xor r1" << endl;
            cudaMemcpy(temp->a, t->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, t->b + i, sizeof(int));
            testCipher("t0", temp, 1, bk, key);

            cout << "bi xor r1" << endl;
            cudaMemcpy(temp->a, t->a + i * n + vLength * n, n * sizeof(int), D2D);
            memcpy(temp->b, t->b + i + vLength, sizeof(int));
            testCipher("t1", temp, 1, bk, key);
            cout << endl;
        }
        cout << "---2---" << endl;*/

        bootsAND_fullGPU_n_Bit(t,
                               t, t1,
                               1 * nCoal * vLength,
                               cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);
        /*for (int i = 0; i < vLength; ++i) {
            cout << "t0 = t0 and t1" << endl;
            cudaMemcpy(temp->a, t->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, t->b + i, sizeof(int));
            testCipher("t0", temp, 1, bk, key);
            cout << endl;
        }*/
        bootsXORXOR_fullGPU_n_Bit_vector(r,
                                         ai, t1,
                                         t, r1,
                                         nCoal * vLength, 1, cudaBkFFTCoalesceExt,
                                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

        for (int i = 0; i < nCoal * vLength; ++i) {
            //copy to the result
            cudaMemcpy(result->a + i * nBits * n + bI * n, r->a + i * n, n * sizeof(int), D2D);
            memcpy(result->b + i * nBits + bI, r->b + i, sizeof(int));

            /*cout << "---3---" << endl;
            cout << "ai xor t1" << endl;
            cudaMemcpy(temp->a, a->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, a->b + i, sizeof(int));
            testCipher("ai", temp, 1, bk, key);

            cudaMemcpy(temp->a, t1->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, t1->b + i, sizeof(int));
            testCipher("t1", temp, 1, bk, key);

            cudaMemcpy(temp->a, r->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, r->b + i, sizeof(int));
            testCipher("r0", temp, 1, bk, key);

            cout << "t0 xor r1" << endl;
            cudaMemcpy(temp->a, t->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, t->b + i, sizeof(int));
            testCipher("t0", temp, 1, bk, key);
            cout << "prev r1 in first line" << endl;

            cudaMemcpy(temp->a, r1->a + i * n, n * sizeof(int), D2D);
            memcpy(temp->b, r1->b + i, sizeof(int));
            testCipher("r1", temp, 1, bk, key);
            cout << "action: copy r0 to result" << endl;
            cout << endl;*/
        }
    }


    //free memories
    freeLweSample_16_gpu(r);
    freeLweSample_16_gpu(t);
    freeLweSample_16_gpu(ai);
    freeLweSample_16_gpu(bi);

    //pointers
    delete r1;
    delete t1;
}

void BOOTS_vectorAddition(LweSample_16 **result,
                          LweSample_16 **inputA, LweSample_16 **inputB,
                          const int nCoal, const int vLength, const int nBits,
                          const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                          Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                          TFheGateBootstrappingSecretKeySet *key) {

    const int n = 500;
    //combine the numbers
    LweSample_16 *data1 = convertBitToNumberZero_GPU(nCoal * nBits * vLength, bk);
    LweSample_16 *data2 = convertBitToNumberZero_GPU(nCoal * nBits * vLength, bk);
    LweSample_16 *dataRes = convertBitToNumberZero_GPU(nCoal * nBits * vLength, bk);
    LweSample_16 *temp = convertBitToNumberZero_GPU(nBits, bk);

    for (int i = 0; i < vLength; ++i) {
        cudaMemcpy(data1->a + (i * nCoal * nBits * n), inputA[i]->a, nCoal * nBits * n * sizeof(int), D2D);
        memcpy(data1->b + (i * nCoal * nBits), inputA[i]->b, nCoal * nBits * sizeof(int));

        /*for (int j = 0; j < nCoal; ++j) {
            cudaMemcpy(testingData->a, data1->a + (i * nCoal * nBits * n) + j * nBits * n, nBits * n * sizeof(int),
                       D2D);
            memcpy(testingData->b, data1->b + (i * nBits * nCoal) + j * nBits, nBits * sizeof(int));
            testCipher("d1", testingData, nBits, bk, key);
        }*/

        cudaMemcpy(data2->a + (i * nCoal * nBits * n), inputB[i]->a, nCoal * nBits * n * sizeof(int), D2D);
        memcpy(data2->b + (i * nCoal * nBits), inputB[i]->b, nCoal * nBits * sizeof(int));

        /*for (int j = 0; j < nCoal; ++j) {
            cudaMemcpy(testingData->a, data2->a + (i * nCoal * nBits * n) + j * nBits * n, nBits * n * sizeof(int),
                       D2D);
            memcpy(testingData->b, data2->b + (i * nBits * nCoal) + j * nBits, nBits * sizeof(int));
            testCipher("d2", testingData, nBits, bk, key);
        }
        cout << endl;*/
    }

    taskLevelParallelAdd_bitwise_vector_coalInput(dataRes, data1, data2,
                                                  vLength, nBits, nCoal,
                                                  bk, cudaBkFFTCoalesceExt,
                                                  ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    for (int i = 0; i < vLength; ++i) {
        cudaMemcpy(result[i]->a, dataRes->a + (i * nCoal * nBits * n), nCoal * nBits * n * sizeof(int), D2D);
        memcpy(result[i]->b, dataRes->b + i * nCoal * nBits, nCoal * nBits * sizeof(int));
    }

    //free memory
    freeLweSample_16_gpu(data1);
    freeLweSample_16_gpu(data2);
    freeLweSample_16_gpu(dataRes);
    freeLweSample_16_gpu(temp);
}

__constant__ int n = 500;

__global__ void leftShiftExpandedFormatForMultiplicationKernel(Torus32 *destination, Torus32 *source, int inputBit, int outputBit) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int index = id / (n * inputBit);
    register int dstStart = index * outputBit * n;
    register int idMod = id % (n * inputBit);
    register int shiftPortion = index * n;
    destination[dstStart + idMod  + shiftPortion] = source[id];
}

__global__ void leftShiftExpandedFormatForMultiplicationKernel_single(Torus32 *destination, Torus32 *source, int nBits) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int nBitShift = id / (n * nBits);
    register int bIndex = (id / n) % nBits;
    register int newVal = bIndex < nBitShift ? 0 : source[id - n * nBitShift];
    destination[id] = newVal;
}

__global__ void leftShiftExpandedFormatForMultiplicationKernel_single_vector(Torus32 *destination, Torus32 *source, int vLength, int nBits) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int nBitShift = id / (n * nBits * vLength);
    register int bIndex = (id / n) % nBits;
    register int newVal = bIndex < nBitShift ? 0 : source[id - n * nBitShift];
    destination[id] = newVal;
}

void leftShiftExpandedFormatForMultiplication(LweSample_16 *andLShifExpanded, LweSample_16 *andExpanded,
                                              int inputBit, int outputBit) {
    const int n = 500, nThreads = 1024;
    int nBlocks = (int) ceil((float) (inputBit * inputBit * n) / nThreads);
    leftShiftExpandedFormatForMultiplicationKernel<<<nBlocks, nThreads>>>
                                                              (andLShifExpanded->a,
                                                                      andExpanded->a,
                                                                      inputBit, outputBit);
    for (int i = 0; i < inputBit; ++i) {
        memcpy(andLShifExpanded->b + i * outputBit + i, andExpanded->b + i * inputBit, inputBit * sizeof(int));
    }
}

void leftShiftExpandedFormatForMultiplication(LweSample_16 *andLShifExpanded, LweSample_16 *andExpanded,
                                              int inputBit, int outputBit, bool isDoublePrecision) {
    const int n = 500, nThreads = 1024;
    int nBlocks = (int) ceil((float) (inputBit * inputBit * n) / nThreads);
    if (isDoublePrecision) {
        leftShiftExpandedFormatForMultiplicationKernel<<<nBlocks, nThreads>>>
                                                                  (andLShifExpanded->a,
                                                                          andExpanded->a,
                                                                          inputBit, outputBit);
    } else {
        leftShiftExpandedFormatForMultiplicationKernel_single<<<nBlocks, nThreads>>>
                                                                  (andLShifExpanded->a,
                                                                          andExpanded->a,
                                                                          inputBit);
    }

    for (int i = 0; i < inputBit; ++i) {
        if (isDoublePrecision) {
            memcpy(andLShifExpanded->b + i * outputBit + i, andExpanded->b + i * inputBit, inputBit * sizeof(int));
        } else {
            memcpy(andLShifExpanded->b + i * inputBit + i, andExpanded->b + i * inputBit,
                   (inputBit - i) * sizeof(int));
        }
    }

}

__global__ void leftShiftExpandedFormatForMultiplicationKernel(Torus32 *destination, Torus32 *source, int vlength, int inputBit, int outputBit) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int index = id / (n * inputBit);
    register int dstStart = index * outputBit * n;
    register int idMod = id % (n * inputBit);
    register int shiftPortion = (index/vlength) * n;//(index % inputBit) * n;
    destination[dstStart + idMod  + shiftPortion] = source[id];
}

void leftShiftExpandedFormatForMultiplication(LweSample_16 *andLShifExpanded, LweSample_16 *andExpanded,
                                              int vLength, int inputBit, int outputBit) {
    const int n = 500, nThreads = 1024;
    int nBlocks = (int) ceil((float) (vLength * inputBit * inputBit * n) / nThreads);
    leftShiftExpandedFormatForMultiplicationKernel<<<nBlocks, nThreads>>>
                                                              (andLShifExpanded->a,
                                                                      andExpanded->a,
                                                                      vLength,
                                                                      inputBit, outputBit);
    for (int i = 0; i < inputBit; ++i) {
        for (int j = 0; j < vLength; ++j) {
            memcpy(andLShifExpanded->b + i * vLength * outputBit + j * outputBit + i,
                   andExpanded->b + i * vLength * inputBit + j * inputBit,
                   inputBit * sizeof(int));
        }
    }
}

void leftShiftExpandedFormatForMultiplication_precision(LweSample_16 *andLShifExpanded, LweSample_16 *andExpanded,
                                              int vLength, int inputBit, int outputBit, bool isDoublePrecision) {
    const int n = 500, nThreads = 1024;
    int nBlocks = (int) ceil((float) (vLength * inputBit * inputBit * n) / nThreads);
    if (isDoublePrecision) {
        leftShiftExpandedFormatForMultiplicationKernel<<<nBlocks, nThreads>>>
                                                                  (andLShifExpanded->a,
                                                                          andExpanded->a,
                                                                          vLength,
                                                                          inputBit, outputBit);
    } else {
        leftShiftExpandedFormatForMultiplicationKernel_single_vector<<<nBlocks, nThreads>>>
                                                                                (andLShifExpanded->a,
                                                                                        andExpanded->a,
                                                                                        vLength,
                                                                                        inputBit);
    }
    for (int i = 0; i < inputBit; ++i) {
        for (int j = 0; j < vLength; ++j) {
            if (isDoublePrecision) {
                memcpy(andLShifExpanded->b + i * vLength * outputBit + j * outputBit + i,
                       andExpanded->b + i * vLength * inputBit + j * inputBit,
                       inputBit * sizeof(int));
            } else {
                memcpy(andLShifExpanded->b + i * vLength * outputBit + j * outputBit + i,
                       andExpanded->b + i * vLength * inputBit + j * inputBit,
                       (inputBit - i) * sizeof(int));
            }
        }
    }
}

void multiplyLweSamples(LweSample_16 *result, LweSample_16 *a, LweSample_16 *b, int nBits, bool isDoublePrecision,
                        const TFheGateBootstrappingCloudKeySet *bk,
                        cufftDoubleComplex *cudaBkFFTCoalesceExt,
                        Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                        TFheGateBootstrappingSecretKeySet *key) {
    assert(nBits % 2 == 0);//nBits has to be even for multuplication
    //divide the numbers into bitsize numbers
    const int n = 500, iBits = nBits, oBits = isDoublePrecision ? nBits * 2 : nBits;
    //expand a and b
    LweSample_16 *aExpanded = convertBitToNumberZero_GPU(iBits * iBits, bk);
    LweSample_16 *bExpanded = convertBitToNumberZero_GPU(iBits * iBits, bk);
//    LweSample_16 *andExpanded = convertBitToNumberZero_GPU(iBits * iBits, bk);
    LweSample_16 *andLShiftExp = convertBitToNumberZero_GPU(iBits * oBits, bk);
    LweSample_16 *iResult = convertBitToNumberZero_GPU(iBits / 2 * oBits, bk);
    //andLShifExpanded = 0
    cudaMemset(andLShiftExp->a, 0, iBits * oBits * n * sizeof(Torus32));
    //temp
    LweSample_16 *temp = convertBitToNumberZero_GPU(iBits, bk);
    LweSample_16 *temp2 = convertBitToNumberZero_GPU(oBits, bk);

    for (int i = 0; i < iBits; ++i) {
        cudaMemcpy(aExpanded->a + i * iBits * n, a->a, iBits * n * sizeof(Torus32), D2D);
        memcpy(aExpanded->b + i * iBits, a->b, iBits * sizeof(int));

        for (int j = 0; j < iBits; ++j) {
            cudaMemcpy(bExpanded->a + i * iBits * n + j * n, b->a + i * n, n * sizeof(Torus32), D2D);
            memcpy(bExpanded->b + i * iBits + j, b->b + i, sizeof(int));
        }
    }

    /*for (int i = 0; i < iBits; ++i) {
        cudaMemcpy(temp->a, aExpanded->a + i * iBits * n, iBits * n * sizeof(Torus32), D2D);
        memcpy(temp->b, aExpanded->b + i * iBits, iBits * sizeof(int));
        testCipher("a", temp, iBits, bk, key);

        cudaMemcpy(temp->a, bExpanded->a + i * iBits * n, iBits * n * sizeof(Torus32), D2D);
        memcpy(temp->b, bExpanded->b + i * iBits, iBits * sizeof(int));
        testCipher("b", temp, iBits, bk, key);
        cout << endl;
    }*/

    bootsAND_fullGPU_n_Bit(aExpanded, aExpanded, bExpanded,
                           iBits * iBits, cudaBkFFTCoalesceExt,
                           ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);

    leftShiftExpandedFormatForMultiplication(andLShiftExp, aExpanded, iBits, oBits, isDoublePrecision);

    /*for (int i = 0; i < iBits; ++i) {
        cudaMemcpy(temp2->a, andLShiftExp->a + i * oBits * n, oBits * n * sizeof(Torus32), D2D);
        memcpy(temp2->b, andLShiftExp->b + i * oBits, oBits * sizeof(int));
        testCipher("&S", temp2, oBits, bk, key);
    }
    cout << endl;*/

    //free a and b
    freeLweSample_16_gpu(aExpanded);
    freeLweSample_16_gpu(bExpanded);

    int vLength = iBits / 2, nCoal = 1;
    LweSample_16 *initAndShiftExp = andLShiftExp;
    LweSample_16 *halfLeAndShiftExp = new LweSample_16;
    //set half len
    halfLeAndShiftExp->a = andLShiftExp->a + vLength * oBits * n;
    halfLeAndShiftExp->b = andLShiftExp->b + vLength * oBits;
    for (int i = 0; vLength > 0; i++) {
        //add
        taskLevelParallelAdd_bitwise_vector_coalInput(iResult, initAndShiftExp, halfLeAndShiftExp,
                                                      vLength, oBits, nCoal,
                                                      bk, cudaBkFFTCoalesceExt,
                                                      ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

        /*for (int i = 0; i < vLength; ++i) {
            cudaMemcpy(temp2->a, iResult->a + i * oBits * n, oBits * n * sizeof(Torus32), D2D);
            memcpy(temp2->b, iResult->b + i * oBits, oBits * sizeof(int));
            testCipher("output", temp2, oBits, bk, key);
        }
        cout << endl;*/

        vLength /= 2;
        initAndShiftExp = iResult;
        halfLeAndShiftExp->a = iResult->a + vLength * oBits * n;
        halfLeAndShiftExp->b = iResult->b + vLength * oBits;

        if (i == 0) {
            freeLweSample_16_gpu(andLShiftExp);
        }
    }

    cudaMemcpy(result->a, iResult->a, oBits * n * sizeof(Torus32), D2D);
    memcpy(result->b, iResult->b, oBits * sizeof(int));

//    freeLweSample_16_gpu(andExpanded);
    freeLweSample_16_gpu(iResult);
    freeLweSample_16_gpu(temp);
    freeLweSample_16_gpu(temp2);
    delete halfLeAndShiftExp;
}


void test_vectorAddition(LweSample *ca, LweSample *cb, 
						 int vLength, int nBits,
                         const TFheGateBootstrappingCloudKeySet *bk,
                         cufftDoubleComplex *cudaBkFFTCoalesceExt,
                         Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                         TFheGateBootstrappingSecretKeySet *key) {

    const int n = 500;
    LweSample_16 *a = convertBitToNumber(ca, nBits, bk);
    LweSample_16 *b = convertBitToNumber(cb, nBits, bk);

    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), nBits * n * sizeof(int));
    cudaMemcpy(a->a, temp, nBits * n * sizeof(int), H2D);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), nBits * n * sizeof(int));
    cudaMemcpy(b->a, temp, nBits * n * sizeof(int), H2D);
    free(temp);

    //vector addition
    LweSample_16 **inputA = new LweSample_16*[vLength];
    LweSample_16 **inputB = new LweSample_16*[vLength];
    LweSample_16 **result = new LweSample_16*[vLength];
    for (int i = 0; i < vLength; ++i) {
        inputA[i] = convertBitToNumberZero_GPU(nBits, bk);
        cudaMemcpy(inputA[i]->a, a->a, nBits * n * sizeof(int), D2D);
        memcpy(inputA[i]->b, a->b, nBits * sizeof(int));
        testCipher("intput A", inputA[i], nBits, bk, key);

        inputB[i] = convertBitToNumberZero_GPU(nBits, bk);
        leftShiftCuda_16(inputB[i], inputA[i], nBits, i % 10, bk);
//        cudaMemcpy(inputB[i]->a, b->a, nBits * n * sizeof(int), D2D);
//        memcpy(inputB[i]->b, b->b, nBits * sizeof(int));
        testCipher("intput B", inputB[i], nBits, bk, key);

        result[i] = convertBitToNumberZero_GPU(nBits, bk);
    }

    cout << "Vector Addition" << endl;
    double sT = omp_get_wtime();
    BOOTS_vectorAddition(result, inputA, inputB,
                                 1, vLength, nBits,
                                 bk, cudaBkFFTCoalesceExt,
                                 ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "Vector Length: " << vLength << " Time Taken: " << omp_get_wtime() - sT << endl;

    for (int i = 0; i < vLength; ++i) {
        testCipher("result", result[i], nBits, bk, key);
    }

    //free memory
    for (int i = 0; i < vLength; ++i) {
        freeLweSample_16_gpu(inputA[i]);
        freeLweSample_16_gpu(inputB[i]);
        freeLweSample_16_gpu(result[i]);
    }
    freeLweSample_16_gpu(a);
    freeLweSample_16_gpu(b);

}

void multiplyLweSamples_test(LweSample *ca, LweSample *cb, int nBits,
                             const TFheGateBootstrappingCloudKeySet *bk,
                             cufftDoubleComplex *cudaBkFFTCoalesceExt,
                             Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                             TFheGateBootstrappingSecretKeySet *key) {

    const int iBits = nBits, n = 500;
    int *temp;//pointer
    int isDoublePrecision = false;
    const int oBits = isDoublePrecision ? nBits * 2 : nBits;

    LweSample_16 *a = convertBitToNumber(ca, iBits, bk);
    LweSample_16 *b = convertBitToNumber(cb, iBits, bk);
    //send a, b, result to cuda
    temp = a->a;
    cudaMalloc(&(a->a), iBits * n * sizeof(int));
    cudaMemcpy(a->a, temp, iBits * n * sizeof(int), H2D);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), iBits * n * sizeof(int));
    cudaMemcpy(b->a, temp, iBits * n * sizeof(int), H2D);
    free(temp);


    LweSample_16 *result = convertBitToNumberZero_GPU(oBits, bk);
    int startTime = omp_get_wtime();
    multiplyLweSamples(result, a, b, iBits, isDoublePrecision, bk, cudaBkFFTCoalesceExt,
                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "Time Taken To Multiply: " << omp_get_wtime() - startTime << endl;
    testCipher("multiplication Result", result, oBits, bk, key);

    freeLweSample_16_gpu(a);
    freeLweSample_16_gpu(b);
    freeLweSample_16_gpu(result);

}

LweSample_16 *padLweSample_16(LweSample_16 *ip, int cLen, int nLen, const TFheGateBootstrappingCloudKeySet *bk) {//assumes the input.a in cuda
    LweSample_16 * output = convertBitToNumberZero_GPU(nLen, bk);
    int n = bk->params->in_out_params->n;
    cudaMemset(output->a, 0, nLen * n * sizeof(int));
    cudaMemcpy(output->a, ip->a, n * cLen * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(output->b, ip->b, cLen * sizeof(int));
    memcpy(output->current_variance, ip->current_variance, cLen * sizeof(double));
    return output;
}

__global__ void cudaLeftShift_vector(int *destination, int *source, int vLen, int bitSize, int n, int nBitShift, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bIndex = (id / n) % bitSize;
        int newVal = bIndex < nBitShift ? 0 : source[id - n * nBitShift];
        destination[id] = newVal;
    }
}

void leftShiftCuda_16_vector(LweSample_16 *output, LweSample_16 *input, int vLen, int bitSize, int nBitShift,
                      const TFheGateBootstrappingCloudKeySet *bk) {

    const int n = bk->params->in_out_params->n;
    int BLOCKSIZE = 1024;
    int length = n * bitSize * vLen;

    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    //shift input->a
    cudaLeftShift_vector<<<gridSize, BLOCKSIZE>>>(output->a, input->a, vLen, bitSize, n, nBitShift, length);
    //shift input->b and input->cv
    memcpy(output->b, input->b - nBitShift, bitSize * vLen * sizeof(int));
    memcpy(output->current_variance, input->current_variance - nBitShift, bitSize * vLen * sizeof(double));
    //set shifted bits
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    int message = 0;
    for (int i = 0; i < vLen; ++i) {
        int sI = i * bitSize;
        for (int j = 0; j < nBitShift; ++j) {
            output->b[j + sI] = message ? MU : -MU;
            output->current_variance[j + sI] = 0.;
        }
    }
}

void expandNbitTo2Nbit(LweSample_16 *out, LweSample_16* in, int nInputBits, int nOutputBits, int nConMul,
                       const TFheGateBootstrappingCloudKeySet *bk) {
//    cout << "KJHGIVU" << endl;
    int n = bk->params->in_out_params->n;

    int dstLenA = nOutputBits * n;
    int sourceLenA = nInputBits * n;
    int dstLenB = nOutputBits;
    int sourceLenB = nInputBits;

    for (int i = 0; i < nConMul; ++i) {
        cudaMemcpy(out->a + i * dstLenA, in->a + i * sourceLenA, sourceLenA * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(out->b + i * dstLenB, in->b + i * sourceLenB, sourceLenB * sizeof(int));
        memcpy(out->current_variance + i * dstLenB, in->current_variance + i * sourceLenB, sourceLenB * sizeof(int));
    }

}

void BOOTS_vectorMultiplication(LweSample_16 **result, LweSample_16 **a, LweSample_16 **b,
                                const int vLength, const int nBits, bool isDoublePrecision,
                                const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                TFheGateBootstrappingSecretKeySet *key) {
    const int iBits = nBits, oBits = isDoublePrecision ? nBits * 2: nBits, n = 500;

    LweSample_16 *aExpanded = convertBitToNumberZero_GPU(vLength * iBits * iBits, bk);
    LweSample_16 *bExpanded = convertBitToNumberZero_GPU(vLength * iBits * iBits, bk);
//    LweSample_16 *andExpanded = convertBitToNumberZero_GPU(vLength * iBits * iBits, bk);
    LweSample_16 *andLShiftExp = convertBitToNumberZero_GPU(vLength * iBits * oBits, bk);
    LweSample_16 *iResult = convertBitToNumberZero_GPU(vLength * iBits / 2 * oBits, bk);
    //andLShiftExp = 0
    cudaMemset(andLShiftExp->a, 0, vLength * iBits * oBits * n * sizeof(Torus32));
    //temp
    LweSample_16 *temp = convertBitToNumberZero_GPU(iBits, bk);
    LweSample_16 *temp2 = convertBitToNumberZero_GPU(oBits, bk);

    //expand a and b
    for (int i = 0; i < iBits; ++i) {
        for (int j = 0; j < vLength; ++j) {
            cudaMemcpy(aExpanded->a + i * vLength * iBits * n + j * iBits * n, a[j]->a, iBits * n * sizeof(Torus32),
                       D2D);
            memcpy(aExpanded->b + i * vLength * iBits + j * iBits, a[j]->b, iBits * sizeof(int));

            for (int k = 0; k < iBits; ++k) {
                cudaMemcpy(bExpanded->a + i * vLength * iBits * n + j * iBits * n + k * n, b[j]->a + i * n,
                           n * sizeof(Torus32), D2D);
                memcpy(bExpanded->b + i * vLength * iBits + j * iBits + k, b[j]->b + i, sizeof(int));
            }
        }
    }

    bootsAND_fullGPU_n_Bit(aExpanded, aExpanded, bExpanded,
                           vLength * iBits * iBits, cudaBkFFTCoalesceExt,
                           ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr);


    /*for (int i = 0; i < vLength; ++i) {
        for (int j = 0; j < iBits; ++j) {
            cudaMemcpy(temp->a, aExpanded->a + i * iBits * iBits * n + j * iBits * n, iBits * n * sizeof(Torus32), D2D);
            memcpy(temp->b, aExpanded->b + i * iBits * iBits + j * iBits, iBits * sizeof(int));
            testCipher("a", temp, iBits, bk, key);

            cudaMemcpy(temp->a, bExpanded->a + i * iBits * iBits * n + j * iBits * n, iBits * n * sizeof(Torus32), D2D);
            memcpy(temp->b, bExpanded->b + i * iBits * iBits + j * iBits, iBits * sizeof(int));
            testCipher("b", temp, iBits, bk, key);

            cudaMemcpy(temp->a, andExpanded->a + i * iBits * iBits * n + j * iBits * n, iBits * n * sizeof(Torus32), D2D);
            memcpy(temp->b, andExpanded->b + i * iBits * iBits + j * iBits, iBits * sizeof(int));
            testCipher("&", temp, iBits, bk, key);
            cout << endl;
        }
        cout << endl;
    }*/

//    exit(1);

    leftShiftExpandedFormatForMultiplication_precision(andLShiftExp, aExpanded, vLength, iBits, oBits, isDoublePrecision);

    /*for (int i = 0; i < vLength; ++i) {
        for (int j = 0; j < iBits; ++j) {
            cudaMemcpy(temp->a, aExpanded->a + i * iBits * iBits * n + j * iBits * n, iBits * n * sizeof(Torus32), D2D);
            memcpy(temp->b, aExpanded->b + i * iBits * iBits + j * iBits, iBits * sizeof(int));
            testCipher("&", temp, iBits, bk, key);

            cudaMemcpy(temp2->a, andLShiftExp->a + i * iBits * oBits * n + j * oBits * n, oBits * n * sizeof(Torus32), D2D);
            memcpy(temp2->b, andLShiftExp->b + i * iBits * oBits + j * oBits, oBits * sizeof(int));
            testCipher("&S", temp2, oBits, bk, key);
            cout << endl;
        }
        cout << endl;
    }*/

    //free a and b
    freeLweSample_16_gpu(aExpanded);
    freeLweSample_16_gpu(bExpanded);
//    freeLweSample_16_gpu(andExpanded);

    int vLengthCoal = iBits / 2, nCoal = vLength;
    LweSample_16 *initAndShiftExp = andLShiftExp;
    LweSample_16 *halfLeAndShiftExp = new LweSample_16;

    //set half len
    halfLeAndShiftExp->a = andLShiftExp->a + vLengthCoal * vLength * oBits * n;
    halfLeAndShiftExp->b = andLShiftExp->b + vLengthCoal * vLength * oBits;

    for (int i = 0; vLengthCoal > 0; i++) {

        taskLevelParallelAdd_bitwise_vector_coalInput(iResult, initAndShiftExp, halfLeAndShiftExp,
                                                      vLengthCoal, oBits, nCoal,
                                                      bk, cudaBkFFTCoalesceExt,
                                                      ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

        /*for (int j = 0; j < vLengthCoal * nCoal; ++j) {
            cudaMemcpy(temp2->a, iResult->a + j * oBits * n, oBits * n * sizeof(Torus32), D2D);
            memcpy(temp2->b, iResult->b + j * oBits, oBits * sizeof(int));
            testCipher("res", temp2, oBits, bk, key);
        }
        cout << endl;*/

        vLengthCoal /= 2;
        initAndShiftExp = iResult;
        halfLeAndShiftExp->a = iResult->a + vLengthCoal * vLength * oBits * n;
        halfLeAndShiftExp->b = iResult->b + vLengthCoal * vLength * oBits;
        if (i == 0) {
            freeLweSample_16_gpu(andLShiftExp);
        }
    }

    for (int i = 0; i < vLength; ++i) {
        cudaMemcpy(result[i]->a, iResult->a + i * oBits * n, oBits * n * sizeof(Torus32), D2D);
        memcpy(result[i]->b, iResult->b + i * oBits, oBits * sizeof(int));
    }

    //free memory
    freeLweSample_16_gpu(iResult);
    freeLweSample_16_gpu(temp);
    freeLweSample_16_gpu(temp2);
}

void karatMasterSuba(LweSample_16 *result, LweSample_16 *a, LweSample_16 *b,
                     int nBit,
                     const TFheGateBootstrappingCloudKeySet *bk,
                     cufftDoubleComplex *cudaBkFFTCoalesceExt,
                     Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                     TFheGateBootstrappingSecretKeySet *key) {

    const int n = 500, oBit = nBit * 2, hBit = nBit / 2, fBit = nBit;

    //divide the numbers into half bitsize numbers
    LweSample_16 *Xl = a;
    LweSample_16 *Xr = new LweSample_16;
    LweSample_16 *Yl = b;
    LweSample_16 *Yr = new LweSample_16;
    LweSample_16 *XYl = convertBitToNumberZero_GPU(fBit, bk);
    LweSample_16 *XYr = convertBitToNumberZero_GPU(fBit, bk);

    LweSample_16 *P3 = convertBitToNumberZero_GPU(fBit, bk);
    LweSample_16 *P3_1 = P3;
    LweSample_16 *P3_2 = new LweSample_16;
    LweSample_16 *PE = convertBitToNumberZero_GPU(2 * fBit, bk);
    LweSample_16 *PE_1 = convertBitToNumberZero_GPU(2 * fBit, bk);
    LweSample_16 *PE_2 = convertBitToNumberZero_GPU(2 * fBit, bk);
    LweSample_16 *E_ADD = PE;

    LweSample_16 *E_1sComplement = convertBitToNumberZero_GPU(fBit, bk);
    LweSample_16 *E_ONE = convertBitToNumberZero_GPU(fBit, bk);
    LweSample_16 *E = convertBitToNumberZero_GPU(fBit, bk);

    LweSample_16 *temp = convertBitToNumberZero_GPU(hBit, bk);
    LweSample_16 *temp2 = convertBitToNumberZero_GPU(fBit, bk);

    int nConMul = 3;
    LweSample_16 **input1 = new LweSample_16 *[nConMul];
    LweSample_16 **input2 = new LweSample_16 *[nConMul];
    LweSample_16 **conMulRes = new LweSample_16 *[nConMul];

    //E_ONE = 1
    static Torus32 MU = modSwitchToTorus32(1, 8);
    E_ONE->b[0] = MU;

    Xr->a = a->a + hBit * n;
    Xr->b = a->b + hBit;

    Yr->a = b->a + hBit * n;
    Yr->b = b->b + hBit;

    cudaMemcpy(XYl->a, Xl->a, hBit * n * sizeof(int), D2D);
    memcpy(XYl->b, Xl->b, hBit * sizeof(int));

    cudaMemcpy(XYl->a + hBit * n, Yl->a, hBit * n * sizeof(int), D2D);
    memcpy(XYl->b + hBit, Yl->b, hBit * sizeof(int));

    cudaMemcpy(XYr->a, Xr->a, hBit * n * sizeof(int), D2D);
    memcpy(XYr->b, Xr->b, hBit * sizeof(int));

    cudaMemcpy(XYr->a + hBit * n, Yr->a, hBit * n * sizeof(int), D2D);
    memcpy(XYr->b + hBit, Yr->b, hBit * sizeof(int));

    taskLevelParallelAdd_bitwise_vector_coalInput(P3, XYl, XYr,
                                                  2, hBit, 1,
                                                  bk, cudaBkFFTCoalesceExt,
                                                  ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

    /*cudaMemcpy(temp->a, Xl->a, h_bitSize * n * sizeof(Torus32), D2D);
    memcpy(temp->b, Xl->b, h_bitSize * sizeof(int));
    testCipher("Xl", temp, h_bitSize, bk, key);

    cudaMemcpy(temp->a, Xr->a, h_bitSize * n * sizeof(Torus32), D2D);
    memcpy(temp->b, Xr->b, h_bitSize * sizeof(int));
    testCipher("Xr", temp, h_bitSize, bk, key);

    cudaMemcpy(temp->a, Yl->a, h_bitSize * n * sizeof(Torus32), D2D);
    memcpy(temp->b, Yl->b, h_bitSize * sizeof(int));
    testCipher("Yl", temp, h_bitSize, bk, key);

    cudaMemcpy(temp->a, Yr->a, h_bitSize * n * sizeof(Torus32), D2D);
    memcpy(temp->b, Yr->b, h_bitSize * sizeof(int));
    testCipher("Yr", temp, h_bitSize, bk, key);

    cudaMemcpy(temp->a, P3->a, h_bitSize * n * sizeof(Torus32), D2D);
    memcpy(temp->b, P3->b, h_bitSize * sizeof(int));
    testCipher("P30", temp, h_bitSize, bk, key);

    cudaMemcpy(temp->a, P3->a + h_bitSize * n, h_bitSize * n * sizeof(Torus32), D2D);
    memcpy(temp->b, P3->b + h_bitSize, h_bitSize * sizeof(int));
    testCipher("P31", temp, h_bitSize, bk, key);*/

    P3_1 = P3;
    P3_2->a = P3->a + hBit * n;
    P3_2->b = P3->b + hBit;

    input1[0] = Xl;
    input1[1] = Xr;
    input1[2] = P3_1;
    input2[0] = Yl;
    input2[1] = Yr;
    input2[2] = P3_2;
    conMulRes[0] = convertBitToNumberZero_GPU(fBit, bk);
    conMulRes[1] = convertBitToNumberZero_GPU(fBit, bk);
    conMulRes[2] = convertBitToNumberZero_GPU(fBit, bk);

    BOOTS_vectorMultiplication(conMulRes, input1, input2, nConMul,
                               hBit, true,
                               bk, cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

    /*for (int i = 0; i < nConMul; ++i) {
        cudaMemcpy(temp->a, input1[i]->a, h_bitSize * n * sizeof(Torus32), D2D);
        memcpy(temp->b, input1[i]->b, h_bitSize * sizeof(int));
        testCipher("input1", temp, h_bitSize, bk, key);

        cudaMemcpy(temp->a, input2[i]->a, h_bitSize * n * sizeof(Torus32), D2D);
        memcpy(temp->b, input2[i]->b, h_bitSize * sizeof(int));
        testCipher("input2", temp, h_bitSize, bk, key);

        cudaMemcpy(temp2->a, conMulRes[i]->a, f_bitSize * n * sizeof(Torus32), D2D);
        memcpy(temp2->b, conMulRes[i]->b, f_bitSize * sizeof(int));
        testCipher("r1", temp2, f_bitSize, bk, key);
        cout << endl;
    }*/

    LweSample_16 *P1 = conMulRes[0];
    LweSample_16 *P2 = conMulRes[1];
    LweSample_16 *E_MUL = conMulRes[2];

    /*testCipher("P1", P1, f_bitSize, bk, key);
    testCipher("P2", P2, f_bitSize, bk, key);
    testCipher("EMUL", E_MUL, f_bitSize, bk, key);*/

    cudaMemcpy(PE_1->a, P1->a, fBit * n * sizeof(int), D2D);
    memcpy(PE_1->b, P1->b, fBit * sizeof(int));

    /*int i =0;
    cudaMemcpy(temp2->a, PE_1->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), D2D);
    memcpy(temp2->b, PE_1->b + i * f_bitSize, f_bitSize * sizeof(int));
    testCipher("PE_1.0", temp2, f_bitSize, bk, key);*/

    cudaMemcpy(PE_1->a + fBit * n, E_MUL->a, fBit * n * sizeof(int), D2D);
    memcpy(PE_1->b + fBit, E_MUL->b, fBit * sizeof(int));

    /*i =1;
    cudaMemcpy(temp2->a, PE_1->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), D2D);
    memcpy(temp2->b, PE_1->b + i * f_bitSize, f_bitSize * sizeof(int));
    testCipher("PE_1.1", temp2, f_bitSize, bk, key);*/

    cudaMemcpy(PE_2->a, P2->a, fBit * n * sizeof(int), D2D);
    memcpy(PE_2->b, P2->b, fBit * sizeof(int));

    /*i =0;
    cudaMemcpy(temp2->a, PE_2->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), D2D);
    memcpy(temp2->b, PE_2->b + i * f_bitSize, f_bitSize * sizeof(int));
    testCipher("PE_2.0", temp2, f_bitSize, bk, key);*/

    cudaMemcpy(PE_2->a + fBit * n, E_ONE->a, fBit * n * sizeof(int), D2D);
    memcpy(PE_2->b + fBit, E_ONE->b, fBit * sizeof(int));

    /*i =1;
    cudaMemcpy(temp2->a, PE_2->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), D2D);
    memcpy(temp2->b, PE_2->b + i * f_bitSize, f_bitSize * sizeof(int));
    testCipher("PE_2.1", temp2, f_bitSize, bk, key);*/

    taskLevelParallelAdd_bitwise_vector_coalInput(PE, PE_1, PE_2,
                                                  2, fBit, 1,
                                                  bk, cudaBkFFTCoalesceExt,
                                                  ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    /*for (int i = 0; i < 2; ++i) {
        cudaMemcpy(temp2->a, PE->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), D2D);
        memcpy(temp2->b, PE->b + i * f_bitSize, f_bitSize * sizeof(int));
        testCipher("sum.1", temp2, f_bitSize, bk, key);
    }*/

    E_ADD = PE;
    E_MUL->a = PE->a + fBit * n;
    E_MUL->b = PE->b + fBit;

    bootsNOT_16(E_1sComplement, E_ADD, fBit, n);

    /*cudaMemcpy(temp2->a, E_ADD->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(temp2->b, E_ADD->b, f_bitSize * sizeof(int));
    testCipher("AC + BD", temp2, f_bitSize, bk, key);
    cudaMemcpy(temp2->a, E_MUL->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(temp2->b, E_MUL->b, f_bitSize * sizeof(int));
    testCipher("E_MUL + 1", temp2, f_bitSize, bk, key);
    testCipher("(AC + BD)", E_1sComplement, f_bitSize, bk, key);*/

    taskLevelParallelAdd_bitwise(E, E_MUL, E_1sComplement,
                                 nBit, bk, cudaBkFFTCoalesceExt,
                                 ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

    cudaMemcpy(result->a + fBit * n, P2->a, fBit * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(result->b + fBit, P2->b, fBit * sizeof(int));


    cudaMemcpy(result->a + hBit * n, E->a, hBit * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(result->b + hBit, E->b, hBit * sizeof(int));

    cudaMemcpy(result->a, P1->a, hBit * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(result->b, P1->b, hBit * sizeof(int));

    //free memory
    delete Xr;
    delete Yr;
    freeLweSample_16_gpu(XYl);
    freeLweSample_16_gpu(XYr);
    freeLweSample_16_gpu(P3);
    delete P3_2;
    freeLweSample_16_gpu(PE);
    freeLweSample_16_gpu(PE_1);
    freeLweSample_16_gpu(PE_2);
    freeLweSample_16_gpu(E_1sComplement);
    freeLweSample_16_gpu(E_ONE);
    freeLweSample_16_gpu(E);
    freeLweSample_16_gpu(temp);
    freeLweSample_16_gpu(temp2);
    freeLweSample_16_gpu(P1);
    freeLweSample_16_gpu(P2);
    delete[] input1;
    delete[] input2;
    delete[] conMulRes;
}

void karatSuba_test(LweSample *ca, LweSample *cb, int nBit,
                    const TFheGateBootstrappingCloudKeySet *bk,
                    cufftDoubleComplex *cudaBkFFTCoalesceExt,
                    Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                    TFheGateBootstrappingSecretKeySet *key) {
    const int n = 500, nOut = nBit * 2;

    LweSample_16 *a = convertBitToNumber(ca, nBit, bk);
    LweSample_16 *b = convertBitToNumber(cb, nBit, bk);
    LweSample_16 *result = convertBitToNumberZero_GPU(nOut, bk);

    //send a, b, result to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), nBit * n * sizeof(int));
    cudaMemcpy(a->a, temp, nBit * n * sizeof(int), H2D);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), nBit * n * sizeof(int));
    cudaMemcpy(b->a, temp, nBit * n * sizeof(int), H2D);
    free(temp);

    cout << "KARAT " << nBit << endl;
    int startTime = omp_get_wtime();
    karatMasterSuba(result, a, b,
                    nBit, bk,
                    cudaBkFFTCoalesceExt,
                    ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "Time Taken To Multiply: " << omp_get_wtime() - startTime << endl;
    testCipher("A", a, nBit, bk, key);
    testCipher("B", b, nBit, bk, key);
    testCipher("A * B", result, nOut, bk, key);

    freeLweSample_16_gpu(a);
    freeLweSample_16_gpu(b);
    freeLweSample_16_gpu(result);
}


void vectorMultiplicationTest(LweSample *ca, LweSample *cb, int vLength, int nBits,
                              const TFheGateBootstrappingCloudKeySet *bk,
                              cufftDoubleComplex *cudaBkFFTCoalesceExt,
                              Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                              TFheGateBootstrappingSecretKeySet *key) {

    bool isDoublePrecision = true;
    const int iBits = nBits, oBits = isDoublePrecision ? nBits * 2 : nBits, n = 500;

    LweSample_16 *a = convertBitToNumber(ca, iBits, bk);
    LweSample_16 *b = convertBitToNumber(cb, iBits, bk);

    //send a, b, result to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), iBits * n * sizeof(Torus32));
    cudaMemcpy(a->a, temp, iBits * n * sizeof(Torus32), H2D);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), iBits * n * sizeof(Torus32));
    cudaMemcpy(b->a, temp, iBits * n * sizeof(Torus32), H2D);
    free(temp);

    //prepare vector
    cout << "Preparing vector" << endl;
    LweSample_16 **inputA = new LweSample_16 *[vLength];
    LweSample_16 **inputB = new LweSample_16 *[vLength];
    LweSample_16 **result = new LweSample_16 *[vLength];
    for (int i = 0; i < vLength; ++i) {
        inputA[i] = convertBitToNumberZero_GPU(iBits, bk);
//        leftShiftCuda_16(inputA[i], a, iBits, i % 10, bk);
        cudaMemcpy(inputA[i]->a, a->a, iBits * n * sizeof(Torus32), D2D);
        memcpy(inputA[i]->b, a->b, iBits * sizeof(int));
//        testCipher("inputA", inputA[i], iBits, bk, key);

        inputB[i] = convertBitToNumberZero_GPU(iBits, bk);
//        leftShiftCuda_16(inputB[i], b, iBits, i % 10, bk);
        cudaMemcpy(inputB[i]->a, b->a, iBits * n * sizeof(Torus32), D2D);
        memcpy(inputB[i]->b, b->b, iBits * sizeof(int));
//        testCipher("inputB", inputB[i], iBits, bk, key);

        result[i] = convertBitToNumberZero_GPU(oBits, bk);
//        cout << endl;
    }

    cout << "Starting Vector Multiplication" << endl;
    double sT = omp_get_wtime();
    BOOTS_vectorMultiplication(result, inputA, inputB, vLength,
                               iBits, isDoublePrecision,
                               bk, cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "Vector Length: " << vLength << " Time taken: " << omp_get_wtime() - sT << endl;
    //show output
    for (int i = 0; i < vLength; ++i) {
        testCipher("inputA", inputA[i], iBits, bk, key);
        testCipher("inputB", inputB[i], iBits, bk, key);
        testCipher("result", result[i], oBits, bk, key);
        cout << endl;
    }

    //free memory
    for (int i = 0; i < vLength; ++i) {
        freeLweSample_16_gpu(inputA[i]);
        freeLweSample_16_gpu(inputB[i]);
        freeLweSample_16_gpu(result[i]);
    }
    freeLweSample_16_gpu(a);
    freeLweSample_16_gpu(b);
    delete[] inputA;
    delete[] inputB;
    delete[] result;
}

int decryptCheck(Cipher cipher, TFheGateBootstrappingSecretKeySet *key) {
    int int_answer = 0;
    for (int i = 0; i < cipher.numberOfBits; i++) {
        int ai = bootsSymDecrypt(&cipher.data[i], key);
#ifdef DEBUG
        printf("%d ",ai);
#endif
        int_answer |= (ai << i);
    }
#ifdef DEBUG
    printf("\n");
#endif
    if (int_answer > pow(2, cipher.numberOfBits - 1)) {
        int_answer = -1 * (pow(2, cipher.numberOfBits) - int_answer);

    }
//    cout << "decrypt check: number of bits:" << cipher.numberOfBits << endl;
    return int_answer;
}

void BOOTS_matrixAddition(LweSample_16 ***result, LweSample_16 ***mA, LweSample_16 ***mB,
                          int row, int col, int nBits,
                          const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                          Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                          TFheGateBootstrappingSecretKeySet *key) {
    const int n = 500;
    LweSample_16 *v1 = convertBitToNumberZero_GPU(row * col * nBits, bk);
    LweSample_16 *v2 = convertBitToNumberZero_GPU(row * col * nBits, bk);
    LweSample_16 *iResult = convertBitToNumberZero_GPU(row * col * nBits, bk);
//    LweSample_16 *temp = convertBitToNumberZero_GPU(nBits, bk);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            cudaMemcpy(v1->a  + (i * col + j) * nBits * n, mA[i][j]->a, nBits * n * sizeof(Torus32), D2D);
            memcpy(v1->b + (i * col + j) * nBits, mA[i][j]->b, nBits * sizeof(int));
            cudaMemcpy(v2->a  + (i * col + j) * nBits * n, mB[i][j]->a, nBits * n * sizeof(Torus32), D2D);
            memcpy(v2->b + (i * col + j) * nBits, mB[i][j]->b, nBits * sizeof(int));
        }
    }
    //addition function
    taskLevelParallelAdd_bitwise_vector_coalInput(iResult, v1, v2,
                                                  row * col, nBits, 1, bk, cudaBkFFTCoalesceExt,
                                                  ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    /*for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int index = i * col + j;
            int sI = index * nBits;
            cudaMemcpy(temp->a, v1->a + sI * n, nBits * n * sizeof(int), D2D);
            memcpy(temp->b, v1->b + sI, nBits * sizeof(int));
            testCipher("v1", temp, nBits, bk, key);

            cudaMemcpy(temp->a, v2->a + sI * n, nBits * n * sizeof(int), D2D);
            memcpy(temp->b, v2->b + sI, nBits * sizeof(int));
            testCipher("v2", temp, nBits, bk, key);

            cudaMemcpy(temp->a, iResult->a + sI * n, nBits * n * sizeof(int), D2D);
            memcpy(temp->b, iResult->b + sI, nBits * sizeof(int));
            testCipher("r", temp, nBits, bk, key);
            cout << endl;
        }
    }*/
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            cudaMemcpy(result[i][j]->a, iResult->a + (i * col + j) * nBits * n, nBits * n * sizeof(Torus32), D2D);
            memcpy(result[i][j]->b, iResult->b + (i * col + j) * nBits, nBits * sizeof(int));

        }
    }
    freeLweSample_16_gpu(v1);
    freeLweSample_16_gpu(v2);
    freeLweSample_16_gpu(iResult);
}

void testMatrixAddition(LweSample *ca, LweSample *cb, int nBits,
                         const TFheGateBootstrappingCloudKeySet *bk,
                         cufftDoubleComplex *cudaBkFFTCoalesceExt,
                         Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                         TFheGateBootstrappingSecretKeySet *key) {
    const int n = 500;
    const int row = 2, col = 2;
    //construct matrices
    LweSample_16 ***m1 = new LweSample_16**[row];
    LweSample_16 ***m2 = new LweSample_16**[row];
    LweSample_16 ***addResult = new LweSample_16**[row];
    for (int i = 0; i < row ; ++i) {
        m1[i] = new LweSample_16*[col];
        m2[i] = new LweSample_16*[col];
        addResult[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            m1[i][j] = convertBitToNumber(ca, nBits, bk);
            m2[i][j] = convertBitToNumber(cb, nBits, bk);
            addResult[i][j] = convertBitToNumberZero_GPU(nBits, bk);

            int *temp;
            temp = m1[i][j]->a;
            cudaMalloc(&(m1[i][j]->a), nBits * n * sizeof(int));
            cudaMemcpy(m1[i][j]->a, temp, nBits * n * sizeof(int), H2D);
            free(temp);

            temp = m2[i][j]->a;
            cudaMalloc(&(m2[i][j]->a), nBits * n * sizeof(int));
            cudaMemcpy(m2[i][j]->a, temp, nBits * n * sizeof(int), H2D);
            free(temp);

            int x = i * col + j;
            if (x > 0 && x < nBits) {
                leftShiftCuda_16_vector(m1[i][j], m1[0][0], 1, nBits, x, bk);
                leftShiftCuda_16_vector(m2[i][j], m2[0][0], 1, nBits, x, bk);
            }

            cout << "i: " << i << " j: " << j << endl;
            testCipher("m1", m1[i][j], nBits, bk, key);
            testCipher("m2", m2[i][j], nBits, bk, key);
        }
    }
    cout << "Matrix Addition" << endl;
    double sT = omp_get_wtime();
    BOOTS_matrixAddition(addResult, m1, m2,
                         row, col, nBits,
                         bk, cudaBkFFTCoalesceExt,
                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "row: " << row << " col: " << col << " Time taken: " << omp_get_wtime() - sT << endl;

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            testCipher("res", addResult[i][j], nBits, bk, key);
        }
    }
    //free memory
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            freeLweSample_16_gpu(m1[i][j]);
            freeLweSample_16_gpu(m2[i][j]);
            freeLweSample_16_gpu(addResult[i][j]);
        }
    }
}


void BOOTS_matrixMultiplication(LweSample_16 ***result, LweSample_16 ***mA, LweSample_16 ***mB,
                          int row, int col, int nBits, bool isDoublePrecision,
                          const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                          Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                          TFheGateBootstrappingSecretKeySet *key) {
    assert(row == col);
    const int n = 500, iBits = nBits, oBits = isDoublePrecision ? nBits * 2 : nBits;
    LweSample_16 **leftMat = matMul_prepareLeftMat(mA, row, col, col, iBits, bk);
    LweSample_16 **rightMat = matMul_prepareRightMat(mB, row, col, row, iBits, bk);
    LweSample_16 **iResult = new LweSample_16*[row * col * row];

    for (int i = 0; i < row * col * row; ++i) {
        iResult[i] = convertBitToNumberZero_GPU(oBits, bk);
    }
    int vLength = row * col * row;
    BOOTS_vectorMultiplication(iResult, leftMat, rightMat,
                               vLength, iBits, isDoublePrecision,
                               bk, cudaBkFFTCoalesceExt,
                               ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

    /*for (int i = 0; i < row * col * row; ++i) {
        testCipher("left", leftMat[i], iBits, bk, key);
        testCipher("right", rightMat[i], iBits, bk, key);
        testCipher("res", iResult[i], iBits, bk, key);
        cout << endl;
    }*/
    LweSample_16 *addV1 = convertBitToNumberZero_GPU(vLength/2 * oBits, bk);
    LweSample_16 *addV2 = convertBitToNumberZero_GPU(vLength/2 * oBits, bk);
    LweSample_16 *iResCoal = convertBitToNumberZero_GPU(vLength/2 * oBits, bk);
//    LweSample_16 *temp = convertBitToNumberZero_GPU(oBits, bk);
    /*for (; vLength > row * col; vLength /= 2) {
        for (int i = 0; i < vLength; ++i) {
            int j = i / 2;
            if(i % 2 == 0) {
                leftMat[j] = iResult[i];
                testCipher("left", leftMat[j], iBits, bk, key);
            } else {
                leftMat[j] = iResult[i];
                testCipher("right", rightMat[j], iBits, bk, key);
            }
        }

//        BOOTS_vectorMultiplication(iResult, leftMat, rightMat,
//                                   vLength, iBits, isDoublePrecision,
//                                   bk, cudaBkFFTCoalesceExt,
//                                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    }*/

    for (int i = 0; i < vLength; ++i) {
        int j = i / 2;
        if(i % 2 == 0) {
            cudaMemcpy(addV1->a + j * oBits * n, iResult[i]->a, oBits * n * sizeof(int), D2D);
            memcpy(addV1->b + j * oBits, iResult[i]->b, oBits * sizeof(int));

            /*cudaMemcpy(temp->a, addV1->a + j * oBits * n, oBits * n * sizeof(int), D2D);
            memcpy(temp->b, addV1->b + j * oBits, oBits * sizeof(int));
            testCipher("0 ", temp, oBits, bk, key);*/
        } else {
            cudaMemcpy(addV2->a + j * oBits * n, iResult[i]->a, oBits * n * sizeof(int), D2D);
            memcpy(addV2->b + j * oBits, iResult[i]->b, oBits * sizeof(int));

            /*cudaMemcpy(temp->a, addV2->a + j * oBits * n, oBits * n * sizeof(int), D2D);
            memcpy(temp->b, addV2->b + j * oBits, oBits * sizeof(int));
            testCipher("1 ", temp, oBits, bk, key);*/
        }

        freeLweSample_16_gpu(iResult[i]);
    }

    taskLevelParallelAdd_bitwise_vector_coalInput(iResCoal, addV1, addV2,
                                                  vLength / 2, oBits, 1,
                                                  bk, cudaBkFFTCoalesceExt,
                                                  ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

    for (vLength /= 2; vLength > row * col; vLength /= 2) {
        for (int i = 0; i < vLength; ++i) {
            int j = i / 2;
            if(i % 2 == 0) {
                cudaMemcpy(addV1->a + j * oBits * n, iResCoal->a + i * oBits * n, oBits * n * sizeof(int), D2D);
                memcpy(addV1->b + j * oBits, iResCoal->b + i * oBits, oBits * sizeof(int));

                /*cudaMemcpy(temp->a, addV1->a + j * oBits * n, oBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
                memcpy(temp->b, addV1->b + j * oBits, oBits * sizeof(int));
                testCipher("0 ", temp, oBits, bk, key);*/
            } else {
                cudaMemcpy(addV2->a + j * oBits * n, iResCoal->a + i * oBits * n, oBits * n * sizeof(int), D2D);
                memcpy(addV2->b + j * oBits, iResCoal->b + i * oBits, oBits * sizeof(int));

                /*cudaMemcpy(temp->a, addV2->a + j * oBits * n, oBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
                memcpy(temp->b, addV2->b + j * oBits, oBits * sizeof(int));
                testCipher("1 ", temp, oBits, bk, key);*/
            }
        }

        taskLevelParallelAdd_bitwise_vector_coalInput(iResCoal, addV1, addV2,
                                                      vLength / 2, oBits, 1,
                                                      bk, cudaBkFFTCoalesceExt,
                                                      ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//        cout << endl;
    }

    /*for (int i = 0; i < row * col; ++i) {
        cudaMemcpy(temp->a, iResCoal->a + i * oBits * n, oBits * n * sizeof(int), D2D);
        memcpy(temp->b, iResCoal->b + i * oBits, oBits * sizeof(int));
        testCipher("iR", temp, oBits, bk, key);
    }*/

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            cudaMemcpy(result[i][j]->a, iResCoal->a + (i * col + j) * oBits * n, oBits * n * sizeof(Torus32), D2D);
            memcpy(result[i][j]->b, iResCoal->b + (i * col + j) * oBits, oBits * sizeof(int));
        }
    }

    delete [] leftMat;
    delete [] rightMat;
    delete [] iResult;
    freeLweSample_16_gpu(addV1);
    freeLweSample_16_gpu(addV2);
    freeLweSample_16_gpu(iResCoal);
}

void testMatrixMultiplication(LweSample *ca, LweSample *cb, int nBits,
                        const TFheGateBootstrappingCloudKeySet *bk,
                        cufftDoubleComplex *cudaBkFFTCoalesceExt,
                        Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                        TFheGateBootstrappingSecretKeySet *key) {
    bool isDoublePrecision = false;
    const int n = 500, iBits = nBits, oBits = isDoublePrecision ? nBits * 2 : nBits;
    const int row = 4, col = 4;

    //construct matrices
    LweSample_16 ***m1 = new LweSample_16**[row];
    LweSample_16 ***m2 = new LweSample_16**[row];
    LweSample_16 ***mulResult = new LweSample_16**[row];
    for (int i = 0; i < row ; ++i) {
        m1[i] = new LweSample_16*[col];
        m2[i] = new LweSample_16*[col];
        mulResult[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            m1[i][j] = convertBitToNumber(ca, iBits, bk);
            m2[i][j] = convertBitToNumber(cb, iBits, bk);
            mulResult[i][j] = convertBitToNumberZero_GPU(oBits, bk);

            int *temp;
            temp = m1[i][j]->a;
            cudaMalloc(&(m1[i][j]->a), iBits * n * sizeof(int));
            cudaMemcpy(m1[i][j]->a, temp, iBits * n * sizeof(int), H2D);
            free(temp);

            temp = m2[i][j]->a;
            cudaMalloc(&(m2[i][j]->a), iBits * n * sizeof(int));
            cudaMemcpy(m2[i][j]->a, temp, iBits * n * sizeof(int), H2D);
            free(temp);

            int x = i * col + j;
            if (x > 0 && x < iBits) {
//                leftShiftCuda_16_vector(m1[i][j], m1[0][0], 1, iBits, x, bk);
//                leftShiftCuda_16_vector(m2[i][j], m2[0][0], 1, iBits, x, bk);
            }

            cout << "i: " << i << " j: " << j << endl;
            testCipher("m1", m1[i][j], iBits, bk, key);
            testCipher("m2", m2[i][j], iBits, bk, key);
        }
    }
    cout << "Matrix Multiplication" << endl;
    double sT = omp_get_wtime();
    BOOTS_matrixMultiplication(mulResult, m1, m2,
                         row, col, nBits, isDoublePrecision,
                         bk, cudaBkFFTCoalesceExt,
                         ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "row: " << row << " col: " << col << " Time taken: " << omp_get_wtime() - sT << endl;

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            testCipher("res", mulResult[i][j], nBits, bk, key);
        }
    }
    //free memory
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            freeLweSample_16_gpu(m1[i][j]);
            freeLweSample_16_gpu(m2[i][j]);
            freeLweSample_16_gpu(mulResult[i][j]);
        }
    }
}

void upRotateMatrix(LweSample_16 ***matrix, int row, int col) {
    for (int i = 0; i < row - 1; i++) {
        swap(matrix[i], matrix[i + 1]);
    }
}

void upRotateVec(LweSample_16 ***mat, int nRow, int colIndex) {
    for (int i = 0; i < nRow - 1; i++) {
        swap(mat[i][colIndex], mat[i + 1][colIndex]);
    }
}

void leftRotateMatrix(LweSample_16 ***matrix, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col - 1; j++) {
            swap(matrix[i][j], matrix[i][j + 1]);
        }
    }
}

void leftRotateVec(LweSample_16 **vec, int nElem) {
    for (int i = 0; i < nElem - 1; i++) {
        swap(vec[i], vec[i + 1]);
    }
}

void printMatrix(LweSample_16 ***mat, int bitSize, int row, int col,
                 const TFheGateBootstrappingCloudKeySet *bk,
                 TFheGateBootstrappingSecretKeySet *key) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            cout << "row: " << i << " col: " << j << "\t";
            testCipher("mat", mat[i][j], bitSize, bk, key);
        }
        cout << endl;
    }
}

void printVector(LweSample_16 **vec, int bitSize, int nElem,
                 const TFheGateBootstrappingCloudKeySet *bk,
                 TFheGateBootstrappingSecretKeySet *key) {
    cout << "vector: " << endl;
    for (int i = 0; i < nElem; ++i) {
        testCipher("vi", vec[i], bitSize, bk, key);
    }
    cout << endl;
}

LweSample_16** vectorFromMatrix_rowMajor(LweSample_16 ***matrix, int row, int col,
                                        const TFheGateBootstrappingCloudKeySet *bk) {
    LweSample_16 **vector = new LweSample_16*[row * col];
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            vector[i * col + j] = matrix[i][j];
        }
    }
    return vector;
}

void BOOTS_CannonsAlgo(LweSample_16 ***mResult, LweSample_16 ***mA, LweSample_16 ***mB,
                     int row, int col, int nBits, bool isDoublePrecision,
                     const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                     Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                     TFheGateBootstrappingSecretKeySet *key) {

    const int n = 500, iBits = nBits, oBits = isDoublePrecision ? nBits * 2: nBits, nElem = row * col;

    LweSample_16 **vRes = vectorFromMatrix_rowMajor(mResult, row, col, bk);
    //temps
    LweSample_16 **tempV1 = new LweSample_16*[row * col];
    LweSample_16 *temp = convertBitToNumberZero_GPU(oBits, bk);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < i; j++) {
            leftRotateVec(mA[i], row);
            upRotateVec(mB, row, i);
        }
    }
    //vRes = 0
    for (int i = 0; i < row * col; ++i) {
        tempV1[i] = convertBitToNumberZero_GPU(oBits, bk);
    }

    for (int k = 0; k < row; ++k) {
        LweSample_16 **vecA = vectorFromMatrix_rowMajor(mA, row, col, bk);
        LweSample_16 **vecB = vectorFromMatrix_rowMajor(mB, row, col, bk);

        if (k == 0) {
            BOOTS_vectorMultiplication(vRes, vecA, vecB,
                                       nElem, iBits, isDoublePrecision,
                                       bk, cudaBkFFTCoalesceExt,
                                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
        } else {
            BOOTS_vectorMultiplication(tempV1, vecA, vecB,
                                       nElem, iBits, isDoublePrecision,
                                       bk, cudaBkFFTCoalesceExt,
                                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
            BOOTS_vectorAddition(vRes, vRes, tempV1, 1, nElem, oBits, bk, cudaBkFFTCoalesceExt,
                                 ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
        }
        leftRotateMatrix(mA, row, col);
        upRotateMatrix(mB, row, col);

        delete [] vecA;
        delete [] vecB;
    }
    //free memory
    for (int i = 0; i < row * col; ++i) {
        freeLweSample_16_gpu(tempV1[i]);
    }
    freeLweSample_16_gpu(temp);
    delete [] tempV1;
    delete [] vRes;
}

void testCannonsAlgo(LweSample *ca, LweSample *cb, int nBits,
                             const TFheGateBootstrappingCloudKeySet *bk,
                             cufftDoubleComplex *cudaBkFFTCoalesceExt,
                             Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                             TFheGateBootstrappingSecretKeySet *key) {
    bool isDoublePrecision = false;
    const int n = 500, iBits = nBits, oBits = isDoublePrecision ? nBits * 2 : nBits;
    const int row = 16, col = 16;

    //construct matrices
    LweSample_16 ***m1 = new LweSample_16**[row];
    LweSample_16 ***m2 = new LweSample_16**[row];
    LweSample_16 ***mulResult = new LweSample_16**[row];
    for (int i = 0; i < row ; ++i) {
        m1[i] = new LweSample_16*[col];
        m2[i] = new LweSample_16*[col];
        mulResult[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            m1[i][j] = convertBitToNumber(ca, iBits, bk);
            m2[i][j] = convertBitToNumber(cb, iBits, bk);
            mulResult[i][j] = convertBitToNumberZero_GPU(oBits, bk);

            int *temp;
            temp = m1[i][j]->a;
            cudaMalloc(&(m1[i][j]->a), iBits * n * sizeof(int));
            cudaMemcpy(m1[i][j]->a, temp, iBits * n * sizeof(int), H2D);
            free(temp);

            temp = m2[i][j]->a;
            cudaMalloc(&(m2[i][j]->a), iBits * n * sizeof(int));
            cudaMemcpy(m2[i][j]->a, temp, iBits * n * sizeof(int), H2D);
            free(temp);

//            int x = i * col + j;
//            if (x > 0 && x < iBits) {
//                leftShiftCuda_16_vector(m1[i][j], m1[0][0], 1, iBits, x, bk);
//                leftShiftCuda_16_vector(m2[i][j], m2[0][0], 1, iBits, x, bk);
//            }

            cout << "i: " << i << " j: " << j << endl;
            testCipher("m1", m1[i][j], iBits, bk, key);
            testCipher("m2", m2[i][j], iBits, bk, key);
        }
    }
    cout << "Cannon's Matrix Multiplication" << endl;
    double sT = omp_get_wtime();
    BOOTS_CannonsAlgo(mulResult, m1, m2,
                    row, col, nBits, isDoublePrecision,
                    bk, cudaBkFFTCoalesceExt,
                    ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
    cout << "row: " << row << " col: " << col << " Time taken: " << omp_get_wtime() - sT << endl;

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            testCipher("res", mulResult[i][j], nBits, bk, key);
        }
    }
    //free memory
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            freeLweSample_16_gpu(m1[i][j]);
            freeLweSample_16_gpu(m2[i][j]);
            freeLweSample_16_gpu(mulResult[i][j]);
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Usage: <binary_file_name> <bit_size> <first_number> <second_number> <vector_length>" << endl;
        return 1;
    }
    const int minimum_lambda = 110;
    TFheGateBootstrappingParameterSet *params = new_default_gate_bootstrapping_parameters(minimum_lambda);


    //generate a random key
    uint32_t seed[] = {314, 1592, 657};
    tfhe_random_generator_setSeed(seed, 3);
    TFheGateBootstrappingSecretKeySet *key = new_random_gate_bootstrapping_secret_keyset(params);

    //generate encrypt the 16 bits
    int bitSize = atoi(argv[1]);//=16
//    cout << "bitSize: " << bitSize << endl;
    int plaintext1 = atoi(argv[2]);
    LweSample *ciphertext1 = new_gate_bootstrapping_ciphertext_array(bitSize, params);

    for (int i = 0; i < bitSize; i++) {
        bootsSymEncrypt(&ciphertext1[i], (plaintext1 >> i) & 1, key);
    }

    //generate encrypt the 16 bits
    int plaintext2 = atoi(argv[3]);
    LweSample *ciphertext2 = new_gate_bootstrapping_ciphertext_array(bitSize, params);
    LweSample *ciphertext3 = new_gate_bootstrapping_ciphertext_array(bitSize, params);

    for (int i = 0; i < bitSize; i++) {
        bootsSymEncrypt(&ciphertext2[i], (plaintext2 >> i) & 1, key);
    }

    int vLength = atoi(argv[4]);

    Cipher a(bitSize);
    Cipher b(bitSize);
    Cipher c(bitSize);
    a.data = ciphertext1;
    b.data = ciphertext2;

    cout << "A: " << decryptCheck(a, key) << endl;
    cout << "B: " << decryptCheck(b, key) << endl;

//    cudaDeviceProp prop;
//    cudaGetDeviceProperties(&prop, 0);
//    cout << endl;
//    cout << prop.multiProcessorCount << endl;//20

    //send all keys to GPU
    cudaDeviceReset();

    cufftDoubleComplex *cudaBkFFTCoalesceExt = sendBootstrappingKeyToGPUCoalesceExt(&key->cloud);
    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, &key->cloud);//bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(&key->cloud);//(bk);
//    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

    test_AND_XOR_CompoundGate_Addition(a.data, b.data, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
                                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//    test_vectorAddition(a.data, b.data, vLength, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                        ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//    multiplyLweSamples_test(a.data, b.data, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                            ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//    vectorMultiplicationTest(a.data, b.data, vLength, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                             ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//    karatSuba_test(a.data, b.data, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//
//    testMatrixAddition(a.data, b.data, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//    testMatrixMultiplication(a.data, b.data, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                             ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);
//    testCannonsAlgo(a.data, b.data, bitSize, &key->cloud, cudaBkFFTCoalesceExt,
//                    ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, key);

    cudaDeviceReset();
    //clean up all pointers
    delete_gate_bootstrapping_ciphertext_array(bitSize, ciphertext2);
    delete_gate_bootstrapping_ciphertext_array(bitSize, ciphertext1);

    //clean up all pointers
    delete_gate_bootstrapping_secret_keyset(key);
    delete_gate_bootstrapping_parameters(params);
    return 0;
}


