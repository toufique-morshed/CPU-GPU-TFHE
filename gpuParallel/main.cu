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

#define D2D cudaMemcpyDeviceToDevice
#define D2H cudaMemcpyDeviceToHost

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
    //test variables
//    cout << "n: " << n << endl;
//    cout << "N: " << N << endl;
//    cout << "Ns2: " << Ns2 << endl;
//    cout << "bitSize: " << bitSize << endl;
//    cout << "tlwe_params->k: " << tlwe_params->k << endl;//1
//    cout << "bk_params->l: " << bk_params->l << endl;//2
//    cout << "bk_params->kpl: " << bk_params->kpl << endl;//4
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

//__global__ void sendKeySwitchKeyToGPU_extendedMemAllocationHelper(Torus32** destination, Torus32 *source, int k,
//                                                                  int bitSize, int n, int length) {
//    int id = blockIdx.x * blockDim.x + threadIdx.x;
//    if (id < length) {
//        cudaMalloc(&destination[k], sizeof(Torus32) * bitSize * n);
//        cudaMemcpyToSymbol(destination[k], source, sizeof(Torus32) * bitSize * n, 0, cudaMemcpyHostToDevice);
//    }
//}

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
    cout <<  "bitSize " << bitSize << endl;
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

//void bootsAND_16_test(LweSample *cresult, const LweSample *ca, const LweSample *cb, int bitSize,
//                      const TFheGateBootstrappingCloudKeySet *bk,
//                      TFheGateBootstrappingSecretKeySet *key) {
//    double startTime = omp_get_wtime();
//    for (int i = 0; i < bitSize; ++i) {
//        bootsXOR(&cresult[i], &ca[i], &cb[i], bk);
//    }
//    cout << endl << "Time for Sequential AND: " << omp_get_wtime() - startTime << endl << endl;
//
//    cufftDoubleComplex ****cudaBkFFT = sendBootstrappingKeyToGPU(bitSize, bk);
//    Torus32 ****ks_a_gpu = NULL;//sendKeySwitchKeyToGPU(bitSize, bk);
//    Torus32 ****ks_a_gpu_extended = sendKeySwitchKeyToGPU_extended(bitSize, bk);
//    int ***ks_b_gpu = sendKeySwitchBtoGPU(bk);
//    double ***ks_cv_gpu = sendKeySwitchCVtoGPU(bk);
//    const LweParams *in_out_params = bk->params->in_out_params;
//
//    LweSample_16* a = convertBitToNumber(ca, bitSize, bk);
//    LweSample_16* b = convertBitToNumber(cb, bitSize, bk);
//    LweSample_16* result = convertBitToNumberZero(bitSize, bk);
//
//    //send a, b, and result to cuda
//    int * temp = new int[bitSize * in_out_params->n];
//
//    temp = a->a;
//    cudaMalloc(&(a->a), bitSize * in_out_params->n * sizeof(int));
//    cudaMemcpy(a->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
//    free(temp);
//
//    temp = b->a;
//    cudaMalloc(&(b->a), bitSize * in_out_params->n * sizeof(int));
//    cudaMemcpy(b->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
//    free(temp);
//
//    temp = result->a;
//    cudaMalloc(&(result->a), bitSize * in_out_params->n * sizeof(int));
//    cudaMemcpy(result->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
//    free(temp);
//
//    cudaCheckErrors("cudaMalloc/cudaMemcpy failed");
//    cudaDeviceSynchronize();
//
//    startTime = omp_get_wtime();
//    bootsXOR_16(result, a, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu);
//    cout << "Time for || AND: " << omp_get_wtime() - startTime << endl;
//    testCipher("result", result, bitSize, bk, key);
//    bootsXOR_16(result, a, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu);
//    testCipher("result", result, bitSize, bk, key);
//    bootsXOR_16(result, a, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu);
//    testCipher("result", result, bitSize, bk, key);
//    bootsXOR_16(result, a, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu);
//    testCipher("result", result, bitSize, bk, key);
//    bootsXOR_16(result, b, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu);
//    testCipher("result", result, bitSize, bk, key);
//    bootsXOR_16(b, a, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu);
//    testCipher("b", b, bitSize, bk, key);
//
//
//    temp = new int[bitSize * in_out_params->n];
//    cudaMemcpy(temp, result->a, sizeof(int) * bitSize * in_out_params->n, cudaMemcpyDeviceToHost);
//    cudaFree(result->a);
//    result->a = temp;
//
//    LweSample* andOutput = convertNumberToBits(result, bitSize, bk);
//
//    Cipher andOutputCipher(bitSize);
//    andOutputCipher.data = andOutput;
//
//    cout << "In bootsAND_16_test : " << decryptCheck(andOutputCipher, key) << endl;
//
//}
//
//void bootsXOR_16_test(LweSample *cresult, const LweSample *ca, const LweSample *cb, int bitSize,
//                      const TFheGateBootstrappingCloudKeySet *bk,
//                      TFheGateBootstrappingSecretKeySet *key) {
//    double startTime = omp_get_wtime();
//    for (int i = 0; i < bitSize; ++i) {
//        bootsXOR(&cresult[i], &ca[i], &cb[i], bk);
//    }
//    cout << endl << "Time for Sequential XOR: " << omp_get_wtime() - startTime << endl << endl;
//    //convert to number
//    LweSample_16* a = convertBitToNumber(ca, bitSize, bk);
//    LweSample_16* b = convertBitToNumber(cb, bitSize, bk);
//    LweSample_16* result = convertBitToNumberZero(bitSize, bk);
//
//    cufftDoubleComplex ****cudaBkFFT = sendBootstrappingKeyToGPU(bitSize, bk);
//
//    Torus32 ****ks_a_gpu = sendKeySwitchKeyToGPU(bitSize, bk);
//    Torus32 ****ks_a_gpu_extended = sendKeySwitchKeyToGPU_extended(bitSize, bk);
//
//    //send a, b, and result to cuda
//    const LweParams *in_out_params = bk->params->in_out_params;
//    int * temp = new int[bitSize * in_out_params->n];
//    temp = a->a;
//    cudaMalloc(&(a->a), bitSize * in_out_params->n * sizeof(int));
//    cudaMemcpy(a->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
//    free(temp);
//
//    temp = b->a;
//    cudaMalloc(&(b->a), bitSize * in_out_params->n * sizeof(int));
//    cudaMemcpy(b->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
//    free(temp);
//
//    temp = result->a;
//    cudaMalloc(&(result->a), bitSize * in_out_params->n * sizeof(int));
//    cudaMemcpy(result->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
//    free(temp);
//
//
//    startTime = omp_get_wtime();
//    bootsXOR_16(result, a, b, bitSize, bk, cudaBkFFT, ks_a_gpu, ks_a_gpu_extended);
//    cout << "Time for || XOR: " << omp_get_wtime() - startTime << endl;
//
//    //get data in cpu from gpu
//    temp = new int[bitSize * in_out_params->n];
//    cudaMemcpy(temp, result->a, sizeof(int) * bitSize * in_out_params->n, cudaMemcpyDeviceToHost);
//    cudaFree(result->a);
//    result->a = temp;
//
//    LweSample* andOutput = convertNumberToBits(result, bitSize, bk);
//
//    Cipher andOutputCipher(bitSize);
//    andOutputCipher.data = andOutput;
//
//    cout << "In bootsXOR_16_test: " << decryptCheck(andOutputCipher, key) << endl;
//}

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


void bootsCircuitADDSequential(LweSample_16 *cudaRes, LweSample_16 *cudaA, LweSample_16 *cudaB, int bitSize,
                               cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                               Torus32 ****ks_a_gpu_extended,
                               const TFheGateBootstrappingCloudKeySet *bk,
                               TFheGateBootstrappingSecretKeySet *key, int ***ks_b_gpu, double ***ks_cv_gpu) {
    const LweParams *in_out_params = bk->params->in_out_params;
    LweSample_16 *carry = convertBitToNumberZero_GPU(bitSize, bk);
    LweSample_16 *tempB = convertBitToNumberZero_GPU(bitSize, bk);


    //make cudaRes equal to cudaA
    cudaMemcpy(cudaRes->a, cudaA->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(cudaRes->b, cudaA->b, sizeof(int) * bitSize);
    memcpy(cudaRes->current_variance, cudaA->current_variance, sizeof(int) * bitSize);

    //make tempB equal B
    cudaMemcpy(tempB->a, cudaB->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(tempB->b, cudaB->b, sizeof(int) * bitSize);
    memcpy(tempB->current_variance, cudaB->current_variance, sizeof(int) * bitSize);


    for (int i = 0; i < bitSize; ++i) {
//        bootsAND_16(carry, cudaRes, tempB, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
//                    ks_cv_gpu);
//        bootsXOR_16(cudaRes, cudaRes, tempB, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended,
//                    ks_b_gpu, ks_cv_gpu);
//        leftShiftCuda_16(tempB, carry, bitSize, bk);
    }
    //free memory
    cudaFree(carry->a);
    carry->a = NULL;
    freeLweSample_16(carry);
    cudaFree(tempB->a);
    tempB->a = NULL;
    freeLweSample_16(tempB);
}

void bootsCircuitADDParallelOpenMP(LweSample_16 *cudaRes, LweSample_16 *cudaA, LweSample_16 *cudaB, int bitSize,
                                   cufftDoubleComplex ****cudaBkFFT, Torus32 ****ks_a_gpu_extended,
                                   const TFheGateBootstrappingCloudKeySet *bk,
                                   TFheGateBootstrappingSecretKeySet *key,
                                   cudaFFTProcessorTest *p1,
                                   cudaFFTProcessorTest *p2) {
    const LweParams *in_out_params = bk->params->in_out_params;
    LweSample_16 *carry = convertBitToNumberZero_GPU(bitSize, bk);
    LweSample_16 *tempB = convertBitToNumberZero_GPU(bitSize, bk);


    //make cudaRes equal to cudaA
    cudaMemcpy(cudaRes->a, cudaA->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(cudaRes->b, cudaA->b, sizeof(int) * bitSize);
    memcpy(cudaRes->current_variance, cudaA->current_variance, sizeof(int) * bitSize);

    //make tempB equal B
    cudaMemcpy(tempB->a, cudaB->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(tempB->b, cudaB->b, sizeof(int) * bitSize);
    memcpy(tempB->current_variance, cudaB->current_variance, sizeof(int) * bitSize);

    testCipher("carry", carry, bitSize, bk, key);
    testCipher("tempB", tempB, bitSize, bk, key);

    //free memory
    cudaFree(carry->a);
    carry->a = NULL;
    freeLweSample_16(carry);
    cudaFree(tempB->a);
    tempB->a = NULL;
    freeLweSample_16(tempB);
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


void taskLevelParallelAdd(LweSample_16 *cudaRes, LweSample_16 *a, LweSample_16 *b, int bitSize,
                          const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                          cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                          int ***ks_b_gpu, double ***ks_cv_gpu,
                          Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr,
                          TFheGateBootstrappingSecretKeySet *key) {

    int nOutputs = 2;
    const LweParams *in_out_params = bk->params->in_out_params;
    LweSample_16 *taskResult = convertBitToNumberZero(bitSize, bk);
    LweSample_16 *tempB = convertBitToNumberZero_GPU(bitSize, bk);
//    LweSample_16* testingVec = convertBitToNumberZero_GPU(bitSize, bk);
    //modify result so that it can accomodate 2 resuts
    cudaMalloc(&(taskResult->a), nOutputs * bitSize * in_out_params->n * sizeof(int));
    taskResult->b = (int *) calloc(nOutputs * bitSize, sizeof(int));
    taskResult->current_variance = (double *) calloc(nOutputs * bitSize, sizeof(double));

    cudaMemcpy(tempB->a, b->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(tempB->b, b->b, bitSize * sizeof(int));
    memcpy(tempB->current_variance, b->current_variance, bitSize * sizeof(double));
//    testCipher("tempB", tempB, bitSize, bk, key);

    cudaMemcpy(cudaRes->a, a->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(cudaRes->b, a->b, bitSize * sizeof(int));
    memcpy(cudaRes->current_variance, a->current_variance, bitSize * sizeof(double));
//    testCipher("cudaRes", cudaRes, bitSize, bk, key);
    //test for vector addition
//    bootsANDXOR_16(taskResult, cudaRes, tempB, nOutputs, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended,
//                   ks_b_gpu, ks_cv_gpu);
//    cudaMemcpy(carry->a, taskResult->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(carry->b, taskResult->b, bitSize * sizeof(int));
//    memcpy(carry->current_variance, taskResult->current_variance, bitSize * sizeof(double));
//    testCipher("a", a, bitSize, bk, key);
//    testCipher("b", b, bitSize, bk, key);
//    testCipher("AND", carry, bitSize, bk, key);
//    cudaMemcpy(carry->a, taskResult->a + (bitSize * in_out_params->n), bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(carry->b, taskResult->b + bitSize, bitSize * sizeof(int));
//    memcpy(carry->current_variance, taskResult->current_variance + bitSize, bitSize * sizeof(double));
//    testCipher("XOR", carry, bitSize, bk, key);
//    oneBitLeftShiftFromTwoOutputs(tempB, taskResult, bitSize, bk);
//    testCipher("XOR", tempB, bitSize, bk, key);
//    cudaMemcpy(cudaRes->a, result->a + (bitSize * in_out_params->n), sizeof(int) * bitSize * in_out_params->n,
//               cudaMemcpyDeviceToDevice);
//    memcpy(cudaRes->b, result->b + bitSize, bitSize * sizeof(int));
//    memcpy(cudaRes->current_variance, result->current_variance + bitSize, bitSize * sizeof(double));
//    testCipher("cudaRes", cudaRes, bitSize, bk, key);
    for (int i = 0; i < bitSize; ++i) {

        bootsANDXOR_16(taskResult, cudaRes, tempB, nOutputs, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
                       ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

        //split the result
//        LweSample_16 *carry = convertBitToNumberZero_GPU(bitSize, bk);
//        cudaMemcpy(carry->a, taskResult->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(carry->b, taskResult->b, bitSize * sizeof(int));
//        memcpy(carry->current_variance, taskResult->current_variance, bitSize * sizeof(double));
//        testCipher("AND", carry, bitSize, bk, key);
        oneBitLeftShiftFromTwoOutputs(tempB, taskResult, bitSize, bk);
//
//        testCipher("ANDshifted", tempB, bitSize, bk, key);


        cudaMemcpy(cudaRes->a, taskResult->a + (bitSize * in_out_params->n), sizeof(int) * bitSize * in_out_params->n,
                   cudaMemcpyDeviceToDevice);
        memcpy(cudaRes->b, taskResult->b + bitSize, bitSize * sizeof(int));
        memcpy(cudaRes->current_variance, taskResult->current_variance + bitSize, bitSize * sizeof(double));
//        testCipher("XOR", cudaRes, bitSize, bk, key);

//        cudaMemcpy(testingVec->a, taskResult->a + (bitSize * in_out_params->n), sizeof(int) * bitSize * in_out_params->n,
//                   cudaMemcpyDeviceToDevice);
//        memcpy(testingVec->b, taskResult->b + bitSize, bitSize * sizeof(int));
//        memcpy(testingVec->current_variance, taskResult->current_variance + bitSize, bitSize * sizeof(double));
//        testCipher("XOR", testingVec, bitSize, bk, key);
//        cout << "I am here" << endl;

//        leftShiftCuda_16(tempB, carry, bitSize, bk);
//        testCipher("tempB", tempB, bitSize, bk, key);
    }
}

void addCudaLweSamplePreProcessing(LweSample *ca, LweSample *cb, int bitSize,
                                   const TFheGateBootstrappingCloudKeySet *bk,
                                   TFheGateBootstrappingSecretKeySet *key) {
    //get keys

    cufftDoubleComplex ****cudaBkFFT = sendBootstrappingKeyToGPU(bitSize, bk);
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(bitSize, bk);
    Torus32 ****ks_a_gpu_extended = sendKeySwitchKeyToGPU_extended(bitSize, bk);
    int ***ks_b_gpu = sendKeySwitchBtoGPU(bk);
    double ***ks_cv_gpu = sendKeySwitchCVtoGPU(bk);
    const LweParams *in_out_params = bk->params->in_out_params;

    //convert to number
    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);
    LweSample_16 *result = convertBitToNumberZero(bitSize, bk);
    LweSample_16 *carry = convertBitToNumberZero(bitSize, bk);
    LweSample_16 *tempB = convertBitToNumberZero(bitSize, bk);

    //send a, b, result, carry and tempB to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    free(carry->a);
    cudaMalloc(&(carry->a), bitSize * in_out_params->n * sizeof(int));

    //send a to result
    free(result->a);
    cudaMalloc(&(result->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(result->a, a->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(result->b, a->b, sizeof(int) * bitSize);
    memcpy(result->current_variance, a->current_variance, sizeof(int) * bitSize);
    //send b to tempB
    free(tempB->a);
    cudaMalloc(&(tempB->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(tempB->a, b->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(tempB->b, b->b, sizeof(int) * bitSize);
    memcpy(tempB->current_variance, b->current_variance, sizeof(int) * bitSize);

    testCipher("a", a, bitSize, bk, key);
    testCipher("b", b, bitSize, bk, key);

    double sTime = omp_get_wtime();
    bootsCircuitADDSequential(result, a, b, bitSize, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu_extended, bk, key, ks_b_gpu,
                              ks_cv_gpu);
    cout << "Time taken to ADD (Sequential): " << omp_get_wtime() - sTime << endl;
    testCipher("ADD", result, bitSize, bk, key);
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





void taskLevelParallelAdd_bitwise(LweSample_16 *mainResult, LweSample_16 *a, LweSample_16 *b,
                                         int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                                         cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                         Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu,
                                         double ***ks_cv_gpu, Torus32 *ks_a_gpu_extendedPtr,
                                         Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr,
                                         TFheGateBootstrappingSecretKeySet *key) {
    int nOutputs = 2, n = bk->params->in_out_params->n, nBits = 1;
    LweSample_16 *r = convertBitToNumberZero_GPU(2, bk);
    LweSample_16 *r1 = convertBitToNumberZero_GPU(1, bk);
    LweSample_16 *t = convertBitToNumberZero_GPU(2, bk);
    LweSample_16 *t1 = convertBitToNumberZero_GPU(1, bk);
    LweSample_16 *ai = convertBitToNumberZero_GPU(1, bk);
    LweSample_16 *bi = convertBitToNumberZero_GPU(1, bk);
    LweSample_16 *temp = convertBitToNumberZero_GPU(1, bk);

    cudaFree(r1->a);
    cudaFree(t1->a);
    cudaFree(ai->a);
    cudaFree(bi->a);

    t1->a = t->a + n;
    t1->b = t->b + nBits;
    t1->current_variance = t->current_variance + nBits;

    r1->a = r->a + n;
    r1->b = r->b + nBits;
    r1->current_variance = r->current_variance + nBits;

    // t0 = a, t1 = b;
//    cudaMemcpy(t->a, a->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(t->b, a->b, sizeof(int));
//    memcpy(t->current_variance, a->current_variance, sizeof(double));
//
//    cudaMemcpy(t->a + nBits * n, b->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(t->b + nBits, b->b, sizeof(int));
//    memcpy(t->current_variance + nBits, b->current_variance, sizeof(double));
//
//    bootsAND_16(t, t, t1, nBits, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
//                ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
//    bootsXOR_16(r, a, t1, nBits, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
//                ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
//    cudaMemcpy(r1->a, t->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(r1->b, t->b, sizeof(int));
//    memcpy(r1->current_variance, t->current_variance, sizeof(double));
    bootsANDXOR_16(r, a, b, nOutputs, nBits, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended,
                   ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    cudaMemcpy(temp->a, r->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp->b, r->b, nBits * sizeof(int));
//    testCipher("R0", temp, nBits, bk, key);
//
//    cudaMemcpy(temp->a, r->a + nBits * n, nBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp->b, r->b + nBits, nBits * sizeof(int));
//    testCipher("R1", temp, nBits, bk, key);

    cudaMemcpy(mainResult->a, r->a + n, n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(mainResult->b, r->b + nBits, sizeof(int));
    memcpy(mainResult->current_variance, r->current_variance + nBits, sizeof(double));

    cudaMemcpy(r->a + n, r->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(r->b + nBits, r->b, sizeof(int));
    memcpy(r->current_variance + nBits, r->current_variance, sizeof(double));


    for (int bI = 1; bI < bitSize; ++bI) {
        ai->a = a->a + n * bI;
        ai->b = a->b + bI;
        ai->current_variance = a->current_variance + bI;
//        cudaMemcpy(ai->a, a->a + n * bI, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(ai->b, a->b + bI, sizeof(int));
//        memcpy(ai->current_variance, a->current_variance + bI, sizeof(double));
//
        bi->a = b->a + n * bI;
        bi->b = b->b + bI;
        bi->current_variance = b->current_variance + bI;
//        cudaMemcpy(bi->a, b->a + n * bI, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(bi->b, b->b + bI, sizeof(int));
//        memcpy(bi->current_variance, b->current_variance + bI, sizeof(double));
//
//        cudaMemcpy(r1->a, r->a + n, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(r1->b, r->b + 1, sizeof(int));
//        memcpy(r1->current_variance, r->current_variance + 1, sizeof(double));

//        cout << "----------" << endl;
//        cudaMemcpy(temp->a, r1->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, r1->b, nBits * sizeof(int));
//        memcpy(temp->current_variance, r1->current_variance, nBits * sizeof(double));
//        testCipher("R1", temp, nBits, bk, key);

//        cudaMemcpy(temp->a, ai->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, ai->b, nBits * sizeof(int));
//        memcpy(temp->current_variance, ai->current_variance, nBits * sizeof(double));
//        testCipher("ai", temp, nBits, bk, key);

//        cudaMemcpy(temp->a, bi->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, bi->b, nBits * sizeof(int));
//        memcpy(temp->current_variance, bi->current_variance, nBits * sizeof(double));
//        testCipher("bi", temp, nBits, bk, key);
//        cout << endl;

//        cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << endl;
        bootsXORXOR_16(t, ai, r1, bi, r1, nOutputs, nBits, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended,
                       ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//        cudaMemcpy(temp->a, t->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, t->b, nBits * sizeof(int));
//        memcpy(temp->current_variance, t->current_variance, nBits * sizeof(double));
//        testCipher("t0", temp, nBits, bk, key);
//
//        cudaMemcpy(temp->a, t->a + nBits * n, nBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, t->b + nBits, nBits * sizeof(int));
//        memcpy(temp->current_variance, t->current_variance + nBits, nBits * sizeof(double));
//        testCipher("t1", temp, nBits, bk, key);

        bootsAND_16(t, t, t1, nBits, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
                    ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);


//        cudaMemcpy(temp->a, t->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, t->b, nBits * sizeof(int));
//        memcpy(temp->current_variance, t->current_variance, nBits * sizeof(double));
//        testCipher("t0t1", temp, nBits, bk, key);

        bootsXORXOR_16(r, ai, t1, t, r1, nOutputs, nBits, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended,
                       ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

        cudaMemcpy(mainResult->a + bI * n, r->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(mainResult->b + bI, r->b, sizeof(int));
        memcpy(mainResult->current_variance + bI, r->current_variance, sizeof(double));

//        cudaMemcpy(temp->a, r->a, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, r->b, nBits * sizeof(int));
//        memcpy(temp->current_variance, r->current_variance, nBits * sizeof(double));
//        testCipher("R0", temp, nBits, bk, key);

//        cudaMemcpy(temp->a, r->a + nBits * n, nBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, r->b + nBits, nBits * sizeof(int));
//        memcpy(temp->current_variance, r->current_variance + nBits, nBits * sizeof(double));
//        testCipher("R1", temp, nBits, bk, key);
//        testCipher("Main result", mainResult, bitSize, bk, key);


//        testCipher("t", t, 2, bk, key);
    }
//    testCipher("Main result", mainResult, bitSize, bk, key);

}


void test_AND_XOR_CompoundGate_Addition(LweSample *ca, LweSample *cb, int bitSize,
                              const TFheGateBootstrappingCloudKeySet *bk,
                              TFheGateBootstrappingSecretKeySet *key) {
    //get keys
    int nOutputs = 2;
    cufftDoubleComplex ****cudaBkFFT = NULL;//sendBootstrappingKeyToGPU(bitSize, bk);
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    cufftDoubleComplex *cudaBkFFTCoalesceExt = sendBootstrappingKeyToGPUCoalesceExt(bk);
    Torus32 ****ks_a_gpu_extended = NULL;//sendKeySwitchKeyToGPU_extended(bitSize, bk);
//    Torus32 ****ks_a_gpu_extended_2 = sendKeySwitchKeyToGPU_extended_2(nOutputs, bitSize, bk);
    int ***ks_b_gpu = NULL;//sendKeySwitchBtoGPU(bk);
    double ***ks_cv_gpu = NULL;//sendKeySwitchCVtoGPU(bk);
    const LweParams *in_out_params = bk->params->in_out_params;
    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);
//    Torus32 *ks_a_gpu_extendedPtr2 = sendKeySwitchKeyToGPU_extendedOnePointer(bitSize, bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);


//    Torus32 *ks_a_gpu_extendedOnePtr = sendKeySwitchKeyToGPU_extendedOnePointer(bitSize, bk);

    //convert to number
    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);
    LweSample_16 *result = convertBitToNumberZero(bitSize, bk);
    LweSample_16 *andResult = convertBitToNumberZero(bitSize, bk);
    LweSample_16 *xorResult = convertBitToNumberZero(bitSize, bk);

    //send a, b, result to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    free(andResult->a);
    free(xorResult->a);
    cudaMalloc(&(andResult->a), bitSize * in_out_params->n * sizeof(int));
    cudaMalloc(&(xorResult->a), bitSize * in_out_params->n * sizeof(int));

    //modify result so that it can accomodate 2 resuts
    cudaMalloc(&(result->a), nOutputs * bitSize * in_out_params->n * sizeof(int));
    result->b = (int *) calloc(nOutputs * bitSize, sizeof(int));
    result->current_variance = (double *) calloc(nOutputs * bitSize, sizeof(double));

    float et;
    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "...Starting Experiments of: A and B of " << bitSize << " bits... " << endl;
    for (int i = 0; i < nExp; ++i) {
        double sTime = omp_get_wtime();
        bootsAND_16(andResult, a, b, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
                    ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

        double eTime = omp_get_wtime();
        cout << " --------" << "AND time taken: " << eTime - sTime << " --------" << endl;
        testCipher("AB", andResult, bitSize, bk, key);
        //gpu bootAND
//        andResult = convertBitToNumberZero_GPU(bitSize, bk);
//        testCipher("AB before GPU and call", andResult, bitSize, bk, key);

//        sTime = omp_get_wtime();
//        bootsAND_fullGPU_OneBit(andResult, a, b, bitSize, cudaBkFFTCoalesceExt, ks_a_gpu_extendedPtr,
//                              ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//        bootsAND_fullGPU_n_Bit(andResult, a, b, bitSize, cudaBkFFTCoalesceExt, ks_a_gpu_extendedPtr,
//                                ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//        cudaDeviceSynchronize();
//        eTime = omp_get_wtime();
//        cout << " --------" << "AND time taken (full GPU): " << eTime - sTime << " --------" << endl;
//        testCipher("AB after and", andResult, bitSize, bk, key);

//        cout << " --------" << "AND time taken(cudaEvent): " << et << " --------" << endl;

        sTime = omp_get_wtime();
        bootsXOR_16(xorResult, a, b, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
                    ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
        eTime = omp_get_wtime();
        cout << " --------" << "XOR time taken: " << eTime - sTime << " --------" << endl;
        testCipher("A+B", xorResult, bitSize, bk, key);
    }
    testCipher("AB", andResult, bitSize, bk, key);


    //ANDXOR Together
    result = convertBitToNumberZero_GPU(bitSize * nOutputs, bk);
    for (int i = 0; i < nExp; ++i) {
        double sTime = omp_get_wtime();
        bootsANDXOR_16(result, a, b, nOutputs, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended,
                       ks_b_gpu,
                       ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
        double eTime = omp_get_wtime();
        cout << " --------" << "ANDXOR time taken: " << eTime - sTime << " --------" << endl;
    }

    //split the result
    cudaMemcpy(andResult->a, result->a, bitSize * in_out_params->n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(andResult->b, result->b, bitSize * sizeof(int));
    memcpy(andResult->current_variance, result->current_variance, bitSize * sizeof(double));
    testCipher("AB v2", andResult, bitSize, bk, key);

    cudaMemcpy(xorResult->a, result->a + (bitSize * in_out_params->n), sizeof(int) * bitSize * in_out_params->n,
               cudaMemcpyDeviceToDevice);
    memcpy(xorResult->b, result->b + bitSize, bitSize * sizeof(int));
    memcpy(xorResult->current_variance, result->current_variance + bitSize, bitSize * sizeof(double));
    testCipher("A+B v2", xorResult, bitSize, bk, key);

    //addition starting
    for (int i = 0; i < nExp; ++i) {
        double sTime = omp_get_wtime();
//        taskLevelParallelAdd(andResult, a, b, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
//                             ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        taskLevelParallelAdd_bitwise(andResult, a, b, 1, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce,
                                            NULL, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                            ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        cout << "--------ADDITION time taken: " << omp_get_wtime() - sTime << " --------" << endl;
        testCipher("Summation: ", andResult, bitSize, bk, key);
    }
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


void taskLevelParallelAdd_bitwise_vector(LweSample_16 *mainResult, LweSample_16 *a, LweSample_16 *b,
                                         int vLength, int bitSize, int nCoal,
                                         const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                                         cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu,
                                         Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                                         Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                         double *ks_cv_gpu_extendedPtr,
                                         TFheGateBootstrappingSecretKeySet *key) {
    int nOutputs = 2, n = bk->params->in_out_params->n;

    LweSample_16 *r = convertBitToNumberZero_GPU(2 * nCoal * vLength, bk);
    LweSample_16 *r1 = convertBitToNumberZero_GPU(1 * nCoal * vLength, bk);
    LweSample_16 *t = convertBitToNumberZero_GPU(2 * nCoal * vLength, bk);
    LweSample_16 *t1 = convertBitToNumberZero_GPU(1 * nCoal * vLength, bk);
    LweSample_16 *ai = convertBitToNumberZero_GPU(1 * nCoal * vLength, bk);
    LweSample_16 *bi = convertBitToNumberZero_GPU(1 * nCoal * vLength, bk);

    cudaFree(r1->a);
    cudaFree(t1->a);

    t1->a = t->a + nCoal * vLength * n;
    t1->b = t->b + nCoal * vLength;
    t1->current_variance = t->current_variance + nCoal * vLength;

    r1->a = r->a + nCoal * vLength * n;
    r1->b = r->b + nCoal * vLength;
    r1->current_variance = r->current_variance + nCoal * vLength;

    int bI = 0;
    for (int i = 0; i < vLength * nCoal; ++i) {
        cudaMemcpy(ai->a + i * n, a->a + i * bitSize * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(ai->b + i, a->b + i * bitSize, sizeof(int));
        memcpy(ai->current_variance + i, a->current_variance + i * bitSize, sizeof(double));
        /*
        cudaMemcpy(temp->a, ai->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, ai->b + i, nBits * sizeof(int));
        memcpy(temp->current_variance, ai->current_variance + i, nBits * sizeof(double));
        testCipher("ai", temp, 1, bk, key);*/

        cudaMemcpy(bi->a + i * n, b->a + i * bitSize * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(bi->b + i, b->b + i * bitSize, sizeof(int));
        memcpy(bi->current_variance + i, b->current_variance + i * bitSize, sizeof(double));
        /*
        cudaMemcpy(temp->a, bi->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, bi->b + i, nBits * sizeof(int));
        memcpy(temp->current_variance, bi->current_variance + i, nBits * sizeof(double));
        testCipher("bi", temp, 1, bk, key);*/
    }
//    cout << "here" << endl;
    bootsANDXOR_16_vector(r, ai, bi, nOutputs, nCoal * vLength, 1, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                          ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                          ks_cv_gpu_extendedPtr);

//    cout << "here" << endl;
    //test copy r0 to main result
    for (int i = 0; i < nCoal * vLength; ++i) {
        cudaMemcpy(mainResult->a + i * bitSize * n, r1->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(mainResult->b + i * bitSize, r1->b + i, sizeof(int));
        memcpy(mainResult->current_variance + i * bitSize, r1->current_variance + i, sizeof(double));
        /*
        cudaMemcpy(temp->a, r->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, r->b + i, sizeof(int));
        memcpy(temp->current_variance, r->current_variance + i, sizeof(double));
        testCipher("r0", temp, 1, bk, key);

        cudaMemcpy(temp->a, mainResult->a + i * bitSize * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, mainResult->b + i * bitSize, sizeof(int));
        memcpy(temp->current_variance, mainResult->current_variance + i * bitSize, sizeof(double));
        testCipher("mainResult0", temp, 1, bk, key);*/
    }
    cudaMemcpy(r1->a, r->a, nCoal * vLength * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(r1->b, r->b, nCoal * vLength * sizeof(int));
    memcpy(r1->current_variance, r->current_variance, nCoal * vLength * sizeof(double));

    /*
    cout << endl;
//    for (int i = 0; i < vLength; ++i) {
//        cudaMemcpy(temp->a, r1->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, r1->b + i, sizeof(int));
//        memcpy(temp->current_variance, r1->current_variance + i, sizeof(double));
//        testCipher("r1", temp, 1, bk, key);
//    }
    cout << endl;*/
    for (bI = 1; bI < bitSize; ++bI) {
        //get ai and bi
        for (int i = 0; i < nCoal * vLength; ++i) {
            /*
            cudaMemcpy(temp->a, r1->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, r1->b + i, sizeof(int));
            memcpy(temp->current_variance, r1->current_variance + i, sizeof(double));
            testCipher("r1", temp, 1, bk, key);*/

            cudaMemcpy(ai->a + i * n, a->a + i * bitSize * n + bI * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(ai->b + i, a->b + i * bitSize + bI, sizeof(int));
            memcpy(ai->current_variance + i, a->current_variance + i * bitSize + bI, sizeof(double));
            /*
            cudaMemcpy(temp->a, ai->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, ai->b + i, sizeof(int));
            memcpy(temp->current_variance, ai->current_variance + i, sizeof(double));
            testCipher("ai", temp, 1, bk, key);*/

            cudaMemcpy(bi->a + i * n, b->a + i * bitSize * n + bI * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(bi->b + i, b->b + i * bitSize + bI, sizeof(int));
            memcpy(bi->current_variance + i, b->current_variance + i * bitSize + bI, sizeof(double));
            /*
            cudaMemcpy(temp->a, bi->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, bi->b + i, sizeof(int));
            memcpy(temp->current_variance, bi->current_variance + i, sizeof(double));
            testCipher("bi", temp, 1, bk, key);
            cout << endl;*/
        }

        bootsXORXOR_16_vector(t, ai, r1, bi, r1, nCoal * vLength, nOutputs, 1, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended,
                       ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
        /*for (int i = 0; i < vLength; ++i) {
            cudaMemcpy(temp->a, t->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, t->b + i, sizeof(int));
            memcpy(temp->current_variance, t->current_variance + i, sizeof(double));
            testCipher("t0", temp, 1, bk, key);

            cudaMemcpy(temp->a, t->a + i * n + vLength * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, t->b + i + vLength, sizeof(int));
            memcpy(temp->current_variance, t->current_variance + i + vLength, sizeof(double));
            testCipher("t1", temp, 1, bk, key);
            cout << endl;
        }*/
        double sT = omp_get_wtime();
        bootsAND_16_vector(t, t, t1, 1, nCoal * vLength, 1, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                           ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                           ks_cv_gpu_extendedPtr);
        double eT = omp_get_wtime();
//        cout << "bits: " << nCoal * vLength << " time: " << eT - sT << endl;
//        bootsAND_16(t, t, t1, nCoal * vLength, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended, ks_b_gpu,
//                    ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

        /*for (int i = 0; i < vLength; ++i) {
            cudaMemcpy(temp->a, t->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, t->b + i, sizeof(int));
            memcpy(temp->current_variance, t->current_variance + i, sizeof(double));
            testCipher("t0", temp, 1, bk, key);
        }*/

        bootsXORXOR_16_vector(r, ai, t1, t, r1, nCoal * vLength, nOutputs, 1, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended,
                              ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

        for (int i = 0; i < nCoal * vLength; ++i) {
            //copy to the result
            cudaMemcpy(mainResult->a + i * bitSize * n + bI * n, r->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(mainResult->b + i * bitSize + bI, r->b + i, sizeof(int));
            memcpy(mainResult->current_variance + i * bitSize + bI, r->current_variance + i, sizeof(double));
            /*
            cudaMemcpy(temp->a, a->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, a->b + i, sizeof(int));
            memcpy(temp->current_variance, a->current_variance + i, sizeof(double));
            testCipher("ai", temp, 1, bk, key);

            cudaMemcpy(temp->a, t1->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, t1->b + i, sizeof(int));
            memcpy(temp->current_variance, t1->current_variance + i, sizeof(double));
            testCipher("t1", temp, 1, bk, key);

            cudaMemcpy(temp->a, r->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, r->b + i, sizeof(int));
            memcpy(temp->current_variance, r->current_variance + i, sizeof(double));
            testCipher("r0", temp, 1, bk, key);

            cudaMemcpy(temp->a, r1->a + i * n, n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, r1->b + i, sizeof(int));
            memcpy(temp->current_variance, r1->current_variance + i, sizeof(double));
            testCipher("r1", temp, 1, bk, key);
            cout << endl;*/
        }
    }
    cudaFree(r->a);
    cudaFree(t->a);
    cudaFree(ai->a);
    cudaFree(bi->a);
}

void BOOTS_Add_vector_bitwise(LweSample_16 *output, LweSample_16 **input, int vecSize, int bitSize, int nCoal,
                      const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                      cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                      int ***ks_b_gpu, double ***ks_cv_gpu, Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                      double *ks_cv_gpu_extendedPtr, TFheGateBootstrappingSecretKeySet *key) {

    int currentVecSize = vecSize / 2;
    cout << "currentVecSize: " << currentVecSize << endl;
    cout << "bitSize (inside): " << bitSize << endl;
    int n = bk->params->in_out_params->n;
    //combine the numbers
    LweSample_16 *data1 = convertBitToNumberZero_GPU(nCoal * bitSize * currentVecSize, bk);
    LweSample_16 *data2 = convertBitToNumberZero_GPU(nCoal * bitSize * currentVecSize, bk);
    LweSample_16 *dataRes = convertBitToNumberZero_GPU(nCoal * bitSize * currentVecSize, bk);
    LweSample_16 *testingData = convertBitToNumberZero_GPU(bitSize, bk);

    for (int i = 0; i < currentVecSize; ++i) {
        cudaMemcpy(data1->a + (i * nCoal * bitSize * n), input[i]->a, nCoal * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data1->b + (i * nCoal * bitSize), input[i]->b, nCoal * bitSize * sizeof(int));
        memcpy(data1->current_variance + (i * nCoal * bitSize), input[i]->current_variance, nCoal * bitSize * sizeof(double));
//        for (int j = 0; j < nCoal; ++j) {
//            cudaMemcpy(testingData->a, data1->a + (i * nCoal * bitSize * n) + j * bitSize * n, bitSize * n * sizeof(int),
//                       cudaMemcpyDeviceToDevice);
//            memcpy(testingData->b, data1->b + (i * bitSize * nCoal) + j * bitSize, bitSize * sizeof(int));
//            memcpy(testingData->current_variance, data1->current_variance + (i * nCoal * bitSize) + j * bitSize, bitSize * sizeof(double));
//            testCipher("d1", testingData, bitSize, bk, key);
//        }
//        cout << endl;

        int j = i + currentVecSize;
        cudaMemcpy(data2->a + (i * nCoal * bitSize * n), input[j]->a, nCoal * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data2->b + (i * nCoal * bitSize), input[j]->b, nCoal * bitSize * sizeof(int));
        memcpy(data2->current_variance + (i * nCoal * bitSize), input[j]->current_variance, nCoal * bitSize * sizeof(double));

//        for (int j = 0; j < nCoal; ++j) {
//            cudaMemcpy(testingData->a, data2->a + (i * nCoal * bitSize * n) + j * bitSize * n, bitSize * n * sizeof(int),
//                       cudaMemcpyDeviceToDevice);
//            memcpy(testingData->b, data2->b + (i * bitSize * nCoal) + j * bitSize, bitSize * sizeof(int));
//            memcpy(testingData->current_variance, data2->current_variance + (i * nCoal * bitSize) + j * bitSize, bitSize * sizeof(double));
//            testCipher("d2", testingData, bitSize, bk, key);
//        }
//        cout << endl;

        //remove input vector start
            cudaFree(input[i]->a);
            cudaFree(input[j]->a);
        //remove input vector end
    }
    double sT = omp_get_wtime();
//        cout << "I am here before: taskLevelParallelAdd_bitwise_vector" << endl;
    taskLevelParallelAdd_bitwise_vector(dataRes, data1, data2, currentVecSize, bitSize, nCoal, bk, cudaBkFFT,
                                        cudaBkFFTCoalesce,
                                        ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                        ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
    cout << "vector size: " << currentVecSize << " time: " << omp_get_wtime() - sT << endl;
//    for (int i = 0; i < currentVecSize; ++i) {
//        for (int j = 0; j < nCoal; ++j) {
//            cudaMemcpy(testingData->a, dataRes->a + (i * nCoal * bitSize * n) + j * bitSize * n, bitSize * n * sizeof(int),
//                       cudaMemcpyDeviceToDevice);
//            memcpy(testingData->b, dataRes->b + (i * bitSize * nCoal) + j * bitSize, bitSize * sizeof(int));
//            memcpy(testingData->current_variance, dataRes->current_variance + (i * nCoal * bitSize) + j * bitSize, bitSize * sizeof(double));
//            testCipher("out_vector", testingData, bitSize, bk, key);
//        }
//        cout << endl;
//    }
//    cout << endl;

//    cout << "XXX" << endl;
    for (currentVecSize = currentVecSize/2; currentVecSize > 0; currentVecSize/=2) {
//        cout << "currentVecSize: " << currentVecSize << endl;
        cudaMemcpy(data1->a, dataRes->a, currentVecSize * nCoal * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(data1->b, dataRes->b, currentVecSize * nCoal * bitSize * sizeof(int));
        memcpy(data1->current_variance, dataRes->current_variance, currentVecSize * nCoal * bitSize * sizeof(double));

        cudaMemcpy(data2->a, dataRes->a + currentVecSize * nCoal * bitSize * n, currentVecSize * nCoal * bitSize * n * sizeof(int),
                   cudaMemcpyDeviceToDevice);
        memcpy(data2->b, dataRes->b + currentVecSize * nCoal * bitSize, currentVecSize * nCoal * bitSize * sizeof(int));
        memcpy(data2->current_variance, dataRes->current_variance + currentVecSize * nCoal * bitSize,
               currentVecSize * nCoal * bitSize * sizeof(double));

        double sT = omp_get_wtime();
        taskLevelParallelAdd_bitwise_vector(dataRes, data1, data2, currentVecSize, bitSize, nCoal, bk, cudaBkFFT,
                                            cudaBkFFTCoalesce,
                                            ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                            ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        cout << "vector size: " << currentVecSize << " time: " << omp_get_wtime() - sT << endl;
//        for (int i = 0; i < currentVecSize; ++i) {
//            for (int j = 0; j < nCoal; ++j) {
//                cudaMemcpy(testingData->a, dataRes->a + (i * nCoal * bitSize * n) + j * bitSize * n, bitSize * n * sizeof(int),
//                           cudaMemcpyDeviceToDevice);
//                memcpy(testingData->b, dataRes->b + (i * bitSize * nCoal) + j * bitSize, bitSize * sizeof(int));
//                memcpy(testingData->current_variance, dataRes->current_variance + (i * nCoal * bitSize) + j * bitSize, bitSize * sizeof(double));
//                testCipher("out_vector", testingData, bitSize, bk, key);
//            }
//            cout << endl;
//        }
//        cout << endl;
    }
    cudaMemcpy(output->a, dataRes->a, nCoal * bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(output->b, dataRes->b, nCoal * bitSize * sizeof(int));
    memcpy(output->current_variance, dataRes->current_variance, nCoal * bitSize * sizeof(double));

    cudaFree(data1->a);
    cudaFree(data2->a);
    cudaFree(dataRes->a);
}


void multiplyLweSamples(LweSample_16 *cudaRes, LweSample_16 *a, LweSample_16 *b, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk,
                        cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ****cudaBkFFT_x2,
                        cufftDoubleComplex ***cudaBkFFTCoalesce, cufftDoubleComplex ***cudaBkFFTCoalesce_x2,
                        Torus32 ****ks_a_gpu,
                        Torus32 ****ks_a_gpu_extended, Torus32 ****ks_a_gpu_extended_x2,
                        int ***ks_b_gpu, double ***ks_cv_gpu,
                        Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr,
                        TFheGateBootstrappingSecretKeySet *key) {
    //divide the numbers into bitsize numbers
    int nInputBits = bitSize, nOutputBits = 2 * bitSize;

    LweSample_16 **andRes = new LweSample_16 *[nInputBits];
    LweSample_16 **andResShifted = new LweSample_16 *[nInputBits];//nInputBits


//    double sT  = omp_get_wtime();
    for (int bIndex = 0; bIndex < nInputBits; ++bIndex) {//nInputBits
        andRes[bIndex] = convertBitToNumberZero_GPU(nOutputBits, bk);
        cudaMemset(andRes[bIndex]->a, 0, nOutputBits * bk->params->in_out_params->n * sizeof(int));
        andResShifted[bIndex] = convertBitToNumberZero_GPU(nOutputBits, bk);

        bootsAND_MULT(andRes[bIndex], a, b, nInputBits, nInputBits, bIndex, bk, cudaBkFFT, cudaBkFFTCoalesce,
                      ks_a_gpu,
                      ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                      ks_cv_gpu_extendedPtr);

        leftShiftCuda_16(andResShifted[bIndex], andRes[bIndex], nOutputBits, bIndex, bk);
//        testCipher("and ", andRes[bIndex], nOutputBits, bk, key);
//        testCipher("and Shifted ", andResShifted[bIndex], nOutputBits, bk, key);
    }


    //this is new implementation using bitwise addition
            int nCoal = 1;
    BOOTS_Add_vector_bitwise(cudaRes, andResShifted, nInputBits, nOutputBits, nCoal, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                     ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
//            testCipher("cudres", cudaRes, nOutputBits, bk, key);

    //this is using 16 bit shifting addition
//    BOOTS_Add_vector(cudaRes, andResShifted, nInputBits, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
//                     ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
    for (int bIndex = 0; bIndex < nInputBits; ++bIndex) {
        cudaFree(andRes[bIndex]->a);
        cudaFree(andResShifted[bIndex]->a);
    }
    free(andRes);
    free(andResShifted);
}


void test_vectorAddition(LweSample *ca, LweSample *cb, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk,
                        TFheGateBootstrappingSecretKeySet *key) {

    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    const int n = 500;

    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);
    LweSample_16 *result = convertBitToNumberZero_GPU(bitSize, bk);//2*inputBitSize

    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);


    //vector addition
    int vecLen = 4;
    LweSample_16 **input = new LweSample_16*[vecLen];
    for (int i = 0; i < vecLen/2; ++i) {
        input[i] = convertBitToNumberZero_GPU(bitSize, bk);
        cudaMemcpy(input[i]->a, a->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(input[i]->b, a->b, bitSize * sizeof(int));


        int j = i + vecLen/2;
        input[j] = convertBitToNumberZero_GPU(bitSize, bk);
        cudaMemcpy(input[j]->a, b->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(input[j]->b, b->b, bitSize * sizeof(int));

    }

    BOOTS_Add_vector_bitwise(result, input, vecLen, bitSize, 1, bk, NULL, cudaBkFFTCoalesce, NULL,
                             NULL, NULL, NULL, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                             ks_cv_gpu_extendedPtr, key);

}

void multiplyLweSamples_test(LweSample *ca, LweSample *cb, int bitSize,
                             const TFheGateBootstrappingCloudKeySet *bk,
                             TFheGateBootstrappingSecretKeySet *key) {
    int inputBitSize = bitSize, outputBitSize = bitSize * 2;

    cufftDoubleComplex ****cudaBkFFT = NULL;//sendBootstrappingKeyToGPU(inputBitSize, bk);
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    Torus32 ****ks_a_gpu_extended = NULL;//sendKeySwitchKeyToGPU_extended(inputBitSize, bk);

    cufftDoubleComplex ****cudaBkFFT_x2 = NULL;//sendBootstrappingKeyToGPU(outputBitSize, bk);
    cufftDoubleComplex ***cudaBkFFTCoalesce_x2 = NULL;// sendBootstrappingKeyToGPUCoalesce(outputBitSize, bk);
    Torus32 ****ks_a_gpu_extended_x2 = NULL;//sendKeySwitchKeyToGPU_extended(outputBitSize, bk);

    int ***ks_b_gpu = NULL;//sendKeySwitchBtoGPU(bk);
    double ***ks_cv_gpu = NULL;//sendKeySwitchCVtoGPU(bk);
    const LweParams *in_out_params = bk->params->in_out_params;

    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

    LweSample_16 *a = convertBitToNumber(ca, inputBitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, inputBitSize, bk);
    LweSample_16 *result = convertBitToNumberZero_GPU(outputBitSize, bk);//2*inputBitSize

    //send a, b, result to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), inputBitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(a->a, temp, inputBitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), inputBitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(b->a, temp, inputBitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);


    int startTime = omp_get_wtime();
    multiplyLweSamples(result, a, b, bitSize, bk, cudaBkFFT, cudaBkFFT_x2, cudaBkFFTCoalesce, cudaBkFFTCoalesce_x2,
                       NULL, ks_a_gpu_extended, ks_a_gpu_extended_x2, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                       ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);//ks_a_gpu
    cout << "Time Taken To Multiply: " << omp_get_wtime() - startTime << endl;
    testCipher("multiplication Result", result, outputBitSize, bk, key);

    cout << "I am here" << endl;
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

//__global__ void expandNbitTo2Nbit_cuda(int *out, int *in, int nInputBitSize, int nOutputBitSize, int n, int length) {
//    int id = blockIdx.x * blockDim.x + threadIdx.x;
//    if (id < length) {
//        out[id] = 0;
//        int bIndex = (id / n) % nOutputBitSize;
//        if(bIndex < nInputBitSize) {
//            int smallIndex = bIndex * n;
//        }
//
//    }
//}

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

void concurrentMultiplication(LweSample_16 *result, LweSample_16 **a, LweSample_16 **b, int nConMul, int bitSize,
                              const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                              cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu,
                              Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                              Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                              double *ks_cv_gpu_extendedPtr, TFheGateBootstrappingSecretKeySet *key) {
    cout << "***********" << endl;
    int nOutputBits = bitSize * 2;
    int n = bk->params->in_out_params->n;
    LweSample_16 *temp = convertBitToNumberZero_GPU(bitSize, bk);
    LweSample_16 *temp2 = convertBitToNumberZero_GPU(nOutputBits, bk);
    LweSample_16 **andRes = new LweSample_16 *[bitSize];
    LweSample_16 **andResEx = new LweSample_16 *[bitSize];
    LweSample_16 **andResShifted = new LweSample_16 *[bitSize];
    /*
    for (int i = 0; i < nConMul; ++i) {
        cudaMemcpy(temp->a, a[i]->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, a[i]->b, bitSize * sizeof(int));
        memcpy(temp->current_variance, a[i]->current_variance, bitSize * sizeof(double));
        testCipher("a0", temp, bitSize, bk, key);

        cudaMemcpy(temp->a, b[i]->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, b[i]->b, bitSize * sizeof(int));
        memcpy(temp->current_variance, b[i]->current_variance, bitSize * sizeof(double));
        testCipher("b0", temp, bitSize, bk, key);
    }
    cout << endl;*/

    LweSample_16 *tempAndDouble = convertBitToNumberZero_GPU(nOutputBits * nConMul, bk);
    cudaMemset(tempAndDouble->a, 0, nOutputBits * nConMul * n * sizeof(int));
    for (int bIndex = 0; bIndex < bitSize; ++bIndex) {//bitSize
        andRes[bIndex] = convertBitToNumberZero_GPU(bitSize * nConMul, bk);
        andResEx[bIndex] = convertBitToNumberZero_GPU(nOutputBits * nConMul, bk);
        cudaMemset(andResEx[bIndex]->a, 0, nOutputBits * nConMul * n * sizeof(int));
        andResShifted[bIndex] = convertBitToNumberZero_GPU(nOutputBits * nConMul, bk);
//        cout << "concurrentMultiplication: bitSize: " << bitSize << endl;
        bootsAND_MULT_con(andRes[bIndex], a, b, nConMul, bitSize, bitSize, bIndex, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                          ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                          ks_cv_gpu_extendedPtr);

        expandNbitTo2Nbit(andResEx[bIndex], andRes[bIndex], bitSize, nOutputBits, nConMul, bk);
        //make double of input bits
        //left shift vector
        leftShiftCuda_16_vector(andResShifted[bIndex], andResEx[bIndex], nConMul, nOutputBits, bIndex, bk);

        cudaFree(andRes[bIndex]->a);
        cudaFree(andResEx[bIndex]->a);

//        cout << "----" << "bIndex: " << bIndex << "----" << endl;
//        for (int i = 0; i < nConMul; ++i) {
//            cudaMemcpy(temp->a, andRes[bIndex]->a + i * bitSize * n, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//            memcpy(temp->b, andRes[bIndex]->b + i * bitSize, bitSize * sizeof(int));
//            memcpy(temp->current_variance, andRes[bIndex]->current_variance + i * bitSize, bitSize * sizeof(double));
//            testCipher("res0", temp, bitSize, bk, key);
//
//            cudaMemcpy(temp2->a, andResEx[bIndex]->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
//            memcpy(temp2->b, andResEx[bIndex]->b + i * nOutputBits, nOutputBits * sizeof(int));
//            memcpy(temp2->current_variance, andResEx[bIndex]->current_variance + i * nOutputBits, nOutputBits* sizeof(double));
//            testCipher("res0Ex", temp2, nOutputBits, bk, key);
//
//            cudaMemcpy(temp2->a, andResShifted[bIndex]->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
//            memcpy(temp2->b, andResShifted[bIndex]->b + i * nOutputBits, nOutputBits * sizeof(int));
//            memcpy(temp2->current_variance, andResShifted[bIndex]->current_variance + i * nOutputBits, nOutputBits * sizeof(double));
//            testCipher("resShifted", temp2, nOutputBits, bk, key);
//            cout << endl;
//        }
//        cout << endl;
    }
    cout << "bitSize: " << bitSize << endl;
    cout << "nOutputBits: " << nOutputBits << endl;
    cout << "reduction start" << endl;
    cout << "nConMul: " << nConMul << endl;
//    result = convertBitToNumberZero_GPU(nConMul * nOutputBits, bk);
    BOOTS_Add_vector_bitwise(result, andResShifted, bitSize, nOutputBits, nConMul, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                             ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
    cout << "Printing Result: " << endl;
    cout << "nOutputBits: " << nOutputBits << endl;
    for (int i = 0; i < nConMul; ++i) {
        cudaMemcpy(temp2->a, result->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp2->b, result->b + i * nOutputBits, nOutputBits * sizeof(int));
        memcpy(temp2->current_variance, result->current_variance + i * nOutputBits, nOutputBits * sizeof(double));
        testCipher("in conMul res0", temp2, nOutputBits, bk, key);
    }

    cout << endl;

}

void karatMasterSuba(LweSample_16 *cudaRes, LweSample_16 *a, LweSample_16 *b, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk,
                        cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ****cudaBkFFT_x2,
                        cufftDoubleComplex ***cudaBkFFTCoalesce, cufftDoubleComplex ***cudaBkFFTCoalesce_x2,
                        Torus32 ****ks_a_gpu,
                        Torus32 ****ks_a_gpu_extended, Torus32 ****ks_a_gpu_extended_x2,
                        int ***ks_b_gpu, double ***ks_cv_gpu,
                        Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr,
                        TFheGateBootstrappingSecretKeySet *key) {

    //make sure the bitSize is 32
//    assert(bitSize == BIT_32);
    int nOutputBits = bitSize * 2;
    int f_bitSize = bitSize, outputBitSize = bitSize * 2;
    int h_bitSize = bitSize/2;
    const LweParams *in_out_params = bk->params->in_out_params;
    const int n = in_out_params->n;

    //divide the numbers into half bitsize numbers

    LweSample_16 *Xl = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *Xr = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *Yl = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *Yr = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *XYl = convertBitToNumberZero_GPU(f_bitSize, bk);
    LweSample_16 *XYr = convertBitToNumberZero_GPU(f_bitSize, bk);

    LweSample_16 *P1 = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *P2 = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *P3_1 = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *P3_2 = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *P3 = convertBitToNumberZero_GPU(f_bitSize, bk);
    LweSample_16 *E_ADD = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *E_1sComplement = convertBitToNumberZero_GPU(f_bitSize, bk);
    LweSample_16 *E_MUL = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *E_ONE = convertBitToNumberZero_GPU(f_bitSize, bk);
    LweSample_16 *E = convertBitToNumberZero_GPU(f_bitSize, bk);
    LweSample_16 *E_padded = convertBitToNumberZero_GPU(2 * f_bitSize, bk);
    LweSample_16 *P2padded = convertBitToNumberZero_GPU(f_bitSize, bk);

    LweSample_16 *temp = convertBitToNumberZero_GPU(h_bitSize, bk);
    LweSample_16 *temp2 = convertBitToNumberZero_GPU(f_bitSize, bk);

    int nConMul = 3;
    LweSample_16 *conMulRes = convertBitToNumberZero_GPU(f_bitSize * nConMul, bk);
    LweSample_16 **input1 = new LweSample_16*[nConMul];
    LweSample_16 **input2 = new LweSample_16*[nConMul];



    cudaMemset(E_padded->a, 0, f_bitSize * n * sizeof(int));
    cudaMemset(P2padded->a, 0, f_bitSize * n * sizeof(int));

    Torus32 MU = modSwitchToTorus32(1, 8);
    E_ONE->b[0] = MU;


    cudaFree(Xl->a);
    cudaFree(P3_1->a);
    cudaFree(P3_2->a);
    free(Xl->b);
    free(Xl->current_variance);
    cudaFree(Yl->a);
    free(Yl->b);
    free(Yl->current_variance);

    Xl = a;
    Yl = b;

    Xr->a = a->a + h_bitSize * n;
    Xr->b = a->b + h_bitSize;
    Xr->current_variance = a->current_variance + h_bitSize;

    Yr->a = b->a + h_bitSize * n;
    Yr->b = b->b + h_bitSize;
    Yr->current_variance = b->current_variance + h_bitSize;

    cudaMemcpy(XYl->a, Xl->a, h_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(XYl->b, Xl->b, h_bitSize * sizeof(int));
    memcpy(XYl->current_variance, Xl->current_variance, h_bitSize * sizeof(double));
    cudaMemcpy(XYl->a + h_bitSize * n, Yl->a, h_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(XYl->b + h_bitSize, Yl->b, h_bitSize * sizeof(int));
    memcpy(XYl->current_variance + h_bitSize, Yl->current_variance, h_bitSize * sizeof(double));

    cudaMemcpy(XYr->a, Xr->a, h_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(XYr->b, Xr->b, h_bitSize * sizeof(int));
    memcpy(XYr->current_variance, Xr->current_variance, h_bitSize * sizeof(double));
    cudaMemcpy(XYr->a + h_bitSize * n, Yr->a, h_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(XYr->b + h_bitSize, Yr->b, h_bitSize * sizeof(int));
    memcpy(XYr->current_variance + h_bitSize, Yr->current_variance, h_bitSize * sizeof(double));

    taskLevelParallelAdd_bitwise_vector(P3, XYl, XYr, 2, h_bitSize, 1, bk, cudaBkFFT,
                                        cudaBkFFTCoalesce,
                                        ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                        ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
    P3_1 = P3;
    P3_2->a = P3->a + h_bitSize * n;
    P3_2->b = P3->b + h_bitSize;
    P3_2->current_variance = P3->current_variance + h_bitSize;

    input1[0] = Xl;
    input1[1] = Xr;
    input1[2] = P3_1;
    input2[0] = Yl;
    input2[1] = Yr;
    input2[2] = P3_2;

    concurrentMultiplication(conMulRes, input1, input2, nConMul, h_bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                             ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                             ks_cv_gpu_extendedPtr, key);

    P1->a = conMulRes->a;
    P1->b = conMulRes->b;
    P1->current_variance = conMulRes->current_variance;

    P2->a = conMulRes->a + f_bitSize * n;
    P2->b = conMulRes->b + f_bitSize;
    P2->current_variance = conMulRes->current_variance + f_bitSize;

    E_MUL->a = conMulRes->a + 2 * f_bitSize * n;
    E_MUL->b = conMulRes->b + 2 * f_bitSize;
    E_MUL->current_variance = conMulRes->current_variance + 2 * f_bitSize;

//    cudaMemcpy(temp2->a, P1->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp2->b, P1->b, f_bitSize * sizeof(int));
//        memcpy(temp2->current_variance, P1->current_variance, f_bitSize * sizeof(double));
//        testCipher("P1", temp2, f_bitSize, bk, key);

//        cudaMemcpy(temp2->a, P2->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, P2->b, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, P2->current_variance, f_bitSize * sizeof(double));
//    testCipher("P2", temp2, f_bitSize, bk, key);

//    cudaMemcpy(temp2->a, E_MUL->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, E_MUL->b, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, E_MUL->current_variance, f_bitSize * sizeof(double));
//    testCipher("EMUL", temp2, f_bitSize, bk, key);



    LweSample_16 *PE = convertBitToNumberZero_GPU(2 * f_bitSize, bk);
    LweSample_16 *PE_1 = convertBitToNumberZero_GPU(2 * f_bitSize, bk);
    LweSample_16 *PE_2 = convertBitToNumberZero_GPU(2 * f_bitSize, bk);

    cudaMemcpy(PE_1->a, P1->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(PE_1->b, P1->b, f_bitSize * sizeof(int));
    memcpy(PE_1->current_variance, P1->current_variance, f_bitSize * sizeof(double));

//    int i =0;
//    cudaMemcpy(temp2->a, PE_1->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, PE_1->b + i * f_bitSize, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, PE_1->current_variance + i * f_bitSize, f_bitSize * sizeof(double));
//    testCipher("PE_1.0", temp2, f_bitSize, bk, key);



    cudaMemcpy(PE_1->a + f_bitSize * n, E_MUL->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(PE_1->b + f_bitSize, E_MUL->b, f_bitSize * sizeof(int));
    memcpy(PE_1->current_variance + f_bitSize, E_MUL->current_variance, f_bitSize * sizeof(double));

//    i =1;
//    cudaMemcpy(temp2->a, PE_1->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, PE_1->b + i * f_bitSize, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, PE_1->current_variance + i * f_bitSize, f_bitSize * sizeof(double));
//    testCipher("PE_1.1", temp2, f_bitSize, bk, key);

    cudaMemcpy(PE_2->a, P2->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(PE_2->b, P2->b, f_bitSize * sizeof(int));
    memcpy(PE_2->current_variance, P2->current_variance, f_bitSize * sizeof(double));

//    i =0;
//    cudaMemcpy(temp2->a, PE_2->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, PE_2->b + i * f_bitSize, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, PE_2->current_variance + i * f_bitSize, f_bitSize * sizeof(double));
//    testCipher("PE_2.0", temp2, f_bitSize, bk, key);

    cudaMemcpy(PE_2->a + f_bitSize * n, E_ONE->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(PE_2->b + f_bitSize, E_ONE->b, f_bitSize * sizeof(int));
    memcpy(PE_2->current_variance + f_bitSize, E_ONE->current_variance, f_bitSize * sizeof(double));

//    i =1;
//    cudaMemcpy(temp2->a, PE_2->a + i * f_bitSize * n, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, PE_2->b + i * f_bitSize, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, PE_2->current_variance + i * f_bitSize, f_bitSize * sizeof(double));
//    testCipher("PE_2.1", temp2, f_bitSize, bk, key);
//    cout << "------------------1" << endl;

    taskLevelParallelAdd_bitwise_vector(PE, PE_1, PE_2, 2, f_bitSize, 1, bk, cudaBkFFT,
                                        cudaBkFFTCoalesce,
                                        ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                        ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
    E_ADD->a = PE->a;
    E_ADD->b = PE->b;
    E_ADD->current_variance = PE->current_variance;
    E_MUL->a = PE->a + f_bitSize * n;
    E_MUL->b = PE->b + f_bitSize;
    E_MUL->current_variance = PE->current_variance + f_bitSize;
//    cout << "------------------2" << endl;


//    cudaMemcpy(temp2->a, E_ADD->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, E_ADD->b, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, E_ADD->current_variance, f_bitSize * sizeof(double));
//    testCipher("AC + BD", temp2, f_bitSize, bk, key);
//
//    cudaMemcpy(temp2->a, E_MUL->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
//    memcpy(temp2->b, E_MUL->b, f_bitSize * sizeof(int));
//    memcpy(temp2->current_variance, E_MUL->current_variance, f_bitSize * sizeof(double));
//    testCipher("E_MUL + 1", temp2, f_bitSize, bk, key);


    bootsNOT_16(E_1sComplement, E_ADD, f_bitSize, n);
//    testCipher("(AC + BD)", E_1sComplement, f_bitSize, bk, key);
//
//    cout << "------------------3" << endl;
//    cout << "here" << endl;

    taskLevelParallelAdd_bitwise(E, E_MUL, E_1sComplement, 1, f_bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce,
                                 NULL, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                 ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
//    taskLevelParallelAdd_bitwise_vector(E, E_MUL, E_1sComplement, 1, f_bitSize, 1, bk, cudaBkFFT,
//                                        cudaBkFFTCoalesce,
//                                        ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
//                                        ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);

//    cout << "------------------4" << endl;
//    testCipher("E", E, f_bitSize, bk, key);

//    cudaMemset(E_padded->a, 0, nOutputBits * n * sizeof(int));

    cudaMemcpy(E_padded->a + f_bitSize * n, P2->a, f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(E_padded->b + f_bitSize, P2->b, f_bitSize * sizeof(int));
    memcpy(E_padded->current_variance + f_bitSize, P2->current_variance, f_bitSize * sizeof(double));
//    testCipher("E_padded (E_padded)", E_padded, f_bitSize, bk, key);

    cudaMemcpy(E_padded->a + h_bitSize * n, E->a, h_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(E_padded->b + h_bitSize, E->b, h_bitSize * sizeof(int));
    memcpy(E_padded->current_variance + h_bitSize, E->current_variance, h_bitSize * sizeof(double));
//    testCipher("E_padded (E_padded)", E_padded, f_bitSize, bk, key);

    cudaMemcpy(E_padded->a, P1->a, h_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(E_padded->b, P1->b, h_bitSize * sizeof(int));
    memcpy(E_padded->current_variance, P1->current_variance, h_bitSize * sizeof(double));
    testCipher("Final result (E_padded)", E_padded, nOutputBits, bk, key);

    cudaMemcpy(cudaRes->a, E_padded->a, 2 * f_bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    memcpy(cudaRes->b, E_padded->b, 2 * f_bitSize * sizeof(int));
    memcpy(cudaRes->current_variance, E_padded->current_variance, 2 * f_bitSize * sizeof(double));
    testCipher("Final result (cudaRes)", cudaRes, nOutputBits, bk, key);


}

void karatSuba_test(LweSample *ca, LweSample *cb, int bitSize,
                             const TFheGateBootstrappingCloudKeySet *bk,
                             TFheGateBootstrappingSecretKeySet *key) {
//    assert(bitSize == BIT_32);

    int nOutputBits = bitSize * 2;

    cufftDoubleComplex ****cudaBkFFT = NULL;//sendBootstrappingKeyToGPU(1, bk);
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    Torus32 ****ks_a_gpu_extended = NULL;//sendKeySwitchKeyToGPU_extended(inputBitSize, bk);

    cufftDoubleComplex ****cudaBkFFT_x2 = NULL;//sendBootstrappingKeyToGPU(outputBitSize, bk);
    cufftDoubleComplex ***cudaBkFFTCoalesce_x2 = NULL;// sendBootstrappingKeyToGPUCoalesce(outputBitSize, bk);
    Torus32 ****ks_a_gpu_extended_x2 = NULL;//sendKeySwitchKeyToGPU_extended(outputBitSize, bk);

    int ***ks_b_gpu = NULL;//sendKeySwitchBtoGPU(bk);
    double ***ks_cv_gpu = NULL;//sendKeySwitchCVtoGPU(bk);
    const LweParams *in_out_params = bk->params->in_out_params;

    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

//    assert(bitSize == BIT_32);
    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);
    LweSample_16 *result = convertBitToNumberZero_GPU(nOutputBits, bk);//2*inputBitSize

    //send a, b, result to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    cout << "KARAT " << bitSize << endl;
    int startTime = omp_get_wtime();
    karatMasterSuba(result, a, b, bitSize, bk, cudaBkFFT, cudaBkFFT_x2, cudaBkFFTCoalesce, cudaBkFFTCoalesce_x2,
                    NULL, ks_a_gpu_extended, ks_a_gpu_extended_x2, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                    ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);//ks_a_gpu
    cout << "Time Taken To Multiply: " << omp_get_wtime() - startTime << endl;
//    testCipher("A", a, bitSize, bk, key);
//    testCipher("B", b, bitSize, bk, key);
    testCipher("multiplication Result", result, nOutputBits, bk, key);

//    cout << "I am here" << endl;
}



void vectorMultiplicationTest(LweSample *ca, LweSample *cb, int bitSize, int nConMul,
                    const TFheGateBootstrappingCloudKeySet *bk,
                    TFheGateBootstrappingSecretKeySet *key) {

    const int n = bk->params->in_out_params->n;
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);
    int nOutputBitSize = bitSize * 2;

    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);
    LweSample_16 *result = convertBitToNumberZero_GPU(nOutputBitSize * nConMul, bk);

    //send a, b, result to cuda
    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    //prepare vector
    cout << "Preparing vector" << endl;
    LweSample_16 **arrayA = new LweSample_16*[nConMul];
    LweSample_16 **arrayB = new LweSample_16*[nConMul];
    for (int i = 0; i < nConMul; ++i) {
        arrayA[i] = convertBitToNumberZero_GPU(bitSize, bk);
        arrayB[i] = convertBitToNumberZero_GPU(bitSize, bk);

        cudaMemcpy(arrayA[i]->a, a->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(arrayA[i]->b, a->b, bitSize * sizeof(int));

        cudaMemcpy(arrayB[i]->a, b->a, bitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(arrayB[i]->b, b->b, bitSize * sizeof(int));
    }

    cout << "Vector Multiplication" << bitSize << endl;
    double sT = omp_get_wtime();
    concurrentMultiplication(result, arrayA, arrayB, nConMul, bitSize, bk, NULL, cudaBkFFTCoalesce, NULL,
                             NULL, NULL, NULL, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                             ks_cv_gpu_extendedPtr, key);
    cout << "#Concurrent Multiplication: " << nConMul << " Time taken: " << omp_get_wtime() - sT << endl;
    LweSample_16 *tempRes = convertBitToNumberZero_GPU(nOutputBitSize, bk);
    for (int i = 0; i < nConMul; ++i) {
        cudaMemcpy(tempRes->a, result->a + i * nOutputBitSize * n, nOutputBitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(tempRes->b, result->b + i * nOutputBitSize, nOutputBitSize * sizeof(int));
        memcpy(tempRes->current_variance, result->current_variance + i * nOutputBitSize, nOutputBitSize * sizeof(double));
        testCipher("result", tempRes, nOutputBitSize, bk, key);
    }/**/
////    testCipher("A", a, bitSize, bk, key);
////    testCipher("B", b, bitSize, bk, key);
//    testCipher("multiplication Result", result, bitSize, bk, key);
//
//    cout << "I am here" << endl;
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


void testMatrixOperation(LweSample *ca, LweSample *cb, int bitSize,
                                const TFheGateBootstrappingCloudKeySet *bk,
                                TFheGateBootstrappingSecretKeySet *key) {
    const int n = bk->bk->in_out_params->n;
    cufftDoubleComplex ****cudaBkFFT = NULL;
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    Torus32 ****ks_a_gpu_extended = NULL;

    int ***ks_b_gpu = NULL;
    double ***ks_cv_gpu = NULL;
    const LweParams *in_out_params = bk->params->in_out_params;
    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);

    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

    int row = 2, col = 2;
    //construct m1 m2
    LweSample_16 ***m1 = new LweSample_16**[row];
    LweSample_16 ***m2 = new LweSample_16**[row];
    for (int i = 0; i < row ; ++i) {
        m1[i] = new LweSample_16*[col];
        m2[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            m1[i][j] = convertBitToNumber(ca, bitSize, bk);
            m2[i][j] = convertBitToNumber(cb, bitSize, bk);

            int *temp;
            temp = m1[i][j]->a;
            cudaMalloc(&(m1[i][j]->a), bitSize * n * sizeof(int));
            cudaMemcpy(m1[i][j]->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
            free(temp);

            temp = m2[i][j]->a;
            cudaMalloc(&(m2[i][j]->a), bitSize * n * sizeof(int));
            cudaMemcpy(m2[i][j]->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
            free(temp);

//            int x = i * col + j;
//            if (x > 0 && x < bitSize) {
//                leftShiftCuda_16_vector(m1[i][j], m1[0][0], 1, bitSize, x, bk);
//                leftShiftCuda_16_vector(m2[i][j], m2[0][0], 1, bitSize, x, bk);
//            }

            cout << "i: " << i << " j: " << j << endl;
            testCipher("m1", m1[i][j], bitSize, bk, key);
            testCipher("m2", m2[i][j], bitSize, bk, key);
        }
    }

    //send data to GPU
//    LweSample_16 ***d_m1 = matrixToDevice(m1, row, col, bitSize, bk);
//    LweSample_16 ***d_m2 = matrixToDevice(m2, row, col, bitSize, bk);
    //construct vector from matrix
    LweSample_16 *v1 = matrixToVector(m1, row, col, bitSize, bk);
    LweSample_16 *v2 = matrixToVector(m2, row, col, bitSize, bk);

    LweSample_16 *result = convertBitToNumberZero_GPU(row * col * bitSize, bk);
    //addition function
    taskLevelParallelAdd_bitwise_vector(result, v1, v2, row * col, bitSize, 1, bk, cudaBkFFT, cudaBkFFTCoalesce,
                                        NULL, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                        ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);

    //get result
    LweSample_16 ***addResult = vectorToMatrix(result, row, col, bitSize, bk);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            testCipher("res", addResult[i][j], bitSize, bk, key);
        }
    }
    //matrix addition complete

    // matrix multiplication start
    //prepare left matrix
    LweSample_16 **leftMatVec = matMul_prepareLeftMat(m1, row, col, col, bitSize, bk);
    //test
    int leftMatLen = row * col * col;
    for (int i = 0; i < leftMatLen; ++i) {
        testCipher("left mat", leftMatVec[i], bitSize, bk, key);
    }
    //prepare right matrix
    LweSample_16 **rightMatVec = matMul_prepareRightMat(m2, row, col, row, bitSize, bk);

    //test
    int rightMatLen = row * col * row;
    for (int i = 0; i < rightMatLen; ++i) {
        testCipher("right mat", rightMatVec[i], bitSize, bk, key);
    }
    int nConMul = row * col * row;
    int nOutputBits = bitSize * 2;
    LweSample_16 *matMulRes = convertBitToNumberZero_GPU(nConMul * nOutputBits, bk);;

    concurrentMultiplication(matMulRes, leftMatVec, rightMatVec, nConMul, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce,
                             NULL, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                             ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
    //prepare vector for multiplication
    LweSample_16 *temp = convertBitToNumberZero_GPU(nOutputBits, bk);
    LweSample_16 *addV1 = convertBitToNumberZero_GPU(rightMatLen/2 * nOutputBits, bk);
    LweSample_16 *addV2 = convertBitToNumberZero_GPU(rightMatLen/2 * nOutputBits, bk);
    for (int i = 0; i < rightMatLen; ++i) {
        int j = i / 2;
        if(i % 2 == 0) {
            cudaMemcpy(addV1->a + j * nOutputBits * n, matMulRes->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(addV1->b + j * nOutputBits, matMulRes->b + i * nOutputBits, nOutputBits * sizeof(int));
            memcpy(addV1->current_variance + j * nOutputBits, matMulRes->current_variance + i * nOutputBits, nOutputBits * sizeof(double));

            cudaMemcpy(temp->a, addV1->a + j * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, addV1->b + j * nOutputBits, nOutputBits * sizeof(int));
            memcpy(temp->current_variance, addV1->current_variance + j * nOutputBits, nOutputBits * sizeof(double));
            testCipher("0 ", temp, nOutputBits, bk, key);
        } else {
            cudaMemcpy(addV2->a + j * nOutputBits * n, matMulRes->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(addV2->b + j * nOutputBits, matMulRes->b + i * nOutputBits, nOutputBits * sizeof(int));
            memcpy(addV2->current_variance + j * nOutputBits, matMulRes->current_variance + i * nOutputBits, nOutputBits * sizeof(double));

            cudaMemcpy(temp->a, addV2->a + j * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, addV2->b + j * nOutputBits, nOutputBits * sizeof(int));
            memcpy(temp->current_variance, addV2->current_variance + j * nOutputBits, nOutputBits * sizeof(double));
            testCipher("1 ", temp, nOutputBits, bk, key);
        }
    }

    taskLevelParallelAdd_bitwise_vector(matMulRes, addV1, addV2, 1, nOutputBits, rightMatLen/2, bk, cudaBkFFT,
                                        cudaBkFFTCoalesce,
                                        NULL, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                        ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);

    for (rightMatLen /= 2; rightMatLen > row * col; rightMatLen /= 2) {
        cout << "need to do more: rightMatLen: " << rightMatLen << endl;
        for (int i = 0; i < rightMatLen; ++i) {
            int j = i / 2;
            if(i % 2 == 0) {
                cudaMemcpy(addV1->a + j * nOutputBits * n, matMulRes->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
                memcpy(addV1->b + j * nOutputBits, matMulRes->b + i * nOutputBits, nOutputBits * sizeof(int));
                memcpy(addV1->current_variance + j * nOutputBits, matMulRes->current_variance + i * nOutputBits, nOutputBits * sizeof(double));

                cudaMemcpy(temp->a, addV1->a + j * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
                memcpy(temp->b, addV1->b + j * nOutputBits, nOutputBits * sizeof(int));
                memcpy(temp->current_variance, addV1->current_variance + j * nOutputBits, nOutputBits * sizeof(double));
                testCipher("0 ", temp, nOutputBits, bk, key);
            } else {
                cudaMemcpy(addV2->a + j * nOutputBits * n, matMulRes->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
                memcpy(addV2->b + j * nOutputBits, matMulRes->b + i * nOutputBits, nOutputBits * sizeof(int));
                memcpy(addV2->current_variance + j * nOutputBits, matMulRes->current_variance + i * nOutputBits, nOutputBits * sizeof(double));

                cudaMemcpy(temp->a, addV2->a + j * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
                memcpy(temp->b, addV2->b + j * nOutputBits, nOutputBits * sizeof(int));
                memcpy(temp->current_variance, addV2->current_variance + j * nOutputBits, nOutputBits * sizeof(double));
                testCipher("1 ", temp, nOutputBits, bk, key);
            }
        }

        taskLevelParallelAdd_bitwise_vector(matMulRes, addV1, addV2, 1, nOutputBits, rightMatLen/2, bk, cudaBkFFT,
                                            cudaBkFFTCoalesce,
                                            NULL, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr,
                                            ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        cout << endl;
    }
    for (int i = 0; i < row * col; ++i) {
        cudaMemcpy(temp->a, matMulRes->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(temp->b, matMulRes->b + i * nOutputBits, nOutputBits * sizeof(int));
        memcpy(temp->current_variance, matMulRes->current_variance + i * nOutputBits, nOutputBits * sizeof(double));
        testCipher("mat mul test int ", temp, nOutputBits, bk, key);

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


void cannonsAlgoComp(LweSample_16 ***res, LweSample_16 ***matA, LweSample_16 ***matB, int bitSize, int row, int col,
                     const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ***cudaBkFFTCoalesce,
                     Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                     double *ks_cv_gpu_extendedPtr, TFheGateBootstrappingSecretKeySet *key) {

    int resBitSize = bitSize * 2, nElem = row * col;
    const int n = bk->params->in_out_params->n;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < i; j++) {
            leftRotateVec(matA[i], row);
            upRotateVec(matB, row, i);
        }
    }

    LweSample_16 **vecRes = vectorFromMatrix_rowMajor(res, row, col, bk);
    //temps
    LweSample_16 *tempV1 = convertBitToNumberZero_GPU(nElem * resBitSize, bk);
    LweSample_16 *temp = convertBitToNumberZero_GPU(resBitSize, bk);

    for (int k = 0; k < row; ++k) {
        LweSample_16 **vecA = vectorFromMatrix_rowMajor(matA, row, col, bk);
        LweSample_16 **vecB = vectorFromMatrix_rowMajor(matB, row, col, bk);

        LweSample_16 *vecMulRes = convertBitToNumberZero_GPU(nElem * resBitSize, bk);
        concurrentMultiplication(vecMulRes, vecA, vecB, nElem, bitSize, bk, NULL, cudaBkFFTCoalesce,
                                 NULL, NULL, NULL, NULL,
                                 ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr, key);
        if (k == 0) {
            cudaMemcpy(tempV1->a, vecMulRes->a, nElem * resBitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(tempV1->b, vecMulRes->b, nElem * resBitSize * sizeof(int));
        } else {
            taskLevelParallelAdd_bitwise_vector(tempV1, tempV1, vecMulRes, nElem, resBitSize, 1, bk, NULL, cudaBkFFTCoalesce,
                                                NULL, NULL, NULL, NULL, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                                                ks_cv_gpu_extendedPtr, key);
        }
        for (int i = 0; i < nElem; ++i) {
            cudaMemcpy(temp->a, vecMulRes->a + i * resBitSize * n, resBitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
            memcpy(temp->b, vecMulRes->b + i * resBitSize, resBitSize * sizeof(int));
            testCipher("v1", temp, resBitSize, bk, key);
        }
        leftRotateMatrix(matA, row, col);
        upRotateMatrix(matB, row, col);
    }

    for (int i = 0; i < nElem; ++i) {
        cudaMemcpy(vecRes[i]->a, tempV1->a + i * resBitSize * n, resBitSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
        memcpy(vecRes[i]->b, tempV1->b + i * resBitSize, resBitSize * sizeof(int));
    }
    printMatrix(res, resBitSize, row,col, bk, key);
}



void cannonsAlgoPreparations(LweSample *ca, LweSample *cb, int bitSize,
                         const TFheGateBootstrappingCloudKeySet *bk,
                         TFheGateBootstrappingSecretKeySet *key) {
    const int n = bk->bk->in_out_params->n;

    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    const LweParams *in_out_params = bk->params->in_out_params;

    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);
    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

    int row = 2, col = 2;
    //construct m1 m2
    LweSample_16 ***matA = new LweSample_16**[row];
    LweSample_16 ***matB = new LweSample_16**[row];
    LweSample_16 ***res = new LweSample_16**[row];
    for (int i = 0; i < row ; ++i) {
        matA[i] = new LweSample_16*[col];
        matB[i] = new LweSample_16*[col];
        res[i] = new LweSample_16*[col];
        for (int j = 0; j < col; ++j) {
            matA[i][j] = convertBitToNumber(ca, bitSize, bk);
            matB[i][j] = convertBitToNumber(cb, bitSize, bk);
            res[i][j] = convertBitToNumberZero_GPU(bitSize * 2, bk); // result bit size is twice of bitsize

            int *temp;
            temp = matA[i][j]->a;
            cudaMalloc(&(matA[i][j]->a), bitSize * n * sizeof(int));
            cudaMemcpy(matA[i][j]->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
            free(temp);

            temp = matB[i][j]->a;
            cudaMalloc(&(matB[i][j]->a), bitSize * n * sizeof(int));
            cudaMemcpy(matB[i][j]->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
            free(temp);

//            int x = i * col + j;
//            if (x > 0 && x < bitSize) {
//                leftShiftCuda_16_vector(matA[i][j], matA[0][0], 1, bitSize, x, bk);
//                leftShiftCuda_16_vector(matB[i][j], matB[0][0], 1, bitSize, x, bk);
//            }

//            cout << "i: " << i << " j: " << j << endl;
//            testCipher("m1", matA[i][j], bitSize, bk, key);
//            testCipher("m2", matB[i][j], bitSize, bk, key);
//            testCipher("res", res[i][j], bitSize, bk, key);
//            cout << endl;
        }
        cout << endl;
    }
    double sT = omp_get_wtime();
    cannonsAlgoComp(res, matA, matB, bitSize, row, col, bk, cudaBkFFTCoalesce, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr,
                    ks_cv_gpu_extendedPtr, key);
    cout << "time taken: " << omp_get_wtime() - sT  << endl;
    //construct vector from matrix
//    for (int i = 0; i < row * col; ++i) {
//        cudaMemcpy(temp->a, matMulRes->a + i * nOutputBits * n, nOutputBits * n * sizeof(int), cudaMemcpyDeviceToDevice);
//        memcpy(temp->b, matMulRes->b + i * nOutputBits, nOutputBits * sizeof(int));
//        memcpy(temp->current_variance, matMulRes->current_variance + i * nOutputBits, nOutputBits * sizeof(double));
//        testCipher("mat mul test int ", temp, nOutputBits, bk, key);
//
//    }


}


void testMUXopertion(LweSample *ca, LweSample *cb, int bitSize,
                         const TFheGateBootstrappingCloudKeySet *bk,
                         TFheGateBootstrappingSecretKeySet *key) {

    const int n = bk->bk->in_out_params->n;
    cufftDoubleComplex ****cudaBkFFT = NULL;
    cufftDoubleComplex ***cudaBkFFTCoalesce = sendBootstrappingKeyToGPUCoalesce(1, bk);
    Torus32 ****ks_a_gpu_extended = NULL;

    int ***ks_b_gpu = NULL;
    double ***ks_cv_gpu = NULL;
    const LweParams *in_out_params = bk->params->in_out_params;
    Torus32 *ks_a_gpu_extendedPtr = sendKeySwitchKeyToGPU_extendedOnePointer(1, bk);

    Torus32 *ks_b_gpu_extendedPtr = sendKeySwitchBtoGPUOnePtr(bk);
    double *ks_cv_gpu_extendedPtr = sendKeySwitchCVtoGPUOnePtr(bk);

    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *b = convertBitToNumber(cb, bitSize, bk);

    int *temp;
    temp = a->a;
    cudaMalloc(&(a->a), bitSize * n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = b->a;
    cudaMalloc(&(b->a), bitSize * n * sizeof(int));
    cudaMemcpy(b->a, temp, bitSize * n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    testCipher("a (MUX test)", a, bitSize, bk, key);
    testCipher("b (MUX test)", b, bitSize, bk, key);

    LweSample_16 * result = convertBitToNumberZero_GPU(bitSize, bk);
    cudaMemset(result->a, 0, sizeof(int) * bitSize * n);
    bootsMUX_16_vector(result, a, result, b, 1, bitSize, bk, cudaBkFFT, cudaBkFFTCoalesce, NULL, ks_a_gpu_extended,
                       ks_b_gpu, ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
    testCipher("result(MUX)", result, bitSize, bk, key);


}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "Usage: <binary_file_name> <bit_size> <first_number> <second_number>" << endl;
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
    cout << "bitSize: " << bitSize << endl;
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

    test_AND_XOR_CompoundGate_Addition(a.data, b.data, bitSize, &key->cloud, key);
//    test_vectorAddition(a.data, b.data, bitSize, &key->cloud, key);
//    multiplyLweSamples_test(a.data, b.data, bitSize, &key->cloud, key);
//    karatSuba_test(a.data, b.data, bitSize, &key->cloud, key);

//    vectorMultiplicationTest(a.data, b.data, bitSize, 4, &key->cloud, key);

//    testMatrixOperation(a.data, b.data, bitSize, &key->cloud, key);
//    cannonsAlgoPreparations(a.data, b.data, bitSize, &key->cloud, key);
//    testMUXopertion(a.data, b.data, bitSize, &key->cloud, key);






    //clean up all pointers
    delete_gate_bootstrapping_ciphertext_array(bitSize, ciphertext2);
    delete_gate_bootstrapping_ciphertext_array(bitSize, ciphertext1);

    //clean up all pointers
    delete_gate_bootstrapping_secret_keyset(key);
    delete_gate_bootstrapping_parameters(params);
    return 0;
}


void leftShiftCuda_16_test(LweSample *cresult, LweSample *ca, int bitSize,
                           const TFheGateBootstrappingCloudKeySet *bk,
                           TFheGateBootstrappingSecretKeySet *key) {

    LweSample_16 *a = convertBitToNumber(ca, bitSize, bk);
    LweSample_16 *result = convertBitToNumberZero(bitSize, bk);


    const LweParams *in_out_params = bk->params->in_out_params;
    int *temp;

    temp = a->a;
    cudaMalloc(&(a->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(a->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    temp = result->a;
    cudaMalloc(&(result->a), bitSize * in_out_params->n * sizeof(int));
    cudaMemcpy(result->a, temp, bitSize * in_out_params->n * sizeof(int), cudaMemcpyHostToDevice);
    free(temp);

    leftShiftCuda_16(result, a, bitSize, 2, bk);

    //get data in cpu from gpu
    temp = new int[bitSize * in_out_params->n];
    cudaMemcpy(temp, result->a, sizeof(int) * bitSize * in_out_params->n, cudaMemcpyDeviceToHost);
    cudaFree(result->a);
    result->a = temp;
    LweSample *leftShiftOutput = convertNumberToBits(result, bitSize, bk);
    Cipher leftShiftCipher(bitSize);
    leftShiftCipher.data = leftShiftOutput;

    cout << "In leftShiftCuda: " << decryptCheck(leftShiftCipher, key) << endl;
}


