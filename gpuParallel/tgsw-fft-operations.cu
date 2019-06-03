#ifndef TFHE_TEST_ENVIRONMENT
/* ***************************************************
TGSW fft operations
*************************************************** */

#include <cstdlib>
#include <iostream>
#include <random>
#include <cassert>
#include <ccomplex>
#include "tfhe_core.h"
#include "numeric_functions.h"
#include "lweparams.h"
#include "lwekey.h"
#include "lwesamples.h"
#include "lwe-functions.h"
#include "tlwe_functions.h"
#include "tgsw_functions.h"
#include "polynomials_arithmetic.h"
#include "lagrangehalfc_arithmetic.h"
#include "lwebootstrappingkey.h"
#include "lagrangehalfc_impl.h"
#include <iostream>
#include <cufftXt.h>
#include <fstream>
#include <omp.h>
#include <ctime>
#include "cudaFFTTest.h"

using namespace std;

#else
#undef EXPORT
#define EXPORT
#endif

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


__global__ void setComplexVectorToConstant(cufftDoubleComplex *destination, double realVal, double imaginaryVal, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        destination[id].x = realVal;
        destination[id].y = imaginaryVal;
    }
}

__global__ void setIntVectorToConstant(int *destination, int val, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        destination[id] = val;
    }
}

//constructor content
EXPORT void init_TGswSampleFFT(TGswSampleFFT *obj, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int l = params->l;
    TLweSampleFFT *all_samples = new_TLweSampleFFT_array((k + 1) * l, params->tlwe_params);
    new(obj) TGswSampleFFT(params, all_samples);
}

//destructor content
EXPORT void destroy_TGswSampleFFT(TGswSampleFFT *obj) {
    int k = obj->k;
    int l = obj->l;
    delete_TLweSampleFFT_array((k + 1) * l, obj->all_samples);
    obj->~TGswSampleFFT();
}


// For all the kpl TLWE samples composing the TGSW sample 
// It computes the inverse FFT of the coefficients of the TLWE sample   
EXPORT void tGswToFFTConvert(TGswSampleFFT *result, const TGswSample *source, const TGswParams *params) {
    const int kpl = params->kpl;

    for (int p = 0; p < kpl; p++)
        tLweToFFTConvert(result->all_samples + p, source->all_sample + p, params->tlwe_params);
}

// For all the kpl TLWE samples composing the TGSW sample 
// It computes the FFT of the coefficients of the TLWEfft sample
EXPORT void tGswFromFFTConvert(TGswSample *result, const TGswSampleFFT *source, const TGswParams *params) {
    const int kpl = params->kpl;

    for (int p = 0; p < kpl; p++)
        tLweFromFFTConvert(result->all_sample + p, source->all_samples + p, params->tlwe_params);
}



// result = result + H
EXPORT void tGswFFTAddH(TGswSampleFFT *result, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int l = params->l;

    for (int j = 0; j < l; j++) {
        Torus32 hj = params->h[j];
        for (int i = 0; i <= k; i++)
            LagrangeHalfCPolynomialAddTorusConstant(&result->sample[i][j].a[i], hj);
    }

}

// result = list of TLWE (0,0)
EXPORT void tGswFFTClear(TGswSampleFFT *result, const TGswParams *params) {
    const int kpl = params->kpl;

    for (int p = 0; p < kpl; p++)
        tLweFFTClear(result->all_samples + p, params->tlwe_params);
}

// External product (*): accum = gsw (*) accum 
EXPORT void tGswFFTExternMulToTLwe(TLweSample *accum, const TGswSampleFFT *gsw, const TGswParams *params) {
    const TLweParams *tlwe_params = params->tlwe_params;
    const int k = tlwe_params->k;
    const int l = params->l;
    const int kpl = params->kpl;
    const int N = tlwe_params->N;
//    cout << "-----------------" << endl;
//    static int i = 1;
//    if (i == 1) {
//        cout << "old: " << endl;
//        cout << "\ttlwe_params->k: " << tlwe_params->k << endl;//1
//        cout << "\tparams->l: " << params->l << endl;//2
//        cout << "\tparams->kpl: " << params->kpl << endl; //4
//        cout << "\ttlwe_params->N: " << tlwe_params->N << endl;//1024
//        i++;
//    }
    //TODO attention, improve these new/delete...
    IntPolynomial *deca = new_IntPolynomial_array(kpl, N); //decomposed accumulator
    LagrangeHalfCPolynomial *decaFFT = new_LagrangeHalfCPolynomial_array(kpl, N); //fft version
    TLweSampleFFT *tmpa = new_TLweSampleFFT(tlwe_params);

    //test morshed start
//    static int tCounter = 0;
//    if (tCounter >= 500 && tCounter < 500 + P_LIMIT) {
//        cout << "old_input: ";
//        int accIndex = 1;
//        for (int i = 0; i < 10; ++i) {
//            cout << (accum->a + accIndex)->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
//    tCounter ++;
    //test morshed end
    for (int i = 0; i <= k; i++) {
        tGswTorus32PolynomialDecompH(deca + i * l, accum->a + i, params);
    }
    //test Morshed start
//    static int tCounter = 0;
//    if (tCounter < P_LIMIT) {
//        for (int i = 0; i < kpl; ++i) {
//            cout << "old decompH: ";
//            for (int j = 0; j < N; ++j) {
//                cout << (deca + i)->coefs[j] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounter++;
    //test Morshed end
    for (int p = 0; p < kpl; p++) {
        IntPolynomial_ifft(decaFFT + p, deca + p);
    }
    //test morshed start
//    static int tCounterX = 0;
//    if (tCounterX < P_LIMIT) {
//        ofstream inputFile, outputFile;
//        inputFile.open("old_ifft_input.txt", ofstream::app | ofstream::out);
//        outputFile.open("old_ifft_output.txt", ofstream::app | ofstream::out);
//        for (int i = 0; i < kpl; ++i) {
//            cout << "old ifft input: ";
//            for (int j = 0; j < N; ++j) {
//                cout << (deca + i)->coefs[j] << " ";
//                inputFile << (deca + i)->coefs[j] << " ";
//            }
//            cout << endl;
//            inputFile << endl;
//            cout << "old ifft output: ";
//            for (int j = 0; j < 5; ++j) {
//                cout << ((LagrangeHalfCPolynomial_IMPL*)(decaFFT + i))->coefsC[j] << " ";
////                outputFile << ((LagrangeHalfCPolynomial_IMPL *) (decaFFT + i))->coefsC[j] << " ";
//            }
//            cout << endl;
//            outputFile << endl;
//        }
//        inputFile.close();
//        outputFile.close();
//    }
//    tCounterX++;
////    //test morshed end
    tLweFFTClear(tmpa, tlwe_params);

    for (int p = 0; p < kpl; p++) {
        //test morshed start
//        static int tCounter = 0;
//        for (int j = 0; j <= k; ++j) {
//            if (tCounter < P_LIMIT) {
//                fstream oldFile;
//                oldFile.open("oldAllSample.txt", ios::app);
////                cout << "old all_samples: ";
//                for (int i = 0; i < 512; ++i) {
////                    cout << ((LagrangeHalfCPolynomial_IMPL *) ((gsw->all_samples + p)->a) + j)->coefsC[i] << " ";
//                    oldFile << ((LagrangeHalfCPolynomial_IMPL *) ((gsw->all_samples + p)->a) + j)->coefsC[i] << " ";
//                }
//                oldFile << endl;
////                cout << endl;
//                oldFile.close();
//            }
//            tCounter++;
//        }
        //test morshed end
        tLweFFTAddMulRTo(tmpa, decaFFT + p, gsw->all_samples + p, tlwe_params);
    }
//    static int tCounterY = 0;
//    int offset = 0;
//    if (tCounterY >= offset && tCounterY < offset + P_LIMIT) {
////        fstream oldFile;
////        oldFile.open("oldComplex.txt", ios::app);
//        for (int index = 0; index <= k; ++index) {
//            cout << "old complex output: ";
//            for (int i = 0; i < 5; ++i) {
//                    cout << ((LagrangeHalfCPolynomial_IMPL *) (tmpa->a + index))->coefsC[i] << " ";
////                oldFile << ((LagrangeHalfCPolynomial_IMPL *) (tmpa->a + index))->coefsC[i] << " ";
//            }
//                cout << endl;
////            oldFile << endl;
//        }
////        oldFile.close();
//    }
//    tCounterY++;




    tLweFromFFTConvert(accum, tmpa, tlwe_params);

//    static int tCounterZ = 0;
//    if (tCounterZ < P_LIMIT) {
//        for (int ind = 0; ind <= k; ++ind) {
//            cout << "old_fft: ";
//            for (int i = 0; i < 10; ++i) {
//                cout << (accum->a + ind)->coefsT[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounterZ++;

    delete_TLweSampleFFT(tmpa);
    delete_LagrangeHalfCPolynomial_array(kpl, decaFFT);
    delete_IntPolynomial_array(kpl, deca);
}

//new
EXPORT void tGswFFTExternMulToTLwe_16(TLweSample *accum, const TGswSampleFFT *gsw, int bitSize, const TGswParams *params,
                                      cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce,
                                      IntPolynomial *deca, IntPolynomial *decaCoalesce,
                                      cufftDoubleComplex **cuDecaFFT, cufftDoubleComplex *cuDecaFFTCoalesce,
                                      cufftDoubleComplex **tmpa_gpu) {
    const TLweParams *tlwe_params = params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = params->l;//2
    const int kpl = params->kpl;//4
    int N = tlwe_params->N;//1024
    //GPU variable start
    int bigN = N * bitSize;
    int Ns2 = N / 2;
    int BLOCKSIZE = 1024;
    int COMPLEX_BLOCKSIZE = Ns2;
    int gridSize_complex = (int)ceil((float)(Ns2 * bitSize)/COMPLEX_BLOCKSIZE);
//    static int decaInitiator = 0;
//    static int cudaDecaFFTInitiator = 0;
//    static int tmpa_gpuInitiator = 0;
    /*
    //GPU variable end
//    if (i == 1) {
//        cout << "new: " << endl;
//        cout << "\ttlwe_params->k: " << tlwe_params->k << endl;//1
//        cout << "\tparams->l: " << params->l << endl;//2
//        cout << "\tparams->kpl: " << params->kpl << endl; //4
//        cout << "\ttlwe_params->N: " << tlwe_params->N << endl;//1024
//        cout << "\tN: " << N << endl;//1024
//        i++;
//    }
//    static IntPolynomial *deca;
//        if(decaInitiator < 1) {
//        deca = new_IntPolynomial_array(kpl, bigN); //decomposed accumulator
//        for (int i = 0; i < kpl; ++i) {
//            free(deca[i].coefs);
//            cudaMalloc(&((deca + i)->coefs), bitSize * N * sizeof(int));
//        }
//        decaInitiator++;
//    }
//    //testing coalesce
//    static IntPolynomial *decaCoalesce;
//    static int decaCoalesceInitiator = 0;
//    if (decaCoalesceInitiator < 1) {
//        decaCoalesce = new_IntPolynomial(kpl * bigN);
//        free(decaCoalesce->coefs);
//        cudaMalloc(&(decaCoalesce->coefs), bigN * kpl * sizeof(int));
////        int gridSize = (int)ceil((float)(bigN * kpl)/BLOCKSIZE);
////        setIntVectorToConstant<<<gridSize, BLOCKSIZE>>>(decaCoalesce->coefs, 0, bigN * kpl);
//        decaCoalesceInitiator++;
//    }
//
//    static cufftDoubleComplex **cuDecaFFT;
//    if(cudaDecaFFTInitiator < 1) {
//        cuDecaFFT = new cufftDoubleComplex *[kpl];
//        for (int i = 0; i < kpl; ++i) {
//            cudaMalloc(&(cuDecaFFT[i]), bitSize * Ns2 * sizeof(cufftDoubleComplex));
//        }
//        cudaDecaFFTInitiator++;
//    }
//    //testing coalesce
//    static cufftDoubleComplex *cuDecaFFTCoalesce;
//    static int cuDecaFFTCoalesceInitiator = 0;
//    if(cuDecaFFTCoalesceInitiator < 1) {
//        cudaMalloc(&cuDecaFFTCoalesce, kpl * bitSize * Ns2 * sizeof(cufftDoubleComplex));
//        cuDecaFFTCoalesceInitiator++;
//    }
//
//    static cufftDoubleComplex **tmpa_gpu;
//    if (tmpa_gpuInitiator < 1) {
//        tmpa_gpu = new cufftDoubleComplex *[k + 1];;
//        for (int i = 0; i <= k; ++i) {
//            cudaMalloc(&(tmpa_gpu[i]), bitSize * Ns2 * sizeof(cufftDoubleComplex));
//        }
//        tmpa_gpuInitiator++;
//    }*/

    tGswTorus32PolynomialDecompH_16_Coalesce(decaCoalesce->coefs, accum, bitSize, params);

//    int *temp = new int[bitSize * N * kpl];
//    cudaMemcpy(temp, decaCoalesce->coefs, kpl * bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//    cout <<  "coalesce: " << endl;
//    for (int i = 0; i < bitSize * kpl; ++i) {
//        int sI = i * N;
//        for (int j = 0; j < 500; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;
    /*
//    cout <<  "old: " << endl;
//    for (int i = 0; i <= k; i++) {
//        tGswTorus32PolynomialDecompH_16(deca + i * l, accum->a + i, bitSize, params);
//        int *temp = new int[bitSize * N];
//        cudaMemcpy(temp, (deca + i * l)->coefs, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < 4; ++j) {
//            for (int m = 0; m < 20; ++m) {
//                cout << temp[j * N + m] << " ";
//            }
//            cout << endl;
//        }
//        cudaMemcpy(temp, (deca + i * l + 1)->coefs, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < 4; ++j) {
//            for (int m = 0; m < 20; ++m) {
//                cout << temp[j * N + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
    //test Morshed start
//    static int tCounter = 0;
//    if (tCounter++ < P_LIMIT) {
//        for (int i = 0; i < kpl; ++i) {
//            int bI = 0;
//            int sI = bI * N;
//            int *temp = new int[bitSize * N];
//            cout << "new decompH: ";
//            cudaMemcpy(temp, (deca + i)->coefs, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            for (int j = 0; j < N; ++j) {
//                cout << temp[sI + j] << " ";
//            }
//            cout << endl;
//        }
//    }
    //test Morshed end*/
//    cout << "I am here" << endl;
    IntPolynomial_ifft_16_Coalesce(cuDecaFFTCoalesce, bitSize, decaCoalesce);

//    cufftDoubleComplex *temp = new cufftDoubleComplex[kpl * bitSize * Ns2];
//    cudaMemcpy(temp, cuDecaFFTCoalesce, kpl * bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int p = 0; p < kpl; ++p) {
//        int sI = p * Ns2 * bitSize;
//        for (int i = 0; i < bitSize; ++i) {
//            int sI2 = sI + i * Ns2;
//            for (int j = 0; j < 500; ++j) {
//                cout << "(" << temp[sI2 + j].x << "," << temp[sI2 + j].y << ") ";
//            }
//            cout << endl;
//        }
////        cout << endl;
//    }
//
//    cout << endl << endl;
    /*
////    for (int p = 0; p < kpl; p++) {
////        IntPolynomial_ifft_16(cuDecaFFT[p], deca + p);
////        cufftDoubleComplex *temp = new cufftDoubleComplex[bitSize * Ns2];
////        cudaMemcpy(temp, cuDecaFFT[p], bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
////        for (int i = 0; i < 1; ++i) {
////            for (int j = 0; j < 10; ++j) {
////                cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ") ";
////            }
////            cout << endl;
////        }
////        cout << endl;
////    }
//
//    //test morshed start
////    static int tCounterX = 0;
////    ofstream inputFile, outputFile;
////    inputFile.open("new_ifft_input.txt", ofstream::app | ofstream::out);
////    outputFile.open("new_ifft_output.txt", ofstream::app | ofstream::out);
////    if(tCounterX < P_LIMIT) {
////    int bI = 0;
////    int sI = bI * N;
////    int sIx = bI * Ns2;
////        for (int i = 0; i < kpl; ++i) {
////            int *temp = new int[N * bitSize];
////            cufftDoubleComplex *temp_outputCmplx = new cufftDoubleComplex[Ns2 * bitSize];
////            cudaMemcpy(temp_outputCmplx, cuDecaFFT[i], Ns2 * bitSize * sizeof(cufftDoubleComplex),
////                       cudaMemcpyDeviceToHost);
////            cudaMemcpy(temp, (deca + i)->coefs, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
////            cout << "new ifft input: ";
////            for (int j = 0; j < N; ++j) {
////                cout << temp[j] << " ";
////                inputFile << temp[sI + j] << " ";
////            }
////            inputFile << endl;
////            cout << "new ifft output: ";
////            for (int j = 0; j < 5; ++j) {
//////                outputFile << "(" << temp_outputCmplx[sIx + j].x << "," << temp_outputCmplx[sIx + j].y << ") ";
////                cout << "(" << temp_outputCmplx[sIx + j].x << "," << temp_outputCmplx[sIx + j].y << ") ";
////            }
//////            outputFile << endl;
////            cout << endl;
////        }
////        inputFile.close();
////        outputFile.close();
////    }
////    tCounterX++;
//    //test morshed end
////    //create data structure corresponding to tmpa
////
//
//    //clear tmpa gpu
////    int gridSize_complex_coalesce = (int)ceil((float)((k + 1) * Ns2 * bitSize)/COMPLEX_BLOCKSIZE);
////    cout << "***gridSize_complex: " << gridSize_complex << endl;
////    cout << "***gridSize_complex_coalesce: " << gridSize_complex_coalesce << endl;
////    setComplexVectorToConstant<<<gridSize_complex_coalesce, COMPLEX_BLOCKSIZE>>>
////                                                           (tmpa_gpuCoalesce, 0., 0., bitSize * Ns2 * (k + 1));
    */
    for (int i = 0; i <= k; ++i) {
        setComplexVectorToConstant<<<gridSize_complex, COMPLEX_BLOCKSIZE>>>(tmpa_gpu[i], 0., 0., bitSize*Ns2);
    }

    tLweFFTAddMulRTo_gpu_coalesce(tmpa_gpu, cuDecaFFTCoalesce, cudaBKiCoalesce, kpl, Ns2, bitSize, tlwe_params);

//    cout << "Coalesce: " << endl;
//    cufftDoubleComplex *temp = new cufftDoubleComplex[bitSize * Ns2];
//    for (int m = 0; m <= 1; ++m) {
//        cudaMemcpy(temp, tmpa_gpu[m], bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            for (int j = 0; j < 500; ++j) {
//                cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
//            }
//            cout << endl;
//        }
////        cout << endl;
//    }
//    cout << endl;
/*
//    for (int i = 0; i <= k; ++i) {
//        setComplexVectorToConstant<<<gridSize_complex, COMPLEX_BLOCKSIZE>>>(tmpa_gpu[i], 0., 0., bitSize * Ns2);
//    }

//    for (int p = 0; p < kpl; p++) {
//        static int counter1 = 0;
//        if(counter1 < P_LIMIT) {
//            int bI = 0;
//            int sI = bI * Ns2;
//            cudaMemcpy(outputCmplx, cuDecaFFT[p], sizeof(cufftDoubleComplex)*Ns2*bitSize, cudaMemcpyDeviceToHost);
//            cudaMemcpy(outputCmplx1, cdGSWAllSamples[p][0], sizeof(cufftDoubleComplex)*Ns2*bitSize, cudaMemcpyDeviceToHost);
//            cout << "new all sample: ";
//
//            for (int i = 0; i < 10; ++i) {
//                cout << "(" << outputCmplx[sI + i].x<< "," << outputCmplx[sI + i].y << ")" << " ";
//                cout << "(" << outputCmplx1[sI + i].x<< "," << outputCmplx1[sI + i].y << ")" << " ";
//            }
//            cout << endl;
//            counter1++;
//        }
//        static int tCounter = 0;
//        for (int j = 0; j <= k; ++j) {
//            if (tCounter < P_LIMIT) {
//                fstream newFile;
//                newFile.open("newAllSample.txt", ios::app);
//                cufftDoubleComplex *outputComplx = new cufftDoubleComplex[Ns2*bitSize];
//                cudaMemcpy(outputComplx, cdGSWAllSamples[p][j], Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
////                cout << "new all_samples: ";
//                for (int i = 0; i < Ns2; ++i) {
////                    cout << "(" << outputComplx[i].x << "," << outputComplx[i].y << ") ";
//                    newFile << "(" << outputComplx[i].x << "," << outputComplx[i].y << ") ";
//                }
////                cout << endl;
//                newFile << endl;
//                newFile.close();
//            }
//            tCounter++;
//        }
//        tLweFFTAddMulRTo_gpu(tmpa_gpu, cuDecaFFT[p], cdGSWAllSamples[p], Ns2, bitSize, tlwe_params);
//        tLweFFTAddMulRTo_gpu(tmpa_gpu, cuDecaFFT[p], cudaBKi[p], Ns2, bitSize, tlwe_params);
//    }
//    cout << "old: " << endl;
//    temp = new cufftDoubleComplex[bitSize * Ns2];
//    cudaMemcpy(temp, tmpa_gpu[1], bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
    //test morshed start
//    int static tCounterY = 0;
//    if (tCounterY < P_LIMIT) {
//        int bI = 0;
//        int sI = bI * Ns2;
//        cufftDoubleComplex *testingout = new cufftDoubleComplex[bitSize * Ns2];
//        fstream newFile1;
//        newFile1.open("newComplexOutput1.txt", ios::app);
//        fstream newFile2;
//        newFile2.open("newComplexOutput2.txt", ios::app);
//        for (int index = 0; index <= k; ++index) {
//            cudaMemcpy(testingout, tmpa_gpu[index], bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//            cout << "new complex output: ";
//            for (int i = 0; i < 4; ++i) {
//                cout << "(" << testingout[sI + i].x << "," << testingout[sI + i].y << ") ";
//
//            }
//            for (int i = 0; i < Ns2; ++i) {
//                newFile1 << testingout[sI + i].x << endl;
//                newFile2 << testingout[sI + i].y << endl;
//            }
//            cout << endl;
////            newFile << endl;
//        }
//        newFile1.close();newFile2.close();
//    }
//    tCounterY++;*/
    //test morshed end
    tLweFromFFTConvert_gpu(accum, tmpa_gpu, bitSize, N, Ns2, tlwe_params);
//
//    int *temp = new int[bitSize * N];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(temp, (accum->a + valK)->coefsT, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                cout << temp[i * N + j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
 /*//    static int tCounterZ = 0;
//    if (tCounterZ < P_LIMIT) {
//        for (int ind = 0; ind <= k; ++ind) {
//            int *testTorus = new int[N * bitSize];
//            cudaMemcpy(testTorus, (accum->a + ind)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//            cout << "new_fft: ";
//            for (int i = 0; i < 10; ++i) {
//                cout << testTorus[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounterZ++;*/

    int error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Cuda error code testing here: %d\n", error);
        exit(1);
    }
}



__global__ void inputData(cufftDoubleReal *des, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < length) {
        if ((id % 2) == 1) {
            des[id] = id;
        } else {
            des[id] = 0;
        }
//        des[id] = 0;
    }
}

EXPORT void
tGswFFTExternMulToTLwe_16V2(TLweSample *accumV2, const TGswSampleFFT *gsw, int bitSize, const TGswParams *params,
                            cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce,
                            IntPolynomial *deca, IntPolynomial *decaCoalesce,
                            cufftDoubleComplex **cuDecaFFT, cufftDoubleComplex *cuDecaFFTCoalesce,
                            cufftDoubleComplex **tmpa_gpu, cufftDoubleComplex *tmpa_gpuCoal,
                            cudaFFTProcessorTest_general *fftProcessor) {
    const TLweParams *tlwe_params = params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = params->l;//2
    const int kpl = params->kpl;//4
    int N = tlwe_params->N;//1024
    int Ns2 = N / 2;
    int BLOCKSIZE = 1024;

    int length = bitSize * N;
    int length_Ns2 = bitSize * Ns2;//16384
    int gridSize_complexV2 = (int) ceil((float) (length_Ns2 * (k + 1)) / BLOCKSIZE);//32

    tGswTorus32PolynomialDecompH_16_Coalesce(decaCoalesce->coefs, accumV2, bitSize, params);

//    int *temp = new int[bitSize * N * kpl];
//    cudaMemcpy(temp, decaCoalesce->coefs, kpl * bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//    cout <<  "coalesce: " << endl;
//    for (int i = 0; i < bitSize * kpl; ++i) {
//        int sI = i * N;
//        for (int j = 0; j < 500; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;
    fftProcessor->execute_reverse_int(cuDecaFFTCoalesce, decaCoalesce->coefs);

//    cufftDoubleComplex *temp = new cufftDoubleComplex[kpl * bitSize * Ns2];
//    cudaMemcpy(temp, cuDecaFFTCoalesce, kpl * bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int p = 0; p < kpl; ++p) {
//        int sI = p * Ns2 * bitSize;
//        for (int i = 0; i < bitSize; ++i) {
//            int sI2 = sI + i * Ns2;
//            for (int j = 0; j < 500; ++j) {
//                cout << "(" << temp[sI2 + j].x << "," << temp[sI2 + j].y << ") ";
//            }
//            cout << endl;
//        }
////        cout << endl;
//    }
    setComplexVectorToConstant<<<gridSize_complexV2, BLOCKSIZE>>>(tmpa_gpuCoal, 0., 0., length_Ns2 * (k + 1));
    tLweFFTAddMulRTo_gpu_coalesce_2V2(tmpa_gpuCoal, cuDecaFFTCoalesce, cudaBKiCoalesce, kpl, Ns2, 1, bitSize, tlwe_params);

//    cufftDoubleComplex *temp = new cufftDoubleComplex[bitSize * Ns2 * (k + 1)];
//    cudaMemcpy(temp, tmpa_gpuCoal, (k + 1) * bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * 2; ++i) {
//        for (int j = 0; j < 500; ++j) {
//            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;

    fftProcessor->execute_direct_Torus32_gpu(accumV2->a->coefsT, tmpa_gpuCoal);

//    int *temp = new int[bitSize * N];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(temp, (accumV2->a + valK)->coefsT, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                cout << temp[i * N + j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }

}

EXPORT void tGswFFTExternMulToTLwe_16_2V2(TLweSample *accumV2, const TGswSampleFFT *gsw, int nOutputs, int bitSize,
                                          const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                          cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                          IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                          cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex *tmpa_gpuCoal,
                                          cudaFFTProcessorTest_general *fftProcessor) {
    const TLweParams *tlwe_params = params->tlwe_params;
    const int k = tlwe_params->k;//1
//    const int l = params->l;//2
    const int kpl = params->kpl;//4
    int N = tlwe_params->N;//1024
    //GPU variable start
    int Ns2 = N / 2;
    int BLOCKSIZE = 1024;

    int length = nOutputs * bitSize * N;
    int length_Ns2 = nOutputs * bitSize * Ns2;//16384
    int gridSize_complexV2 = (int) ceil((float) (length_Ns2 * (k + 1)) / BLOCKSIZE);//32

    tGswTorus32PolynomialDecompH_16_2_CoalesceV2(decaCoalesce->coefs, accumV2, nOutputs, bitSize, params);
    /*int *temp = new int[kpl * length];
    cudaMemcpy(temp, decaCoalesce->coefs, kpl * length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < kpl; ++i) {
        int sI = i * length;
        for (int j = 0; j < bitSize * nOutputs; ++j) {
            int sI2 = sI + j * N;
            for (int m = 0; m < 10; ++m) {
                cout << temp[sI2 + m] << " ";
            }
            cout << endl;
            if (j == bitSize - 1) cout << endl;
        }
        cout << endl;
    }
    cout << endl;*/

//    cout << "bangladesh" << endl;

//    int i
//    int *idata;
//    cufftDoubleComplex *odata;
////    int BLOCKSIZE  = 1024;
////    int gridSize = (NX * BATCH)/BLOCKSIZE;
//
//    cudaMalloc((void**)&idata, sizeof(int)* decaCoalesce->N);
//    cudaMalloc((void**)&odata, sizeof(cufftDoubleComplex) * 1024 * bitSize* 2);
//    decaCoalesce->coefs = idata;
//    std::clock_t start;
//    double duration;



//    double sT = omp_get_wtime();
//    start = std::clock();

//        IntPolynomial_ifft_16_2_Coalesce(cuDecaFFTCoalesce, bitSize, decaCoalesce);
    fftProcessor->execute_reverse_int(cuDecaFFTCoalesce, decaCoalesce->coefs);

//    cout << bitSize << endl;
//    cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_1(1024, nOutputs, bitSize, 4, 1024, 2);
//    IntPolynomial_ifft_16_2_Coalesce(cuDecaFFTCoalesce, bitSize, decaCoalesce);
//    cudaFFTProcessorTestTest_general_coal_2_1.execute_reverse_int(cuDecaFFTCoalesce, decaCoalesce->coefs);

//    cudaDeviceSynchronize();
//    cout << "time: " << omp_get_wtime() - sT << endl;
//
//    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//
//    std::cout<<"printf: "<< duration <<'\n';

//    cufftDoubleComplex *temp = new cufftDoubleComplex[kpl * length_Ns2];
//    cudaMemcpy(temp, cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length_Ns2;
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            int sI2 = sI + j * Ns2;
//            for (int m = 0; m < 5; ++m) {
//                cout << "(" << temp[sI2 + m].x << "," << temp[sI2 + m].y << ") ";
//            }
//            cout << endl;
//            if(j == bitSize -1)cout << endl;
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cudaDeviceSynchronize();
    setComplexVectorToConstant<<<gridSize_complexV2, BLOCKSIZE>>>(tmpa_gpuCoal, 0., 0., length_Ns2 * (k + 1));
    tLweFFTAddMulRTo_gpu_coalesce_2V2(tmpa_gpuCoal, cuDecaFFTCoalesce, cudaBKiCoalesce, kpl, Ns2, nOutputs, bitSize, tlwe_params);
    /*cufftDoubleComplex *temp = new cufftDoubleComplex[bitSize * Ns2 * nOutputs * 2];
    cudaMemcpy(temp, tmpa_gpuCoal, 2 * nOutputs * bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 5; ++j) {
            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
        }
        cout << endl;
    }
    cout << endl;*/

//    TorusPolynomial_fft_gpu_2(accumV2->a->coefsT, tmpa_gpuCoal, nOutputs, bitSize, N, Ns2);
    fftProcessor->execute_direct_Torus32_gpu(accumV2->a->coefsT, tmpa_gpuCoal);
//    cudaFFTProcessorTestTest_general_coal_2_1.execute_direct_Torus32_gpu(accumV2->a->coefsT, tmpa_gpuCoal);
//    cudaDeviceSynchronize();
//    cudaDeviceSynchronize();

//    int *temp = new int[nOutputs * bitSize * N];
//    cudaMemcpy(temp, (accumV2->a + 0)->coefsT, nOutputs * bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nOutputs * bitSize; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[i * N + j] << " ";
//        }
//        cout << endl;
//        if (i == bitSize - 1)cout << endl;
//    }
//    cout << endl;
//    delete cudaFFTProcessorTestTest_general_coal_2_1;

}

EXPORT void tGswFFTExternMulToTLwe_16_2(TLweSample *accum, const TGswSampleFFT *gsw, int nOutputs, int bitSize,
                                        const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                        cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                        IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                        cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu,
                                        cufftDoubleComplex *tmpa_gpuCoal) {
    const TLweParams *tlwe_params = params->tlwe_params;
    const int k = tlwe_params->k;//1
//    const int l = params->l;//2
    const int kpl = params->kpl;//4
    int N = tlwe_params->N;//1024
    //GPU variable start
    int bigN = N * bitSize;
    int Ns2 = N / 2;
    int BLOCKSIZE = 1024;
    int COMPLEX_BLOCKSIZE = 1024;
//    static int decaInitiator = 0;
//    static int cudaDecaFFTInitiator = 0;
//    static int tmpa_gpuInitiator = 0;
    int length = nOutputs * bitSize * N;
    int length_Ns2 = nOutputs * bitSize * Ns2;//16384
    int gridSize_complex = (int)ceil((float)(length_Ns2)/COMPLEX_BLOCKSIZE);//32
    int gridSize_complexV2 = (int)ceil((float)(length_Ns2 * (k + 1))/COMPLEX_BLOCKSIZE);//32



//    tGswTorus32PolynomialDecompH_16_2_Coalesce(decaCoalesce->coefs, accumV2, nOutputs, bitSize, params);

//    cout << "coalesce: " << endl;
//    int *temp = new int[kpl * length];
//    cudaMemcpy(temp, decaCoalesce->coefs, kpl * length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length;
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            int sI2 = sI + j * N;
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[sI2 + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    cout << endl;


    tGswTorus32PolynomialDecompH_16_2_Coalesce(decaCoalesce->coefs, accum, nOutputs, bitSize, params);

//    cout << "coalesce: " << endl;
//    temp = new int[kpl * length];
//    cudaMemcpy(temp, decaCoalesce->coefs, kpl * length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length;
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            int sI2 = sI + j * N;
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[sI2 + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//
//    tGswTorus32PolynomialDecompH_16_2_CoalesceV2(decaCoalesce->coefs, accum, nOutputs, bitSize, params);
//    cout << "V2" << endl << endl;
//    cudaMemcpy(temp, decaCoalesce->coefs, kpl * length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length;
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            int sI2 = sI + j * N;
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[sI2 + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    cout << "new: " << endl;
//    for (int i = 0; i <= k; i++) {
//        tGswTorus32PolynomialDecompH_16_2(deca + i * l, accum->a + i, nOutputs, bitSize, params);
//        int *temp = new int[length];
//        cudaMemcpy(temp, (deca + i * l)->coefs, length * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[j * N + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//
//        cudaMemcpy(temp, (deca + i * l + 1)->coefs, length * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[j * N + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
    IntPolynomial_ifft_16_2_Coalesce(cuDecaFFTCoalesce, bitSize, decaCoalesce);

//    cufftDoubleComplex *temp = new cufftDoubleComplex[kpl * length_Ns2];
//    cudaMemcpy(temp, cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    cout << "Coalesce: " << endl;
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length_Ns2;
//        for (int j = 0; j < bitSize * nOutputs; ++j) {
//            int sI2 = sI + j * Ns2;
//            for (int m = 0; m < 8; ++m) {
//                cout << "(" << temp[sI2 + m].x << "," << temp[sI2 + m].y << ") ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    cout << "new: "<< endl;
//    for (int p = 0; p < kpl; p++) {
//        IntPolynomial_ifft_16_2(cuDecaFFT[p], deca + p);
//        cufftDoubleComplex *temp = new cufftDoubleComplex[length_Ns2];
//        cudaMemcpy(temp, cuDecaFFT[p], length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize * nOutputs; ++i) {
//            for (int j = 0; j < 8; ++j) {
//                cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ") ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }

    //clear tmpa gpu
    for (int i = 0; i <= k; ++i) {
//        setComplexVectorToConstant<<<gridSize_complex, COMPLEX_BLOCKSIZE>>>(tmpa_gpu[i], 0., 0., length_Ns2);
        cudaMemset(tmpa_gpu[i], 0, length_Ns2 * sizeof(cufftDoubleComplex));
    }

//    setComplexVectorToConstant<<<gridSize_complexV2, COMPLEX_BLOCKSIZE>>>(tmpa_gpuCoal, 0., 0., length_Ns2 * (k + 1));

    tLweFFTAddMulRTo_gpu_coalesce_2(tmpa_gpu, cuDecaFFTCoalesce, cudaBKiCoalesce, kpl, Ns2, nOutputs, bitSize, tlwe_params);

//    tLweFFTAddMulRTo_gpu_coalesce_2V2(tmpa_gpuCoal, cuDecaFFTCoalesce, cudaBKiCoalesce, kpl, Ns2, nOutputs, bitSize, tlwe_params);
/*
    cufftDoubleComplex *tmpa_gpuCombined;
    length_Ns2 = nOutputs * bitSize * Ns2;//16384
    cudaMalloc(&tmpa_gpuCombined, sizeof(cufftDoubleComplex) * length_Ns2 * (k + 1));
//
//    cout << "Coalesce: " << endl;
    cufftDoubleComplex *temp = new cufftDoubleComplex[length_Ns2];
    cudaMemcpy(temp, tmpa_gpu[0], length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
        }
        cout << endl;
    }
    cudaMemcpy(temp, tmpa_gpu[1], length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << endl;
    temp = new cufftDoubleComplex[length_Ns2 * (k + 1)];
    cudaMemcpy(temp, tmpa_gpuCoal, length_Ns2 * (k + 1) * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs * (k + 1); ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
        }
        cout << endl;
    }
    cout << endl << "------" << endl;
    */
//    for (int i = 0; i <= k; ++i) {
//        setComplexVectorToConstant<<<gridSize_complex, COMPLEX_BLOCKSIZE>>>(tmpa_gpu[i], 0., 0., length_Ns2);
//    }
//
//    for (int p = 0; p < kpl; p++) {
//        tLweFFTAddMulRTo_gpu_2(tmpa_gpu, cuDecaFFT[p], cudaBKi[p], Ns2, nOutputs, bitSize, tlwe_params);
//    }
//    cout << "new: " << endl;
//    temp = new cufftDoubleComplex[length_Ns2];
//    cudaMemcpy(temp, tmpa_gpu[0], length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * nOutputs; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << "(" << temp[i * Ns2 + j].x << "," << temp[i * Ns2 + j].y << ")" << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;

//    for (int i = 0; i <= k; ++i) {
//        int sI = i * length_Ns2;
//        cudaMemcpy(tmpa_gpuCoal + sI, tmpa_gpu[i], length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
//    }
//    int error = cudaGetLastError();
//    if (error != cudaSuccess){
//        fprintf(stderr, "****ERRORfftBefore: %d\n", error);
//        fprintf(stderr, "Cuda error code testing here: %d\n", error);
//        exit(1);
//    }
//
//    cout << "here" << endl;
    tLweFromFFTConvert_gpu_2(accum, tmpa_gpu, NULL, nOutputs, bitSize, N, Ns2, tlwe_params);
//    cout << "here2" << endl;
//    error = cudaGetLastError();
//    if (error != cudaSuccess){
//        fprintf(stderr, "****ERRORfft: %d\n", error);
//        fprintf(stderr, "Cuda error code testing here: %d\n", error);
//        exit(1);
//    }
//    TorusPolynomial_fft_gpu_2(accum->a->coefsT, tmpa_gpuCoal, nOutputs, bitSize, N, Ns2);
//    cudaMemcpy((accum->a + 0)->coefsT , (accumV2->a + 0)->coefsT, nOutputs * bitSize * N * sizeof(Torus32), cudaMemcpyDeviceToDevice);
//    cudaMemcpy((accum->a + 1)->coefsT , (accumV2->a + 1)->coefsT, nOutputs * bitSize * N * sizeof(Torus32), cudaMemcpyDeviceToDevice);

    //    (accum->a + 0)->coefsT = (accumV2->a + 0)->coefsT;
//    (accum->a + 1)->coefsT = (accumV2->a + 1)->coefsT;
    /*
//    cout << bitSize << endl;
    cout << "FFT: " << endl;
    int *temp = new int[length];
    cudaMemcpy(temp, (accum->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << temp[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cudaMemcpy(temp, (accumV2->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << temp[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;*/



//    int error = cudaGetLastError();
//    if (error != cudaSuccess){
//        fprintf(stderr, "Cuda error code testing here: %d\n", error);
//        exit(1);
//    }

}



EXPORT void tGswFFTExternMulToTLwe_16_2_vector(TLweSample *accum, const TGswSampleFFT *gsw, int vLength, int nOutputs,
                                               int bitSize, const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                               cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                               IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                               cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu) {
    const TLweParams *tlwe_params = params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int kpl = params->kpl;//4
    int N = tlwe_params->N;//1024
    //GPU variable start
    int Ns2 = N / 2;
    int BLOCKSIZE = 1024;
    int COMPLEX_BLOCKSIZE = Ns2;

    int length = vLength * nOutputs * bitSize * N;
    int length_Ns2 = vLength * nOutputs * bitSize * Ns2;
    int gridSize_complex = (int)ceil((float)(length_Ns2)/COMPLEX_BLOCKSIZE);

//    cout << "tGswFFTExternMulToTLwe_16_2_vector" << endl;
//    cout << vLength << " " << nOutputs << " " << bitSize << " " << gridSize_complex << " " << length_Ns2 << endl;

    tGswTorus32PolynomialDecompH_16_2_Coalesce_vector(decaCoalesce->coefs, accum, vLength, nOutputs, bitSize, params);
//    int *temp = new int[kpl * length];
//    cudaMemcpy(temp, decaCoalesce->coefs, kpl * length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length;
//        for (int j = 0; j < 2 * bitSize; ++j) {
//            int sI2 = sI + j * N;
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[sI2 + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//
//    cout << "XOR PART" << endl;
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length + vLength * bitSize * N;;
//        for (int j = 0; j < 2 * bitSize; ++j) {
//            int sI2 = sI + j * N;
//            for (int m = 0; m < 10; ++m) {
//                cout << temp[sI2 + m] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }

    IntPolynomial_ifft_16_2_Coalesce_vector(cuDecaFFTCoalesce, vLength, bitSize, decaCoalesce);
    //test code
//    cufftDoubleComplex *temp = new cufftDoubleComplex[kpl * length_Ns2];
//    cudaMemcpy(temp, cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    cout << "Coalesce: " << endl;
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length_Ns2;
//        for (int j = 0; j < bitSize * 2; ++j) {
//            int sI2 = sI + j * Ns2;
//            for (int m = 0; m < 8; ++m) {
//                cout << "(" << temp[sI2 + m].x << "," << temp[sI2 + m].y << ") ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    cout << "XOR part: " << endl;
//    for (int i = 0; i < kpl; ++i) {
//        int sI = i * length_Ns2 + vLength * bitSize * Ns2;
//        for (int j = 0; j < bitSize * 2; ++j) {
//            int sI2 = sI + j * Ns2;
//            for (int m = 0; m < 8; ++m) {
//                cout << "(" << temp[sI2 + m].x << "," << temp[sI2 + m].y << ") ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
    for (int i = 0; i <= k; ++i) {
//        setComplexVectorToConstant<<<gridSize_complex, COMPLEX_BLOCKSIZE>>>(tmpa_gpu[i], 0., 0., length_Ns2);
        cudaMemset(tmpa_gpu[i], 0, length_Ns2 * sizeof(cufftDoubleComplex));
    }

//    cout << "tGswFFTExternMulToTLwe_16_2_vector:bitsize: " << bitSize << endl;
    tLweFFTAddMulRTo_gpu_coalesce_2_vector(tmpa_gpu, cuDecaFFTCoalesce, cudaBKiCoalesce, kpl, Ns2, vLength, nOutputs, bitSize, tlwe_params);
    //test code
//    cufftDoubleComplex *temp = new cufftDoubleComplex[length_Ns2];
//    cudaMemcpy(temp, tmpa_gpu[1], length_Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    cout << "AND PART: " << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = i * Ns2;
//        for (int j = 0; j < 10; ++j) {
//            cout << "(" << temp[sI + j].x << "," << temp[sI + j].y << ")" << " ";
//        }
//        cout << endl;
//    }
//    cout << endl << "XOR PART: " << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = i * Ns2 + vLength * bitSize * Ns2;
//        for (int j = 0; j < 10; ++j) {
//            cout << "(" << temp[sI + j].x << "," << temp[sI + j].y << ")" << " ";
//        }
//        cout << endl;
//    }
    tLweFromFFTConvert_gpu_2_vector(accum, tmpa_gpu, vLength, nOutputs, bitSize, N, Ns2, tlwe_params);
//    int *temp = new int[length];
//    cudaMemcpy(temp, (accum->a + 0)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = i * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << "XOR PART" << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = i * N + vLength * bitSize * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
}

// result = (X^ai -1)*bki  
/*
//This function is not used, but may become handy in a future release
//
EXPORT void tGswFFTMulByXaiMinusOne(TGswSampleFFT* result, const int ai, const TGswSampleFFT* bki, const TGswParams* params) {
    const TLweParams* tlwe_params=params->tlwe_params;
    const int k = tlwe_params->k;
    //const int l = params->l;
    const int kpl = params->kpl;
    const int N = tlwe_params->N;
    //on calcule x^ai-1 en fft
    //TODO attention, this prevents parallelization...
    //TODO: parallelization
    static LagrangeHalfCPolynomial* xaim1=new_LagrangeHalfCPolynomial(N);
    LagrangeHalfCPolynomialSetXaiMinusOne(xaim1,ai);
    for (int p=0; p<kpl; p++) {
        const LagrangeHalfCPolynomial* in_s = bki->all_samples[p].a;
        LagrangeHalfCPolynomial* out_s = result->all_samples[p].a;
        for (int j=0; j<=k; j++)
            LagrangeHalfCPolynomialMul(&out_s[j], xaim1, &in_s[j]); 
    }
}
*/

//-------------------------------------------------------------------------------------
// autogenerated memory-related functions
//-------------------------------------------------------------------------------------

USE_DEFAULT_CONSTRUCTOR_DESTRUCTOR_IMPLEMENTATIONS1(TGswSampleFFT, TGswParams);

//
//----------------------------------------------------------------------------------------







#if 0
// BOOTSTRAPPING (as in CGGI16b - algo 3)
//  - modswitch: torus coefs multiplied by N/2
//  - set the test vector
//  - blind rotation by the phase
//  - sample extract 
//  - keyswitch
EXPORT void tfhe_bootstrapFFT(LweSample* result, const LweBootstrappingKeyFFT* bk, Torus32 mu1, Torus32 mu0, const LweSample* x){
    const Torus32 ab=(mu1+mu0)/2;
    const Torus32 aa = mu0-ab; // aa=(mu1-mu0)/2;
    const TGswParams* bk_params = bk->bk_params;
    const TLweParams* accum_params = bk_params->tlwe_params;
    const LweParams* extract_params = &accum_params->extracted_lweparams;
    const LweParams* in_out_params = bk->in_out_params;
    const int n=in_out_params->n;
    const int N=accum_params->N;
    const int Ns2=N/2;
    const int Nx2= 2*N;
    

    // Set the test vector (aa + aaX + ... + aaX^{N/2-1} -aaX^{N/2} - ... -aaX^{N-1})*X^{b}
    TorusPolynomial* testvect=new_TorusPolynomial(N);
    TorusPolynomial* testvectbis=new_TorusPolynomial(N);

    int barb=modSwitchFromTorus32(x->b,Nx2);
    //je definis le test vector (multiplié par a inclus !
    for (int i=0;i<Ns2;i++)
       testvect->coefsT[i]=aa;
    for (int i=Ns2;i<N;i++)
       testvect->coefsT[i]=-aa;
    torusPolynomialMulByXai(testvectbis, barb, testvect);



    // Accumulateur acc = fft((0, testvect))
    TLweSample* acc = new_TLweSample(accum_params);

    // acc will be used for tfhe_bootstrapFFT, acc1=acc will be used for tfhe_bootstrap
    tLweNoiselessTrivial(acc, testvectbis, accum_params);

    TGswSample* temp = new_TGswSample(bk_params);
    TGswSampleFFT* tempFFT = new_TGswSampleFFT(bk_params);


    // Blind rotation
//NICOLAS: j'ai ajouté ce bloc
#ifndef NDEBUG
    TorusPolynomial* phase = new_TorusPolynomial(N);
    int correctOffset = barb;
//    cout << "starting the test..." << endl;
#endif
    // the index 1 is given when we don't use the fft
    for (int i=0; i<n; i++) {
        int bara=modSwitchFromTorus32(-x->a[i],Nx2);
        
        if (bara!=0) {
            tGswFFTMulByXaiMinusOne(tempFFT, bara, bk->bkFFT+i, bk_params);
            tGswFFTAddH(tempFFT, bk_params);
            tGswFFTExternMulToTLwe(acc, tempFFT, bk_params);
        }

//NICOLAS: et surtout, j'ai ajouté celui-ci!
#ifndef NDEBUG
    tLwePhase(phase,acc,debug_accum_key);  //celui-ci, c'est la phase de acc (FFT)
    if (debug_in_key->key[i]==1) correctOffset = (correctOffset+bara)%Nx2; 
        torusPolynomialMulByXai(testvectbis, correctOffset, testvect); //celui-ci, c'est la phase idéale (calculée sans bruit avec la clé privée)
    for (int j=0; j<N; j++) {
           printf("Iteration %d, index %d: phase %d vs noiseless %d\n",i,j,phase->coefsT[j], testvectbis->coefsT[j]);
    }
#endif

    }


    // Sample extract
    LweSample* u = new_LweSample(extract_params);
    tLweExtractLweSample(u, acc, extract_params, accum_params);
    u->b += ab;
    

    // KeySwitching
    lweKeySwitch(result, bk->ks, u);
    


    delete_LweSample(u);
    delete_TGswSampleFFT(tempFFT); 
    delete_TGswSample(temp);
    delete_TLweSample(acc);
    delete_TorusPolynomial(testvectbis);
    delete_TorusPolynomial(testvect);
}
#endif

#undef INCLUDE_ALL

//old bki
//    cufftDoubleComplex ***cdGSWAllSamples = new cufftDoubleComplex **[kpl];
//    for (int i = 0; i < kpl; ++i) {
//        cdGSWAllSamples[i] = new cufftDoubleComplex *[k + 1];
//        for (int j = 0; j <= k; ++j) {
//            cudaMalloc(&(cdGSWAllSamples[i][j]), bitSize * Ns2 * sizeof(cufftDoubleComplex));
//            cufftDoubleComplex *temp_host_copy = new cufftDoubleComplex[Ns2 * bitSize];
//
//            for (int m = 0; m < Ns2; ++m) {
//                temp_host_copy[m].x = ((LagrangeHalfCPolynomial_IMPL *) ((gsw->all_samples + i)->a +
//                                                                         j))->coefsC[m].real();
//                temp_host_copy[m].y = ((LagrangeHalfCPolynomial_IMPL *) ((gsw->all_samples + i)->a +
//                                                                         j))->coefsC[m].imag();
//
//            }
//            //duplicate the copy
//            for (int n = 1; n < bitSize; ++n) { //because this has been copied once
//                int sI = n * Ns2;
//                memcpy(temp_host_copy + sI, temp_host_copy, Ns2 * sizeof(cufftDoubleComplex));
//            }
//            cudaMemcpy(cdGSWAllSamples[i][j], temp_host_copy, bitSize * Ns2 * sizeof(cufftDoubleComplex),
//                       cudaMemcpyHostToDevice);
//
//            //testing new integration starts
//            cufftDoubleComplex *temp_cudaBKi_testing = new cufftDoubleComplex[bitSize * Ns2];
//            cudaMemcpy(temp_cudaBKi_testing, cudaBKi[i][j], bitSize * Ns2 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//            for (int a = 0; a < bitSize * Ns2; ++a) {
//                assert(temp_cudaBKi_testing[a].x == temp_host_copy[a].x);
//                assert(temp_cudaBKi_testing[a].y == temp_host_copy[a].y);
//            }
//            delete[] temp_cudaBKi_testing;
//            //testing new integration ends
//            delete temp_host_copy;
//        }
//    }