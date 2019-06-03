//
// Created by morshed on 7/13/2018.
//

#ifndef V1_CUDAFFTMORSHED_H
#define V1_CUDAFFTMORSHED_H

#include <cassert>
#include <cmath>
#include <ccomplex>
typedef std::complex<double> cplx;
//#include "tfhe_core.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

typedef std::complex<double> cplx;

class cudaFFTProcessorTest {
public:
    int _2N = 2048;
    int N = 1024;
    int Ns2 = 512;
    int BATCH = 16;
    int bitSize = 16;
    int gridSize = bitSize;
    int BLOCKSIZE = 1024;
//private:
    cufftDoubleReal* d_rev_in;
    cufftDoubleComplex *d_rev_out;
    cufftDoubleComplex *d_in;
    cufftDoubleReal *d_out;

    cufftHandle p;
    cufftHandle rev_p;

public:
    cudaFFTProcessorTest(int bitSize);
    void execute_reverse_int(cufftDoubleComplex* res, const int* a);
    void execute_reverse_torus32(cplx *res, const int32_t *a);
    void execute_direct_Torus32(int32_t *res, const cplx *a);
    void execute_direct_Torus32_gpu(int32_t *res, cufftDoubleComplex *a);
    ~cudaFFTProcessorTest();
};

class cudaFFTProcessorTest_2 {
public:
    int _2N = 2048;
    int N = 1024;
    int Ns2 = 512;
    int BATCH = 32;
    int gridSize = BATCH;
    int BLOCKSIZE = 1024;
//private:
    cufftDoubleReal* d_rev_in;
    cufftDoubleComplex *d_rev_out;
    cufftDoubleComplex *d_in;
    cufftDoubleReal *d_out;

    cufftHandle p;
    cufftHandle rev_p;

public:
    cudaFFTProcessorTest_2(int bitSize);
    void execute_reverse_int(cufftDoubleComplex* res, const int* a);
    void execute_reverse_torus32(cplx *res, const int32_t *a);
    void execute_direct_Torus32(int32_t *res, const cplx *a);
    void execute_direct_Torus32_gpu(int32_t *res, cufftDoubleComplex *a);
    ~cudaFFTProcessorTest_2();
};

class cudaFFTProcessorTest_general {
public:
    int _2N;
    int N;
    int Ns2;
    int BATCH;
    int gridSize;
    int blockSize;
    int dParts;
//private:
    cufftDoubleReal* d_rev_in;
    cufftDoubleComplex *d_rev_out;
    cufftDoubleComplex *d_in;
    cufftDoubleReal *d_out;

    cufftHandle p;
    cufftHandle rev_p;

public:
    cudaFFTProcessorTest_general(int N, int BATCH, int blockSize);
    cudaFFTProcessorTest_general(int N, int nOutpus, int bitSize, int kpl, int blockSize, int dParts);
    cudaFFTProcessorTest_general(int N, int nOutpus, int vLen, int bitSize, int kpl, int blockSize, int dParts);
    void execute_reverse_int(cufftDoubleComplex* res, const int* a);
    void execute_reverse_torus32(cplx *res, const int32_t *a);
    void execute_direct_Torus32(int32_t *res, const cplx *a);
    void execute_direct_Torus32_gpu(int32_t *res, cufftDoubleComplex *a);
//    void execute_direct_Torus32_gpu(int32_t *out, cufftDoubleComplex *in, int dParts);
    ~cudaFFTProcessorTest_general();
};



#endif //V1_CUDAFFTMORSHED_H
