//
// Created by morshed on 7/13/2018.
//
#include "cudaFFTTest.h"
#include <cufft.h>
#include <iostream>
#include <ctime>
#include <cassert>
#include <cmath>
#include <inttypes.h>
#include <stdio.h>
#include <fstream>
#include <cstdint>
#include <fstream>

//CUDA ERROR CHECK
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;
typedef std::complex<double> cplx;

//cuda functions
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void cudaCheckErrorCustom() {
    int error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: Cuda error code: %d\n", error);
        exit(1);
    }
}

__global__ void setComplexVectorTo(cufftDoubleComplex *destination, double real, double img, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id].x = real;
        destination[id].y = img;
    }
}

__global__ void execute_reverse_intHelper1(cufftDoubleReal *destination, const int *source, int N, int _2N, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / _2N;
        int tIndex = id % _2N;
        int startIndexSmall = bitIndex * N;
        if (tIndex < N) {
            destination[id] = source[startIndexSmall + tIndex] / 2.;
        } else {
            destination[id] = -source[startIndexSmall + tIndex - N] / 2.;
        }

//        destination[id] = source[startIndexSmall + tIndex];//
    }
}

__global__  void execute_reverse_intHelper2(cufftDoubleComplex *destination, cufftDoubleComplex *source, int N, int Ns2, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        int bitIndex = id/Ns2;
        destination[id] = source[2*id + 1 + bitIndex];
    }
}

__global__ void execute_direct_Torus32_gpu_helper1(cufftDoubleComplex *destination, cufftDoubleComplex *source, int N_p_1, int Ns2, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / Ns2;
        int tIndex = id % Ns2;
//        int startIndexSmall = bitIndex * Ns2;
        int startIndexLarge = bitIndex * N_p_1;
        destination[startIndexLarge + 2 * tIndex + 1] = source[id];
    }
}

__global__ void execute_direct_Torus32_gpu_helper2(int *destination, cufftDoubleReal *source, double _2p32, double _1sN, int N, int _2N, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int bitIndex = id / N;
        int tIndex = id % N;
        int startIndexLarge = bitIndex * _2N;
//        int startIndexSmall = bitIndex * N;
        destination[id] = int32_t(int64_t(source[startIndexLarge + tIndex] * _1sN * _2p32));//
    }
}

//FFT Processor functions
cudaFFTProcessorTest::cudaFFTProcessorTest(int bitSize) {
    //variable initialization
    N = 1024;
    _2N = N * 2;
    Ns2 = N / 2;
    BATCH = bitSize;//1 for 1 bit, 16 fot 16 bit, 32 for 32 bit
    BLOCKSIZE = 1024;
    //memory allocation
    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
    //plan
    cufftPlan1d(&p, _2N, CUFFT_Z2D, BATCH);
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, BATCH);
    //setup values
    int length = (N + 1) * BATCH;
    int gridSize = (int) ceil((float) ((N + 1) * BATCH) / BLOCKSIZE);
    setComplexVectorTo<<<gridSize, BLOCKSIZE>>>(d_in, 0., 0., length);
}

cudaFFTProcessorTest_2::cudaFFTProcessorTest_2(int bitSize) {
    //variable initialization
    _2N = 2048;
    N = 1024;
    Ns2 = 512;
    BATCH = bitSize * 2;//32 for 16 bit, 64 for 32 bit
    gridSize = BATCH;
    BLOCKSIZE = 1024;
    //memory allocation
    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
    //plan
    cufftPlan1d(&p, _2N, CUFFT_Z2D, BATCH);
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, BATCH);
    //setup values
    int length = (N + 1) * BATCH;
    int gridSize = (int) ceil((float) ((N + 1) * BATCH) / BLOCKSIZE);
    setComplexVectorTo<<<gridSize, BLOCKSIZE>>>(d_in, 0., 0., length);
}

cudaFFTProcessorTest_general::cudaFFTProcessorTest_general(int N, int BATCH, int blockSize) {
    this->N = N;
    this->Ns2 = N/2;
    this->_2N = N * 2;

    this->BATCH = BATCH;
    this->gridSize = BATCH;
    this->blockSize = blockSize;
    this->dParts = 4;

    cout << "constructing........." << endl;

    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, BATCH);// - (BATCH/dParts));//64 for 32 * 2, 48 for 24 bit
//change here change 2 to 4//this was previous

    BATCH = BATCH/dParts;
    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
    cufftPlan1d(&p, _2N, CUFFT_Z2D, BATCH);//_2N
    //setup values
    int length = (N + 1) * BATCH;
    int gridSize = (int) ceil((float) ((N + 1) * BATCH) / blockSize);
    setComplexVectorTo<<<gridSize, blockSize>>>(d_in, 0., 0., length);


}





cudaFFTProcessorTest_general::cudaFFTProcessorTest_general(int N, int nOutputs, int bitSize,
                                                                 int kpl, int blockSize, int dParts) {
    this->N = N;
    this->Ns2 = N/2;
    this->_2N = N * 2;

    int BATCH = nOutputs * bitSize * kpl;
    this->BATCH = BATCH;
    this->gridSize = BATCH;
    this->blockSize = blockSize;
    this->dParts = dParts;

    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, BATCH);//-bitSize);// - (nOutputs * bitSize));//64 for 32 * 2, 48 for 24 bit
//change here change 2 to 4//this was previous

    BATCH = BATCH/dParts;
    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
    cufftPlan1d(&p, _2N, CUFFT_Z2D, BATCH);//_2N
    //setup values
    int length = (N + 1) * BATCH;
    int gridSize = (int) ceil((float) ((N + 1) * BATCH) / blockSize);
    setComplexVectorTo<<<gridSize, blockSize>>>(d_in, 0., 0., length);
}



//this is for vector addition;
cudaFFTProcessorTest_general::cudaFFTProcessorTest_general(int N, int nOutputs, int vLen, int bitSize, int kpl, int blockSize, int dParts){
    this->N = N;
    this->Ns2 = N/2;
    this->_2N = N * 2;

    int BATCH = nOutputs * vLen * bitSize * kpl;
    this->BATCH = BATCH;
    this->gridSize = BATCH;
    this->blockSize = blockSize;
    this->dParts = dParts;

    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, BATCH - (nOutputs * bitSize * vLen));//64 for 32 * 2, 48 for 24 bit

//change here change 2 to 4//this was previous

    BATCH = BATCH/dParts;
    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
    cufftPlan1d(&p, _2N, CUFFT_Z2D, BATCH);//_2N
    //setup values
    int length = (N + 1) * BATCH;
    int gridSize = (int) ceil((float) ((N + 1) * BATCH) / blockSize);
    setComplexVectorTo<<<gridSize, blockSize>>>(d_in, 0., 0., length);
}

//cudaFFTProcessorTest_general::cudaFFTProcessorTest_general(int N, int nOutputs, int bitSize, int kpl, int blockSize) {
//    this->N = N;
//    this->Ns2 = N/2;
//    this->_2N = N * 2;
//    int BATCH = nOutputs * bitSize * kpl;
//    this->BATCH = BATCH;
//    this->gridSize = BATCH;
//    this->blockSize = blockSize;
//
//    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
//    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
//    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, BATCH - );//64 for 32 * 2
////change here change 2 to 4//this was previous
//
//    BATCH = BATCH/2;
//    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
//    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
//    cufftPlan1d(&p, _2N, CUFFT_Z2D, BATCH);//_2N
//    //setup values
//    int length = (N + 1) * BATCH;
//    int gridSize = (int) ceil((float) ((N + 1) * BATCH) / blockSize);
//    setComplexVectorTo<<<gridSize, blockSize>>>(d_in, 0., 0., length);
//}

void cudaFFTProcessorTest::execute_reverse_int(cufftDoubleComplex *out, const int *in) {
    int length = BATCH * _2N;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    execute_reverse_intHelper1<<<gridSize, BLOCKSIZE>>>(d_rev_in, in, N, _2N, length);
    cufftExecD2Z(rev_p, d_rev_in, d_rev_out);

    length = Ns2 * BATCH;
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    execute_reverse_intHelper2<<<gridSize, BLOCKSIZE>>>(out, d_rev_out, N, Ns2, length);
}


void cudaFFTProcessorTest_general::execute_reverse_int(cufftDoubleComplex *out, const int *in) {
//    cout << "XXXXXXXXXXXXXXXXXXXXXXXXXX" << endl;
//    cout << "BATCH iFFT" << this->BATCH << endl;
    int length = this->BATCH * _2N;
//    cout << "BATCH: " << this->BATCH << endl;
    int gridSize = (int) ceil((float) (length) / blockSize);
//    cout << "yyyyy" << endl;
//    cout << "fft grid: " << gridSize << endl;
//    cout << "gridSize: " << gridSize << endl;
//    cudaMalloc(&d_rev_in, sizeof(cufftDoubleReal) * _2N * BATCH);
//    cudaMalloc(&d_rev_out, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);

    execute_reverse_intHelper1<<<gridSize, blockSize>>>(d_rev_in, in, N, _2N, length);
    cufftExecD2Z(rev_p, d_rev_in, d_rev_out);//0.02
//    cudaDeviceSynchronize();


//    cout << "mmmmmmmmmmmmmmmBATCH: " << this->BATCH << endl;
//    ofstream myfile;
//    myfile.open ("halfGPU.txt", ios::out | ios::app);
//    static int counter = 0;
//    myfile << "j: " << counter << " output: ";
//    cufftDoubleReal *temp = new cufftDoubleReal[length];
//    cudaMemcpy(temp, d_rev_in, length * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->BATCH; ++i) {
//        int sI = i * _2N;
//        for (int j = 0; j < 10; ++j) {
////            myfile << temp[sI + j] << " ";
//            cout << temp[sI + j] << " ";
////            cout << "(" << temp[sI + j].x << ", " <<  temp[sI + j].y << ") ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;
//    cout << endl;
//    myfile.close();
//    counter++;
//    length = (N + 1) * this->BATCH;
//    cufftDoubleComplex *temp2 = new cufftDoubleComplex[length];
//    cudaMemcpy(temp2, d_rev_out, length * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->BATCH; ++i) {
//        int sI = i * (N + 1);
//        for (int j = 0; j < 10; ++j) {
////            cout << temp[sI + j] << " ";
//            cout << "(" << temp2[sI + j].x << "," <<  temp2[sI + j].y << ") ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;





    length = Ns2 * BATCH;
    gridSize = (int) ceil((float) (length) / blockSize);
//    cout << "blockSize: " << blockSize << endl;
//    cout << "gridSize: " << gridSize << endl;
    execute_reverse_intHelper2<<<gridSize, blockSize>>>(out, d_rev_out, N, Ns2, length);
//    cudaFree(d_rev_in);
//    cudaFree(d_rev_out);




}

void cudaFFTProcessorTest_2::execute_reverse_int(cufftDoubleComplex *out, const int *in) {
    int length = BATCH * _2N; //32 * 2048
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    execute_reverse_intHelper1<<<gridSize, BLOCKSIZE>>>(d_rev_in, in, N, _2N, length);
    cufftExecD2Z(rev_p, d_rev_in, d_rev_out);

    length = Ns2 * BATCH;//512 * 32
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    execute_reverse_intHelper2<<<gridSize, BLOCKSIZE>>>(out, d_rev_out, N, Ns2, length);
}

void cudaFFTProcessorTest::execute_reverse_torus32(cplx *res, const int32_t *a) {}

void cudaFFTProcessorTest::execute_direct_Torus32(int32_t *res, const cplx *a) {

//    static const double _2p32 = double(INT64_C(1) << 32);
//    static const double _1sN = double(1) / double(N);
//
//    for (int i = 0; i < Ns2; i++) {
//        h_in[2 * i + 1].x = a[i].real();
//        h_in[2 * i + 1].y = a[i].imag();
//    }
//
//    cudaMemcpy(d_in, h_in, sizeof(cufftDoubleComplex) * (N + 1), cudaMemcpyHostToDevice);
//    cufftExecZ2D(p, d_in, d_out);
////    cudaDeviceSynchronize();
//    cudaMemcpy(h_out, d_out, sizeof(cufftDoubleReal) * _2N, cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < N; i++) {
//        res[i] = int32_t(int64_t(h_out[i] * _1sN * _2p32));
//    }
}

void cudaFFTProcessorTest::execute_direct_Torus32_gpu(int32_t *out, cufftDoubleComplex *in) {
    static const double _2p32 = double(INT64_C(1) << 32);
    static const double _1sN = double(1) / double(N);

    int length = BATCH * Ns2;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    execute_direct_Torus32_gpu_helper1<<<gridSize, BLOCKSIZE>>>(d_in, in, (N + 1), Ns2, length);

    cufftExecZ2D(p, d_in, d_out);
//    cudaDeviceSynchronize();

    length = N * BATCH;
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    execute_direct_Torus32_gpu_helper2<<<gridSize, BLOCKSIZE>>>(out, d_out, _2p32, _1sN, N, _2N, length);
}

void cudaFFTProcessorTest_2::execute_direct_Torus32_gpu(int32_t *out, cufftDoubleComplex *in) {
    static const double _2p32 = double(INT64_C(1) << 32);
    static const double _1sN = double(1) / double(N);

    int length = BATCH * Ns2;//32 * 512
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    execute_direct_Torus32_gpu_helper1<<<gridSize, BLOCKSIZE>>>(d_in, in, (N + 1), Ns2, length);

    cufftExecZ2D(p, d_in, d_out);

    length = N * BATCH;
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    execute_direct_Torus32_gpu_helper2<<<gridSize, BLOCKSIZE>>>(out, d_out, _2p32, _1sN, N, _2N, length);
}

void cudaFFTProcessorTest_general::execute_direct_Torus32_gpu(int32_t *out, cufftDoubleComplex *in) {
//    cout << "here" << endl;
    static const double _2p32 = double(INT64_C(1) << 32);
    static const double _1sN = double(1) / double(N);
//change this one to 4//4 was previous code
    int BATCH = this->BATCH/dParts;
//    cout << "BATCH iFFT" << BATCH << endl;
//    cout << "BATCH: " << BATCH << endl;
    int length = BATCH * Ns2;
    int gridSize = (int) ceil((float) (length) / blockSize);
//    cout << "grid: " << gridSize << endl;
//    cout << "BLK: " << blockSize<< endl;
//    cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * (N + 1) * BATCH);
//    cudaMalloc(&d_out, sizeof(cufftDoubleReal) * _2N * BATCH);
//    cudaMemset(d_in, 0, (N + 1) * BATCH * sizeof(cufftDoubleComplex));
    execute_direct_Torus32_gpu_helper1<<<gridSize, blockSize>>>(d_in, in, (N + 1), Ns2, length);

//    cout << "xxxxxxxxxxxBBBBBBBBBAAAATCH: " << BATCH << endl;
//        length = (N + 1) * BATCH;
//    cufftDoubleComplex *temp = new cufftDoubleComplex[length];
//    cudaMemcpy(temp, d_in, length * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < BATCH; ++i) {
//        int sI = i * (N + 1);
//        for (int j = 0; j < 10; ++j) {
//            cout << "(" << temp[sI + j].x << "," <<  temp[sI + j].y << ") ";
//        }
//        cout << endl;
//    }
//    cout << endl;


    cufftExecZ2D(p, d_in, d_out);

    length = N * BATCH;
    gridSize = (int) ceil((float) (length) / blockSize);
//    cout << "len:-> "  << length << endl;
    execute_direct_Torus32_gpu_helper2<<<gridSize, blockSize>>>(out, d_out, _2p32, _1sN, N, _2N, length);


//    cudaFree(d_in);
//    cudaFree(d_out);
}

//destructors
cudaFFTProcessorTest::~cudaFFTProcessorTest() {
    cudaFree(d_rev_in);
    cudaFree(d_rev_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cufftDestroy(rev_p);
    cufftDestroy(p);
}

cudaFFTProcessorTest_2::~cudaFFTProcessorTest_2() {
    cudaFree(d_rev_in);
    cudaFree(d_rev_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cufftDestroy(rev_p);
    cufftDestroy(p);
}

cudaFFTProcessorTest_general::~cudaFFTProcessorTest_general() {
    cudaFree(d_rev_in);
    cudaFree(d_rev_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cufftDestroy(rev_p);
    cufftDestroy(p);
}