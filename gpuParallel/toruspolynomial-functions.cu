#include <cassert>
#include <cmath>
#include "tfhe_core.h"
#include "numeric_functions.h"
#include "polynomials.h"
#include <iostream>
using namespace std;

using namespace std;

// TorusPolynomial = 0
EXPORT void torusPolynomialClear(TorusPolynomial *result) {
    const int N = result->N;
//    cout << "Original N: torusPolynomialClear: " << N << endl;

    for (int i = 0; i < N; ++i) result->coefsT[i] = 0;
}

// TorusPolynomial = random
EXPORT void torusPolynomialUniform(TorusPolynomial *result) {
    const int N = result->N;
    Torus32 *x = result->coefsT;

    for (int i = 0; i < N; ++i)
        x[i] = uniformTorus32_distrib(generator);
}

// TorusPolynomial = TorusPolynomial
EXPORT void torusPolynomialCopy(
        TorusPolynomial *result,
        const TorusPolynomial *sample) {
    assert(result != sample);
    const int N = result->N;
    const Torus32 *__restrict s = sample->coefsT;
    Torus32 *__restrict r = result->coefsT;

    for (int i = 0; i < N; ++i) {
        r[i] = s[i];
    }
}

// TorusPolynomial + TorusPolynomial
EXPORT void torusPolynomialAdd(TorusPolynomial *result, const TorusPolynomial *poly1, const TorusPolynomial *poly2) {
    const int N = poly1->N;
    assert(result != poly1); //if it fails here, please use addTo
    assert(result != poly2); //if it fails here, please use addTo
    Torus32 *__restrict r = result->coefsT;
    const Torus32 *__restrict a = poly1->coefsT;
    const Torus32 *__restrict b = poly2->coefsT;

    for (int i = 0; i < N; ++i)
        r[i] = a[i] + b[i];
}

// TorusPolynomial += TorusPolynomial
EXPORT void torusPolynomialAddTo(TorusPolynomial *result, const TorusPolynomial *poly2) {
    const int N = poly2->N;
    Torus32 *r = result->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i)
        r[i] += b[i];
}

//EXPORT void torusPolynomialAddTo_16(TorusPolynomial *result, int bitSize, int N, const TorusPolynomial *poly2) {
//    const int N = poly2->N;
//    Torus32 *r = result->coefsT;
//    const Torus32 *b = poly2->coefsT;
//
////    for (int i = 0; i < N; ++i)
////        r[i] += b[i];
//    for (int i = startIndex; i < endIndex; ++i)
//        r[i] += b[i];
//}

__global__ void vectorAddToSelf(int * destination, const int * source, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < length) {
        destination[id] += source[id];
    }
}


EXPORT void torusPolynomialAddTo_gpu(TorusPolynomial *result, int bitSize, int N, const TorusPolynomial *poly2) {
//    const int N = poly2->N;
    int *r = result->coefsT;
    const int *b = poly2->coefsT;
    int length = N * bitSize;
    int BLOCKSIZE  = 1024;
    int gridSize = (int)ceil((float)(N*bitSize)/BLOCKSIZE);

    vectorAddToSelf<<<gridSize, BLOCKSIZE>>>(r, b, length);
//    for (int i = 0; i < N; ++i)
//        r[i] += b[i];
//    for (int i = startIndex; i < endIndex; ++i)
//        r[i] += b[i];
}

EXPORT void torusPolynomialAddTo_gpu_2(TorusPolynomial *result, int nOutputs, int bitSize, int N, const TorusPolynomial *poly2) {
//    const int N = poly2->N;
    int *r = result->coefsT;
    const int *b = poly2->coefsT;
    int length = nOutputs * bitSize * N;
    int BLOCKSIZE  = 1024;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);

    vectorAddToSelf<<<gridSize, BLOCKSIZE>>>(r, b, length);
//    for (int i = 0; i < N; ++i)
//        r[i] += b[i];
//    for (int i = startIndex; i < endIndex; ++i)
//        r[i] += b[i];
}

EXPORT void torusPolynomialAddTo_gpu_2_vector(TorusPolynomial *result, int vLength, int nOutputs, int bitSize, int N, const TorusPolynomial *poly2) {
//    const int N = poly2->N;
    int *r = result->coefsT;
    const int *b = poly2->coefsT;
    int length = vLength * nOutputs * bitSize * N;
    int BLOCKSIZE  = 1024;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);

    vectorAddToSelf<<<gridSize, BLOCKSIZE>>>(r, b, length);
//    for (int i = 0; i < N; ++i)
//        r[i] += b[i];
//    for (int i = startIndex; i < endIndex; ++i)
//        r[i] += b[i];
}


// TorusPolynomial - TorusPolynomial
EXPORT void torusPolynomialSub(TorusPolynomial *result, const TorusPolynomial *poly1, const TorusPolynomial *poly2) {
    const int N = poly1->N;
    assert(result != poly1); //if it fails here, please use subTo
    assert(result != poly2); //if it fails here, please use subTo
    Torus32 *__restrict r = result->coefsT;
    const Torus32 *a = poly1->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i)
        r[i] = a[i] - b[i];
}

// TorusPolynomial -= TorusPolynomial
EXPORT void torusPolynomialSubTo(TorusPolynomial *result, const TorusPolynomial *poly2) {
    const int N = poly2->N;
    Torus32 *r = result->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i)
        r[i] -= b[i];
}

// TorusPolynomial + p*TorusPolynomial
EXPORT void
torusPolynomialAddMulZ(TorusPolynomial *result, const TorusPolynomial *poly1, int p, const TorusPolynomial *poly2) {
    const int N = poly1->N;
    Torus32 *r = result->coefsT;
    const Torus32 *a = poly1->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i)
        r[i] = a[i] + p * b[i];
}

// TorusPolynomial += p*TorusPolynomial
EXPORT void torusPolynomialAddMulZTo(TorusPolynomial *result, const int p, const TorusPolynomial *poly2) {
    const int N = poly2->N;
    Torus32 *r = result->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i) r[i] += p * b[i];
}

// TorusPolynomial - p*TorusPolynomial
EXPORT void torusPolynomialSubMulZ(TorusPolynomial *result, const TorusPolynomial *poly1, const int p,
                                   const TorusPolynomial *poly2) {
    const int N = poly1->N;
    Torus32 *r = result->coefsT;
    const Torus32 *a = poly1->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i) r[i] = a[i] - p * b[i];
}
/**
 *
 * @param result : result
 * @param a : barai
 * @param source : accum
 */
//result= (X^{a}-1)*source
EXPORT void torusPolynomialMulByXaiMinusOne(TorusPolynomial *result, int a, const TorusPolynomial *source) {
    const int N = source->N;
    Torus32 *out = result->coefsT;
    Torus32 *in = source->coefsT;

    assert(a >= 0 && a < 2 * N);



    if (a < N) {
        for (int i = 0; i < a; i++)//sur que i-a<0
            out[i] = -in[i - a + N] - in[i];
        for (int i = a; i < N; i++)//sur que N>i-a>=0
            out[i] = in[i - a] - in[i];
    } else {
        const int aa = a - N;
        for (int i = 0; i < aa; i++)//sur que i-a<0
            out[i] = in[i - aa + N] - in[i];
        for (int i = aa; i < N; i++)//sur que N>i-a>=0
            out[i] = -in[i - aa] - in[i];
    }
    //check intput
//    static int counter = 0;
//    int offset = 1000;
//    if (counter >= offset && counter < offset + 20) {
//        cout << "old input: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << source->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
//    counter++;

    //check output
//    static int counter = 0;
//    int offset = 1000;
//    if (counter >= offset && counter < offset + 20) {
//        cout << "old: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << result->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
//    counter++;
}

//new
__global__ void torusPolynomialMulByXaiMinusOne_16_GPU(int* destination, const int* bara, int baraIndex, int bitSize, int N, int* source) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < bitSize * N) {
        int bitIndex = id / N;
        int startIndex = bitIndex * N;
        int a = bara[startIndex + baraIndex];
        int threadIdModN = id % N;
        if (a < N) {
            if(threadIdModN < a) {
                destination[id] = -source[id - a + N] - source[id];
            } else {
                destination[id] = source[id - a] - source[id];
            }
        } else {
            const int aa = a - N;
            if(threadIdModN < aa) {
                destination[id] = source[id - aa + N] - source[id];
            } else {
                destination[id] = -source[id - aa] - source[id];
            }
        }
    }

}

__global__ void torusPolynomialMulByXaiMinusOne_16_GPU_2(int* destination, const int* bara, int baraIndex,
                                                         int nOutputs, int bitSize, int N, int* source) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int length = nOutputs * bitSize * N;
    if(id < length) {
        //bibel
        int bitIndex = id / N;
        int startIndex = bitIndex * N;
        int a = bara[startIndex + baraIndex];
        int threadIdModN = id % N;
        if (a < N) {
            if(threadIdModN < a) {
                destination[id] = -source[id - a + N] - source[id];
            } else {
                destination[id] = source[id - a] - source[id];
            }
        } else {
            int aa = a - N;
            if(threadIdModN < aa) {
                destination[id] = source[id - aa + N] - source[id];
            } else {
                destination[id] = -source[id - aa] - source[id];
            }
        }
//        testing for v2
//        destination[id] = bitIndex;
    }
}


__global__ void torusPolynomialMulByXaiMinusOne_16_GPU_2v2(int* destination, const int* bara, int baraIndex,
                                                         int nOutputs, int bitSize, int N, int k, int* source) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int length = nOutputs * bitSize * N *  (k + 1);
    if(id < length) {
        int bitIndex = (id / N) % (bitSize *nOutputs);
        int startIndex = bitIndex * N;
        int a = bara[startIndex + baraIndex];
        int threadIdModN = id % N;
        if (a < N) {
            if(threadIdModN < a) {
                destination[id] = -source[id - a + N] - source[id];
            } else {
                destination[id] = source[id - a] - source[id];
            }
        } else {
            int aa = a - N;
            if(threadIdModN < aa) {
                destination[id] = source[id - aa + N] - source[id];
            } else {
                destination[id] = -source[id - aa] - source[id];
            }
        }
//        __syncthreads();
    }
}

__global__ void torusPolynomialMulByXaiMinusOne_16_GPU_2_vector(int* destination, const int* bara, int baraIndex,
                                                           int nOutputs, int vLength, int bitSize, int N, int* source) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int length = nOutputs * vLength * bitSize * N;
    if(id < length) {
        int bitIndex = (id / N) % (bitSize *nOutputs * vLength);
        int startIndex = bitIndex * N;
        int a = bara[startIndex + baraIndex];
        int threadIdModN = id % N;
        if (a < N) {
            if(threadIdModN < a) {
                destination[id] = -source[id - a + N] - source[id];
            }
            __syncthreads();
            if(threadIdModN >= a){
                destination[id] = source[id - a] - source[id];
            }
            __syncthreads();
        }
        __syncthreads();
        if ((a >= N)) {
            int aa = a - N;
            if(threadIdModN < aa) {
                destination[id] = source[id - aa + N] - source[id];
            }
            __syncthreads();
            if(threadIdModN >= aa) {
                destination[id] = -source[id - aa] - source[id];
            }
            __syncthreads();
        }
    }
}

EXPORT void torusPolynomialMulByXaiMinusOne_16(TorusPolynomial *result, const int *bara, int baraIndex, int bitSize,
                                               int N, const TorusPolynomial *source) {

//    cout << "new: ";
//    for (int i = startIndex; i < startIndex + 10 ; ++i) {
//        cout << source->coefsT[i] << " ";
//    }
//    cout << endl;
//    Torus32 *out = result->coefsT;
    int *out = result->coefsT;
//    Torus32 *in = source->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int gridSize = (int)ceil((float)(N*bitSize)/BLOCKSIZE);

    torusPolynomialMulByXaiMinusOne_16_GPU<<<gridSize, BLOCKSIZE>>>(out, bara, baraIndex, bitSize, N, in);
//    cudaDeviceSynchronize();

    //input
//    int *temp_a = new int[N*bitSize];
//    cudaMemcpy(temp_a, result->coefsT, N*bitSize* sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_a[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    static int counter = 0;
//    int bitIndex = 1;
//    int startIndex = bitIndex*N;
//    if (counter < 20) {
//        cout << "new input: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[startIndex + i] << " ";
//        }
//        cout << endl;
//    }
//    counter++;

    //output
//    int *temp_a = new int[N*bitSize];
//    cudaMemcpy(temp_a, result->coefsT, N*bitSize* sizeof(int), cudaMemcpyDeviceToHost);
//    static int counter = 0;
//    int bitIndex = 1;
//    int startIndex = bitIndex*N;
//    if (counter < 20) {
//        cout << "new: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[startIndex + i] << " ";
//        }
//        cout << endl;
//    }
//    counter++;
}


EXPORT void torusPolynomialMulByXaiMinusOne_16_2(TorusPolynomial *result, const int *bara, int baraIndex, int nOutputs,
                                                 int bitSize, int N, const TorusPolynomial *source) {


    int *out = result->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int length = nOutputs * bitSize * N;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);
//    cout << "gridSize: " << gridSize << endl;
    torusPolynomialMulByXaiMinusOne_16_GPU_2<<<gridSize, BLOCKSIZE>>>(out, bara, baraIndex, nOutputs, bitSize, N, in);
/*
    //input
//    int *temp_a = new int[N*bitSize];
//    cudaMemcpy(temp_a, source->coefsT, N*bitSize* sizeof(int), cudaMemcpyDeviceToHost);
//    static int counter = 0;
//    int bitIndex = 1;
//    int startIndex = bitIndex*N;
//    if (counter < 20) {
//        cout << "new input: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[startIndex + i] << " ";
//        }
//        cout << endl;
//    }
//    counter++;


    //output
//    int *temp_a = new int[length];
//    cudaMemcpy(temp_a, result->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nOutputs * bitSize; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_a[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    static int counter = 0;
//    int bitIndex = 1;
//    int startIndex = bitIndex*N;
//    if (counter < 20) {
//        cout << "new: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[startIndex + i] << " ";
//        }
//        cout << endl;
//    }
//    counter++;*/
}

EXPORT void torusPolynomialMulByXaiMinusOne_16_2v2(TorusPolynomial *resultV2, const int *bara, int baraIndex, int nOutputs,
                                                 int bitSize, int N, const TorusPolynomial *source) {


    int *out = resultV2->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int k = 1;
    int length = nOutputs * bitSize * N * (k + 1);
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);

    torusPolynomialMulByXaiMinusOne_16_GPU_2v2<<<gridSize, BLOCKSIZE>>>(out, bara, baraIndex, nOutputs, bitSize, N, k, in);
}

EXPORT void torusPolynomialMulByXaiMinusOne_16_2_vector(TorusPolynomial *result, const int *bara, int baraIndex,
                                                        int vLength, int nOutputs, int bitSize, int N,
                                                        const TorusPolynomial *source) {


    int *out = result->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int length = vLength * nOutputs * bitSize * N;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);
    torusPolynomialMulByXaiMinusOne_16_GPU_2_vector<<<gridSize, BLOCKSIZE>>>(out, bara, baraIndex, nOutputs, vLength, bitSize, N, in);
}




//result= X^{a}*source
EXPORT void torusPolynomialMulByXai(TorusPolynomial *result, int a, const TorusPolynomial *source) {
    const int N = source->N;
    Torus32 *out = result->coefsT;
    Torus32 *in = source->coefsT;

    assert(a >= 0 && a < 2 * N);
    assert(result != source);

    if (a < N) {
        for (int i = 0; i < a; i++)//sur que i-a<0
            out[i] = -in[i - a + N];
        for (int i = a; i < N; i++)//sur que N>i-a>=0
            out[i] = in[i - a];
    } else {
        const int aa = a - N;
        for (int i = 0; i < aa; i++)//sur que i-a<0
            out[i] = in[i - aa + N];
        for (int i = aa; i < N; i++)//sur que N>i-a>=0
            out[i] = -in[i - aa];
    }
    //test morshed start
//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << out[i] << " ";
//    }
//    cout << endl;
    //test morshed end
}
/**
 *
 * @param destination
 * @param N
 * @param _2N
 * @param barb
 * @param bitSize
 * @param source
 * @return
 */
//new
__global__ void torusPolynomialMulByXai_16_GPU(int *destination, int N, int _2N, const int *barb, int bitSize, int *source) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int bIndex = id / N;
    int baraIndex = id % N;
    if (id < (bitSize * N)) {
//        destination[id] = 1;
        int a = _2N - barb[bIndex];

        if (a < N) {
            if (baraIndex < a) {
                destination[id] = -source[id - a + N];
            } else {
                destination[id] = source[id - a];
            }
        } else {
            const int aa = a - N;
            if (baraIndex < aa) {
                destination[id] = source[id - aa + N];
            } else {
                destination[id] =  -source[id - aa];
            }
        }
    }
}

__global__ void torusPolynomialMulByXai_16_GPU_2(int *destination, int N, int _2N, const int *barb,
                                                 int nOutputs, int bitSize, int *source) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int bIndex = id / N;
    int baraIndex = id % N;
    if (id < nOutputs * bitSize * N) {
        int a = _2N - barb[bIndex];

        if (a < N) {
            if (baraIndex < a) {
                destination[id] = -source[id - a + N];//barb[bIndex];//
            } else {
                destination[id] = source[id - a];//barb[bIndex];//
            }
        } else {
            const int aa = a - N;
            if (baraIndex < aa) {
                destination[id] = source[id - aa + N];//barb[bIndex];
            } else {
                destination[id] =  -source[id - aa];//barb[bIndex] ;//
            }
        }
    }
}

EXPORT void torusPolynomialMulByXai_16(TorusPolynomial *result, int _2N, const int *barb, int bitSize, const TorusPolynomial *source) {
    const int N = source->N/bitSize;
    int *out = result->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int gridSize = (int)ceil((float)(N * bitSize)/BLOCKSIZE);

    int *barb_GPU;
    cudaMalloc(&barb_GPU, bitSize * sizeof(int));
    cudaMemcpy(barb_GPU, barb, bitSize * sizeof(int), cudaMemcpyHostToDevice);

    torusPolynomialMulByXai_16_GPU<<<gridSize, BLOCKSIZE>>>(out, N, _2N, barb_GPU, bitSize, in);
    cudaFree(barb_GPU);
//    int *_tempBara = (int*)malloc(sizeof(int)*bitSize*N);
//    cudaMemcpy(_tempBara, out, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i*N;
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
}

EXPORT void torusPolynomialMulByXai_16_2(TorusPolynomial *result, int _2N, const int *barb,
                                         int nOutputs, int bitSize, const TorusPolynomial *source) {

    const int N = source->N/(bitSize * nOutputs); // 1024
    int *out = result->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int length = nOutputs * bitSize * N;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);//32

    int *barb_GPU;
    cudaMalloc(&barb_GPU, nOutputs * bitSize * sizeof(int));
    cudaMemcpy(barb_GPU, barb, nOutputs * bitSize * sizeof(int), cudaMemcpyHostToDevice);

    torusPolynomialMulByXai_16_GPU_2<<<gridSize, BLOCKSIZE>>>(out, N, _2N, barb_GPU, nOutputs, bitSize, in);
    cudaFree(barb_GPU);
//    int *_tempBara = new int[length];
//    cudaMemcpy(_tempBara, out, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * nOutputs; ++i) {
//        int startIndex = i * N;
////        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
}

EXPORT void torusPolynomialMulByXai_16_2_vector(TorusPolynomial *result, int _2N, const int *barb,
                                         int vLength, int nOutputs, int bitSize, const TorusPolynomial *source) {
//    cout << "torusPolynomialMulByXai_16_2_vector" << endl;
//    cout << "vLength: " << vLength << " nOutputs: " << nOutputs << " bitSize: " << bitSize << endl;
    const int N = source->N/(bitSize * nOutputs * vLength); // 1024
    int *out = result->coefsT;
    int *in = source->coefsT;
    int BLOCKSIZE = 1024;
    int length = vLength * nOutputs * bitSize * N;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);
//    cout << "gridSize: " << gridSize << endl;

    int *barb_GPU;
    cudaMalloc(&barb_GPU, vLength * nOutputs * bitSize * sizeof(int));
    cudaMemcpy(barb_GPU, barb, vLength * nOutputs * bitSize * sizeof(int), cudaMemcpyHostToDevice);
    torusPolynomialMulByXai_16_GPU_2<<<gridSize, BLOCKSIZE>>>(out, N, _2N, barb_GPU, nOutputs, bitSize * vLength, in);
    cudaFree(barb_GPU);
}


// TorusPolynomial -= p*TorusPolynomial
EXPORT void torusPolynomialSubMulZTo(TorusPolynomial *result, int p, const TorusPolynomial *poly2) {
    const int N = poly2->N;
    Torus32 *r = result->coefsT;
    const Torus32 *b = poly2->coefsT;

    for (int i = 0; i < N; ++i) r[i] -= p * b[i];
}


// Norme Euclidienne d'un IntPolynomial
EXPORT double intPolynomialNormSq2(const IntPolynomial *poly) {
    const int N = poly->N;
    int temp1 = 0;

    for (int i = 0; i < N; ++i) {
        int temp0 = poly->coefs[i] * poly->coefs[i];
        temp1 += temp0;
    }
    return temp1;
}

// Sets to zero
EXPORT void intPolynomialClear(IntPolynomial *poly) {
    const int N = poly->N;
    for (int i = 0; i < N; ++i)
        poly->coefs[i] = 0;
}

// Sets to zero
EXPORT void intPolynomialCopy(IntPolynomial *result, const IntPolynomial *source) {
    const int N = source->N;
    for (int i = 0; i < N; ++i)
        result->coefs[i] = source->coefs[i];
}

/** accum += source */
EXPORT void intPolynomialAddTo(IntPolynomial *accum, const IntPolynomial *source) {
    const int N = source->N;
    for (int i = 0; i < N; ++i)
        accum->coefs[i] += source->coefs[i];
}

/**  result = (X^ai-1) * source */
EXPORT void intPolynomialMulByXaiMinusOne(IntPolynomial *result, int ai, const IntPolynomial *source) {
    const int N = source->N;
    int *out = result->coefs;
    int *in = source->coefs;

    assert(ai >= 0 && ai < 2 * N);

    if (ai < N) {
        for (int i = 0; i < ai; i++)//sur que i-a<0
            out[i] = -in[i - ai + N] - in[i];
        for (int i = ai; i < N; i++)//sur que N>i-a>=0
            out[i] = in[i - ai] - in[i];
    } else {
        const int aa = ai - N;
        for (int i = 0; i < aa; i++)//sur que i-a<0
            out[i] = in[i - aa + N] - in[i];
        for (int i = aa; i < N; i++)//sur que N>i-a>=0
            out[i] = -in[i - aa] - in[i];
    }
}



// Norme infini de la distance entre deux TorusPolynomial
EXPORT double torusPolynomialNormInftyDist(const TorusPolynomial *poly1, const TorusPolynomial *poly2) {
    const int N = poly1->N;
    double norm = 0;

    // Max between the coefficients of abs(poly1-poly2)
    for (int i = 0; i < N; ++i) {
        double r = abs(t32tod(poly1->coefsT[i] - poly2->coefsT[i]));
        if (r > norm) { norm = r; }
    }
    return norm;
}






// Norme 2 d'un IntPolynomial
EXPORT double intPolynomialNorm2sq(const IntPolynomial *poly) {
    const int N = poly->N;
    double norm = 0;

    for (int i = 0; i < N; ++i) {
        double r = poly->coefs[i];
        norm += r * r;
    }
    return norm;
}

// Norme infini de la distance entre deux IntPolynomial
EXPORT double intPolynomialNormInftyDist(const IntPolynomial *poly1, const IntPolynomial *poly2) {
    const int N = poly1->N;
    double norm = 0;


    // Max between the coefficients of abs(poly1-poly2)
    for (int i = 0; i < N; ++i) {
        double r = abs(poly1->coefs[i] - poly2->coefs[i]);
        if (r > norm) { norm = r; }
    }
    return norm;
}


