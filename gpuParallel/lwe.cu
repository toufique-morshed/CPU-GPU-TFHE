#include <cstdlib>
#include <iostream>
#include <random>
#include <cassert>
#include "tfhe_core.h"
#include "numeric_functions.h"
#include "lweparams.h"
#include "lwekey.h"
#include "lwesamples.h"
#include "lwe-functions.h"
#include "tlwe_functions.h"
#include "tgsw_functions.h"
#include "lwekeyswitch.h"
#include "lwebootstrappingkey.h"
#include "polynomials_arithmetic.h"
#include "lagrangehalfc_arithmetic.h"

using namespace std;


#ifndef NDEBUG
const TLweKey* debug_accum_key;
const LweKey* debug_extract_key;
const LweKey* debug_in_key;
#endif



//TODO: mettre les mêmes fonctions arithmétiques que pour Lwe
//      pour les opérations externes, prévoir int et intPolynomial


/*//calcule l'arrondi inférieur d'un élément Torus32
  int bar(uint64_t b, uint64_t Nx2){
  uint64_t xx=b*Nx2+(1l<<31);
  return (xx>>32)%Nx2;
  }*/



EXPORT void tLweExtractLweSampleIndex(LweSample* result, const TLweSample* x, const int index, const LweParams* params,  const TLweParams* rparams) {
    const int N = rparams->N;
    const int k = rparams->k;
//    cout << "old: rparams->N: " << N << endl;
//    cout << "old: rparams->k: " << k << endl;
//    cout << "old: params->n: " << params->n << endl;
    assert(params->n == k*N);

    for (int i=0; i<k; i++) {
      for (int j=0; j<=index; j++)
        result->a[i*N+j] = x->a[i].coefsT[index-j];
      for (int j=index+1; j<N; j++)
        result->a[i*N+j] = -x->a[i].coefsT[N+index-j];
    }
    result->b = x->b->coefsT[index];
}

//new
__global__ void extract_gpu(int *destination, int *source, int index, int i, int N, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        int bitIndex = id / N;
        int tIndex = id % N;//corresponding to j
        int startIndex = bitIndex * N;
        if (id % N == 0) {
            destination[id] = source[index - tIndex + startIndex];
        }
        else {
            destination[id] = -source[index - tIndex + startIndex + N];
        }
    }
}

EXPORT void tLweExtractLweSampleIndex_16(LweSample_16* result, const TLweSample* x, const int index, const LweParams* params,
                                         int bitSize, const TLweParams* rparams) {
    const int N = rparams->N;
    const int k = rparams->k;
    //test morshed start
//    cout << "rparams->N: " << rparams->N << endl;//1024
//    cout << "rparams->k: " << rparams->k << endl;//1
//    cout << "params->n: " << params->n << endl;//1024
//    cout << "rparams->extracted_lweparams.n: " << rparams->extracted_lweparams.n << endl; //1024
//    cout << "x->a->N: " << x->a->N << endl;//16*1024
//    cout << "x->b->N: " << x->b->N << endl;//16*1024
    //test morshed end
    assert(params->n == k * N);

    int BLOCKSIZE = 1024;
    int gridSize = (int) ceil((float) (N * bitSize) / BLOCKSIZE);
    int length = bitSize * N;
    for (int i = 0; i < k; ++i) {
        extract_gpu<<<gridSize, BLOCKSIZE>>>(result->a, x->a[i].coefsT, index, i, N, length);
    }

    //copy x->b->coefs from gpu to cpu
    int *x_b_coefs = new int[x->b->N];
    cudaMemcpy(x_b_coefs, x->b->coefsT, sizeof(int) * bitSize * N, cudaMemcpyDeviceToHost);
    for (int j = 0; j < bitSize; ++j) {
        result->b[j] = x_b_coefs[j * N];
    }

    int error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Cuda error code testing here: %d\n", error);
        exit(1);
    }
    delete[] x_b_coefs;
//    for (int z = 0; z < bitSize; ++z) {
//        int startIndex = z * N;
//        int endIndex = z * N + N;
//        for (int i = 0; i < k; ++i) {
//            for (int j = 0; j <= index; ++j) {
//                result->a[i*N+j + startIndex] = x->a[i].coefsT[index-j + startIndex];
//            }
//            for (int j = index + 1; j < N; ++j) {
//                result->a[i*N+j + startIndex] = -x->a[i].coefsT[N+index-j + startIndex];
//            }
//        }
//        result->b[z] = x->b->coefsT[z * N];
//
//    }

//    for (int i=0; i<k; i++) {
//        for (int j=0; j<=index; j++)
//            result->a[i*N+j] = x->a[i].coefsT[index-j];
//        for (int j=index+1; j<N; j++)
//            result->a[i*N+j] = -x->a[i].coefsT[N+index-j];
//    }
//    result->b = x->b->coefsT[index];
}

EXPORT void tLweExtractLweSampleIndex_16_2(LweSample_16* result, const TLweSample* x, const int index, const LweParams* params,
                                         int nOutputs, int bitSize, const TLweParams* rparams) {


    const int N = rparams->N;
    const int k = rparams->k;
    //test morshed start
//    cout << "rparams->N: " << rparams->N << endl;//1024
//    cout << "rparams->k: " << rparams->k << endl;//1
//    cout << "params->n: " << params->n << endl;//1024
//    cout << "rparams->extracted_lweparams.n: " << rparams->extracted_lweparams.n << endl; //1024
//    cout << "x->a->N: " << x->a->N << endl;//16*1024
//    cout << "x->b->N: " << x->b->N << endl;//16*1024
    //test morshed end
    assert(params->n == k * N);
    int BLOCKSIZE = 1024;
    int length = nOutputs * bitSize * N;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    for (int i = 0; i < k; ++i) {
        extract_gpu<<<gridSize, BLOCKSIZE>>>(result->a, x->a[i].coefsT, index, i, N, length);
    }

    //copy x->b->coefs from gpu to cpu
    int *x_b_coefs = new int[length];
    cudaMemcpy(x_b_coefs, x->b->coefsT, sizeof(int) * length, cudaMemcpyDeviceToHost);
    for (int j = 0; j < nOutputs * bitSize; ++j) {
        result->b[j] = x_b_coefs[j * N];
    }

    int error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Cuda error code testing here: %d\n", error);
        exit(1);
    }
    delete[] x_b_coefs;
//    for (int z = 0; z < bitSize; ++z) {
//        int startIndex = z * N;
//        int endIndex = z * N + N;
//        for (int i = 0; i < k; ++i) {
//            for (int j = 0; j <= index; ++j) {
//                result->a[i*N+j + startIndex] = x->a[i].coefsT[index-j + startIndex];
//            }
//            for (int j = index + 1; j < N; ++j) {
//                result->a[i*N+j + startIndex] = -x->a[i].coefsT[N+index-j + startIndex];
//            }
//        }
//        result->b[z] = x->b->coefsT[z * N];
//
//    }

//    for (int i=0; i<k; i++) {
//        for (int j=0; j<=index; j++)
//            result->a[i*N+j] = x->a[i].coefsT[index-j];
//        for (int j=index+1; j<N; j++)
//            result->a[i*N+j] = -x->a[i].coefsT[N+index-j];
//    }
//    result->b = x->b->coefsT[index];
}

EXPORT void tLweExtractLweSampleIndex_16_2_vector(LweSample_16* result, const TLweSample* x, const int index, const LweParams* params,
                                                  int vLength, int nOutputs, int bitSize, const TLweParams* rparams) {
//    cout << "_______________" << endl;
//    cout << "tLweExtractLweSampleIndex_16_2_vector" << endl;
//    cout << "vLength: " << vLength << " nOutputs: " << nOutputs << " bitSize: " << bitSize << endl;
    const int N = rparams->N;
    const int k = rparams->k;
    assert(params->n == k * N);
    int BLOCKSIZE = 1024;
    int length = vLength * nOutputs * bitSize * N;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "++++++++++++++++++" << endl;
    for (int i = 0; i < k; ++i) {
        extract_gpu<<<gridSize, BLOCKSIZE>>>(result->a, x->a[i].coefsT, index, i, N, length);
    }

//    cout << "++++++++++++++++++" << endl;
//    cout << "x->b->N: " << x->b->N << " length: " << length << endl;
    //copy x->b->coefs from gpu to cpu
    int *x_b_coefs = new int[length];
    cudaMemcpy(x_b_coefs, x->b->coefsT, sizeof(int) * length, cudaMemcpyDeviceToHost);
    for (int j = 0; j < vLength * nOutputs * bitSize; ++j) {
        result->b[j] = x_b_coefs[j * N];
    }

    int error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "I am here ERROR ERROR: %d\n", error);
        fprintf(stderr, "Cuda error code testing here: %d\n", error);
        exit(1);
    }
    delete[] x_b_coefs;
}



EXPORT void tLweExtractLweSample(LweSample* result, const TLweSample* x, const LweParams* params,  const TLweParams* rparams) {
    tLweExtractLweSampleIndex(result, x, 0, params, rparams);
    //test morshed start
//    cout << "old: ";
//    for (int j = 0; j < 10; ++j) {
//        cout << result->a[j] << " ";
//    }
//    cout << endl;
//    cout << "old: " << result->b << endl;
    //test morshed end
}

//new
EXPORT void tLweExtractLweSample_16(LweSample_16* result, const TLweSample* x, const LweParams* params, int bitSize, const TLweParams* rparams) {
    tLweExtractLweSampleIndex_16(result, x, 0, params, bitSize, rparams);
    //test morshed start
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i * rparams->N;
//        int endIndex = i * rparams->N + rparams->N;
//        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << result->a[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    for (int i = 0; i < bitSize; ++i) {
//        cout << "new: " << result->b[i] << endl;
//    }
    //test morshed end
}

EXPORT void tLweExtractLweSample_16_2(LweSample_16* result, const TLweSample* x, const LweParams* params, int nOutputs, int bitSize, const TLweParams* rparams) {
    tLweExtractLweSampleIndex_16_2(result, x, 0, params, nOutputs, bitSize, rparams);
    //test morshed start
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i * rparams->N;
//        int endIndex = i * rparams->N + rparams->N;
//        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << result->a[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    for (int i = 0; i < bitSize; ++i) {
//        cout << "new: " << result->b[i] << endl;
//    }
    //test morshed end
}

EXPORT void tLweExtractLweSample_16_2_vector(LweSample_16* result, const TLweSample* x, const LweParams* params,
                                             int vLength, int nOutputs, int bitSize, const TLweParams* rparams) {
//    cout << "*****************************" << endl;
    tLweExtractLweSampleIndex_16_2_vector(result, x, 0, params, vLength, nOutputs, bitSize, rparams);
//    cout << "*****************************" << endl;
}




//extractions Ring Lwe -> Lwe
EXPORT void tLweExtractKey(LweKey* result, const TLweKey* key) //sans doute un param supplémentaire
{
    const int N = key->params->N;
    const int k = key->params->k;
    assert(result->params->n == k*N);
    for (int i=0; i<k; i++) {
	for (int j=0; j<N; j++)
	    result->key[i*N+j]=key->key[i].coefs[j];
    }
}

