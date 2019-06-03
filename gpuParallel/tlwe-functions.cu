#include <cstdlib>
#include <iostream>
#include <random>
#include <cassert>
#include "tlwe_functions.h"
#include "numeric_functions.h"
#include "polynomials_arithmetic.h"
#include "lagrangehalfc_arithmetic.h"
#include "tfhe_generic_templates.h"

using namespace std;


// TLwe
EXPORT void tLweKeyGen(TLweKey *result) {
    const int N = result->params->N;
    const int k = result->params->k;
    uniform_int_distribution<int> distribution(0, 1);

    for (int i = 0; i < k; ++i)
        for (int j = 0; j < N; ++j)
            result->key[i].coefs[j] = distribution(generator);
}

/*create an homogeneous tlwe sample*/
EXPORT void tLweSymEncryptZero(TLweSample *result, double alpha, const TLweKey *key) {
    const int N = key->params->N;
    const int k = key->params->k;

    for (int j = 0; j < N; ++j)
        result->b->coefsT[j] = gaussian32(0, alpha);

    for (int i = 0; i < k; ++i) {
        torusPolynomialUniform(&result->a[i]);
        torusPolynomialAddMulR(result->b, &key->key[i], &result->a[i]);
    }

    result->current_variance = alpha * alpha;
}

EXPORT void tLweSymEncrypt(TLweSample *result, TorusPolynomial *message, double alpha, const TLweKey *key) {
    const int N = key->params->N;

    tLweSymEncryptZero(result, alpha, key);

    for (int j = 0; j < N; ++j)
        result->b->coefsT[j] += message->coefsT[j];
}

/**
 * encrypts a constant message
 */
EXPORT void tLweSymEncryptT(TLweSample *result, Torus32 message, double alpha, const TLweKey *key) {
    tLweSymEncryptZero(result, alpha, key);

    result->b->coefsT[0] += message;
}



/**
 * This function computes the phase of sample by using key : phi = b - a.s
 */
EXPORT void tLwePhase(TorusPolynomial *phase, const TLweSample *sample, const TLweKey *key) {
    const int k = key->params->k;

    torusPolynomialCopy(phase, sample->b); // phi = b

    for (int i = 0; i < k; ++i)
        torusPolynomialSubMulR(phase, &key->key[i], &sample->a[i]);
}


/**
 * This function computes the approximation of the phase 
 * à revoir, surtout le Msize
 */
EXPORT void tLweApproxPhase(TorusPolynomial *message, const TorusPolynomial *phase, int Msize, int N) {
    for (int i = 0; i < N; ++i) message->coefsT[i] = approxPhase(phase->coefsT[i], Msize);
}




EXPORT void tLweSymDecrypt(TorusPolynomial *result, const TLweSample *sample, const TLweKey *key, int Msize) {
    tLwePhase(result, sample, key);
    tLweApproxPhase(result, result, Msize, key->params->N);
}


EXPORT Torus32 tLweSymDecryptT(const TLweSample *sample, const TLweKey *key, int Msize) {
    TorusPolynomial *phase = new_TorusPolynomial(key->params->N);

    tLwePhase(phase, sample, key);
    Torus32 result = approxPhase(phase->coefsT[0], Msize);

    delete_TorusPolynomial(phase);
    return result;
}




//Arithmetic operations on TLwe samples
/** result = (0,0) */
EXPORT void tLweClear(TLweSample *result, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i) torusPolynomialClear(&result->a[i]);
    torusPolynomialClear(result->b);
    result->current_variance = 0.;
}


/** result = sample */
EXPORT void tLweCopy(TLweSample *result, const TLweSample *sample, const TLweParams *params) {
    const int k = params->k;
    const int N = params->N;

    for (int i = 0; i <= k; ++i)
        for (int j = 0; j < N; ++j)
            result->a[i].coefsT[j] = sample->a[i].coefsT[j];

    result->current_variance = sample->current_variance;
}



/** result = (0,mu) */
EXPORT void tLweNoiselessTrivial(TLweSample *result, const TorusPolynomial *mu, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i){
        torusPolynomialClear(&result->a[i]);
    }
    torusPolynomialCopy(result->b, mu);
    result->current_variance = 0.;
}

//new
EXPORT void tLweNoiselessTrivial_16(TLweSample *result, const TorusPolynomial *mu, const TLweParams *params) {
//    cout << " I am in another inside" << endl;
    const int k = params->k;
//    cout << "params->k: " << params->k << endl;
//    cout << "result->a->N: " << result->a->N << endl;
//    cout << "result->b->N: " << result->b->N << endl;
//    cout << "mu->N: " << mu->N << endl;

//    for (int i = 0; i < k; ++i){
//        torusPolynomialClear(&result->a[i]);
//    }
    for (int i = 0; i < k; ++i){
        torusPolynomialClear(&result->a[i]);
    }
    torusPolynomialCopy(result->b, mu);
    result->current_variance = 0.;
}

/** result = (0,mu) where mu is constant*/
EXPORT void tLweNoiselessTrivialT(TLweSample *result, const Torus32 mu, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i) torusPolynomialClear(&result->a[i]);
    torusPolynomialClear(result->b);
    result->b->coefsT[0] = mu;
    result->current_variance = 0.;
}

/** result = result + sample */
EXPORT void tLweAddTo(TLweSample *result, const TLweSample *sample, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i)
        torusPolynomialAddTo(&result->a[i], &sample->a[i]);
    //test morshed start
//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        int j = 0;
//        cout << (result->a + j)->coefsT[i] << " ";
//    }
//    cout << endl;
    //test morshed end
    torusPolynomialAddTo(result->b, sample->b);
    //test morshed start
//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << result->b->coefsT[i] << " ";
//    }
//    cout << endl;
    //test morshed end
    result->current_variance += sample->current_variance;
}

//new
EXPORT void tLweAddTo_16(TLweSample *result, const TLweSample *sample, int bitSize, int N, const TLweParams *params) {
    const int k = params->k;

//    cout << "tLweAddTo_16" << endl;
//    cout << "params->k: " << params->k << endl;
//    cout << "result->a->N: " << result->a->N << endl;
//    cout << "sample->a->N: " << sample->a->N << endl;
    for (int i = 0; i < k; ++i)
        torusPolynomialAddTo_gpu(&result->a[i], bitSize, N, &sample->a[i]);
    //test morshed start
//    cout << "new: ";
//    for (int i = 0; i < 10; ++i) {
//        int j = 0;
//        cout << (result->a + j)->coefsT[startIndex + i] << " ";
//    }
//    cout << endl;
    //test morshed end
    torusPolynomialAddTo_gpu(result->b, bitSize, N, sample->b);
//    //test morshed start
//    cout << "new: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << result->b->coefsT[startIndex + i] << " ";
//    }
//    cout << endl;
//    //test morshed end
    result->current_variance += sample->current_variance;
}

EXPORT void tLweAddTo_16_2(TLweSample *result, const TLweSample *sample, int nOutputs, int bitSize, int N, const TLweParams *params) {

    const int k = params->k;
//    cout << "QWEQWEQWEQWEQWEQWEQWEQWEQWEQWEQWE" << endl;
//    cout << "tLweAddTo_16" << endl;
//    cout << "params->k: " << params->k << endl;
//    cout << "result->a->N: " << result->a->N << endl;
//    cout << "sample->a->N: " << sample->a->N << endl;
    for (int i = 0; i < k; ++i)
        torusPolynomialAddTo_gpu_2(&result->a[i], nOutputs, bitSize, N, &sample->a[i]);
    //test morshed start
//    cout << "new: ";
//    for (int i = 0; i < 10; ++i) {
//        int j = 0;
//        cout << (result->a + j)->coefsT[startIndex + i] << " ";
//    }
//    cout << endl;
    //test morshed end
    torusPolynomialAddTo_gpu_2(result->b, nOutputs, bitSize, N, sample->b);
//    //test morshed start
//    cout << "new: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << result->b->coefsT[startIndex + i] << " ";
//    }
//    cout << endl;
//    //test morshed end
    result->current_variance += sample->current_variance;
}


__global__ void tlweVectorAddition(int *destination, int *source, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        destination[id] += source[id];
    }
}

EXPORT void tLweAddTo_16_2v2(TLweSample *result, const TLweSample *sample, int nOutputs, int bitSize, int N, const TLweParams *params) {

    const int k = params->k;
    int length = nOutputs * bitSize * N * (k + 1);
    int BLOCKSIZE = 1024;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "here";

    tlweVectorAddition<<<gridSize, BLOCKSIZE>>>(result->a->coefsT, sample->a->coefsT, length);
    result->current_variance += sample->current_variance;
}


EXPORT void tLweAddTo_16_2_vector(TLweSample *result, const TLweSample *sample, int vLength, int nOutputs, int bitSize,
                                  int N, const TLweParams *params) {

    const int k = params->k;
    for (int i = 0; i < k; ++i) {
        torusPolynomialAddTo_gpu_2_vector(&result->a[i], vLength, nOutputs, bitSize, N, &sample->a[i]);
    }
    torusPolynomialAddTo_gpu_2_vector(result->b, vLength, nOutputs, bitSize, N, sample->b);
    result->current_variance += sample->current_variance;
}

/** result = result - sample */
EXPORT void tLweSubTo(TLweSample *result, const TLweSample *sample, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i)
        torusPolynomialSubTo(&result->a[i], &sample->a[i]);
    torusPolynomialSubTo(result->b, sample->b);
    result->current_variance += sample->current_variance;
}

/** result = result + p.sample */
EXPORT void tLweAddMulTo(TLweSample *result, int p, const TLweSample *sample, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i)
        torusPolynomialAddMulZTo(&result->a[i], p, &sample->a[i]);
    torusPolynomialAddMulZTo(result->b, p, sample->b);
    result->current_variance += (p * p) * sample->current_variance;
}

/** result = result - p.sample */
EXPORT void tLweSubMulTo(TLweSample *result, int p, const TLweSample *sample, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i < k; ++i)
        torusPolynomialSubMulZTo(&result->a[i], p, &sample->a[i]);
    torusPolynomialSubMulZTo(result->b, p, sample->b);
    result->current_variance += (p * p) * sample->current_variance;
}


/** result = result + p.sample */
EXPORT void
tLweAddMulRTo(TLweSample *result, const IntPolynomial *p, const TLweSample *sample, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i <= k; ++i)
        torusPolynomialAddMulR(result->a + i, p, sample->a + i);
    result->current_variance += intPolynomialNormSq2(p) * sample->current_variance;
}


/**
 *
 * @param result : result
 * @param ai : barai
 * @param bk : accum
 * @param params : tlweParams
 */
//mult externe de X^ai-1 par bki/
EXPORT void tLweMulByXaiMinusOne(TLweSample *result, int ai, const TLweSample *bk, const TLweParams *params) {
    const int k = params->k;
//    static int counter = 0;
//    int offset = 500;
    for (int i = 0; i <= k; i++) {
        torusPolynomialMulByXaiMinusOne(&result->a[i], ai, &bk->a[i]);
//        if (counter >= offset && counter < offset + 10) {
//            cout << "old: ";
//            for (int j = 0; j < 10; ++j) {
//                cout << (&bk->a[i])->coefsT[j] << " ";
//            }
//            cout << endl;
//        }
    }
//    counter++;
}

//new
EXPORT void tLweMulByXaiMinusOne_16(TLweSample *result, const int* bara, int baraIndex, const TLweSample *bk, int bitSize, int N,
                                    const TLweParams *params) {
    const int k = params->k;
//    static int counter = 0;
    for (int i = 0; i <= k; i++) {
        torusPolynomialMulByXaiMinusOne_16(&result->a[i], bara, baraIndex, bitSize, N, &bk->a[i]);

//        int *temp_a = new int[bitSize*N];
//        cudaMemcpy(temp_a, result->a[i].coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//        if (counter < 10) {
//            int bI = 1;
//            int sI = bI * N;
//            cout << "new: ";
//            for (int l = 0; l < 10; ++l) {
//                cout << temp_a[sI + l] << " ";
//            }
//            cout << endl;
//        }
     }
//    counter++;
}

EXPORT void tLweMulByXaiMinusOne_16_2(TLweSample *result, const int* bara, int baraIndex, const TLweSample *bk,
                                    int nOutputs, int bitSize, int N, const TLweParams *params) {
    const int k = params->k;
//    static int counter = 0;
    for (int i = 0; i <= k; i++) {
        torusPolynomialMulByXaiMinusOne_16_2(&result->a[i], bara, baraIndex, nOutputs, bitSize, N, &bk->a[i]);

/*
//        int *temp_a = new int[bitSize*N];
//        cudaMemcpy(temp_a, result->a[i].coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//        if (counter < 10) {
//            int bI = 1;
//            int sI = bI * N;
//            cout << "new: ";
//            for (int l = 0; l < 10; ++l) {
//                cout << temp_a[sI + l] << " ";
//            }
//            cout << endl;
//        }*/
    }
//    counter++;
}

EXPORT void tLweMulByXaiMinusOne_16_2v2(TLweSample *resultV2, const int* bara, int baraIndex, const TLweSample *bkV2,
                                      int nOutputs, int bitSize, int N, const TLweParams *params) {
    const int k = params->k;
    torusPolynomialMulByXaiMinusOne_16_2v2(resultV2->a, bara, baraIndex, nOutputs, bitSize, N, bkV2->a);
}

EXPORT void tLweMulByXaiMinusOne_16_2_vector(TLweSample *result, const int* bara, int baraIndex, const TLweSample *bk,
                                      int vLength, int nOutputs, int bitSize, int N, const TLweParams *params) {
    const int k = params->k;
    for (int i = 0; i <= k; i++) {
        torusPolynomialMulByXaiMinusOne_16_2_vector(&result->a[i], bara, baraIndex, vLength,
                                                    nOutputs, bitSize, N, &bk->a[i]);
    }

}

/** result += (0,x) */
EXPORT void tLweAddTTo(TLweSample *result, const int pos, const Torus32 x, const TLweParams *params) {
    result->a[pos].coefsT[0] += x;
}

/** result += p*(0,x) */
EXPORT void
tLweAddRTTo(TLweSample *result, const int pos, const IntPolynomial *p, const Torus32 x, const TLweParams *params) {
    const int N = params->N;

    for (int i = 0; i < N; i++)
        result->a[pos].coefsT[i] += p->coefs[i] * x;
}


EXPORT void init_TLweKey(TLweKey *obj, const TLweParams *params) {
    new(obj) TLweKey(params);
}
EXPORT void destroy_TLweKey(TLweKey *obj) {
    (obj)->~TLweKey();
}

EXPORT void init_TLweSample(TLweSample *obj, const TLweParams *params) {
    new(obj) TLweSample(params);
}
EXPORT void destroy_TLweSample(TLweSample *obj) {
    (obj)->~TLweSample();
}

USE_DEFAULT_CONSTRUCTOR_DESTRUCTOR_IMPLEMENTATIONS1(TLweKey, TLweParams);
USE_DEFAULT_CONSTRUCTOR_DESTRUCTOR_IMPLEMENTATIONS1(TLweSample, TLweParams);










