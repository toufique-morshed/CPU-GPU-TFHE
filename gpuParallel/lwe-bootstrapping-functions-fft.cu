/*
 * Bootstrapping FFT functions
 */


#ifndef TFHE_TEST_ENVIRONMENT

#include <iostream>
#include <cassert>
#include "tfhe.h"
#include <iostream>
#include <fstream>

#include <cufftXt.h>
//#include "cudaFFTMorshed.h"
#include <omp.h>

using namespace std;

using namespace std;
#define INCLUDE_ALL
#else
#undef EXPORT
#define EXPORT
#endif

#define BLOCKSIZE 1024
#define P_LIMIT 3

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


__global__ void setVectorTo(int *destination, int val, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = val;
    }
}

__global__ void copyVector(int *destination, int *source, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = source[id];
    }
}

#if defined INCLUDE_ALL || defined INCLUDE_TFHE_INIT_LWEBOOTSTRAPPINGKEY_FFT
#undef INCLUDE_TFHE_INIT_LWEBOOTSTRAPPINGKEY_FFT
//(equivalent of the C++ constructor)
EXPORT void init_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *obj, const LweBootstrappingKey *bk) {

    const LweParams *in_out_params = bk->in_out_params;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int n = in_out_params->n;
    const int t = bk->ks->t;
    const int basebit = bk->ks->basebit;
    const int base = bk->ks->base;
    const int N = extract_params->n;

    LweKeySwitchKey *ks = new_LweKeySwitchKey(N, t, basebit, in_out_params);
    // Copy the KeySwitching key
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < t; j++) {
            for (int p = 0; p < base; p++) {
                lweCopy(&ks->ks[i][j][p], &bk->ks->ks[i][j][p], in_out_params);
            }
        }
    }

    // Bootstrapping Key FFT
    TGswSampleFFT *bkFFT = new_TGswSampleFFT_array(n, bk_params);
    for (int i = 0; i < n; ++i) {
        tGswToFFTConvert(&bkFFT[i], &bk->bk[i], bk_params);
    }

    new(obj) LweBootstrappingKeyFFT(in_out_params, bk_params, accum_params, extract_params, bkFFT, ks);
}

#endif



//destroys the LweBootstrappingKeyFFT structure
//(equivalent of the C++ destructor)
EXPORT void destroy_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *obj) {
    delete_LweKeySwitchKey((LweKeySwitchKey *) obj->ks);
    delete_TGswSampleFFT_array(obj->in_out_params->n, (TGswSampleFFT *) obj->bkFFT);

    obj->~LweBootstrappingKeyFFT();
}

//params: temp2, temp3, bkFFt + i, barai
void tfhe_MuxRotate_FFT(TLweSample *result, const TLweSample *accum, const TGswSampleFFT *bki, const int barai,
                        const TGswParams *bk_params) {
    // ACC = BKi*[(X^barai-1)*ACC]+ACC
//    ofstream input_old;
//    input_old.open("input_old.txt", ofstream::app);
//    for (int j = 0; j < 1024; ++j) {
//        input_old << (accum->a + 1)->coefsT[j] << " ";
//    }
//    input_old << endl;
//    input_old.close();
    //check input
//    static int tCounter = 0;
//    int offset = 500;
//    if (tCounter >= offset && tCounter < offset + P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            cout << "old_accum: ";
//            for (int i = 0; i < 20; ++i) {
//                cout << (accum->a + j)->coefsT[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounter++;
    // temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne(result, barai, accum, bk_params->tlwe_params);
//    static int tCounter = 0;
//    int offset = 0;
//    if (tCounter >= offset && tCounter < offset + P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            cout << "old: ";
//            for (int i = 0; i < 10; ++i) {
//                cout << (result->a + j)->coefsT[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounter++;
    // temp *= BKi
    tGswFFTExternMulToTLwe(result, bki, bk_params);
    //test morshed start
//    static int counter = 0;
//    if(counter < P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            cout << "old: ";
//            for (int i = 0; i < 10; ++i) {
//                cout << (result->a + j)->coefsT[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    counter++;
//    test morshed end
    //test morshed start
//    static int tCounter = 0;
//    if (tCounter < 10) {
//        cout << "old: ";
//        for (int i = 0; i < 10; ++i) {
//            int j = 0;
//            cout << (result->a + j)->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
//    tCounter++;
    //test morshed end

    // ACC += temp
    tLweAddTo(result, accum, bk_params->tlwe_params);
    //test morshed start
//    static int counterx = 0;
//    if (counterx++ < 10) {
//        cout << "old tLweAddTo: ";
//
//        for (int i=0; i < 10; ++i) {
//            int j = 1;
//            cout << (result->b)->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
//    cout << result->current_variance << " " << endl;
    //test morshed end
}

void tfhe_MuxRotate_FFT_16(TLweSample *result, const TLweSample *accum, const TGswSampleFFT *bki, const int *bara,
                           int baraIndex, int bitSize, int N, const TGswParams *bk_params,
                           cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce,
                           IntPolynomial *deca, IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                           cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu) {
    // ACC = BKi*[(X^barai-1)*ACC]+ACC
//    ofstream input_new;
//    input_new.open("input_new.txt", ofstream::app);
//    int * temp = new int[bitSize * N];
//    cudaMemcpy(temp, (accum->a + 1)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i * N;
//        for (int j = 0; j < N; ++j) {
//            input_new << temp[startIndex + j] << " ";
//        }
//        input_new << endl;
//    }
//    input_new.close();
//    static int tCounter = 0;
//    int * temp = new int[bitSize * N];
//    if (tCounter++ < P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            int bI = 1;
//            int sI = bI * N;
//            cudaMemcpy(temp, (accum->a + j)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//            cout << "new_accum: ";
//            for (int i = 0; i < 20; ++i) {
//                cout << temp[sI + i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    int *exp = new int[bitSize * N];
//    for (int i = 0; i < bitSize * N; ++i) {
//        exp[i] = i << 3 + 1 << 16;
//    }
//    cudaMemcpy(accum->a[0].coefsT, exp, bitSize * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(accum->a[1].coefsT, exp, bitSize * N * sizeof(int), cudaMemcpyHostToDevice);

    // temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne_16(result, bara, baraIndex, accum, bitSize, N, bk_params->tlwe_params);

//    int *temp_result = new int[bitSize * N];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(temp_result, (result->a + valK)->coefsT, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                cout << temp_result[i * N + j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }

//    cout << endl;
//    static int tCounter = 0;
//    int * temp = new int[bitSize * N];
//    if (tCounter++ < P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            cout << "new: ";
//            int bI = 0;
//            int sI = bI * N;
//            cudaMemcpy(temp, (result->a + j)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//                cout << temp[sI + i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    ofstream output_new;
//    output_new.open("output_new.txt", ofstream::app);
//    cudaMemcpy(temp, (result->a + 1)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i * N;
//        for (int j = 0; j < N; ++j) {
//            output_new << temp[startIndex + j] << " ";
//        }
//        output_new << endl;
//    }
//    output_new.close();
    // temp *= BKi
    tGswFFTExternMulToTLwe_16(result, bki, bitSize, bk_params, cudaBKi, cudaBKiCoalesce, deca, decaCoalesce, cuDecaFFT,
                              cuDecaFFTCoalesce, tmpa_gpu);
//    int *temp = new int[bitSize * N];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(temp, (result->a + valK)->coefsT, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                cout << temp[i * N + j] << " ";
//            }
//            cout << endl;
//        }
//    }
//    cout << endl;
//    test morshed start
//    static int tCounter = 0;
//    if(tCounter < P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            cout << "new: ";
//            int *temp = new int[N * bitSize];
//            cudaMemcpy(temp, (result->a + j)->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//                cout << temp[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounter++;
    //test morshed end
    // ACC += temp
    tLweAddTo_16(result, accum, bitSize, N, bk_params->tlwe_params);
//    cout << "result->current_variance: " << result->current_variance;
//    int *temp = new int[bitSize * N];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(temp, result->a[valK].coefsT, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                cout << temp[i * N + j] << " ";
//            }
//            cout << endl;
//        }
//    }
//    cout << endl;
    //test morshed start
//    static int counterx = 0;
//    if(counterx++ < 10) {
//        cout << "new tLweAddTo_16: ";
//        for (int i = 0; i < 10; ++i) {
//        int j = 1;
//            cout << (result->b)->coefsT[startIndex + i] << " ";
//        }
//        cout << endl;
//    }
//    cout << result->current_variance << " " << endl;
    //test morshed end
}

void tfhe_MuxRotate_FFT_16_2(TLweSample *result, const TLweSample *accum,
                             const TGswSampleFFT *bki, const int *bara,
                             int baraIndex, int nOutputs, int bitSize, int N, const TGswParams *bk_params,
                             cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                             IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                             cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu,
                             cufftDoubleComplex *tmpa_gpuCoal) {

        int length = nOutputs * bitSize * N;
    // ACC = BKi*[(X^barai-1)*ACC]+ACC
//    ofstream input_new;
//    input_new.open("input_new.txt", ofstream::app);
//    int * temp = new int[bitSize * N];
//    cudaMemcpy(temp, (accum->a + 1)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i * N;
//        for (int j = 0; j < N; ++j) {
//            input_new << temp[startIndex + j] << " ";
//        }
//        input_new << endl;
//    }
//    input_new.close();
//    static int tCounter = 0;
//    int * temp = new int[bitSize * N];
//    if (tCounter++ < P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            int bI = 1;
//            int sI = bI * N;
//            cudaMemcpy(temp, (accum->a + j)->coefsT, sizeof(int) * N * bitSize, cudaMemcpyDeviceToHost);
//            cout << "new_accum: ";
//            for (int i = 0; i < 20; ++i) {
//                cout << temp[sI + i] << " ";
//            }
//            cout << endl;
//        }
//    }
    // temp = (X^barai-1)*ACC
//            cout << "bitSize: " << bitSize << endl;
//            exit(0);
    tLweMulByXaiMinusOne_16_2(result, bara, baraIndex, accum, nOutputs, bitSize, N, bk_params->tlwe_params);
//    tLweMulByXaiMinusOne_16_2v2(resultV2, bara, baraIndex, accumV2, nOutputs, bitSize, N, bk_params->tlwe_params);
/*
    cout << "(X^barai-1)*ACC" << endl;
    int *temp_result = new int[length];
    int *temp_resultV2 = new int[length];
    int *temp_accum = new int[length];
    int *temp_accumV2 = new int[length];
    for (int x = 0; x <= 1; ++x) {
        cudaMemcpy(temp_result, (result->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_resultV2, (resultV2->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_accum, (accum->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_accumV2, (accumV2->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        for (int i = 0; i < bitSize * nOutputs; ++i) {
            int sI = 100;
            for (int j = 0; j < N; ++j) {
//                cout << "x: " << x << " i: " << i << " j :" << j << " length: " << (result->a + x)->N << "index: " << i * N + j << endl;
                assert(temp_result[i * N + j] == temp_resultV2[i * N + j]);
                assert(temp_accum[i * N + j] == temp_accumV2[i * N + j]);
//                cout << temp_result[i * N + j + sI] << " ";
            }
//            cout << endl;
        }
    }*/
//    cout << endl;
//    cout << endl;
//    for (int x = 0; x <= 1; ++x) {
//        cudaMemcpy(temp_result, (resultV2->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize * nOutputs; ++i) {
//            int sI = 100;
//            for (int j = 0; j < 30; ++j) {
//                cout << temp_result[i * N + j + sI] << " ";
//            }
//            cout << endl;
//        }
//    }

    // temp *= BKi
    tGswFFTExternMulToTLwe_16_2(result, bki, nOutputs, bitSize, bk_params, cudaBKi, cudaBKiCoalesce, deca, decaCoalesce,
                                cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu, tmpa_gpuCoal);

//    tGswFFTExternMulToTLwe_16_2V2(resultV2, bki, nOutputs, bitSize, bk_params, cudaBKi, cudaBKiCoalesce, deca, decaCoalesce,
//                                cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpuCoal);
/*
    cout << "temp *= BKi" << endl;
    int *temp = new int[length];
    int *temp2 = new int[length];
    for (int x = 0; x <= 1; ++x) {
        cudaMemcpy(temp, (result->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp2, (resultV2->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < bitSize * nOutputs; ++i) {
            for (int j = 0; j < N; ++j) {
//            cout << temp[i * N + j] << " ";
                assert(temp[i * N + j] == temp2[i * N + j]);
            }
//        cout << endl;
        }
    }
//    cout << endl;
        */

//    test morshed start
//    static int tCounter = 0;
//    if(tCounter < P_LIMIT) {
//        for (int j = 0; j <= 1; ++j) {
//            cout << "new: ";
//            int *temp = new int[N * bitSize];
//            cudaMemcpy(temp, (result->a + j)->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            for (int i = 0; i < 10; ++i) {
//                cout << temp[i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    tCounter++;
    //test morshed end
     //ACC += temp
    tLweAddTo_16_2(result, accum, nOutputs, bitSize, N, bk_params->tlwe_params);

//    tLweAddTo_16_2v2(resultV2, accumV2, nOutputs, bitSize, N, bk_params->tlwe_params);
/*
    int *tempRes = new int[length];
    int *tempResV2 = new int[length];
    for (int x = 0; x <= 1; ++x) {
        cudaMemcpy(tempRes, (result->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempResV2, (resultV2->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < bitSize * nOutputs; ++i) {
            for (int j = 0; j < N; ++j) {
                assert(tempRes[i * N + j] == tempResV2[i * N + j]);
//                cout << temp[i * N + j] << " ";
            }
//            cout << endl;
        }
//        cout << endl;
    }
    cudaMemcpy(temp, (resultV2->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << temp[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;*/
}



void tfhe_MuxRotate_FFT_16V2(TLweSample *resultV2, const TLweSample *accumV2, const TGswSampleFFT *bki, const int *bara,
                           int baraIndex, int bitSize, int N, const TGswParams *bk_params,
                           cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce,
                           IntPolynomial *deca, IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                           cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu,
                             cufftDoubleComplex *tmpa_gpuCoal, cudaFFTProcessorTest_general *fftProcessor) {
    int length = bitSize * N;
    tLweMulByXaiMinusOne_16_2v2(resultV2, bara, baraIndex, accumV2, 1, bitSize, N, bk_params->tlwe_params);

    tGswFFTExternMulToTLwe_16V2(resultV2, bki, bitSize, bk_params, cudaBKi, cudaBKiCoalesce, deca, decaCoalesce,
                                cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu, tmpa_gpuCoal, fftProcessor);

    tLweAddTo_16_2v2(resultV2, accumV2, 1, bitSize, N, bk_params->tlwe_params);

















}

void tfhe_MuxRotate_FFT_16_2V2(TLweSample *resultV2, const TLweSample *accumV2,
                               const TGswSampleFFT *bki, const int *bara,
                               int baraIndex, int nOutputs, int bitSize, int N, const TGswParams *bk_params,
                               cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                               IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                               cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu,
                               cufftDoubleComplex *tmpa_gpuCoal,
                               cudaFFTProcessorTest_general *fftProcessor) {

    int length = nOutputs * bitSize * N;
    // temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne_16_2v2(resultV2, bara, baraIndex, accumV2, nOutputs, bitSize, N, bk_params->tlwe_params);
/*
    cout << "(X^barai-1)*ACC" << endl;
    int *temp_result = new int[length];
    int *temp_resultV2 = new int[length];
    int *temp_accum = new int[length];
    int *temp_accumV2 = new int[length];
    for (int x = 0; x <= 1; ++x) {
        cudaMemcpy(temp_result, (result->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_resultV2, (resultV2->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_accum, (accum->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_accumV2, (accumV2->a + x)->coefsT, length * sizeof(Torus32), cudaMemcpyDeviceToHost);
        for (int i = 0; i < bitSize * nOutputs; ++i) {
            int sI = 100;
            for (int j = 0; j < N; ++j) {
//                cout << "x: " << x << " i: " << i << " j :" << j << " length: " << (result->a + x)->N << "index: " << i * N + j << endl;
                assert(temp_result[i * N + j] == temp_resultV2[i * N + j]);
                assert(temp_accum[i * N + j] == temp_accumV2[i * N + j]);
//                cout << temp_result[i * N + j + sI] << " ";
            }
//            cout << endl;
        }
    }*/

    // temp *= BKi
    tGswFFTExternMulToTLwe_16_2V2(resultV2, bki, nOutputs, bitSize, bk_params, cudaBKi, cudaBKiCoalesce, deca, decaCoalesce,
                                  cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpuCoal, fftProcessor);
//    cudaFree(d_out_test);
//    cudaFree(d_in_test);
/*
    cout << "temp *= BKi" << endl;
    int *temp = new int[length];
    int *temp2 = new int[length];
    for (int x = 0; x <= 1; ++x) {
        cudaMemcpy(temp, (result->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp2, (resultV2->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < bitSize * nOutputs; ++i) {
            for (int j = 0; j < N; ++j) {
//            cout << temp[i * N + j] << " ";
                assert(temp[i * N + j] == temp2[i * N + j]);
            }
//        cout << endl;
        }
    }
//    cout << endl;*/
    tLweAddTo_16_2v2(resultV2, accumV2, nOutputs, bitSize, N, bk_params->tlwe_params);
/*
    int *tempRes = new int[length];
    int *tempResV2 = new int[length];
    for (int x = 0; x <= 1; ++x) {
        cudaMemcpy(tempRes, (result->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempResV2, (resultV2->a + x)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < bitSize * nOutputs; ++i) {
            for (int j = 0; j < N; ++j) {
                assert(tempRes[i * N + j] == tempResV2[i * N + j]);
//                cout << temp[i * N + j] << " ";
            }
//            cout << endl;
        }
//        cout << endl;
    }
    cudaMemcpy(temp, (resultV2->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bitSize * nOutputs; ++i) {
        for (int j = 0; j < 10; ++j) {
            cout << temp[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;*/

}


void tfhe_MuxRotate_FFT_16_2_vector(TLweSample *result, const TLweSample *accum, const TGswSampleFFT *bki, const int *bara,
                             int baraIndex, int vLength, int nOutputs, int bitSize, int N, const TGswParams *bk_params,
                             cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                             IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                             cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu) {

//        cout << "tfhe_MuxRotate_FFT_16_2_vector" << endl;
//        cout << "vLength: " << vLength << " nOutputs: " << nOutputs << " bitSize: " << bitSize << endl;
    int length = vLength * nOutputs * bitSize * N;
    // temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne_16_2_vector(result, bara, baraIndex, accum, vLength, nOutputs, bitSize, N, bk_params->tlwe_params);
//    tLweMulByXaiMinusOne_16_2(result, bara, baraIndex, accum, nOutputs, vLength * bitSize, N, bk_params->tlwe_params);

    //test
//    int *temp_result = new int[length];
//    cudaMemcpy(temp_result, (result->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < bitSize * 2; ++i) {
//        int sI = i * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_result[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << "XOR PART" << endl;
//    for (int i = 0; i < bitSize * 2; ++i) {
//        int sI = (i + vLength * bitSize) * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_result[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << "tfhe_MuxRotate_FFT_16_2_vector:bitSize: " << bitSize << endl;
    // temp *= BKi
    tGswFFTExternMulToTLwe_16_2_vector(result, bki, vLength, nOutputs, bitSize, bk_params, cudaBKi, cudaBKiCoalesce,
                                       deca, decaCoalesce, cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu);

//    int *temp = new int[length];
//    cout << "AND PART:" << endl;
//    cudaMemcpy(temp, (result->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << "XOR PART:" << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = (i + vLength * bitSize) * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
    // ACC += temp
    tLweAddTo_16_2_vector(result, accum, vLength, nOutputs, bitSize, N, bk_params->tlwe_params);
//    int *temp = new int[length];
//    cudaMemcpy(temp, (result->a)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = i * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//
//    cout << "XOR PART" << endl;
//    for (int i = 0; i < 3 * bitSize; ++i) {
//        int sI = i * N  + vLength * bitSize * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
}

#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BLIND_ROTATE_FFT
#undef INCLUDE_TFHE_BLIND_ROTATE_FFT
/**
 * multiply the accumulator by X^sum(bara_i.s_i)
 * @param accum the TLWE sample to multiply
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
 */
EXPORT void tfhe_blindRotate_FFT(TLweSample *accum,
                                 const TGswSampleFFT *bkFFT,
                                 const int *bara,
                                 const int n,
                                 const TGswParams *bk_params) {

    //TGswSampleFFT* temp = new_TGswSampleFFT(bk_params);
    TLweSample *temp = new_TLweSample(bk_params->tlwe_params);
    TLweSample *temp2 = temp;
    TLweSample *temp3 = accum;

//    for (int x = 0; x <= 1; ++x) {
//        for (int i = 0; i < 1024; ++i) {
//            (temp2->a + x)->coefsT[i] = 0;
//        }
//    }


//    for (int k = 0; k <= bk_params->tlwe_params->k; ++k) {
//        cout << "new_temp2_test: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << (temp->a + k)->coefsT[j] << " ";
//        }
//        cout << endl;
//    }


    for (int i = 0; i < n; i++) {//n
        const int barai = bara[i];
        if (barai == 0) continue; //indeed, this is an easy case!
        tfhe_MuxRotate_FFT(temp2, temp3, bkFFT + i, barai, bk_params);
        swap(temp2, temp3);
        //test morshed start
//        static int counter = 0;
//        if(counter < 3){
//            if(counter < P_LIMIT ) {
//                for (int valK = 0; valK <= 1; ++valK) {
//                    cout << "old: ";
//                    for (int i = 0; i < 10; ++i) {
//                        cout << (temp2->a + valK)->coefsT[i] << " ";
//                    }
//                    cout << endl;
//                }
//            }
//        }
//        counter++;
    }
    if (temp3 != accum) {
        tLweCopy(accum, temp3, bk_params->tlwe_params);
    }
//    test morshed start
//    for (int valK = 0; valK <= 1; ++valK) {
//        cout << "old: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << (temp2->a + valK)->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
    //test morshed end
    delete_TLweSample(temp);
//    delete_TGswSampleFFT(temp);
}

//new
//__global__ void swap_16_GPU(int *a, int *b, int length) {
//    int id = blockIdx.x * blockDim.x + threadIdx.x;
//    if(id < length) {
//        int temp = a[id];
//        a[id] = b[id];
//        b[id] = temp;
//    }
//}
void delete_TLweSampleFromCuda(TLweSample *sample, const TLweParams * params) {
//    for (int i = 0; i <= params->k; ++i) {
//        cudaFree((sample->a + i)->coefsT);
//        (sample->a + i)->coefsT = NULL;
//    }
//    delete_TLweSample(sample);
}

EXPORT void tfhe_blindRotate_FFT_16(TLweSample *accum,
                                    const TGswSampleFFT *bkFFT,
                                    const int *bara,
                                    const int n,
                                    int bitSize,
                                    const TGswParams *bk_params,
                                    cufftDoubleComplex ****cudaBkFFT,
                                    cufftDoubleComplex ***cudaBkFFTCoalesce) {

    int N = bk_params->tlwe_params->N;//1024
    const TLweParams *temp_accum_params = new_TLweParams(N * bitSize, bk_params->tlwe_params->k,
                                                         bk_params->tlwe_params->alpha_min,
                                                         bk_params->tlwe_params->alpha_max);
    int gridSize = (int) ceil((float) (N * bitSize) / BLOCKSIZE);

    TLweSample *temp = new_TLweSample(temp_accum_params);

    for (int i = 0; i <= temp->k; ++i) {
        free(temp->a[i].coefsT);
        cudaMalloc(&(temp->a[i].coefsT), temp->a[i].N * sizeof(int));
        setVectorTo<<<gridSize, BLOCKSIZE>>>(temp->a[i].coefsT, 0, temp->a[i].N);
    }
    TLweSample *temp2 = temp;
    TLweSample *temp3 = accum;
//    cout << "bk_params->tlwe_params->k: " << bk_params->tlwe_params->k << endl;
//    cout << "(&accum->a[k])->N: " << (&accum->a[0])->N << endl;
//    int *testtemp2 = new int[bitSize * N];
//    for (int i = 0; i < bitSize; ++i) {
//        for (int k = 0; k <= bk_params->tlwe_params->k; ++k) {
//            cudaMemcpy(testtemp2 , (temp3->a + k)->coefsT, N * bitSize* sizeof(int), cudaMemcpyDeviceToHost);
//            cout << "new_temp2_test: ";
//            for (int j = 0; j < 10; ++j) {
//                cout << testtemp2 [i * N + j] << " ";
//            }
//            cout << endl;
//        }
//
//    }
    //create variables for fft operations
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int bigN = N * bitSize;
    const int Ns2 = N / 2;


    IntPolynomial *deca;
//    deca = new_IntPolynomial_array(kpl, bigN); //decomposed accumulator
//    for (int i = 0; i < kpl; ++i) {
//        free(deca[i].coefs);
//        cudaMalloc(&((deca + i)->coefs), bitSize * N * sizeof(int));
//    }

    IntPolynomial *decaCoalesce;
    decaCoalesce = new_IntPolynomial(kpl * bigN);
    free(decaCoalesce->coefs);
    cudaMalloc(&(decaCoalesce->coefs), bigN * kpl * sizeof(int));

    cufftDoubleComplex **cuDecaFFT;
//    cuDecaFFT = new cufftDoubleComplex *[kpl];
//    for (int i = 0; i < kpl; ++i) {
//        cudaMalloc(&(cuDecaFFT[i]), bitSize * Ns2 * sizeof(cufftDoubleComplex));
//    }

    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, kpl * bitSize * Ns2 * sizeof(cufftDoubleComplex));

    cufftDoubleComplex **tmpa_gpu = new cufftDoubleComplex *[k + 1];;
    for (int i = 0; i <= k; ++i) {
        cudaMalloc(&(tmpa_gpu[i]), bitSize * Ns2 * sizeof(cufftDoubleComplex));
    }

//    int *exp = new int[bitSize * N];
//    for (int i = 0; i < bitSize * N; ++i) {
//        exp[i] = i << 5 + 1 << 20;
//    }
//    cudaMemcpy(temp3->a[0].coefsT, exp, bitSize * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(temp2->a[0].coefsT, exp, bitSize * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(temp3->a[1].coefsT, exp, bitSize * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(temp2->a[1].coefsT, exp, bitSize * N * sizeof(int), cudaMemcpyHostToDevice);

//    ofstream myfile;
//    myfile.open ("halfGPU.txt", ios::out);

    for (int j = 0; j < 500; ++j) {//n
        tfhe_MuxRotate_FFT_16(temp2, temp3, bkFFT + j, bara, j, bitSize, N, bk_params, NULL,
                              cudaBkFFTCoalesce[j], deca, decaCoalesce, cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu);
//        //test morshed start
//        myfile << "j: " << j << " input: ";
//        int *tempx = new int[N * bitSize];
//        for (int valK = 0; valK < 2; ++valK) {
//            cudaMemcpy(tempx, temp2->a[valK].coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            for (int i = 0; i < bitSize; ++i) {
//                int sI = i * N;
//                for (int j = 0; j < 500; ++j) {
//                    cout << tempx[sI + j] << " ";
//                }
//                cout << endl;
//            }
//        }
//
//        myfile << endl;
//        int length = N * bitSize * kpl;
//        myfile << "j: " << j << " output: ";
//        tempx = new int[length];
//        cudaMemcpy(tempx, decaCoalesce->coefs, length * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize * kpl; ++i) {
//            int sI = i * N;
//            for (int j = 0; j < 10; ++j) {
//                myfile << tempx[sI + j] << " ";
//            }
//        }
//        myfile << endl;

        swap(temp2, temp3);
    }
//    myfile << endl;
//    myfile.close();
    //test morshed start
//    int *tempx = new int[N * bitSize];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(tempx, accum->a[valK].coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            int sI = i * N;
//            for (int j = 0; j < 500; ++j) {
//                cout << tempx[sI + j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;
//
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(tempx, temp2->a[valK].coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            int sI = i * N;
//            for (int j = 0; j < 10; ++j) {
//                cout << tempx[sI + j] << " ";
//            }
//            cout << endl;
//        }
////        cout << endl;
//    }
//    cout << endl;
//    cout << endl;
//    test morshed end
    //free memory
    delete_TLweSampleFromCuda(temp, temp_accum_params);
//    cudaFree(decaCoalesce->coefs);
    cudaFree(cuDecaFFTCoalesce);
    for (int i = 0; i <= k; ++i) {
        cudaFree(tmpa_gpu[i]);
    }
    //delete_TGswSampleFFT(temp);
}

EXPORT void tfhe_blindRotate_FFT_16_2(TLweSample *accum,
                                      const TGswSampleFFT *bkFFT,
                                      const int *bara,
                                      const int n,
                                      int nOutputs,
                                      int bitSize,
                                      const TGswParams *bk_params,
                                      cufftDoubleComplex ****cudaBkFFT,
                                      cufftDoubleComplex ***cudaBkFFTCoalesce) {

    int N = bk_params->tlwe_params->N;//1024
    int length = nOutputs * bitSize * N;//32768 = nOutputs * bitSize * N
    const TLweParams *temp_accum_params = new_TLweParams(length, bk_params->tlwe_params->k,
                                                         bk_params->tlwe_params->alpha_min,
                                                         bk_params->tlwe_params->alpha_max);
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    TLweSample *temp = new_TLweSample(temp_accum_params);

    for (int i = 0; i <= temp->k; ++i) {
        free(temp->a[i].coefsT);
        cudaMalloc(&(temp->a[i].coefsT), temp->a[i].N * sizeof(int));
        setVectorTo<<<gridSize, BLOCKSIZE>>>(temp->a[i].coefsT, 0, temp->a[i].N);
    }
    TLweSample *temp2 = temp;
    TLweSample *temp3 = accum;

    //create new tempV2
//    TLweSample *tempV2 = new_TLweSample(temp_accum_params);
//    free(tempV2->a[0].coefsT);
//    free(tempV2->a[1].coefsT);
//    cudaMalloc(&(tempV2->a[0].coefsT), tempV2->a[0].N * sizeof(Torus32) * (tempV2->k + 1));
//    cout << "tempV2->k: " << tempV2->k<< endl;
//    cout << "tempV2->N: " << tempV2->a->N<< endl;
//    cout << "length " << length << endl;
//    cout << "temp2.N " << temp2->a->N<< endl;
//    cudaMemset(tempV2->a[0].coefsT, 0, tempV2->a[0].N * sizeof(Torus32) * (tempV2->k + 1));
//    tempV2->a[1].coefsT = tempV2->a[0].coefsT + length;
//    //create new accum
//    TLweSample *accumV2 = new_TLweSample(temp_accum_params);
//    free(accumV2->a[0].coefsT);
//    free(accumV2->a[1].coefsT);
//    cudaMalloc(&(accumV2->a[0].coefsT), accumV2->a[0].N * sizeof(Torus32) * (accumV2->k + 1));
//    accumV2->a[1].coefsT = accumV2->a[0].coefsT + length;
//    cudaMemcpy(accumV2->a[0].coefsT, accum->a[0].coefsT, accum->a[0].N * sizeof(Torus32), cudaMemcpyDeviceToDevice);
//    cudaMemcpy(accumV2->a[1].coefsT, accum->a[1].coefsT, accum->a[1].N * sizeof(Torus32), cudaMemcpyDeviceToDevice);
//
//    TLweSample *temp2v2 = tempV2;
//    TLweSample *temp3v2 = accumV2;


    //create variables for fft operations
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;
    const int length_Ns2 = nOutputs * bitSize * Ns2;//16384


    IntPolynomial *deca;
//    deca = new_IntPolynomial_array(kpl, length); //decomposed accumulator
//    for (int i = 0; i < kpl; ++i) {
//        free(deca[i].coefs);
//        cudaMalloc(&((deca + i)->coefs), length * sizeof(int));
//    }

    IntPolynomial *decaCoalesce;
    decaCoalesce = new_IntPolynomial(kpl * length);
    free(decaCoalesce->coefs);
    cudaMalloc(&(decaCoalesce->coefs), kpl * length * sizeof(int));


    cufftDoubleComplex **cuDecaFFT;
//    cuDecaFFT = new cufftDoubleComplex *[kpl];
//    for (int i = 0; i < kpl; ++i) {
//        cudaMalloc(&(cuDecaFFT[i]), length_Ns2 * sizeof(cufftDoubleComplex));
//    }

    //testing coalesce
    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex));

    cufftDoubleComplex **tmpa_gpu;
    tmpa_gpu = new cufftDoubleComplex *[k + 1];;
    for (int i = 0; i <= k; ++i) {
        cudaMalloc(&(tmpa_gpu[i]), length_Ns2 * sizeof(cufftDoubleComplex));
    }

    cufftDoubleComplex *tmpa_gpuCoal;
    cudaMalloc(&tmpa_gpuCoal, length_Ns2 * sizeof(cufftDoubleComplex) * (k + 1));





//    temp2 = NULL;
//    temp3 = NULL;

    for (int j = 0; j < n; ++j) {//n
//        cout << "j: " << j << endl << endl;
        tfhe_MuxRotate_FFT_16_2(temp2, temp3, bkFFT + j, bara, j, nOutputs, bitSize, N, bk_params,
                                cudaBkFFT[j], cudaBkFFTCoalesce[j], deca, decaCoalesce,
                                cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu, tmpa_gpuCoal);
        swap(temp2, temp3);
    }


//    for (int i = 0; i < k; ++i) {
//        temp3v2->b->coefsT = (temp3v2->a + 1)->coefsT;
//        cudaMemcpy((accum->a + 0)->coefsT, (temp3v2->a)->coefsT + bitSize * N * nOutputs, bitSize * N * nOutputs * sizeof(Torus32), cudaMemcpyDeviceToDevice);
//        cudaMemcpy((accum->a + 1)->coefsT, (temp3v2->a + 1)->coefsT + bitSize * N * nOutputs, bitSize * N * nOutputs * sizeof(Torus32), cudaMemcpyDeviceToDevice);
//        cudaMemcpy((accum->b)->coefsT, (temp3v2->b)->coefsT + bitSize * N * nOutputs, bitSize * N * nOutputs * sizeof(Torus32), cudaMemcpyDeviceToDevice);
//    }
//    accum->b->coefsT = (accum->a + 1)->coefsT;

    //test morshed start
//    int *tempx = new int[length];
//    for (int bI = 0; bI < bitSize * nOutputs; ++bI) {
//        int valK = 0;
////        for (int valK = 0; valK <= 1; ++valK) {
//            cudaMemcpy(tempx, (accum->a + valK)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//            int sI = bI * N;
//            cout << "new: " << valK << ": ";
//            for (int i = 0; i < 10; ++i) {
//                cout << tempx[sI + i] << " ";
//            }
//            cout << endl;
////        }
//    }
//    cout << endl;
//    for (int bI = 0; bI < bitSize * nOutputs; ++bI) {
//        int valK = 0;
////        for (int valK = 0; valK <= 1; ++valK) {
//        cudaMemcpy(tempx, (accumV2->a + valK)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N;
//        cout << "new: " << valK << ": ";
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        }
//    }
    //test morshed end
//    delete_TLweSampleFromCuda(temp, temp_accum_params);
    cudaFree(decaCoalesce->coefs);
    decaCoalesce->coefs = NULL;
    delete_IntPolynomial(decaCoalesce);
    cudaFree(cuDecaFFTCoalesce);
    for (int i = 0; i <= k; ++i) {
        cudaFree(tmpa_gpu[i]);
    }
//    //delete_TGswSampleFFT(temp);
}


EXPORT void tfhe_blindRotate_FFT_16V2(TLweSample *accumV2,
                                    const TGswSampleFFT *bkFFT,
                                    const int *bara,
                                    const int n,
                                    int bitSize,
                                    const TGswParams *bk_params,
                                    cufftDoubleComplex ****cudaBkFFT,
                                    cufftDoubleComplex ***cudaBkFFTCoalesce) {

    int N = bk_params->tlwe_params->N;//1024
    int length = bitSize * N;//32768 = nOutputs * bitSize * N
    const TLweParams *temp_accum_params = new_TLweParams(length, bk_params->tlwe_params->k,
                                                         bk_params->tlwe_params->alpha_min,
                                                         bk_params->tlwe_params->alpha_max);
    TLweSample *tempV2 = new_TLweSample(temp_accum_params);
    free(tempV2->a[0].coefsT);
    free(tempV2->a[1].coefsT);
    cudaMalloc(&(tempV2->a[0].coefsT), tempV2->a[0].N * (tempV2->k + 1) * sizeof(Torus32));
    cudaMemset(tempV2->a[0].coefsT, 0, tempV2->a[0].N * (tempV2->k + 1) * sizeof(Torus32));
    (tempV2->a + 1)->coefsT = (tempV2->a + 0)->coefsT + (tempV2->a + 0)->N;

    TLweSample *temp2v2 = tempV2;
    TLweSample *temp3v2 = accumV2;

    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;
    const int length_Ns2 = bitSize * Ns2;//16384

    IntPolynomial *decaCoalesce;
    decaCoalesce = new_IntPolynomial(kpl * length);
    free(decaCoalesce->coefs);
    cudaMalloc(&(decaCoalesce->coefs), kpl * length * sizeof(int));

    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex));

    cufftDoubleComplex *tmpa_gpuCoal;
    cudaMalloc(&tmpa_gpuCoal, length_Ns2 * sizeof(cufftDoubleComplex) * (k + 1));

    cudaFFTProcessorTest_general *fftProcessor =
            new cudaFFTProcessorTest_general(1024, 1, bitSize, 4, 1024, 2);


    for (int j = 0; j < 500; ++j) {//n
        tfhe_MuxRotate_FFT_16V2(temp2v2, temp3v2, bkFFT + j, bara, j, bitSize, N, bk_params, NULL,
                              cudaBkFFTCoalesce[j], NULL, decaCoalesce, NULL, cuDecaFFTCoalesce, NULL, tmpa_gpuCoal, fftProcessor);



        swap(temp2v2, temp3v2);
    }

//    cout << endl;
//    int *tempx = new int[N * bitSize];
//    for (int valK = 0; valK < 2; ++valK) {
//        cudaMemcpy(tempx, accumV2->a[valK].coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < bitSize; ++i) {
//            int sI = i * N;
//            for (int j = 0; j < 500; ++j) {
//                cout << tempx[sI + j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }


    cudaFree((temp2v2->a + 0)->coefsT);
    cudaFree(decaCoalesce->coefs);
    decaCoalesce->coefs = NULL;
    delete_IntPolynomial(decaCoalesce);
    cudaFree(cuDecaFFTCoalesce);
    cudaFree(tmpa_gpuCoal);
    delete fftProcessor;

}


EXPORT void tfhe_blindRotate_FFT_16_2v2(TLweSample *accumV2,
                                      const TGswSampleFFT *bkFFT,
                                      const int *bara,
                                      const int n,
                                      int nOutputs,
                                      int bitSize,
                                      const TGswParams *bk_params,
                                      cufftDoubleComplex ****cudaBkFFT,
                                      cufftDoubleComplex ***cudaBkFFTCoalesce) {

    int N = bk_params->tlwe_params->N;//1024
    int length = nOutputs * bitSize * N;//32768 = nOutputs * bitSize * N
    const TLweParams *temp_accum_params = new_TLweParams(length, bk_params->tlwe_params->k,
                                                         bk_params->tlwe_params->alpha_min,
                                                         bk_params->tlwe_params->alpha_max);

    TLweSample *tempV2 = new_TLweSample(temp_accum_params);

//    free(tempV2->a[0].coefsT);
//    free(tempV2->a[1].coefsT);
    cudaMalloc(&(tempV2->a[0].coefsT), tempV2->a[0].N * (tempV2->k + 1) * sizeof(Torus32));
    cudaMemset(tempV2->a[0].coefsT, 0, tempV2->a[0].N * (tempV2->k + 1) * sizeof(Torus32));
    (tempV2->a + 1)->coefsT = (tempV2->a + 0)->coefsT + (tempV2->a + 0)->N;

    TLweSample *temp2v2 = tempV2;
    TLweSample *temp3v2 = accumV2;

    //create variables for fft operations
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;
    const int length_Ns2 = nOutputs * bitSize * Ns2;//16384



    IntPolynomial *decaCoalesce;
    decaCoalesce = new_IntPolynomial(kpl * length);
    free(decaCoalesce->coefs);
    cudaMalloc(&(decaCoalesce->coefs), kpl * length * sizeof(int));



    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex));



    cufftDoubleComplex *tmpa_gpuCoal;
    cudaMalloc(&tmpa_gpuCoal, length_Ns2 * sizeof(cufftDoubleComplex) * (k + 1));

    cudaFFTProcessorTest_general *fftProcessor =
            new cudaFFTProcessorTest_general(1024, nOutputs, bitSize, 4, 1024, 2);

//    int *tempx = new int[length];
//    for (int bI = 0; bI < bitSize * nOutputs; ++bI) {
//        for (int valK = 0; valK <= 1; ++valK) {
//        cudaMemcpy(tempx, (accumV2->a + valK)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N;
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
//        }
//    }

//    cufftDoubleReal *d_in_test;
//    cufftDoubleComplex *d_out_test;
//
//    cudaMalloc(&d_in_test, sizeof(cufftDoubleReal) * 2048 * 128);
//    cudaMalloc(&d_out_test, sizeof(cufftDoubleComplex) * (1024 + 1) * 128);
    for (int j = 0; j < n; ++j) {//n
        tfhe_MuxRotate_FFT_16_2V2(temp2v2, temp3v2, bkFFT + j, bara, j, nOutputs, bitSize, N, bk_params,
                                  NULL, cudaBkFFTCoalesce[j], NULL, decaCoalesce,
                                  NULL, cuDecaFFTCoalesce, NULL, tmpa_gpuCoal, fftProcessor);
        swap(temp2v2, temp3v2);
    }

    //test morshed start
//    int *tempx = new int[length];
//    for (int bI = 0; bI < bitSize * nOutputs; ++bI) {
//        for (int valK = 0; valK <= 1; ++valK) {
//            cudaMemcpy(tempx, (accumV2->a + valK)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//            int sI = bI * N;
//            for (int i = 0; i < 10; ++i) {
//                cout << tempx[sI + i] << " ";
//            }
//            cout << endl;
//        }
//    }

    delete fftProcessor;
    cudaFree((temp2v2->a + 0)->coefsT);
    cudaFree(decaCoalesce->coefs);
    decaCoalesce->coefs = NULL;
    delete_IntPolynomial(decaCoalesce);
    cudaFree(cuDecaFFTCoalesce);
    cudaFree(tmpa_gpuCoal);
}




EXPORT void tfhe_blindRotate_FFT_16_2_vector(TLweSample *accum,
                                      const TGswSampleFFT *bkFFT,
                                      const int *bara,
                                      const int n,
                                      int vLength,
                                      int nOutputs,
                                      int bitSize,
                                      const TGswParams *bk_params,
                                      cufftDoubleComplex ****cudaBkFFT,
                                      cufftDoubleComplex ***cudaBkFFTCoalesce) {

    int N = bk_params->tlwe_params->N;//1024
    int length = vLength * nOutputs * bitSize * N;
    const TLweParams *temp_accum_params = new_TLweParams(length, bk_params->tlwe_params->k,
                                                         bk_params->tlwe_params->alpha_min,
                                                         bk_params->tlwe_params->alpha_max);
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    TLweSample *temp = new_TLweSample(temp_accum_params);

//    cout << "tfhe_blindRotate_FFT_16_2_vector" << endl;

    for (int i = 0; i <= temp->k; ++i) {
        free(temp->a[i].coefsT);
        cudaMalloc(&(temp->a[i].coefsT), temp->a[i].N * sizeof(int));
        cudaMemset(temp->a[i].coefsT, 0, temp->a[i].N * sizeof(Torus32));
    }
    /*
//    //test out
//    int *tempx = new int[length];
//    cout << "AND part: "<< endl;
//    int valK = 0;
//    for (int bI = 0; bI < bitSize * 4; ++bI) {
////        for (int valK = 0; valK < 1; ++valK) {
//        cudaMemcpy(tempx, (accum->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N;
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        }
//    }
//    cout << endl << "XOR part: "<< endl;
//    for (int bI = 0; bI < bitSize * 4; ++bI) {
////        for (int valK = 0; valK < 1; ++valK) {
//        cudaMemcpy(tempx, (accum->a + 1)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N + vLength * bitSize * N;
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        }
//    }*/
    TLweSample *temp2 = temp;
    TLweSample *temp3 = accum;


    //create variables for fft operations
    const TLweParams *tlwe_params = bk_params->tlwe_params;
    const int k = tlwe_params->k;//1
    const int l = bk_params->l;//2
    const int kpl = bk_params->kpl;//4
    const int Ns2 = N / 2;
    const int length_Ns2 = vLength * nOutputs * bitSize * Ns2;

    IntPolynomial *deca;

    IntPolynomial *decaCoalesce;
    decaCoalesce = new_IntPolynomial(kpl * length);
    free(decaCoalesce->coefs);
    cudaMalloc(&(decaCoalesce->coefs), kpl * length * sizeof(int));

    cufftDoubleComplex **cuDecaFFT;

    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, kpl * length_Ns2 * sizeof(cufftDoubleComplex));

    cufftDoubleComplex **tmpa_gpu;
    tmpa_gpu = new cufftDoubleComplex *[k + 1];;
    for (int i = 0; i <= k; ++i) {
        cudaMalloc(&(tmpa_gpu[i]), length_Ns2 * sizeof(cufftDoubleComplex));
    }



//    cout << "tfhe_blindRotate_FFT_16_2_vector:bitSize: " << bitSize << endl;
    for (int j = 0; j < n; ++j) {//n
//        tfhe_MuxRotate_FFT_16_2_vector(temp2, temp3, bkFFT + j, bara, j, vLength, nOutputs, bitSize, N, bk_params,
//                                       cudaBkFFT[j], cudaBkFFTCoalesce[j], deca, decaCoalesce,
//                                       cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu);
        tfhe_MuxRotate_FFT_16_2_vector(temp2, temp3, bkFFT + j, bara, j, vLength, nOutputs, bitSize, N, bk_params,
                                       NULL, cudaBkFFTCoalesce[j], deca, decaCoalesce,
                                       cuDecaFFT, cuDecaFFTCoalesce, tmpa_gpu);
        swap(temp2, temp3);

    }

/*
//    int *tempx = new int[length];
//    cout << "AND part: "<< endl;
//    int valK = 0;
//    for (int bI = 0; bI < bitSize * 4; ++bI) {
////        for (int valK = 0; valK < 1; ++valK) {
//            cudaMemcpy(tempx, (accum->a + 0)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//            int sI = bI * N;
////            cout << "new: " << valK << ": ";
//            for (int i = 0; i < 10; ++i) {
//                cout << tempx[sI + i] << " ";
//            }
//            cout << endl;
////        }
//    }
//    cout << endl << "XOR part: "<< endl;
//    for (int bI = 0; bI < bitSize * 4; ++bI) {
////        for (int valK = 0; valK < 1; ++valK) {
//        cudaMemcpy(tempx, (accum->a + 0)->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N + vLength * bitSize * N;
////        cout << "new: " << valK << ": ";
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        }
//    }
//
*/
//    delete_TLweSampleFromCuda(temp, temp_accum_params);
    cudaFree(decaCoalesce->coefs);
    decaCoalesce->coefs = NULL;
    delete_IntPolynomial(decaCoalesce);
    cudaFree(cuDecaFFTCoalesce);
    for (int i = 0; i <= k; ++i) {
        cudaFree(tmpa_gpu[i]);
        cudaFree((temp2->a + i)->coefsT);
    }
}

#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BLIND_ROTATE_AND_EXTRACT_FFT
#undef INCLUDE_TFHE_BLIND_ROTATE_AND_EXTRACT_FFT

/**
 * result = LWE(v_p) where p=barb-sum(bara_i.s_i) mod 2N
 * @param result the output LWE sample
 * @param v a 2N-elt anticyclic function (represented by a TorusPolynomial)
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param barb A coefficients between 0 and 2N-1
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
 */
EXPORT void tfhe_blindRotateAndExtract_FFT(LweSample *result,
                                           const TorusPolynomial *v,
                                           const TGswSampleFFT *bk,
                                           const int barb,
                                           const int *bara,
                                           const int n,
                                           const TGswParams *bk_params) {

    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int N = accum_params->N;
    const int _2N = 2 * N;

    // Test polynomial
    TorusPolynomial *testvectbis = new_TorusPolynomial(N);
    // Accumulator
    TLweSample *acc = new_TLweSample(accum_params);

    // testvector = X^{2N-barb}*v
    if (barb != 0) torusPolynomialMulByXai(testvectbis, _2N - barb, v);
    else torusPolynomialCopy(testvectbis, v);


    tLweNoiselessTrivial(acc, testvectbis, accum_params);
    // Blind rotation
    tfhe_blindRotate_FFT(acc, bk, bara, n, bk_params);
//    //    test morshed start
//    for (int valK = 0; valK <= 1; ++valK) {
//        cout << "old: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << (acc->a + valK)->coefsT[i] << " ";
//        }
//        cout << endl;
//    }
//    //test morshed end
    // Extraction
    tLweExtractLweSample(result, acc, extract_params, accum_params);
    //    test morshed start
//    cout << "old: ";
//    for (int i = 0; i < N; ++i) {
//        cout << result->a[i] << " ";
//    }
//    cout << endl;
    //test morshed end
//    cout << "old: " << result->b << endl;

    delete_TLweSample(acc);
    delete_TorusPolynomial(testvectbis);
}

//////new
EXPORT void tfhe_blindRotateAndExtract_FFT_16(LweSample_16 *result,
                                              const TorusPolynomial *v,
                                              const TGswSampleFFT *bk,
                                              const int *barb,
                                              const int *bara,
                                              const int n,
                                              int bitSize,
                                              const TGswParams *bk_params,
                                              cufftDoubleComplex ****cudaBkFFT,
                                              cufftDoubleComplex ***cudaBkFFTCoalesce) {

    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int N = accum_params->N;//1024
    const int _2N = 2 * N;
    int length = bitSize * N;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    // Test polynomial
    TorusPolynomial *testvectbis = new_TorusPolynomial(length);
    free(testvectbis->coefsT);
    cudaMalloc(&(testvectbis->coefsT), length * sizeof(int));

    // Accumulator
    const TLweParams *temp_accum_params = new_TLweParams(length, accum_params->k, accum_params->alpha_min,
                                                         accum_params->alpha_max);
//    TLweSample *acc = new_TLweSample(temp_accum_params);
//    for (int i = 0; i <= acc->k; ++i) {
//        free(acc->a[i].coefsT);
//        cudaMalloc(&(acc->a[i].coefsT), acc->a[i].N * sizeof(int));
//        setVectorTo<<<gridSize, BLOCKSIZE>>>(acc->a[i].coefsT, 0, acc->a[i].N);
//    }
    /*version 2 start*/
    TLweSample *accV2 = new_TLweSample(temp_accum_params);
    free(accV2->a[0].coefsT);
    free(accV2->a[1].coefsT);
    cudaMalloc(&(accV2->a[0].coefsT), accV2->a[0].N * (accV2->k + 1) * sizeof(Torus32));
    cudaMemset(accV2->a[0].coefsT, 0, accV2->a[0].N * (accV2->k + 1) * sizeof(Torus32));
    (accV2->a + 1)->coefsT = (accV2->a + 0)->coefsT + (accV2->a + 0)->N;
    /*version 2 end*/
    // testvector = X^{2N-barb}*v
    torusPolynomialMulByXai_16(testvectbis, _2N, barb, bitSize, v);
//    cudaMemcpy(acc->b->coefsT, testvectbis->coefsT, length * sizeof(int), cudaMemcpyDeviceToDevice);

//    cout << "barb[0]: " << barb[0] << endl;

//    tLweNoiselessTrivial_16(acc, testvectbis, accum_params);
    /*version 2 start*/
    cudaMemcpy(accV2->b->coefsT, testvectbis->coefsT, length * sizeof(int), cudaMemcpyDeviceToDevice);
    /*version 2 end*/

    //testCode
//    int *_tempBara = (int *) malloc(sizeof(int) * bitSize * N);
//    cudaMemcpy(_tempBara, acc->a[1].coefsT, bitSize * N * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "acc_a_1" << endl;
//    for (int i = 0; i < bitSize; ++i) {
//        int startIndex = i * N;
////        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
////            cout << _source[startIndex + j] << " ";
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
    // Blind rotation
//    tfhe_blindRotate_FFT_16(acc, bk, bara, n, bitSize, bk_params, cudaBkFFT, cudaBkFFTCoalesce);
    tfhe_blindRotate_FFT_16V2(accV2, bk, bara, n, bitSize, bk_params, cudaBkFFT, cudaBkFFTCoalesce);

    //test morshed start
//    int *tempx = new int[N * bitSize];
//    for (int bI = 0; bI < bitSize; ++bI) {
//        for (int valK = 0; valK <= 1; ++valK) {
//            cudaMemcpy(tempx, (acc->a + valK)->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            int sI = bI * N;
////            cout << "new: ";
//            for (int i = 0; i < 10; ++i) {
//                cout << tempx[sI + i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    cout << endl;
//    for (int bI = 0; bI < bitSize; ++bI) {
//        for (int valK = 0; valK <= 1; ++valK) {
//            cudaMemcpy(tempx, (accV2->a + valK)->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            int sI = bI * N;
////            cout << "new: ";
//            for (int i = 0; i < 10; ++i) {
//                cout << tempx[sI + i] << " ";
//            }
//            cout << endl;
//        }
//    }
//    cout << endl;
    // Extraction
    tLweExtractLweSample_16(result, accV2, extract_params, bitSize, accum_params);
//    int *tempx = new int[N * bitSize];
//    for (int bI = 0; bI < bitSize; ++bI) {
//        cudaMemcpy(tempx, result->a, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N;
////        cout << "new: ";
//        for (int i = 0; i < N; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
//        cout << "new: " << result->b[bI] << endl;
//    }
//    cout << endl;

//    delete_TLweSampleFromCuda(acc, accum_params);
//    delete_TLweSampleFromCuda(accV2, accum_params);
    cudaFree(accV2->a->coefsT);
    cudaFree(testvectbis->coefsT);
    testvectbis->coefsT = NULL;
    delete_TorusPolynomial(testvectbis);
}

EXPORT void tfhe_blindRotateAndExtract_FFT_16_2(LweSample_16 *result,
                                                const TorusPolynomial *v,
                                                const TGswSampleFFT *bk,
                                                const int *barb,
                                                const int *bara,
                                                const int n,
                                                int nOutputs,
                                                int bitSize,
                                                const TGswParams *bk_params,
                                                cufftDoubleComplex ****cudaBkFFT,
                                                cufftDoubleComplex ***cudaBkFFTCoalesce) {

//    cout << "Inside tfhe_blindRotateAndExtract_FFT_16_2..." << endl;
    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int N = accum_params->N;//1024
    const int _2N = 2 * N;
    int length = nOutputs * bitSize * N;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE); //32
//    cout << "gridSize: " << gridSize << endl; //32
//    cout << "BLOCKSIZE: " << BLOCKSIZE << endl; //1024
//    cout << "N: " << N << endl;
//    cout << "length: " << length << endl;

    // Test polynomial
    TorusPolynomial *testvectbis = new_TorusPolynomial(N * bitSize);
    free(testvectbis->coefsT);
    cudaMalloc(&(testvectbis->coefsT), length * sizeof(int));

    // Accumulator
    const TLweParams *temp_accum_params = new_TLweParams(length, accum_params->k, accum_params->alpha_min,
                                                         accum_params->alpha_max);
//    TLweSample *acc = new_TLweSample(temp_accum_params);
//    for (int i = 0; i <= acc->k; ++i) {
//        free(acc->a[i].coefsT);
//        cudaMalloc(&(acc->a[i].coefsT), acc->a[i].N * sizeof(int));
//        setVectorTo<<<gridSize, BLOCKSIZE>>>(acc->a[i].coefsT, 0, acc->a[i].N);
//    }

    //accV2
    TLweSample *accV2 = new_TLweSample(temp_accum_params);
    free(accV2->a[0].coefsT);
    free(accV2->a[1].coefsT);
    cudaMalloc(&(accV2->a[0].coefsT), accV2->a[0].N * (accV2->k + 1) * sizeof(Torus32));
    cudaMemset(accV2->a[0].coefsT, 0, accV2->a[0].N * (accV2->k + 1) * sizeof(Torus32));
    (accV2->a + 1)->coefsT = (accV2->a + 0)->coefsT + (accV2->a + 0)->N;


    // testvector = X^{2N-barb}*v
    torusPolynomialMulByXai_16_2(testvectbis, _2N, barb, nOutputs, bitSize, v);
//    cudaMemcpy(acc->b->coefsT, testvectbis->coefsT, length * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(accV2->b->coefsT, testvectbis->coefsT, length * sizeof(int), cudaMemcpyDeviceToDevice);

    //testCode
//    int *_tempBara = new int[length];
//    cudaMemcpy(_tempBara, accV2->b->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "-----------" << endl;
//    for (int i = 0; i < bitSize * nOutputs; ++i) {
//        int startIndex = i * N;
////        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
    // Blind rotation
//    tfhe_blindRotate_FFT_16_2(acc, bk, bara, n, nOutputs, bitSize, bk_params, cudaBkFFT, cudaBkFFTCoalesce);
//
//    cout << "caliing v1" << endl;
    tfhe_blindRotate_FFT_16_2v2(accV2, bk, bara, n, nOutputs, bitSize, bk_params, cudaBkFFT, cudaBkFFTCoalesce);
//    cout << "fin v1" << endl;


    //test morshed start
//    int *temp1 = new int[N * bitSize * nOutputs];
//    int *temp2 = new int[N * bitSize * nOutputs];
//    int *temp3 = new int[N * bitSize * nOutputs];
//    int *temp4 = new int[N * bitSize * nOutputs];
//    for (int i = 0; i <= 1; ++i) {
////    int i = 0;
////        cudaMemcpy(temp1, (acc->a + i)->coefsT, (acc->a + i)->N * sizeof(Torus32), cudaMemcpyDeviceToHost);
//        cudaMemcpy(temp2, (accV2->a + i)->coefsT, (accV2->a + i)->N * sizeof(Torus32), cudaMemcpyDeviceToHost);
////        cudaMemcpy(temp3, (acc->b)->coefsT, (acc->b)->N * sizeof(Torus32), cudaMemcpyDeviceToHost);
//        cudaMemcpy(temp4, (accV2->b)->coefsT, (accV2->b)->N * sizeof(Torus32), cudaMemcpyDeviceToHost);
//        for (int j = 0; j < nOutputs * bitSize; ++j) {
//            for (int k = 0; k < 10; ++k) {
//                cout << temp2[j * N + k] << " ";
////                assert(temp1[j * N + k] == temp2[j * N + k]);
////                assert(temp3[j * N + k] == temp4[j * N + k]);
//            }
//            cout << endl;
//        }
//    }
//    cout << endl;
//    cout << "Before extraction" << endl;
    //Extraction
    tLweExtractLweSample_16_2(result, accV2, extract_params, nOutputs, bitSize, accum_params);
//    cout << "after extraction" << endl;
//    int *tempx = new int[length];
//    for (int bI = 0; bI < nOutputs * bitSize; ++bI) {
//        cudaMemcpy(tempx, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N;
////        cout << "new: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        cout << "new: " << result->b[bI] << endl;
//    }

//    delete_TLweSampleFromCuda(acc, accum_params);
    cudaFree((accV2->a + 0)->coefsT);
    cudaFree(testvectbis->coefsT);
    testvectbis->coefsT = NULL;
    delete_TorusPolynomial(testvectbis);
}


EXPORT void tfhe_blindRotateAndExtract_FFT_16_2_vector(LweSample_16 *result,
                                                const TorusPolynomial *v,
                                                const TGswSampleFFT *bk,
                                                const int *barb,
                                                const int *bara,
                                                const int n,
                                                int vLength,
                                                int nOutputs,
                                                int bitSize,
                                                const TGswParams *bk_params,
                                                cufftDoubleComplex ****cudaBkFFT,
                                                cufftDoubleComplex ***cudaBkFFTCoalesce) {


    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int N = accum_params->N;//1024
    const int _2N = 2 * N;
    int length = vLength * nOutputs * bitSize * N;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    TorusPolynomial *testvectbis = new_TorusPolynomial(length);
    free(testvectbis->coefsT);
    cudaMalloc(&(testvectbis->coefsT), length * sizeof(int));

    // Accumulator
    const TLweParams *temp_accum_params = new_TLweParams(length, accum_params->k, accum_params->alpha_min,
                                                         accum_params->alpha_max);

    TLweSample *acc = new_TLweSample(temp_accum_params);
    for (int i = 0; i <= acc->k; ++i) {
        free(acc->a[i].coefsT);
        cudaMalloc(&(acc->a[i].coefsT), acc->a[i].N * sizeof(int));
//        setVectorTo<<<gridSize, BLOCKSIZE>>>(acc->a[i].coefsT, 0, acc->a[i].N);
        cudaMemset(acc->a[i].coefsT, 0, acc->a[i].N * sizeof(int));
    }

    // testvector = X^{2N-barb}*v
    torusPolynomialMulByXai_16_2_vector(testvectbis, _2N, barb, vLength, nOutputs, bitSize, v);

    cudaMemcpy(acc->b->coefsT, testvectbis->coefsT, length * sizeof(int), cudaMemcpyDeviceToDevice);

//    cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
    //testCode
//    int *_tempBara = new int[length];
//    cudaMemcpy(_tempBara, acc->b->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * 4; ++i) {
//        int startIndex = i * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    for (int i = 0; i < bitSize * 4; ++i) {
//        int startIndex = (vLength * bitSize + i) * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//
//     Blind rotation
//    cout << "tfhe_blindRotateAndExtract_FFT_16_2_vector:bitSize: " << bitSize << endl;
    tfhe_blindRotate_FFT_16_2_vector(acc, bk, bara, n, vLength, nOutputs, bitSize, bk_params, cudaBkFFT, cudaBkFFTCoalesce);

//    cout << "n: " << n << endl;

//    int i = 0;
//    int *_tempBara = new int[length];
//    cudaMemcpy(_tempBara, acc->a[i].coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < bitSize * 4; ++i) {
//        int startIndex = i * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << "XOR PART" << endl;
//    for (int i = 0; i < bitSize * 4; ++i) {
//        int startIndex = (vLength * bitSize + i) * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << _tempBara[startIndex + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;


    // Extraction
    tLweExtractLweSample_16_2_vector(result, acc, extract_params, vLength, nOutputs, bitSize, accum_params);

//    int *tempx = new int[length];
//    cout << "AND PART" << endl;
//    for (int bI = 0; bI < 4 * bitSize; ++bI) {
//        cudaMemcpy(tempx, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N;
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        cout << "new: " << result->b[bI] << endl;
//    }
//    cout << "XOR PART" << endl;
//    for (int bI = 0; bI < 4 * bitSize; ++bI) {
//        cudaMemcpy(tempx, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//        int sI = bI * N + vLength * bitSize * N;
//        for (int i = 0; i < 10; ++i) {
//            cout << tempx[sI + i] << " ";
//        }
//        cout << endl;
////        cout << "new: " << result->b[bI + vLength * bitSize] << endl;
//    }

    delete_TLweSampleFromCuda(acc, accum_params);
    cudaFree(acc->a->coefsT);
    cudaFree(acc->b->coefsT);
    cudaFree(testvectbis->coefsT);
    testvectbis->coefsT = NULL;
    delete_TorusPolynomial(testvectbis);
}


#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BOOTSTRAP_WO_KS_FFT
#undef INCLUDE_TFHE_BOOTSTRAP_WO_KS_FFT
/**
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
 */
EXPORT void tfhe_bootstrap_woKS_FFT(LweSample *result,
                                    const LweBootstrappingKeyFFT *bk,
                                    Torus32 mu,
                                    const LweSample *x) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int N = accum_params->N;
    const int Nx2 = 2 * N;
    const int n = in_params->n;

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    int *bara = new int[N];


    // Modulus switching
    int barb = modSwitchFromTorus32(x->b, Nx2);

//    cout << "old: ";
    for (int i = 0; i < n; i++) {
        bara[i] = modSwitchFromTorus32(x->a[i], Nx2);
//        if( i < 10)
//            cout << bara[i] << " ";
    }
//    cout << endl;

    // the initial testvec = [mu,mu,mu,...,mu]
    for (int i = 0; i < N; i++) testvect->coefsT[i] = mu;

    // Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, bk_params);


    delete[] bara;
    delete_TorusPolynomial(testvect);
}

#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BOOTSTRAP_FFT
#undef INCLUDE_TFHE_BOOTSTRAP_FFT
/**
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
 */
EXPORT void tfhe_bootstrap_FFT(LweSample *result,
                               const LweBootstrappingKeyFFT *bk,
                               Torus32 mu,
                               const LweSample *x) {

    LweSample *u = new_LweSample(&bk->accum_params->extracted_lweparams);

    tfhe_bootstrap_woKS_FFT(u, bk, mu, x);
    //test morshed start
//    cout << "oldu: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << u->a[i] << " ";
//    }
//    cout << endl;
    //test morshed end
    // Key switching
    lweKeySwitch(result, bk->ks, u);
    //test morshed start
//    cout << "old after ks: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << result->a[i] << " ";
//    }
//    cout << endl;
//    //test morshed end

    delete_LweSample(u);
}








////new
EXPORT void tfhe_bootstrap_FFT_16(LweSample_16 *result,
                                  const LweBootstrappingKeyFFT *bk,
                                  Torus32 mu,
                                  int bitSize,
                                  const LweSample_16 *x,
                                  cufftDoubleComplex ****cudaBkFFT,
                                  cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu,
                                  Torus32 ****ks_a_gpu_extended,
                                  int ***ks_b_gpu,
                                  double ***ks_cv_gpu,
                                  Torus32* ks_a_gpu_extendedPtr,
                                  Torus32 *ks_b_gpu_extendedPtr,
                                  double *ks_cv_gpu_extendedPtr) {

    int length = bitSize * (&bk->accum_params->extracted_lweparams)->n;

    LweSample_16 *u = newLweSample_16(bitSize, &bk->accum_params->extracted_lweparams);
    free(u->a);
    cudaMalloc(&(u->a), length * sizeof(int));
//    cudaCheckErrors("cudaMalloc/cudaMemcpy failed");
//    double sT = omp_get_wtime();
//    if (bitSize == 1) {
//        tfhe_bootstrap_woKS_FFT_n_shared<<<bitSize, 1024>>>();
//    } else {
        tfhe_bootstrap_woKS_FFT_16(u, bk, mu, bitSize, x, cudaBkFFT, cudaBkFFTCoalesce);
//    }

//    cout << "Bootstrapping Time: " << omp_get_wtime() - sT << endl;
    //test morshed start
//    int number = (&bk->accum_params->extracted_lweparams)->n;//1024
//    int *temp_a = new int[number * bitSize];
//    cudaMemcpy(temp_a, u->a, number * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int j = 0; j < bitSize; ++j) {
//        cout << "newu: ";
//        for (int i = 0; i < 1024; ++i) {
//            cout << temp_a[j * number + i] << " ";
//        }
//        cout << endl;
//        cout << u->b[j] << " ";
//    }
//    cout << endl;
//    //test morshed end

    // Key switching
//    sT = omp_get_wtime();
    lweKeySwitch_16(result, bk->ks, u, bitSize, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                    ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//    cout << "KS Time: " << omp_get_wtime() - sT << endl;
    //test morshed start
//    int number = 500;
//    int *temp_a = new int[number * bitSize];
//    cudaMemcpy(temp_a, result->a, number * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int j = 0; j < bitSize; ++j) {
////        cout << "new after ks: ";
//        for (int i = 0; i < 500; ++i) {
//            cout << temp_a[j * number + i] << " ";
//        }
//        cout << endl;
//        cout << result->b[j] << " " << result->current_variance[j] << " ";
//    }
//    cout << endl;
    //test morshed end
    cudaFree(u->a);
    u->a = NULL;
    freeLweSample_16(u);
}

EXPORT void tfhe_bootstrap_FFT_16_2(LweSample_16 *result,
                                    const LweBootstrappingKeyFFT *bk,
                                    Torus32 mu,
                                    int nOutputs,
                                    int bitSize,
                                    const LweSample_16 *x,
                                    cufftDoubleComplex ****cudaBkFFT,
                                    cufftDoubleComplex ***cudaBkFFTCoalesce,
                                    Torus32 ****ks_a_gpu,
                                    Torus32 ****ks_a_gpu_extended,
                                    int ***ks_b_gpu,
                                    double ***ks_cv_gpu,
                                    Torus32 *ks_a_gpu_extendedPtr,
                                    Torus32 *ks_b_gpu_extendedPtr,
                                    double *ks_cv_gpu_extendedPtr) {

    int length = nOutputs * bitSize * (&bk->accum_params->extracted_lweparams)->n;
//    cout << "(&bk->accum_params->extracted_lweparams)->n: " << (&bk->accum_params->extracted_lweparams)->n << endl;

    LweSample_16 *u = newLweSample_16_2(nOutputs, bitSize, &bk->accum_params->extracted_lweparams);
    free(u->a);
    cudaMalloc(&(u->a), length * sizeof(int));

//    double sT = omp_get_wtime();
    tfhe_bootstrap_woKS_FFT_16_2(u, bk, mu, nOutputs, bitSize, x, cudaBkFFT, cudaBkFFTCoalesce);
//    double eT = omp_get_wtime();
//    cout << "Boot Time: " << eT - sT << endl;
    //test morshed start
//    int *temp_a = new int[length];
//    cudaMemcpy(temp_a, u->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int j = 0; j < nOutputs * bitSize; ++j) {
////        cout << "newu: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[j * 1024 + i] << " ";
//        }
//        cout << endl;
////        cout << u->b[j] << " ";
//    }
//    cout << endl;
    //test morshed end

    //Key switching
//    sT = omp_get_wtime();
    lweKeySwitch_16_2(result, bk->ks, u, nOutputs, bitSize, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                      ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//    eT = omp_get_wtime();
//    cout << "KS Time: " << eT - sT << endl;
    //test morshed start
//    int number = 500;
//    length = nOutputs * bitSize * number;
//    int *temp_a = new int[length];
//    cudaMemcpy(temp_a, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int j = 0; j < nOutputs * bitSize; ++j) {
//        cout << "new after ks: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[j * number + i] << " ";
//        }
//        cout << endl;
////        cout << result->b[j] << " " << result->current_variance[j] << endl;
//    }
//    cout << endl;
//    //test morshed end
    cudaFree(u->a);
    u->a = NULL;
    freeLweSample_16(u);
}


EXPORT void tfhe_bootstrap_FFT_16_2_vector(LweSample_16 *result,
                                           const LweBootstrappingKeyFFT *bk,
                                           Torus32 mu,
                                           int vLength,
                                           int nOutputs,
                                           int bitSize,
                                           const LweSample_16 *x,
                                           cufftDoubleComplex ****cudaBkFFT,
                                           cufftDoubleComplex ***cudaBkFFTCoalesce,
                                           Torus32 ****ks_a_gpu,
                                           Torus32 ****ks_a_gpu_extended,
                                           int ***ks_b_gpu,
                                           double ***ks_cv_gpu,
                                           Torus32 *ks_a_gpu_extendedPtr,
                                           Torus32 *ks_b_gpu_extendedPtr,
                                           double *ks_cv_gpu_extendedPtr) {
//    cout << "inside: " << "tfhe_bootstrap_FFT_16_2_vector" << endl;
//    cout << "vLength: " << vLength << " nOutputs: " << nOutputs << " bitSize: " << bitSize << endl;
    int length = vLength * bitSize * nOutputs * (&bk->accum_params->extracted_lweparams)->n;

    LweSample_16 *u = newLweSample_16_2(nOutputs, vLength * bitSize, &bk->accum_params->extracted_lweparams);
    free(u->a);
    cudaMalloc(&(u->a), length * sizeof(int));

    tfhe_bootstrap_woKS_FFT_16_2_vector(u, bk, mu, vLength, nOutputs, bitSize, x, cudaBkFFT, cudaBkFFTCoalesce);
//    cudaCheckErrors("ERROR... EXIT");
//    int *temp_a = new int[length];
//    cudaMemcpy(temp_a, u->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND part" << endl;
//    int testNumber = 4;
//    for (int j = 0; j < testNumber * bitSize; ++j) {
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[j * 1024 + i] << " ";
//        }
//        cout << endl;
//        cout << u->b[j] << " ";
//    }
//    cout << endl;
//    cout << "XOR part" << endl;
//    for (int j = 0; j < testNumber * bitSize; ++j) {
//        cout << "newu: ";
//        int sI = j * 1024 + vLength * bitSize * 1024;
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[sI + i] << " ";
//        }
//        cout << endl;
//        cout << u->b[j] << " ";
//    }
//    cout << endl;

    //key switch left
    lweKeySwitch_16_2_vector(result, bk->ks, u, vLength, nOutputs, bitSize, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu,
                             ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    int number = 500;
//    length = vLength * nOutputs * bitSize * number;
//    int *temp_a = new int[length];
//    cudaMemcpy(temp_a, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    testNumber = 5;
//    for (int j = 0; j < testNumber * bitSize; ++j) {
//        cout << "new after ks: ";
//        int sI = j * number;
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[sI + i] << " ";
//        }
//        cout << endl;
////        cout << result->b[j] << " " << result->current_variance[j] << endl;
//    }
//    cout << endl;
//
//    cout << "XOR PART" << endl;
//    for (int j = 0; j < testNumber * bitSize; ++j) {
//        cout << "new after ks: ";
//        int sI = j * number + vLength * bitSize * number;
////        int sIB = vLength * bitSize;
//        for (int i = 0; i < 10; ++i) {
//            cout << temp_a[sI + i] << " ";
//        }
//        cout << endl;
////        cout << result->b[sIB + j] << " " << result->current_variance[sI + j] << endl;
//    }
//    cout << endl;
//


    cudaFree(u->a);
    u->a = NULL;
    freeLweSample_16(u);
}


#endif
















//allocate memory space for a LweBootstrappingKeyFFT

EXPORT LweBootstrappingKeyFFT *alloc_LweBootstrappingKeyFFT() {
    return (LweBootstrappingKeyFFT *) malloc(sizeof(LweBootstrappingKeyFFT));
}

EXPORT LweBootstrappingKeyFFT *alloc_LweBootstrappingKeyFFT_array(int nbelts) {
    return (LweBootstrappingKeyFFT *) malloc(nbelts * sizeof(LweBootstrappingKeyFFT));
}

//free memory space for a LweKey
EXPORT void free_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *ptr) {
    free(ptr);
}

EXPORT void free_LweBootstrappingKeyFFT_array(int nbelts, LweBootstrappingKeyFFT *ptr) {
    free(ptr);
}

//initialize the key structure

EXPORT void init_LweBootstrappingKeyFFT_array(int nbelts, LweBootstrappingKeyFFT *obj, const LweBootstrappingKey *bk) {
    for (int i = 0; i < nbelts; i++) {
        init_LweBootstrappingKeyFFT(obj + i, bk);
    }
}


EXPORT void destroy_LweBootstrappingKeyFFT_array(int nbelts, LweBootstrappingKeyFFT *obj) {
    for (int i = 0; i < nbelts; i++) {
        destroy_LweBootstrappingKeyFFT(obj + i);
    }
}

//allocates and initialize the LweBootstrappingKeyFFT structure
//(equivalent of the C++ new)
EXPORT LweBootstrappingKeyFFT *new_LweBootstrappingKeyFFT(const LweBootstrappingKey *bk) {
    LweBootstrappingKeyFFT *obj = alloc_LweBootstrappingKeyFFT();
    init_LweBootstrappingKeyFFT(obj, bk);
    return obj;
}

EXPORT LweBootstrappingKeyFFT *new_LweBootstrappingKeyFFT_array(int nbelts, const LweBootstrappingKey *bk) {
    LweBootstrappingKeyFFT *obj = alloc_LweBootstrappingKeyFFT_array(nbelts);
    init_LweBootstrappingKeyFFT_array(nbelts, obj, bk);
    return obj;
}

//destroys and frees the LweBootstrappingKeyFFT structure
//(equivalent of the C++ delete)
EXPORT void delete_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *obj) {
    destroy_LweBootstrappingKeyFFT(obj);
    free_LweBootstrappingKeyFFT(obj);
}

EXPORT void delete_LweBootstrappingKeyFFT_array(int nbelts, LweBootstrappingKeyFFT *obj) {
    destroy_LweBootstrappingKeyFFT_array(nbelts, obj);
    free_LweBootstrappingKeyFFT_array(nbelts, obj);
}

//new
//__global__ void setVectorTo(int *destination, int val, int length) {
//    int id = blockIdx.x*blockDim.x+threadIdx.x;
//    if(id < length) {
//        destination[id] = val;
//    }
//}

__global__ void modSwitchFromTorus32_GPU(int *destination, int *phase_source, int Msize, int N, int n, int bitSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int phaseSourceLength = bitSize * n;
    int destinationLength = bitSize * N;
    if (id < destinationLength) {
        destination[id] = 0;
    }
//    __syncthreads();
    if (id < phaseSourceLength) {
        int bitIndex = (int) id / n;
        int arrayIndex = id % n;
        int destinationIndex = bitIndex * N + arrayIndex;
        //get bit Index
        unsigned long one = 1;
        unsigned long phase = phase_source[id];
        unsigned long interv = ((one << 63) / Msize) * 2; // width of each intervall
        unsigned long half_interval = interv / 2; // begin of the first intervall
        unsigned long phase64 = (uint64_t(phase) << 32) + half_interval;
        destination[destinationIndex] = (int) (phase64 / interv);
    }
//    __syncthreads();
}



//__global__ void tfhe_bootstrap_woKS_FFT_n_shared(Torus32 *result, Torus32 mu, int vLength, int bitSize, int nOutputs,
//                                                Torus32 x, ) {
//    register int N = 1024;//accum_params->N;//
//    register int Nx2 = 2048;//2 * N;
//    register int n = 500;//in_params->n;//500
//
//}



EXPORT void tfhe_bootstrap_woKS_FFT_16(LweSample_16 *result,
                                       const LweBootstrappingKeyFFT *bk,
                                       Torus32 mu,
                                       int bitSize,
                                       const LweSample_16 *x,
                                       cufftDoubleComplex ****cudaBkFFT,
                                       cufftDoubleComplex ***cudaBkFFTCoalesce) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int N = accum_params->N;
    const int Nx2 = 2 * N;
    const int n = in_params->n;
    //required for GPU
    int length = N * bitSize;
    int lweSampleLength = bitSize * n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "N: " << N << " n: " << n << endl; //1024 500


    TorusPolynomial *testvect = new_TorusPolynomial(N * bitSize);
    int *barb = new int[bitSize];
    int *bara;
    cudaMalloc(&bara, length * sizeof(int));
//    cout << "bara length: " << length << endl;

    // Modulus switching
    for (int i = 0; i < bitSize; ++i) {
        barb[i] = modSwitchFromTorus32(x->b[i], Nx2);
    }

    modSwitchFromTorus32_GPU<<<gridSize, BLOCKSIZE>>>(bara, x->a, Nx2, N, n, bitSize);

//    int *temp_a = new int[lweSampleLength];
//    cudaMemcpy(temp_a, x->a, lweSampleLength * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "x->a: " << endl;
//    for (int i = 0; i < 1; ++i) {
////        cout << "new: ";
//        for (int j = 0; j < n; ++j) {
//            cout << temp_a[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;

//    int *temp_bara = new int[length];
//    cudaMemcpy(temp_bara, bara, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "bara: " << endl;
//    for (int i = 0; i < bitSize; ++i) {
////        cout << "new: ";
//        for (int j = 0; j < 1024; ++j) {
//            cout << temp_bara[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;

    free(testvect->coefsT);
    cudaMalloc(&(testvect->coefsT), length * sizeof(int));
    setVectorTo<<<gridSize, BLOCKSIZE>>>(testvect->coefsT, mu, length);
//    cout << "mu: " <<  mu << endl;

//    int *testvect_coefsT = new int[length];
//    cudaMemcpy(testvect_coefsT, testvect->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
////        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << testvect_coefsT[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;

    // Bootstrapping rotation and extraction
//    cout << "BBBBBB st" << endl;
    tfhe_blindRotateAndExtract_FFT_16(result, testvect, bk->bkFFT, barb, bara, n, bitSize, bk_params,
                                      cudaBkFFT, cudaBkFFTCoalesce);
//    cout << "BBBBBB end" << endl;


    cudaFree(bara);
    cudaFree(testvect->coefsT);
    testvect->coefsT = NULL;
    delete[] barb;
    delete_TorusPolynomial(testvect);
}


EXPORT void tfhe_bootstrap_woKS_FFT_16_2(LweSample_16 *result,
                                         const LweBootstrappingKeyFFT *bk,
                                         Torus32 mu,
                                         int nOutputs,
                                         int bitSize,
                                         const LweSample_16 *x,
                                         cufftDoubleComplex ****cudaBkFFT,
                                         cufftDoubleComplex ***cudaBkFFTCoalesce) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int N = accum_params->N;//1024
    const int Nx2 = 2 * N;
    const int n = in_params->n;//500
    //required for GPU
    int length = nOutputs * bitSize * N;
    int lweSampleLength = nOutputs * bitSize * n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    TorusPolynomial *testvect = new_TorusPolynomial(nOutputs * bitSize * N);
    int *barb = new int[nOutputs * bitSize];
    int *bara;
    cudaMalloc(&bara, length * sizeof(int));

    // Modulus switching
    for (int i = 0; i < nOutputs * bitSize; ++i) {
        barb[i] = modSwitchFromTorus32(x->b[i], Nx2);
    }

    modSwitchFromTorus32_GPU<<<gridSize, BLOCKSIZE>>>(bara, x->a, Nx2, N, n, bitSize * nOutputs);

//    int *temp_bara = new int[length];
//    cudaMemcpy(temp_bara, bara, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * nOutputs; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_bara[i * N + j] << " ";
//        }
//        cout << endl;
//    }

    free(testvect->coefsT);
    cudaMalloc(&(testvect->coefsT), length * sizeof(int));
    setVectorTo<<<gridSize, BLOCKSIZE>>>(testvect->coefsT, mu, length);

//    cout <<"tfhe_bootstrap_woKS_FFT_16_2" << endl;
//    int *testvect_coefsT = new int[length];
//    cudaMemcpy(testvect_coefsT, testvect->coefsT, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * nOutputs; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            cout << testvect_coefsT[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;

    // Bootstrapping rotation and extraction
//    cout << "Bootstrapping rotation and extraction st:" << endl;
    tfhe_blindRotateAndExtract_FFT_16_2(result, testvect, bk->bkFFT, barb, bara, n, nOutputs, bitSize, bk_params,
                                        cudaBkFFT, cudaBkFFTCoalesce);
//    cout << "Bootstrapping rotation and extraction end" << endl;

    cudaFree(bara);
    cudaFree(testvect->coefsT);
    testvect->coefsT = NULL;
    delete_TorusPolynomial(testvect);
    delete[] barb;
}

EXPORT void tfhe_bootstrap_woKS_FFT_16_2_vector(LweSample_16 *result,
                                         const LweBootstrappingKeyFFT *bk,
                                         Torus32 mu,
                                         int vLength,
                                         int nOutputs,
                                         int bitSize,
                                         const LweSample_16 *x,
                                         cufftDoubleComplex ****cudaBkFFT,
                                         cufftDoubleComplex ***cudaBkFFTCoalesce) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int N = accum_params->N;//1024
    const int Nx2 = 2 * N;
    const int n = in_params->n;//500
    //required for GPU
    int length = vLength * nOutputs * bitSize * N;
    int lweSampleLength = vLength * nOutputs * bitSize * n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    TorusPolynomial *testvect = new_TorusPolynomial(vLength * nOutputs * bitSize * N);

    int *barb = new int[vLength * bitSize * nOutputs];
    int *bara;
    cudaMalloc(&bara, length * sizeof(int));

    for (int i = 0; i < vLength * nOutputs * bitSize; ++i) {
        barb[i] = modSwitchFromTorus32(x->b[i], Nx2);
    }

    gridSize = gridSize/vLength;
    for (int i = 0; i < vLength; ++i) {
        int xIndex = bitSize * nOutputs * n * i;
        int baraIndex = bitSize * nOutputs * N * i;
        modSwitchFromTorus32_GPU<<<gridSize, BLOCKSIZE>>>(bara + baraIndex, x->a + xIndex, Nx2, N, n, bitSize * nOutputs);
    }
    //test out
//    int *temp_bara = new int[length];
//    cudaMemcpy(temp_bara, bara, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < 2 * bitSize; ++i) {
//        int sI = i * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_bara[i * N + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << "XOR PART" << endl;
//    for (int i = 0; i < 2 * bitSize; ++i) {
//        int sI = (vLength * bitSize + i) * N;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp_bara[sI + j] << " ";
//        }
//        cout << endl;
//    }

    gridSize = gridSize * vLength;
    free(testvect->coefsT);
    cudaMalloc(&(testvect->coefsT), length * sizeof(int));
    setVectorTo<<<gridSize, BLOCKSIZE>>>(testvect->coefsT, mu, length);

    // Bootstrapping rotation and extraction
//    cout << "tfhe_bootstrap_woKS_FFT_16_2_vector:bitSize: " << bitSize << endl;
    tfhe_blindRotateAndExtract_FFT_16_2_vector(result, testvect, bk->bkFFT, barb, bara, n, vLength, nOutputs, bitSize,
                                               bk_params, cudaBkFFT, cudaBkFFTCoalesce);

    cudaFree(bara);
    cudaFree(testvect->coefsT);
    testvect->coefsT = NULL;
    delete[] barb;
    delete_TorusPolynomial(testvect);
}



__global__ void fullGPUBootstrapping(int *result, int nBits, cufftDoubleComplex *cudaBkFFTCoalesce_single, Torus32* ks_a_gpu_extendedPtr,
                                     Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

}







