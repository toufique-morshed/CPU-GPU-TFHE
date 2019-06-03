#ifndef TFHE_TEST_ENVIRONMENT
/* ***************************************************
   TLWE fft operations
 *************************************************** */

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
#include "polynomials_arithmetic.h"
#include "lagrangehalfc_arithmetic.h"
#include "lagrangehalfc_impl.h"
#include <fstream>

#define P_LIMIT 10

using namespace std;
#define INCLUDE_ALL

#else
#undef EXPORT
#define EXPORT
#endif


#if defined INCLUDE_ALL || defined INCLUDE_INIT_TLWESAMPLE_FFT
#undef INCLUDE_INIT_TLWESAMPLE_FFT
EXPORT void init_TLweSampleFFT(TLweSampleFFT *obj, const TLweParams *params) {
    //a is a table of k+1 polynomials, b is an alias for &a[k]
    const int k = params->k;
    LagrangeHalfCPolynomial *a = new_LagrangeHalfCPolynomial_array(k + 1, params->N);
    double current_variance = 0;
    new(obj) TLweSampleFFT(params, a, current_variance);
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_DESTROY_TLWESAMPLE_FFT
#undef INCLUDE_DESTROY_TLWESAMPLE_FFT
EXPORT void destroy_TLweSampleFFT(TLweSampleFFT *obj) {
    const int k = obj->k;
    delete_LagrangeHalfCPolynomial_array(k + 1, obj->a);
    obj->~TLweSampleFFT();
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TLWE_TO_FFT_CONVERT
#undef INCLUDE_TLWE_TO_FFT_CONVERT
// Computes the inverse FFT of the coefficients of the TLWE sample
EXPORT void tLweToFFTConvert(TLweSampleFFT *result, const TLweSample *source, const TLweParams *params) {
    const int k = params->k;

    for (int i = 0; i <= k; ++i)
        TorusPolynomial_ifft(result->a + i, source->a + i);
    result->current_variance = source->current_variance;
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TLWE_FROM_FFT_CONVERT
#undef INCLUDE_TLWE_FROM_FFT_CONVERT

#include <iostream>
using namespace std;
// Computes the FFT of the coefficients of the TLWEfft sample
EXPORT void tLweFromFFTConvert(TLweSample *result, const TLweSampleFFT *source, const TLweParams *params) {
    const int k = params->k;
    //test start morshed
//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        int j = 1;
//        cout << (result->a + j)->coefsT[i] << " ";
//    }
//    cout << endl;
    //test end morshed

//    static int counter1 = 0;
//    static int counter2 = 0;
    for (int i = 0; i <= k; ++i) {
//        if (counter1 < P_LIMIT) {
//            cout << "old input_fft: ";
//            for (int x = 0; x < 30; ++x) {
//                cout << ((LagrangeHalfCPolynomial_IMPL *) (source->a + i))->coefsC[x] << " ";
//            }
//            cout << endl;
//            counter1++;
//        }
        TorusPolynomial_fft(result->a + i, source->a + i);
//        if(counter2 < P_LIMIT) {
//            cout << "old output_fft: ";
//            for (int x = 0; x < 30; ++x) {
//                cout << (result->a + i)->coefsT[x] << " ";
//            }
//            cout << endl;
//            counter2++;
//        }
    }

    //test start morshed
//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        int j = 1;
//        cout << (result->a + j)->coefsT[i] << " ";
//    }
//    cout << endl;
    //test end morshed
    result->current_variance = source->current_variance;
}

//new
EXPORT void tLweFromFFTConvert_gpu(TLweSample *result, cufftDoubleComplex **source, int bitSize, int N, int Ns2,
                                   const TLweParams *params) {
    const int k = params->k;
    for (int i = 0; i <= k; ++i) {
//        static int counter1 = 0;
//        if(counter1 < P_LIMIT) {
//            cout << "new input_fft: ";
//            cufftDoubleComplex * s2 = new cufftDoubleComplex[Ns2*bitSize];
//            cudaMemcpy(s2, source[i], Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//            for (int x = 0; x < 30; ++x) {
//                cout << "(" << s2[x].x << "," << s2[x].y << ") ";
//            }
//            cout << endl;
//        }
        TorusPolynomial_fft_gpu(result->a + i, source[i], bitSize, N, Ns2);
//        if(counter1 < P_LIMIT) {
//            cout << "new output_fft: ";
//            int * temp_int = new int[N * bitSize];
//            cudaMemcpy(temp_int, (result->a + i)->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//            for (int x = 0; x < 30; ++x) {
//                cout << temp_int[x] << " ";
//            }
//            cout << endl;
//        }

//        counter1++;

    }
//    result->current_variance = source->current_variance;
}

EXPORT void tLweFromFFTConvert_gpu_2(TLweSample *result, cufftDoubleComplex **source, cufftDoubleComplex *sourceSingle, int nOutputs, int bitSize, int N, int Ns2,
                                   const TLweParams *params) {
    const int k = params->k;
//    Torus32 *fftPoly;
//    cudaMalloc(&fftPoly, result->a->N * sizeof(int) * (k + 1));
//    cudaFree(result->a->coefsT);
//    cudaFree((result->a + 1)->coefsT);
//    cudaMalloc(&(result->a->coefsT), result->a->N * sizeof(int) * (k + 1));
//    (result->a + 1)->coefsT = (result->a + 0)->coefsT + result->a->N;
//    result->b = (result->a + 1);

//    TorusPolynomial_fft_gpu_2(fftPoly, sourceSingle, nOutputs, bitSize, N, Ns2);
//    int sI = 0;
//    cudaMemcpy(fft)

    for (int i = 0; i <= k; ++i) {
////        cout << "i: " << i << endl;
////        static int counter1 = 0;
////        if(counter1 < P_LIMIT) {
////            cout << "new input_fft: ";
////            cufftDoubleComplex * s2 = new cufftDoubleComplex[Ns2*bitSize];
////            cudaMemcpy(s2, source[i], Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
////            for (int x = 0; x < 30; ++x) {
////                cout << "(" << s2[x].x << "," << s2[x].y << ") ";
////            }
////            cout << endl;
////        }
        TorusPolynomial_fft_gpu_2((result->a + i)->coefsT, source[i], nOutputs, bitSize, N, Ns2);
//        int sI = i * result->a->N;
//        cout << " Hi";
//        cudaMemcpy((result->a + i)->coefsT, fftPoly + sI, result->a->N * sizeof(Torus32), cudaMemcpyDeviceToDevice);
////        cudaMemcpy((result->a + i)->coefsT, source[i], Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//
////        if(counter1 < P_LIMIT) {
////            cout << "new output_fft: ";
////            int * temp_int = new int[N * bitSize];
////            cudaMemcpy(temp_int, (result->a + i)->coefsT, N * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
////            for (int x = 0; x < 30; ++x) {
////                cout << temp_int[x] << " ";
////            }
////            cout << endl;
////        }
//
////        counter1++;
//
    }
//    cudaFree(fftPoly);
//    result->current_variance = source->current_variance;
}

EXPORT void tLweFromFFTConvert_gpu_2_vector(TLweSample *result, cufftDoubleComplex **source, int vLength, int nOutputs,
                                            int bitSize, int N, int Ns2, const TLweParams *params) {
    const int k = params->k;
    for (int i = 0; i <= k; ++i) {
        TorusPolynomial_fft_gpu_16_2_Coalesce_vector(result->a + i, source[i], vLength, nOutputs, bitSize, N, Ns2);
    }
//    cout << "tLweFromFFTConvert_gpu_2_vector:bitSize: " << bitSize << endl;
//    result->current_variance = source->current_variance;
}


EXPORT void tLweFromFFTConvert_16(TLweSample *result, const TLweSampleFFT *source, int startIndex, int endIndex,
                                  int bitSize, const TLweParams *params) {
    const int k = params->k;
//    cout << "params->k: " << params->k << endl;
//    cout << "result->a->N: " << result->a->N << endl;
    //test start morshed
//    cout << "new: ";
//    for (int i = startIndex; i < startIndex + 10; ++i) {
//        int j = 1;
//        cout << (result->a + j)->coefsT[i] << " ";
//    }
//    cout << endl;
    //test end morshed
//    int x = 0; //equivalent to k
//    Torus32* startCoefsTResulta0 = (result->a + x)->coefsT;
//    x++;
//    Torus32* startCoefsTResulta1 = (result->a + x)->coefsT;

//    static int counter1 = 0;
//    static int counter2 = 0;
    for (int i = 0; i <= k; ++i) {
//        (result->a + i)->coefsT = &(((result->a + i)->coefsT)[startIndex]);

//        if(counter1 < P_LIMIT) {
//            cout << "new input_fft: ";
//            for (int x = 0; x < 10; ++x) {
//                cout << ((LagrangeHalfCPolynomial_IMPL*)(source->a + i))->coefsC[x] << " ";
//            }
//            cout << endl;
//            counter1++;
//        }
        TorusPolynomial_fft_16(result->a + i, source->a + i, startIndex, endIndex, bitSize);
//        if(counter2 < P_LIMIT) {
//            cout << "new output_fft: ";
//            for (int x = 0; x < 10; ++x) {
//                cout << (result->a + i)->coefsT[x] << " ";
//            }
//            cout << endl;
//            counter2++;
//        }
    }

//    x = 0; //equivalent to k
//    (result->a + x)->coefsT = startCoefsTResulta0;
//    x++;
//    (result->a + x)->coefsT = startCoefsTResulta1;
    //test start morshed
//    cout << "new: ";
//    for (int i = startIndex; i < startIndex + 10; ++i) {
//        int j = 1;
//        cout << (result->a + j)->coefsT[i] << " ";
//    }
//    cout << endl;
    //test end morshed

    result->current_variance = source->current_variance;
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TLWE_FFT_CLEAR
#undef INCLUDE_TLWE_FFT_CLEAR
//Arithmetic operations on TLwe samples
/** result = (0,0) */
EXPORT void tLweFFTClear(TLweSampleFFT *result, const TLweParams *params) {
    int k = params->k;

    for (int i = 0; i <= k; ++i)
        LagrangeHalfCPolynomialClear(&result->a[i]);
    result->current_variance = 0.;
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TLWE_FFT_ADDMULRTO
#undef INCLUDE_TLWE_FFT_ADDMULRTO
// result = result + p*sample
EXPORT void tLweFFTAddMulRTo(TLweSampleFFT *result, const LagrangeHalfCPolynomial *p, const TLweSampleFFT *sample,
                             const TLweParams *params) {
    const int k = params->k;

//    static int test = 0;
    for (int i = 0; i <= k; i++) {
        LagrangeHalfCPolynomialAddMul(result->a + i, p, sample->a + i);
//        if (test < 50) {
//            fstream oldFile;
//            if (i == k) {
//                oldFile.open("oldComplex.txt", ios::app);
//                for (int j = 0; j < 10; ++j) {
////                cout << "old: ";
//                    oldFile << ((LagrangeHalfCPolynomial_IMPL *) p)->coefsC[j] << "\t";
//                    oldFile << ((LagrangeHalfCPolynomial_IMPL *) sample->a + i)->coefsC[j] << "\t";
//                    oldFile << ((LagrangeHalfCPolynomial_IMPL *) result->a + i)->coefsC[j] << endl;
//                }
//                oldFile.close();
//            }
//        }
    }
//    test++;
    //result->current_variance += sample->current_variance; 
    //TODO: how to compute the variance correctly?
}

__global__ void multiplyComplexVectors(cufftDoubleComplex *result, cufftDoubleComplex *source1,
                                       cufftDoubleComplex *source2, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < length) {
        cufftDoubleComplex temp = cuCmul(source1[id], source2[id]);
        result[id].x += temp.x;
        result[id].y += temp.y;
    }
}

EXPORT void tLweFFTAddMulRTo_gpu(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                 int Ns2, int bitSize, const TLweParams *params) {
    const int k = params->k;
    int BLOCKSIZE = Ns2;
    int gridSize = (int)ceil((float)(Ns2*bitSize)/BLOCKSIZE);

//    cout << "tLweFFTAddMulRTo_gpu: " << "gridSize: " << gridSize << " BLOCKSIZE: " << BLOCKSIZE << endl;
//    cout << "bitSize * Ns2: " << bitSize * Ns2 << endl;
    //test morshed start
//    cufftDoubleComplex *test_morshed = new cufftDoubleComplex[Ns2*bitSize];
//    cudaMemcpy(test_morshed, p, sizeof(cufftDoubleComplex) * bitSize * Ns2, cudaMemcpyDeviceToHost);
//    int sI = 512;
//    for (int i = 0; i <  10; ++i) {
//        cout << "(" << test_morshed[sI + i].x << "," << test_morshed[sI + i].y << ") ";
//    }
//    cout << endl;
    //test morshed end
    for (int i = 0; i <= k; ++i) {
        multiplyComplexVectors<<<gridSize, BLOCKSIZE>>>(result[i], p, sample[i], bitSize * Ns2);
//        cudaThreadSynchronize();
//        if (test < 50){
//            if(i == k) {
//                fstream newFile;
//                newFile.open("newComplex.txt", ios::app);
//                cudaMemcpy(s2, result[i], Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//                cudaMemcpy(temp1, p, Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//                cudaMemcpy(temp2, sample[i], Ns2 * bitSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//                for (int j = 0; j < 10; ++j) {
//
////                cout << "new: ";
//                    newFile << "(" << temp1[j].x << "," << temp1[j].y << ")\t";
//                    newFile << "(" << temp2[j].x << "," << temp2[j].y << ")\t";
//                    newFile << "(" << s2[j].x << "," << s2[j].y << ")" << endl;
//                }
//                newFile.close();
//            }
//        }
    }
}


__global__ void multiplyComplexVectors_coalesce(cufftDoubleComplex *result, cufftDoubleComplex *source1,
                                         cufftDoubleComplex *source2, int kpl, int Ns2, int bitSize, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < length) {
        for (int i = 0; i < 4; ++i) {//kpl
            int offset = i * (Ns2 * bitSize);
            cufftDoubleComplex temp = cuCmul(source1[offset + id], source2[i * Ns2 + (id % Ns2) ]);
            result[id].x += temp.x;//source2[i * Ns2 + (id % Ns2) ].x;//temp.x;
            result[id].y += temp.y;//source2[i * Ns2 + (id % Ns2) ].y;//temp.y;
//            result[id].x = temp.x;//source2[i * Ns2 + (id % Ns2) ].x;//source2[i * Ns2 + (id % Ns2) ].x;//temp.x;
//            result[id].y = temp.y;//source2[i * Ns2 + (id % Ns2) ].y;//source2[i * Ns2 + (id % Ns2) ].y;//temp.y;
        }
    }
}

EXPORT void tLweFFTAddMulRTo_gpu_coalesce(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                          int kpl, int Ns2, int bitSize, const TLweParams *params) {
    const int k = params->k;
    int BLOCKSIZE = Ns2;
    int length = Ns2 * bitSize;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "qweqweqweqwe" << endl;
    for (int i = 0; i <= k; ++i) {
        multiplyComplexVectors_coalesce<<<gridSize, BLOCKSIZE>>>(result[i], p, sample[i], kpl, Ns2, bitSize, length);
    }
}

__global__ void multiplyComplexVectors_coalesce_2(cufftDoubleComplex *result, cufftDoubleComplex *source1,
                                                  cufftDoubleComplex *source2, int kpl, int Ns2, int nOutputs,
                                                  int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        for (int i = 0; i < kpl; ++i) {
            int offsetB = i * bitSize * Ns2 * nOutputs;
            int offsetS = i * bitSize * Ns2;
            int idS = id % (Ns2 * bitSize);
//            int offset = i * (nOutputs * Ns2 * bitSize);
            //correct one //source2[offset/nOutputs + (id % (length/nOutputs))];
            cufftDoubleComplex temp = cuCmul(source1[offsetB + id], source2[offsetS + idS]);
//            result[id].x += temp.x;
//            result[id].y += temp.y;
            result[id].x += temp.x;//source2[offsetS + idS].x;//source1[offsetB + id].x;//source2[offsetS + idS].x;//source1[offsetB + id].x;
            result[id].y += temp.y;//source2[offsetS + idS].y;//source1[offsetB + id].y;//source2[offsetS + idS].y;//source1[offsetB + id].y;
        }
    }
}

__global__ void multiplyComplexVectors_coalesce_2v2(cufftDoubleComplex *result, cufftDoubleComplex *source1,
                                                    cufftDoubleComplex *source2_0, cufftDoubleComplex *source2_1,
                                                    int kpl, int Ns2, int nOutputs, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int k = id / (nOutputs * bitSize * Ns2);
        for (int i = 0; i < kpl; ++i) {
            int offsetB = i * bitSize * Ns2 * nOutputs ;
            int offsetS = i * bitSize * Ns2;
            int idS = id % (Ns2 * bitSize);
//                    int sID = offsetS + idS;
                    int sID = (i * Ns2) + id % (Ns2);
            cufftDoubleComplex source2_temp = k == 0 ? source2_0[sID] : source2_1[sID];
            cufftDoubleComplex temp = cuCmul(source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))], source2_temp);
            result[id].x += temp.x;
            result[id].y += temp.y;
//            result[id].x += source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))].x;//source2_temp.x;//temp.x;//source2_temp.x;//source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))].x;//source2_temp.x;//offsetB + (id % (nOutputs * bitSize * Ns2 ));//source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))].x;
//            result[id].y += source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))].y;//source2_temp.y;//temp.y;//source2_temp.y;//source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))].y;//source2_temp.y;//offsetB + (id % (nOutputs * bitSize * Ns2 ));//source1[offsetB + (id % (nOutputs * bitSize * Ns2 ))].y;
        }
    }
}

__global__ void multiplyComplexVectors_coalesce_2_vector(cufftDoubleComplex *result, cufftDoubleComplex *source1,
                                                  cufftDoubleComplex *source2, int kpl, int Ns2, int nOutputs,
                                                  int vLength, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        for (int i = 0; i < kpl; ++i) {
            int offsetB = i * bitSize * Ns2 * nOutputs * vLength;
            int offsetS = i * bitSize * Ns2;
            int idS = id % (Ns2 * bitSize);
//            int offset = i * (nOutputs * Ns2 * bitSize);
            //correct one //source2[offset/nOutputs + (id % (length/nOutputs))];
            int sID = (i * Ns2) + id % (Ns2);
            cufftDoubleComplex temp = cuCmul(source1[offsetB + id], source2[sID]);
            result[id].x += temp.x;
            result[id].y += temp.y;
        }
    }
}

EXPORT void tLweFFTAddMulRTo_gpu_coalesce_2(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                          int kpl, int Ns2, int nOutputs, int bitSize, const TLweParams *params) {
//    cout << "I am here 2" << endl;
    const int k = params->k;
    int BLOCKSIZE = Ns2;
    int length = nOutputs * Ns2 * bitSize;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    for (int i = 0; i <= k; ++i) {
        multiplyComplexVectors_coalesce_2<<<gridSize, BLOCKSIZE>>>
                                                      (result[i], p, sample[i], kpl, Ns2, nOutputs, bitSize, length);
    }
}

EXPORT void tLweFFTAddMulRTo_gpu_coalesce_2V2(cufftDoubleComplex *result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                            int kpl, int Ns2, int nOutputs, int bitSize, const TLweParams *params) {
//    cout << "I am here 2" << endl;
    const int k = params->k;
    int BLOCKSIZE = Ns2;
    int length = nOutputs * Ns2 * bitSize * (k + 1);
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "PPPPPPPPPPPPPPPPPPPPPPPPPP" << endl;
    multiplyComplexVectors_coalesce_2v2<<<gridSize, BLOCKSIZE>>>
                                                  (result, p, sample[0], sample[1], kpl, Ns2, nOutputs, bitSize, length);
}

EXPORT void tLweFFTAddMulRTo_gpu_coalesce_2_vector(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                            int kpl, int Ns2, int vLength, int nOutputs, int bitSize, const TLweParams *params) {
    const int k = params->k;
    int BLOCKSIZE = Ns2;
    int length = vLength * nOutputs * Ns2 * bitSize;
    int totalBitSize = bitSize * vLength;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    for (int i = 0; i <= k; ++i) {
        multiplyComplexVectors_coalesce_2_vector<<<gridSize, BLOCKSIZE>>>
                                                      (result[i], p, sample[i], kpl, Ns2, nOutputs, vLength, bitSize, length);
    }
}


__global__ void multiplyComplexVectors_2(cufftDoubleComplex *result, cufftDoubleComplex *source1,
                                       cufftDoubleComplex *source2, int nOutputs, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < length) {
        cufftDoubleComplex s2 = source2[id % (length/nOutputs)];
        cufftDoubleComplex temp = cuCmul(source1[id], s2);

        result[id].x += temp.x;
        result[id].y += temp.y;
    }
}

EXPORT void tLweFFTAddMulRTo_gpu_2(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                 int Ns2, int nOutputs, int bitSize, const TLweParams *params) {
    const int k = params->k;
    int length_Ns2 = nOutputs * bitSize * Ns2;
    int BLOCKSIZE = Ns2;
    int gridSize = (int)ceil((float)(length_Ns2)/BLOCKSIZE);

    for (int i = 0; i <= k; ++i) {
        multiplyComplexVectors_2<<<gridSize, BLOCKSIZE>>>(result[i], p, sample[i], nOutputs, length_Ns2);
    }
}
#endif


//autogenerated memory functions (they will always be included, even in
//tests)

USE_DEFAULT_CONSTRUCTOR_DESTRUCTOR_IMPLEMENTATIONS1(TLweSampleFFT, TLweParams);

#undef INCLUDE_ALL
