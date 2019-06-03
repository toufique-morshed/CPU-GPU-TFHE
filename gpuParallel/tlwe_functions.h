#ifndef TLWE_FUNCTIONS_H
#define TLWE_FUNCTIONS_H
///@file
///@brief This file contains the declaration of TLWE related functions

#include "tlwe.h"
#include <cufftXt.h>



// Ring
EXPORT void tLweKeyGen(TLweKey *result);
EXPORT void tLweSymEncrypt(TLweSample *result, TorusPolynomial *message, double alpha, const TLweKey *key);
EXPORT void tLweSymEncryptT(TLweSample *result, Torus32 message, double alpha, const TLweKey *key);
EXPORT void tLwePhase(TorusPolynomial *phase, const TLweSample *sample, const TLweKey *key);
EXPORT void tLweApproxPhase(TorusPolynomial *message, const TorusPolynomial *phase, int Msize, int N);
EXPORT void tLweSymDecrypt(TorusPolynomial *result, const TLweSample *sample, const TLweKey *key, int Msize);
EXPORT Torus32 tLweSymDecryptT(const TLweSample *sample, const TLweKey *key, int Msize);

//Arithmetic operations on TLwe samples
/** result = (0,0) */
EXPORT void tLweClear(TLweSample *result, const TLweParams *params);
/** result = sample */
EXPORT void tLweCopy(TLweSample *result, const TLweSample *sample, const TLweParams *params);
/** result = (0,mu) */
EXPORT void tLweNoiselessTrivial(TLweSample *result, const TorusPolynomial *mu, const TLweParams *params);
/** result = result + sample */
EXPORT void tLweAddTo(TLweSample *result, const TLweSample *sample, const TLweParams *params);
/** result = result - sample */
EXPORT void tLweSubTo(TLweSample *result, const TLweSample *sample, const TLweParams *params);
/** result = result + p.sample */
EXPORT void tLweAddMulTo(TLweSample *result, int p, const TLweSample *sample, const TLweParams *params);
/** result = result - p.sample */
EXPORT void tLweSubMulTo(TLweSample *result, int p, const TLweSample *sample, const TLweParams *params);

/*create an homogeneous tlwe sample*/
EXPORT void tLweSymEncryptZero(TLweSample *result, double alpha, const TLweKey *key);


/** result = result + p.sample */
EXPORT void
tLweAddMulRTo(TLweSample *result, const IntPolynomial *p, const TLweSample *sample, const TLweParams *params);

/** result += (0...,0,x,0,...,0) */
EXPORT void tLweAddTTo(TLweSample *result, const int pos, const Torus32 x, const TLweParams *params);

/** result += p*(0...,0,x,0,...,0) */
EXPORT void
tLweAddRTTo(TLweSample *result, const int pos, const IntPolynomial *p, const Torus32 x, const TLweParams *params);

// EXPORT void tLwePolyCombination(TLweSample* result, const int* combi, const TLweSample* samples, const TLweParams* params);

EXPORT void tLweMulByXaiMinusOne(TLweSample *result, int ai, const TLweSample *bk, const TLweParams *params);

EXPORT void tLweExtractLweSampleIndex(LweSample *result, const TLweSample *x, const int index, const LweParams *params,
                                      const TLweParams *rparams);
EXPORT void
tLweExtractLweSample(LweSample *result, const TLweSample *x, const LweParams *params, const TLweParams *rparams);


//extractions TLwe -> Lwe
EXPORT void tLweExtractKey(LweKey *result, const TLweKey *); //sans doute un param supplémentaire
//EXPORT void tLweExtractSample(LweSample* result, const TLweSample* x);

//FFT operations

EXPORT void tLweToFFTConvert(TLweSampleFFT *result, const TLweSample *source, const TLweParams *params);
EXPORT void tLweFromFFTConvert(TLweSample *result, const TLweSampleFFT *source, const TLweParams *params);
EXPORT void tLweFFTClear(TLweSampleFFT *result, const TLweParams *params);
EXPORT void tLweFFTAddMulRTo(TLweSampleFFT *result, const LagrangeHalfCPolynomial *p, const TLweSampleFFT *sample,
                             const TLweParams *params);

//new
EXPORT void tLweNoiselessTrivial_16(TLweSample *result, const TorusPolynomial *mu, const TLweParams *params);
EXPORT void tLweMulByXaiMinusOne_16(TLweSample *result, const int* bara, int baraIndex, const TLweSample *bk, int bitSize, int N, const TLweParams *params);
EXPORT void tLweMulByXaiMinusOne_16_2(TLweSample *result, const int* bara, int baraIndex, const TLweSample *bk, int nOutputs, int bitSize, int N, const TLweParams *params);
EXPORT void tLweMulByXaiMinusOne_16_2v2(TLweSample *resultV2, const int* bara, int baraIndex, const TLweSample *bk,
                                        int nOutputs, int bitSize, int N, const TLweParams *params);
EXPORT void tLweMulByXaiMinusOne_16_2_vector(TLweSample *result, const int* bara, int baraIndex, const TLweSample *bk,
                                             int vLength, int nOutputs, int bitSize, int N, const TLweParams *params);
EXPORT void tLweFromFFTConvert_16(TLweSample *result, const TLweSampleFFT *source, int startIndex, int endIndex,
                               int bitSize, const TLweParams *params);
EXPORT void tLweAddTo_16(TLweSample *result, const TLweSample *sample, int bitSize, int N, const TLweParams *params);
EXPORT void tLweAddTo_16_2(TLweSample *result, const TLweSample *sample, int nOutputs, int bitSize, int N, const TLweParams *params);
EXPORT void tLweAddTo_16_2v2(TLweSample *result, const TLweSample *sample, int nOutputs, int bitSize, int N, const TLweParams *params);
EXPORT void tLweAddTo_16_2_vector(TLweSample *result, const TLweSample *sample, int vLength, int nOutputs, int bitSize,
                                  int N, const TLweParams *params);

EXPORT void tLweExtractLweSample_16(LweSample_16 *result, const TLweSample *x, const LweParams *params, int bitSize, const TLweParams *rparams);
EXPORT void tLweExtractLweSample_16_2(LweSample_16 *result, const TLweSample *x, const LweParams *params, int nOutputs,
                                      int bitSize, const TLweParams *rparams);
EXPORT void tLweExtractLweSample_16_2_vector(LweSample_16* result, const TLweSample* x, const LweParams* params,
                                             int vLength, int nOutputs, int bitSize, const TLweParams* rparams);

EXPORT void tLweExtractLweSampleIndex_16(LweSample_16 *result, const TLweSample *x, const int index, const LweParams *params,
                                      int bitSize, const TLweParams *rparams);
EXPORT void tLweExtractLweSampleIndex_16_2(LweSample_16 *result, const TLweSample *x, const int index, const LweParams *params,
                                         int nOutputs, int bitSize, const TLweParams *rparams);
EXPORT void tLweExtractLweSampleIndex_16_2_vector(LweSample_16* result, const TLweSample* x, const int index, const LweParams* params,
                                                  int vLength, int nOutputs, int bitSize, const TLweParams* rparams);
EXPORT void tLweFFTAddMulRTo_gpu(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                 int Ns2, int bitSize, const TLweParams *params);
EXPORT void tLweFFTAddMulRTo_gpu_coalesce(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                          int kpl, int Ns2, int bitSize, const TLweParams *params);
EXPORT void tLweFFTAddMulRTo_gpu_2(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                   int Ns2, int nOutputs, int bitSize, const TLweParams *params);
EXPORT void tLweFFTAddMulRTo_gpu_coalesce_2(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                            int kpl, int Ns2, int nOutputs, int bitSize, const TLweParams *params);
EXPORT void tLweFFTAddMulRTo_gpu_coalesce_2V2(cufftDoubleComplex *result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                              int kpl, int Ns2, int nOutputs, int bitSize, const TLweParams *params);
EXPORT void tLweFFTAddMulRTo_gpu_coalesce_2_vector(cufftDoubleComplex **result, cufftDoubleComplex *p, cufftDoubleComplex **sample,
                                                   int kpl, int Ns2, int vLength, int nOutputs, int bitSize, const TLweParams *params);
EXPORT void tLweFromFFTConvert_gpu(TLweSample *result, cufftDoubleComplex **tmpa_gpu, int bitSize, int N, int Ns2,
                                   const TLweParams *params);
EXPORT void tLweFromFFTConvert_gpu_2(TLweSample *result, cufftDoubleComplex **tmpa_gpu, cufftDoubleComplex *sourceSingle, int nOutputs, int bitSize,
                                     int N, int Ns2, const TLweParams *params);
EXPORT void tLweFromFFTConvert_gpu_2_vector(TLweSample *result, cufftDoubleComplex **source, int vLength, int nOutputs,
                                            int bitSize, int N, int Ns2, const TLweParams *params);

#endif// TLWE_FUNCTIONS_H
