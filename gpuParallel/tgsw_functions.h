#ifndef TGSW_FUNCTIONS_H
#define TGSW_FUNCTIONS_H

///@file
///@brief This file contains the declaration of TGSW related functions


#include "tfhe_core.h"
#include "tgsw.h"
#include "tlwe.h"
#include "lwebootstrappingkey.h"
//#include <cufft.h>
//#include <cufftXt.h>
#include "cudaFFTTest.h"

// Gsw
EXPORT void tGswKeyGen(TGswKey *result);
EXPORT void tGswSymEncrypt(TGswSample *result, const IntPolynomial *message, double alpha, const TGswKey *key);
EXPORT void tGswSymEncryptInt(TGswSample *result, const int message, double alpha, const TGswKey *key);
EXPORT void tGswSymDecrypt(IntPolynomial *result, const TGswSample *sample, const TGswKey *key, const int Msize);
EXPORT int tGswSymDecryptInt(const TGswSample *sample, const TGswKey *key);
//do we really decrypt Gsw samples?

// support Functions for TGsw
// Result = 0
EXPORT void tGswClear(TGswSample *result, const TGswParams *params);
// Result += H
EXPORT void tGswAddH(TGswSample *result, const TGswParams *params);
// Result += mu*H
EXPORT void tGswAddMuH(TGswSample *result, const IntPolynomial *message, const TGswParams *params);
// Result += mu*H, mu integer
EXPORT void tGswAddMuIntH(TGswSample *result, const int message, const TGswParams *params);
// Result = tGsw(0)
EXPORT void tGswEncryptZero(TGswSample *result, double alpha, const TGswKey *key);

//fonction de decomposition
EXPORT void tGswTLweDecompH(IntPolynomial *result, const TLweSample *sample, const TGswParams *params);

EXPORT void
tGswTorus32PolynomialDecompH(IntPolynomial *result, const TorusPolynomial *sample, const TGswParams *params);
EXPORT void tGswTLweDecompH(IntPolynomial *result, const TLweSample *sample, const TGswParams *params);

//TODO: Ilaria.Theoreme3.5
EXPORT void tGswExternProduct(TLweSample *result, const TGswSample *a, const TLweSample *b, const TGswParams *params);

// result=result+ (X^ai-1)*bi (ligne 5 de l'algo)
EXPORT void tGswMulByXaiMinusOne(TGswSample *result, int ai, const TGswSample *bk, const TGswParams *params);

//ligne 5 algo,mult externe
EXPORT void tGswExternMulToTLwe(TLweSample *accum, const TGswSample *sample, const TGswParams *params);

/** result = (0,mu) */
EXPORT void tGswNoiselessTrivial(TGswSample *result, const IntPolynomial *mu, const TGswParams *params);

/** result = result + sample */
EXPORT void tGswAddTo(TGswSample *result, const TGswSample *sample, const TGswParams *params);

/** result = result - sample */
//EXPORT void tGswSubTo(TLweSample* result, const TLweSample* sample, const TLweParams* params);
/** result = result + p.sample */
//EXPORT void tGswAddMulTo(TLweSample* result, int p, const TLweSample* sample, const TLweParams* params);
/** result = result - p.sample */
//EXPORT void tGswSubMulTo(TLweSample* result, int p, const TLweSample* sample, const TLweParams* params);


EXPORT void tGswToFFTConvert(TGswSampleFFT *result, const TGswSample *source, const TGswParams *params);
EXPORT void tGswFromFFTConvert(TGswSample *result, const TGswSampleFFT *source, const TGswParams *params);
EXPORT void tGswFFTAddH(TGswSampleFFT *result, const TGswParams *params);
EXPORT void tGswFFTClear(TGswSampleFFT *result, const TGswParams *params);
EXPORT void tGswFFTExternMulToTLwe(TLweSample *accum, const TGswSampleFFT *gsw, const TGswParams *params);
EXPORT void
tGswFFTMulByXaiMinusOne(TGswSampleFFT *result, const int ai, const TGswSampleFFT *bki, const TGswParams *params);




EXPORT void
tfhe_blindRotate(TLweSample *accum, const TGswSample *bk, const int *bara, const int n, const TGswParams *bk_params);
EXPORT void
tfhe_blindRotateAndExtract(LweSample *result, const TorusPolynomial *v, const TGswSample *bk, const int barb,
                           const int *bara, const int n, const TGswParams *bk_params);
EXPORT void tfhe_bootstrap(LweSample *result, const LweBootstrappingKey *bk, Torus32 mu, const LweSample *x);
EXPORT void tfhe_createLweBootstrappingKey(LweBootstrappingKey *bk, const LweKey *key_in, const TGswKey *rgsw_key);


EXPORT void tfhe_blindRotate_FFT(TLweSample *accum, const TGswSampleFFT *bk, const int *bara, const int n,
                                 const TGswParams *bk_params);
EXPORT void
tfhe_blindRotateAndExtract_FFT(LweSample *result, const TorusPolynomial *v, const TGswSampleFFT *bk, const int barb,
                               const int *bara, const int n, const TGswParams *bk_params);
EXPORT void tfhe_bootstrap_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, Torus32 mu, const LweSample *x);
// EXPORT void tfhe_bootstrapFFT(LweSample* result, const LweBootstrappingKeyFFT* bk, Torus32 mu1, Torus32 mu0, const LweSample* x);
// EXPORT void tfhe_createLweBootstrappingKeyFFT(LweBootstrappingKeyFFT* bk, const LweKey* key_in, const TGswKey* rgsw_key);


////new
EXPORT void tfhe_bootstrap_FFT_16(LweSample_16 *result, const LweBootstrappingKeyFFT *bk, Torus32 mu, int bitSize,
                                  const LweSample_16 *x, cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu,
                                  double ***ks_cv_gpu, Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                  double *ks_cv_gpu_extendedPtr);
EXPORT void tfhe_bootstrap_FFT_16_2(LweSample_16 *result, const LweBootstrappingKeyFFT *bk, Torus32 mu, int nOutputs,
                                    int bitSize, const LweSample_16 *x, cufftDoubleComplex ****cudaBkFFT,
                                    cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu,
                                    Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                                    Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void tfhe_bootstrap_FFT_16_2_vector(LweSample_16 *result, const LweBootstrappingKeyFFT *bk, Torus32 mu,
                                           int vLength, int nOutputs, int bitSize, const LweSample_16 *x,
                                           cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                           Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                                           int ***ks_b_gpu, double ***ks_cv_gpu, Torus32 *ks_a_gpu_extendedPtr,
                                           Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);

EXPORT void tfhe_blindRotate_FFT_16(TLweSample *accum, const TGswSampleFFT *bk, const int *bara, const int n,
                                    int bitSize, const TGswParams *bk_params, cufftDoubleComplex ****cudaBkFFT,
                                    cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotate_FFT_16V2(TLweSample *accumV2, const TGswSampleFFT *bk, const int *bara, const int n,
                                    int bitSize, const TGswParams *bk_params, cufftDoubleComplex ****cudaBkFFT,
                                    cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotate_FFT_16_2(TLweSample *accum, const TGswSampleFFT *bkFFT, const int *bara, const int n,
                                      int nOutputs, int bitSize, const TGswParams *bk_params,
                                      cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotate_FFT_16_2v2(TLweSample *accum, const TGswSampleFFT *bkFFT, const int *bara, const int n,
                                        int nOutputs, int bitSize, const TGswParams *bk_params,
                                        cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotate_FFT_16_2_vector(TLweSample *accum, const TGswSampleFFT *bkFFT, const int *bara, const int n,
                                             int vLength, int nOutputs, int bitSize, const TGswParams *bk_params,
                                             cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tGswFFTExternMulToTLwe_16(TLweSample *accum, const TGswSampleFFT *gsw, int bitSize,
                                      const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                      cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                      IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                      cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu);

EXPORT void
tGswFFTExternMulToTLwe_16V2(TLweSample *accumV2, const TGswSampleFFT *gsw, int bitSize, const TGswParams *params,
                            cufftDoubleComplex ***cudaBKi, cufftDoubleComplex **cudaBKiCoalesce,
                            IntPolynomial *deca, IntPolynomial *decaCoalesce,
                            cufftDoubleComplex **cuDecaFFT, cufftDoubleComplex *cuDecaFFTCoalesce,
                            cufftDoubleComplex **tmpa_gpu, cufftDoubleComplex *tmpa_gpuCoal,
                            cudaFFTProcessorTest_general *fftProcessor);

EXPORT void tGswFFTExternMulToTLwe_16_2(TLweSample *accum, const TGswSampleFFT *gsw, int nOutputs, int bitSize,
                                        const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                        cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                        IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                        cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu,
                                        cufftDoubleComplex *tmpa_gpuCoal);
//EXPORT void tGswFFTExternMulToTLwe_16_2V2(TLweSample *accumV2, const TGswSampleFFT *gsw, int nOutputs, int bitSize,
//                                          const TGswParams *params, cufftDoubleComplex ***cudaBKi,
//                                          cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
//                                          IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
//                                          cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex *tmpa_gpuCoal);
EXPORT void tGswFFTExternMulToTLwe_16_2V2(TLweSample *accumV2, const TGswSampleFFT *gsw, int nOutputs, int bitSize,
                                          const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                          cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                          IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                          cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex *tmpa_gpuCoal,
                                          cudaFFTProcessorTest_general *fftProcessor);
EXPORT void tGswFFTExternMulToTLwe_16_2_vector(TLweSample *accum, const TGswSampleFFT *gsw, int vLength, int nOutputs,
                                               int bitSize, const TGswParams *params, cufftDoubleComplex ***cudaBKi,
                                               cufftDoubleComplex **cudaBKiCoalesce, IntPolynomial *deca,
                                               IntPolynomial *decaCoalesce, cufftDoubleComplex **cuDecaFFT,
                                               cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex **tmpa_gpu);
EXPORT void
tGswTorus32PolynomialDecompH_16(IntPolynomial *result, const TorusPolynomial *sample,
                                int bitSize, const TGswParams *params);
EXPORT void
tGswTorus32PolynomialDecompH_16_2(IntPolynomial *result, const TorusPolynomial *sample, int nOutputs,
                                  int bitSize, const TGswParams *params);
EXPORT void
tGswTorus32PolynomialDecompH_16_2_Coalesce(int *result, const TLweSample *sample, int nOutputs,
                                           int bitSize, const TGswParams *params);
EXPORT void
tGswTorus32PolynomialDecompH_16_2_CoalesceV2(int *resultV2, const TLweSample *sample, int nOutputs,
                                                         int bitSize, const TGswParams *params);
EXPORT void
tGswTorus32PolynomialDecompH_16_2_Coalesce_vector(int *result, const TLweSample *sample, int vLength,
                                                  int nOutputs, int bitSize, const TGswParams *params);
EXPORT void
tGswTorus32PolynomialDecompH_16_Coalesce(int *result, TLweSample *sample,
                                         int bitSize, const TGswParams *params);
EXPORT void tGswFFTExternMulToTLwe_testing(TLweSample *accum, const TGswSampleFFT *gsw, const TGswParams *params);


#endif //TGSW_FUNCTIONS_H
