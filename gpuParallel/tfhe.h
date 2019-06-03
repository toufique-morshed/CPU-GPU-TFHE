#ifndef TFHE_H
#define TFHE_H

///@file
///@brief This file declares almost everything

#include "tfhe_core.h"

#include "numeric_functions.h"

#include "polynomials_arithmetic.h"
#include "lagrangehalfc_arithmetic.h"

#include "lwe-functions.h"

#include "tlwe_functions.h"

#include "tgsw_functions.h"

#include "lwekeyswitch.h"

#include "lwebootstrappingkey.h"

#include "tfhe_gate_bootstrapping_functions.h"

#include "tfhe_io.h"

///////////////////////////////////////////////////
//  TFHE bootstrapping internal functions
//////////////////////////////////////////////////


/** sets the seed of the random number generator to the given values */
EXPORT void tfhe_random_generator_setSeed(uint32_t* values, int size);

EXPORT void tfhe_blindRotate(TLweSample* accum, const TGswSample* bk, const int* bara, const int n, const TGswParams* bk_params);
EXPORT void tfhe_blindRotateAndExtract(LweSample* result, const TorusPolynomial* v, const TGswSample* bk, const int barb, const int* bara, const int n, const TGswParams* bk_params);
EXPORT void tfhe_bootstrap_woKS(LweSample* result, const LweBootstrappingKey* bk, Torus32 mu, const LweSample* x);
EXPORT void tfhe_bootstrap(LweSample* result, const LweBootstrappingKey* bk, Torus32 mu, const LweSample* x);
EXPORT void tfhe_createLweBootstrappingKey(LweBootstrappingKey* bk, const LweKey* key_in, const TGswKey* rgsw_key);

EXPORT void tfhe_blindRotate_FFT(TLweSample* accum, const TGswSampleFFT* bk, const int* bara, const int n, const TGswParams* bk_params);
EXPORT void tfhe_blindRotateAndExtract_FFT(LweSample* result, const TorusPolynomial* v, const TGswSampleFFT* bk, const int barb, const int* bara, const int n, const TGswParams* bk_params);
EXPORT void tfhe_bootstrap_woKS_FFT(LweSample* result, const LweBootstrappingKeyFFT* bk, Torus32 mu, const LweSample* x);
EXPORT void tfhe_bootstrap_FFT(LweSample* result, const LweBootstrappingKeyFFT* bk, Torus32 mu, const LweSample* x);

//new
EXPORT void tfhe_bootstrap_FFT_16(LweSample_16* result, const LweBootstrappingKeyFFT* bk, Torus32 mu, int bitSize,
                                  const LweSample_16* x, cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu,
                                  double ***ks_cv_gpu, Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                  double *ks_cv_gpu_extendedPtr);
EXPORT void tfhe_bootstrap_woKS_FFT_16(LweSample_16* result, const LweBootstrappingKeyFFT* bk, Torus32 mu, int bitSize,
                                       const LweSample_16* x, cufftDoubleComplex ****cudaBkFFT,
                                       cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_bootstrap_woKS_FFT_16_2(LweSample_16 *result, const LweBootstrappingKeyFFT *bk, Torus32 mu,
                                         int nOutputs, int bitSize, const LweSample_16 *x,
                                         cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_bootstrap_woKS_FFT_16_2_vector(LweSample_16 *result, const LweBootstrappingKeyFFT *bk, Torus32 mu,
                                                int vLength, int nOutputs, int bitSize, const LweSample_16 *x,
                                                cufftDoubleComplex ****cudaBkFFT,
                                                cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotateAndExtract_FFT_16(LweSample_16* result, const TorusPolynomial* v, const TGswSampleFFT* bk,
                                              const int *barb, const int* bara, const int n, int bitSize,
                                              const TGswParams* bk_params, cufftDoubleComplex ****cudaBkFFT,
                                              cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotateAndExtract_FFT_16_2(LweSample_16 *result, const TorusPolynomial *v, const TGswSampleFFT *bk,
                                                const int *barb, const int *bara, const int n, int nOutputs, int bitSize,
                                                const TGswParams *bk_params, cufftDoubleComplex ****cudaBkFFT,
                                                cufftDoubleComplex ***cudaBkFFTCoalesce);
EXPORT void tfhe_blindRotateAndExtract_FFT_16_2_vector(LweSample_16 *result, const TorusPolynomial *v,
                                                       const TGswSampleFFT *bk, const int *barb, const int *bara,
                                                       const int n, int vLength, int nOutputs, int bitSize,
                                                       const TGswParams *bk_params, cufftDoubleComplex ****cudaBkFFT,
                                                       cufftDoubleComplex ***cudaBkFFTCoalesce);

#endif //TFHE_H
