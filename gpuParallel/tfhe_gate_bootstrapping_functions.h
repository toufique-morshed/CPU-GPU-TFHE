#ifndef TFHE_GATE_BOOTSTRAPPING_FUNCTIONS_H
#define TFHE_GATE_BOOTSTRAPPING_FUNCTIONS_H

///@file
///@brief tfhe gate bootstrapping api

#include "tfhe_gate_bootstrapping_structures.h"

//////////////////////////////////////////
// Gate bootstrapping public interface
//////////////////////////////////////////

/** generate default gate bootstrapping parameters */
EXPORT TFheGateBootstrappingParameterSet *new_default_gate_bootstrapping_parameters(int minimum_lambda);

/** generate a random gate bootstrapping secret key */
EXPORT TFheGateBootstrappingSecretKeySet *
new_random_gate_bootstrapping_secret_keyset(const TFheGateBootstrappingParameterSet *params);

/** deletes gate bootstrapping parameters */
EXPORT void delete_gate_bootstrapping_parameters(TFheGateBootstrappingParameterSet *params);

/** deletes a gate bootstrapping secret key */
EXPORT void delete_gate_bootstrapping_secret_keyset(TFheGateBootstrappingSecretKeySet *keyset);

/** deletes a gate bootstrapping secret key */
EXPORT void delete_gate_bootstrapping_cloud_keyset(TFheGateBootstrappingCloudKeySet *keyset);

/** generate a new unititialized ciphertext */
EXPORT LweSample *new_gate_bootstrapping_ciphertext(const TFheGateBootstrappingParameterSet *params);

/** generate a new unititialized ciphertext array of length nbelems */
EXPORT LweSample *new_gate_bootstrapping_ciphertext_array(int nbelems, const TFheGateBootstrappingParameterSet *params);

/** deletes a ciphertext */
EXPORT void delete_gate_bootstrapping_ciphertext(LweSample *sample);

/** deletes a ciphertext array of length nbelems */
EXPORT void delete_gate_bootstrapping_ciphertext_array(int nbelems, LweSample *samples);

/** encrypts a boolean */
EXPORT void bootsSymEncrypt(LweSample *result, int message, const TFheGateBootstrappingSecretKeySet *params);

/** decrypts a boolean */
EXPORT int bootsSymDecrypt(const LweSample *sample, const TFheGateBootstrappingSecretKeySet *params);

/** bootstrapped Constant (true or false) trivial Gate */
EXPORT void bootsCONSTANT(LweSample *result, int value, const TFheGateBootstrappingCloudKeySet *bk);

/** bootstrapped Nand Gate */
EXPORT void
bootsNAND(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped Or Gate:  */
EXPORT void
bootsOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped And Gate: result = a and b */
EXPORT void
bootsAND(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped Xor Gate: result = a xor b */
EXPORT void
bootsXOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped Xnor Gate: result = (a==b) */
EXPORT void
bootsXNOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped Not Gate: result = not(a) */
EXPORT void bootsNOT(LweSample *result, const LweSample *ca, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped Copy Gate: result = a */
EXPORT void bootsCOPY(LweSample *result, const LweSample *ca, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped Nor Gate: result = not(a or b) */
EXPORT void
bootsNOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped AndNY Gate: not(a) and b */
EXPORT void
bootsANDNY(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped AndYN Gate: a and not(b) */
EXPORT void
bootsANDYN(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped OrNY Gate: not(a) or b */
EXPORT void
bootsORNY(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);
/** bootstrapped OrYN Gate: a or not(b) */
EXPORT void
bootsORYN(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk);

/** bootstrapped Mux(a,b,c) = a?b:c */
EXPORT void bootsMUX(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *c,
                     const TFheGateBootstrappingCloudKeySet *bk);



////new
EXPORT void bootsAND_16(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                        Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                        Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsXOR_16(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                        Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                        Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsANDXOR_16(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nOutputs,
                           int bitSize, const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                           cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                           int ***ks_b_gpu, double ***ks_cv_gpu,
                           Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsXORXOR_16(LweSample_16 *result,
                           const LweSample_16 *ca1, const LweSample_16 *ca2,
                           const LweSample_16 *cb1, const LweSample_16 *cb2,
                           int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                           cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                           Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                           Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsXORXOR_16_vector(LweSample_16 *result,
                                  const LweSample_16 *ca1, const LweSample_16 *ca2,
                                  const LweSample_16 *cb1, const LweSample_16 *cb2,
                                  int vLen, int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                                  cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                                  Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsANDXOR_16_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nOutputs,
                                  int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                                  cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                                  int ***ks_b_gpu, double ***ks_cv_gpu, Torus32 *ks_a_gpu_extendedPtr,
                                  Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsAND_MULT(LweSample_16 *result,
                          const LweSample_16 *ca, const LweSample_16 *cb,
                          int resBitSize, int bitSize_A, int bIndex,
                          const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                          cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                          int ***ks_b_gpu, double ***ks_cv_gpu,
                          Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsAND_MULT_con(LweSample_16 *result,
                              LweSample_16 **ca, LweSample_16 **cb,
                              int nConMul, int resBitSize, int bitSize_A, int bIndex,
                              const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                              cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu,
                              Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                              Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                              double *ks_cv_gpu_extendedPtr);
EXPORT void bootsAND_16_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nOutputs,
                               int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                               cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                               Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                               int ***ks_b_gpu, double ***ks_cv_gpu,
                               Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                               double *ks_cv_gpu_extendedPtr);
EXPORT void bootsMUX_16_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb,
                               const LweSample_16 *cc, int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                               cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                               Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                               int ***ks_b_gpu, double ***ks_cv_gpu,
                               Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                               double *ks_cv_gpu_extendedPtr);
EXPORT LweSample_16 *convertBitToNumberZero_GPU(int bitSize, const TFheGateBootstrappingCloudKeySet *bk);
EXPORT LweSample_16 *convertBitToNumberZero_GPU_2(int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk);
EXPORT LweSample_16* convertBitToNumberZero(int bitSize, const TFheGateBootstrappingCloudKeySet *bk);

EXPORT LweSample_16* convertBitToNumber(const LweSample* input, int bitSize, const TFheGateBootstrappingCloudKeySet *bk);

EXPORT void freeLweSample_16(LweSample_16* input);

EXPORT LweSample_16* newLweSample_16(int bitSize, const LweParams* params);
EXPORT LweSample_16* newLweSample_n_Bit(int bitSize, const int nElem);
EXPORT LweSample_16* newLweSample_16_2(int nOutputs, int bitSize, const LweParams* params);

EXPORT LweSample* convertNumberToBits(LweSample_16* number, int bitSize, const TFheGateBootstrappingCloudKeySet *bk);
EXPORT void bootsXOR_AND(LweSample *result, const LweSample *ca, const LweSample *cb, const LweSample *cc, const TFheGateBootstrappingCloudKeySet *bk);
void bootsNOT_16(LweSample_16 *output, LweSample_16 *input, int bitSize, int params_n);

//full gpu Functions
EXPORT void bootsAND_fullGPU_OneBit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                  cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                  Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr);
EXPORT void bootsAND_fullGPU_1_Bit_Stream(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                  cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                  Torus32 *ks_b_gpu_extendedPtr);
EXPORT void bootsAND_fullGPU_n_Bit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                   cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                   Torus32 *ks_b_gpu_extendedPtr);
EXPORT void bootsXOR_fullGPU_n_Bit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                   cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                   Torus32 *ks_b_gpu_extendedPtr);
EXPORT void bootsXNOR_fullGPU_n_Bit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                   cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                   Torus32 *ks_b_gpu_extendedPtr);
EXPORT void bootsMUX_fullGPU_n_Bit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb,
                                   const LweSample_16 *cc, int nBits, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                   Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr);

EXPORT void bootsANDXOR_fullGPU_n_Bit_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb,
                                             const int vLength, const int nBits,
                                             cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                             Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr);
EXPORT void bootsXORXOR_fullGPU_n_Bit_vector(LweSample_16 *result,
                                             const LweSample_16 *ca1, const LweSample_16 *ca2,
                                             const LweSample_16 *cb1, const LweSample_16 *cb2,
                                             const int vLength, const int nBits, cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                             Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr);



#endif// TFHE_GATE_BOOTSTRAPPING_FUNCTIONS_H
