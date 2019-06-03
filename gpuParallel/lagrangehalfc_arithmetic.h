#ifndef LAGRANGEHALFC_ARITHMETIC_H
#define LAGRANGEHALFC_ARITHMETIC_H

///@file
///@brief This file contains the declaration of operations on LagrangeHalfC polynomials

#include "polynomials.h"
#include <cufftXt.h>


//initialize the LagrangeHalfCPolynomial structure
//(equivalent of the C++ constructor)
EXPORT void init_LagrangeHalfCPolynomial(LagrangeHalfCPolynomial* obj, const int N);

//destroys the LagrangeHalfCPolynomial structure
//(equivalent of the C++ destructor)
EXPORT void destroy_LagrangeHalfCPolynomial(LagrangeHalfCPolynomial* obj);


/**
 * FFT functions 
 */
EXPORT void IntPolynomial_ifft(LagrangeHalfCPolynomial* result, const IntPolynomial* p);
EXPORT void TorusPolynomial_ifft(LagrangeHalfCPolynomial* result, const TorusPolynomial* p);
EXPORT void TorusPolynomial_fft(TorusPolynomial* result, const LagrangeHalfCPolynomial* p);

//MISC OPERATIONS
/** sets to zero */
EXPORT void LagrangeHalfCPolynomialClear(LagrangeHalfCPolynomial* result);

/** sets to this torus32 constant */
EXPORT void LagrangeHalfCPolynomialSetTorusConstant(LagrangeHalfCPolynomial* result, const Torus32 mu);
EXPORT void LagrangeHalfCPolynomialAddTorusConstant(LagrangeHalfCPolynomial* result, const Torus32 cst);

// /* sets to X^ai-1 */
//This function is commented, because it is not used 
//in the current version. However, it may be included in future releases
//EXPORT void LagrangeHalfCPolynomialSetXaiMinusOne(LagrangeHalfCPolynomial* result, const int ai);


/** multiplication via direct FFT */
EXPORT void torusPolynomialMultFFT(TorusPolynomial* result, const IntPolynomial* poly1, const TorusPolynomial* poly2);
EXPORT void torusPolynomialAddMulRFFT(TorusPolynomial* result, const IntPolynomial* poly1, const TorusPolynomial* poly2);
EXPORT void torusPolynomialSubMulRFFT(TorusPolynomial* result, const IntPolynomial* poly1, const TorusPolynomial* poly2);

/** termwise multiplication in Lagrange space */
EXPORT void LagrangeHalfCPolynomialMul(
	LagrangeHalfCPolynomial* result, 
	const LagrangeHalfCPolynomial* a, 
	const LagrangeHalfCPolynomial* b);

/** termwise multiplication and addTo in Lagrange space */
EXPORT void LagrangeHalfCPolynomialAddTo(
	LagrangeHalfCPolynomial* accum, 
	const LagrangeHalfCPolynomial* a);

EXPORT void LagrangeHalfCPolynomialAddMul(
	LagrangeHalfCPolynomial* accum, 
	const LagrangeHalfCPolynomial* a, 
	const LagrangeHalfCPolynomial* b);

EXPORT void LagrangeHalfCPolynomialSubMul(
	LagrangeHalfCPolynomial* accum, 
	const LagrangeHalfCPolynomial* a, 
	const LagrangeHalfCPolynomial* b);


/*
 * FFT functions
 */

EXPORT void IntPolynomial_ifft_16(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p);
EXPORT void IntPolynomial_ifft_16_Coalesce(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p);
EXPORT void IntPolynomial_ifft_16_2(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p);
EXPORT void IntPolynomial_ifft_16_2_Coalesce(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p);
//EXPORT void IntPolynomial_ifft_16_2_Coalesce_one_out(cufftDoubleComplex* result, int nOutputs, int bitSize, IntPolynomial* p);
EXPORT void IntPolynomial_ifft_16_2_Coalesce_localMem(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p,
													  cufftDoubleReal *d_in_test, cufftDoubleComplex *d_out_test);
EXPORT void IntPolynomial_ifft_16_2_Coalesce_vector(cufftDoubleComplex* result, int vLength, int bitSize, const IntPolynomial* p);
EXPORT void TorusPolynomial_fft_16(TorusPolynomial* result, const LagrangeHalfCPolynomial* p, int startIndex, int endIndex, int bitSize);
EXPORT void TorusPolynomial_fft_gpu(TorusPolynomial* result, cufftDoubleComplex *source, int bitSize, int N, int Ns2);
EXPORT void TorusPolynomial_fft_gpu_2(Torus32* result, cufftDoubleComplex *source, int nOutputs, int bitSize, int N, int Ns2);
EXPORT void TorusPolynomial_fft_gpu_16_2_Coalesce_vector(TorusPolynomial* result, cufftDoubleComplex *source,
														 int vLength, int nOutputs, int bitSize, int N, int Ns2);
#endif // LAGRANGEHALFC_ARITHMETIC_H
