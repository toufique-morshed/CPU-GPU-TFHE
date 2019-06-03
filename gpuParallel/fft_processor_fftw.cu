#include <complex>
//#define complex _Complex
#include <fftw3.h>
#include "polynomials.h"
#include "lagrangehalfc_impl.h"
#include <cassert>
#include <cmath>
#include "cudaFFTTest.h"
#include <iostream>
#include <inttypes.h>
#include <stdio.h>
#include <cstdint>

using namespace std;
typedef std::complex<double> cplx;

#define BIT_16 16
#define BIT_32 32

//global
/*
cudaFFTProcessorTest cudaFFTProcessorTestTest_16(16);
cudaFFTProcessorTest cudaFFTProcessorTestTest_32(32);
cudaFFTProcessorTest_2 cudaFFTProcessorTestTest_2_16(16);
cudaFFTProcessorTest_2 cudaFFTProcessorTestTest_2_32(32);
cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_4(1024, 16, 1024);//for 4 bit//coalescing
cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_8(1024, 32, 1024);//for 8 bit//coalescing
cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_32(1024, 128, 1024);//for 32 bit//coalescing
*/
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_1(1024, 4, 1024);//for 1 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_2(1024, 8, 1024);//for 1 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_3(1024, 12, 1024);//for 1 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_4(1024, 16, 1024);//for 4 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_6(1024, 24, 1024);//for 6 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_8(1024, 32, 1024);//for 8 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_12(1024, 48, 1024);//for 12 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_16(1024, 64, 1024);//for 16 bit//coalescing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_24(1024, 96, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_32(1024, 128, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_48(1024, 192, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_64(1024, 256, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_80(1024, 320, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_128(1024, 512, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_256(1024, 1024, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_512(1024, 2048, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_1024(1024, 4096, 1024);
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_2048(1024, 8192, 1024);
//int nBits = 24;
//n bit single gate testing
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_bBitsTesting(1024, nBits * 4, 1024);//for nBit bit//coalescing

//int nOuts = 2;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_1(1024, nOuts, 1, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_1_1(1024, 1, 1, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_2(1024, nOuts, 2, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_4(1024, nOuts, 4, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_6(1024, nOuts, 6, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_8(1024, nOuts, 8, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16(1024, nOuts, 16, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_1_16(1024, 1, 16, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_24(1024, nOuts, 24, 4, 1024, 2);//for 16 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_32(1024, nOuts, 32, 4, 1024, 2);//for 32 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_bBitsTesting_2_output(1024, nOuts, 24, 4, 1024, 2);//(1024, nBits * nOuts * 4, 1024);//for nBit bit//coalescing
//for vector operations
//int vLen = 8;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_8(1024, nOuts, 8, BIT_16, 4, 1024, 4);//8 numbers each 16 bit//coalescing with 2 output
//// vLen = 4;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_4(1024, nOuts, 4, BIT_16, 4, 1024, 4);//4 numbers each 16 bit//coalescing with 2 output
////vLen = 2;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_2(1024, nOuts, 2, BIT_16, 4, 1024, 4);//2 numbers each 16 bit//coalescing with 2 output
////vLen = 1;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_1(1024, nOuts, 1, BIT_16, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 32;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_32(1024, nOuts, 32, BIT_16, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 16;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_16(1024, nOuts, 16, BIT_16, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 16;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_256(1024, nOuts, 256, BIT_16, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
//vLen = 16
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_16_32(1024, nOuts, 16, BIT_32, 4, 1024, 4);//16 numbers each 32 bit//coalescing with 2 output
////vLen = 8
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_8_32(1024, nOuts, 8, BIT_32, 4, 1024, 4);//8 numbers each 32 bit//coalescing with 2 output
////vLen = 4
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_4_32(1024, nOuts, 4, BIT_32, 4, 1024, 4);//4 numbers each 32 bit//coalescing with 2 output
////vLen = 2
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_2_32(1024, nOuts, 2, BIT_32, 4, 1024, 4);//2 numbers each 32 bit//coalescing with 2 output
////vLen = 1
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_1_32(1024, nOuts, 1, BIT_32, 4, 1024, 4);//1 numbers each 32 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_p5_32(1024, 128, 1024);//0.5 numbers each 32 bit//coalescing with 2 output
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_p25_32(1024, 64, 1024);//0.25 numbers each 32 bit//coalescing with 2 output


//nbits = 8
//vLen = 4;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_4_8(1024, nOuts, 4, 8, 4, 1024, 4);//4 numbers each 16 bit//coalescing with 2 output
////vLen = 2;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_2_8(1024, nOuts, 2, 8, 4, 1024, 4);//2 numbers each 16 bit//coalescing with 2 output
////vLen = 1;
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_1_8(1024, nOuts, 1, 8, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
//
////nbits = 1
////vLen = 16
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_16_1(1024, nOuts, 16, 1, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 8
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_8_1(1024, nOuts, 8, 1, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 4
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_4_1(1024, nOuts, 4, 1, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 2
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_2_1(1024, nOuts, 2, 1, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 1
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_1_1(1024, nOuts, 1, 1, 4, 1024, 4);//1 numbers each 16 bit//coalescing with 2 output
////vLen = 24
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_24_1(1024, nOuts, 24, 1, 4, 1024, 4);
////vLen = 12
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_12_1(1024, nOuts, 12, 1, 4, 1024, 4);
////vLen = 6
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_6_1(1024, nOuts, 6, 1, 4, 1024, 4);
////vLen = 3
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_3_1(1024, nOuts, 3, 1, 4, 1024, 4);
////vLen = 32
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_32_1(1024, nOuts, 32, 1, 4, 1024, 4);
////vLen = 64
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_64_1(1024, nOuts, 64, 1, 4, 1024, 4);
////vLen = 128
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_128_1(1024, nOuts, 128, 1, 4, 1024, 4);
////vLen = 256
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_256_1(1024, nOuts, 256, 1, 4, 1024, 4);
////vLen = 512
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_512_1(1024, nOuts, 512, 1, 4, 1024, 4);
//vLen = 4096
//cudaFFTProcessorTest_general cudaFFTProcessorTestTest_general_coal_2_16_vector_4096_1(1024, nOuts, 4096, 1, 4, 1024, 4);



FFT_Processor_fftw::FFT_Processor_fftw(const int N): _2N(2*N),N(N),Ns2(N/2) {
    rev_in = (double*) malloc(sizeof(double) * _2N);
    out = (double*) malloc(sizeof(double) * _2N);
    rev_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1));
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1));
    rev_p = fftw_plan_dft_r2c_1d(_2N, rev_in, rev_out, FFTW_ESTIMATE);	
    p = fftw_plan_dft_c2r_1d(_2N, in, out, FFTW_ESTIMATE);	
    omegaxminus1 = new cplx[_2N];
    for (int x=0; x<_2N; x++) {
	omegaxminus1[x]=cplx(cos(x*M_PI/N)-1.,-sin(x*M_PI/N)); // instead of cos(x*M_PI/N)-1. + sin(x*M_PI/N) * I
	//exp(i.x.pi/N)-1
    }
}
void FFT_Processor_fftw::execute_reverse_int(cplx* res, const int* a) {
    cplx* rev_out_cplx = (cplx*) rev_out; //fftw_complex and cplx are layout-compatible
    for (int i=0; i<N; i++) rev_in[i]=a[i]/2.;
    for (int i=0; i<N; i++) rev_in[N+i]=-rev_in[i];
    fftw_execute(rev_p);

    for (int i=0; i<Ns2; i++) res[i]=rev_out_cplx[2*i+1];

    for (int i=0; i<=Ns2; i++) assert(abs(rev_out_cplx[2*i])<1e-20);
}
void FFT_Processor_fftw::execute_reverse_torus32(cplx* res, const Torus32* a) {
    static const double _2pm33 = 1./double(INT64_C(1)<<33);
    int32_t* aa = (int32_t*) a;
    cplx* rev_out_cplx = (cplx*) rev_out; //fftw_complex and cplx are layout-compatible
    for (int i=0; i<N; i++) rev_in[i]=aa[i]*_2pm33;
    for (int i=0; i<N; i++) rev_in[N+i]=-rev_in[i];
    fftw_execute(rev_p);
    for (int i=0; i<Ns2; i++) res[i]=rev_out_cplx[2*i+1];
    for (int i=0; i<=Ns2; i++) assert(abs(rev_out_cplx[2*i])<1e-20);
}
void FFT_Processor_fftw::execute_direct_Torus32(Torus32* res, const cplx* a) {
    static const double _2p32 = double(INT64_C(1)<<32);
    static const double _1sN = double(1)/double(N);

    cplx* in_cplx = (cplx*) in; //fftw_complex and cplx are layout-compatible
    for (int i=0; i<=Ns2; i++) in_cplx[2*i]=0;
    for (int i=0; i<Ns2; i++) in_cplx[2*i+1]=a[i];
    fftw_execute(p);
    for (int i=0; i<N; i++) {
        res[i]=Torus32(int64_t(out[i] * _1sN * _2p32));//
    }
    //pas besoin du fmod... Torus32(int64_t(fmod(rev_out[i]*_1sN,1.)*_2p32));
    for (int i=0; i<N; i++) assert(fabs(out[N+i]+out[i])<1e-20);
}

FFT_Processor_fftw::~FFT_Processor_fftw() {
    fftw_destroy_plan(p);
    fftw_destroy_plan(rev_p);
    fftw_free(in); fftw_free(rev_out);	
    free(rev_in); free(out);
    delete[] omegaxminus1;
}

/**
 * FFT functions 
 */
EXPORT void IntPolynomial_ifft(LagrangeHalfCPolynomial* result, const IntPolynomial* p) {
    fp1024_fftw.execute_reverse_int(((LagrangeHalfCPolynomial_IMPL*)result)->coefsC, p->coefs);
}

EXPORT void TorusPolynomial_ifft(LagrangeHalfCPolynomial* result, const TorusPolynomial* p) {
    fp1024_fftw.execute_reverse_torus32(((LagrangeHalfCPolynomial_IMPL*)result)->coefsC, p->coefsT);
}

EXPORT void TorusPolynomial_fft(TorusPolynomial* result, const LagrangeHalfCPolynomial* p) {
    fp1024_fftw.execute_direct_Torus32(result->coefsT, ((LagrangeHalfCPolynomial_IMPL*)p)->coefsC);
}

EXPORT void IntPolynomial_ifft_16(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p) {
    cout << "IntPolynomial_ifft_16" << endl;
    /*
    if (bitSize == 16) {
        cudaFFTProcessorTestTest_16.execute_reverse_int(result, p->coefs);
    } else if(bitSize == 32) {
        cudaFFTProcessorTestTest_32.execute_reverse_int(result, p->coefs);
    }
     */
//    cudaDeviceSynchronize();
}

EXPORT void IntPolynomial_ifft_16_Coalesce(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p) {
//    if(bitSize == 16){
//        cudaFFTProcessorTestTest_general_16.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 32) {
//        cudaFFTProcessorTestTest_general_32.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 1) {
//        cudaFFTProcessorTestTest_general_1.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 8) {
//        cudaFFTProcessorTestTest_general_8.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 4) {
//        cudaFFTProcessorTestTest_general_4.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 2) {
//        cudaFFTProcessorTestTest_general_2.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 48) {
//        cudaFFTProcessorTestTest_general_48.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 24) {
//        cudaFFTProcessorTestTest_general_24.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 12) {
//        cudaFFTProcessorTestTest_general_12.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 6) {
//        cudaFFTProcessorTestTest_general_6.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 3) {
//        cudaFFTProcessorTestTest_general_3.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 64) {
//        cudaFFTProcessorTestTest_general_64.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 128) {
//        cudaFFTProcessorTestTest_general_128.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 256) {
//        cudaFFTProcessorTestTest_general_256.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 512) {
//        cudaFFTProcessorTestTest_general_512.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 1024) {
//        cudaFFTProcessorTestTest_general_1024.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 2048) {
//        cudaFFTProcessorTestTest_general_2048.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 80) {
//        cudaFFTProcessorTestTest_general_80.execute_reverse_int(result, p->coefs);
//    } else {
        cout << "IntPolynomial_ifft_16_Coalesce: " << bitSize << endl;
//    }
//    cudaDeviceSynchronize();
}










EXPORT void IntPolynomial_ifft_16_2(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p) {
    cout << "IntPolynomial_ifft_16_2" << endl;
/*
    if(bitSize == 16){
        cudaFFTProcessorTestTest_2_16.execute_reverse_int(result, p->coefs);
    } else if (bitSize == 32) {
        cudaFFTProcessorTestTest_2_32.execute_reverse_int(result, p->coefs);
    }*/
//    cudaDeviceSynchronize();
}

EXPORT void IntPolynomial_ifft_16_2_Coalesce(cufftDoubleComplex* result, int bitSize, const IntPolynomial* p) {
//    cout << "IntPolynomial_ifft_16_2_Coalesce" << endl;
//    if(bitSize == 16){
//        cudaFFTProcessorTestTest_general_coal_2_16.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 32) {
//        cudaFFTProcessorTestTest_general_coal_2_32.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 8) {
//        cudaFFTProcessorTestTest_general_coal_2_8.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 1) {
//        cudaFFTProcessorTestTest_general_coal_2_1.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 4) {
//        cudaFFTProcessorTestTest_general_coal_2_4.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 24) {
//        cudaFFTProcessorTestTest_general_coal_2_24.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 2) {
//        cudaFFTProcessorTestTest_general_coal_2_2.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 6) {
//        cudaFFTProcessorTestTest_general_coal_2_6.execute_reverse_int(result, p->coefs);
//    } else {
        cout << "IntPolynomial_ifft_16_2_Coalesce: " << bitSize << endl;
//    }
//    cudaDeviceSynchronize();
}


//EXPORT void IntPolynomial_ifft_16_2_Coalesce_one_out(cufftDoubleComplex* result, int nOutputs, int bitSize, IntPolynomial* p) {
//    if(bitSize == 16){
//        cudaFFTProcessorTestTest_general_coal_2_8.execute_reverse_int(result, p->coefs);
//    } else if (bitSize == 1) {
//        cudaFFTProcessorTestTest_general_coal_1_1.execute_reverse_int(result, p->coefs);
//    } else {
//        cout << "IntPolynomial_ifft_16_2_Coalesce: " << bitSize << endl;
//    }
////    else if (bitSize == 32) {
////        cudaFFTProcessorTestTest_general_coal_2_32.execute_reverse_int(result, p->coefs);
////    } else if (bitSize == 8) {
////        cudaFFTProcessorTestTest_general_coal_2_8.execute_reverse_int(result, p->coefs);
//
//// else if (bitSize == 4) {
////        cudaFFTProcessorTestTest_general_coal_2_4.execute_reverse_int(result, p->coefs);
////    } else if (bitSize == 24) {
////        cudaFFTProcessorTestTest_general_coal_2_24.execute_reverse_int(result, p->coefs);
////    } else if (bitSize == 2) {
////        cudaFFTProcessorTestTest_general_coal_2_2.execute_reverse_int(result, p->coefs);
////    }
//
//}

EXPORT void IntPolynomial_ifft_16_2_Coalesce_vector(cufftDoubleComplex* result, int vLength, int bitSize, const IntPolynomial* p) {

//    if (bitSize == 16) {
//        if (vLength == 8) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_8.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_2.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 32) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_32.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 16) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_16.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 256) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_256.execute_reverse_int(result, p->coefs);
//        } else {
//            cout << "IntPolynomial_ifft_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//        }
//    } else if (bitSize == 8) {
//        if (vLength == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4_8.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_2_8.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_1_8.execute_reverse_int(result, p->coefs);
//        } else {
//            cout << "IntPolynomial_ifft_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//        }
//    } else if (bitSize == 1) {
//        if (vLength == 16) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_16_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 8) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_8_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_2_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_1_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 24) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_24_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 12) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_12_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 6) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_6_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 3) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_3_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 32) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_32_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 64) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_64_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 128) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_128_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 256) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_256_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 512) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_512_1.execute_reverse_int(result, p->coefs);
//        } else if (vLength == 4096) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4096_1.execute_reverse_int(result, p->coefs);
//        } else {
//            cout << "IntPolynomial_ifft_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//        }
//    } else {
        cout << "IntPolynomial_ifft_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//    }
//    cudaDeviceSynchronize();
}


EXPORT void TorusPolynomial_fft_16(TorusPolynomial* result, const LagrangeHalfCPolynomial* p, int startIndex,
                                   int endIndex, int bitSize) {
    cout << "TorusPolynomial_fft_16" << endl;

    /*
    if(bitSize == 16){
        cudaFFTProcessorTestTest_16.execute_direct_Torus32(&(result->coefsT[startIndex]),
                                                           ((LagrangeHalfCPolynomial_IMPL*)p)->coefsC);
    } else if (bitSize == 32) {
        cudaFFTProcessorTestTest_32.execute_direct_Torus32(&(result->coefsT[startIndex]),
                                                              ((LagrangeHalfCPolynomial_IMPL*)p)->coefsC);
    }*/
//    cudaDeviceSynchronize();
}
EXPORT void TorusPolynomial_fft_gpu(TorusPolynomial* result, cufftDoubleComplex *source, int bitSize, int N, int Ns2) {

//    if(bitSize == 16){
//        cudaFFTProcessorTestTest_general_16.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if (bitSize == 32) {
//        cudaFFTProcessorTestTest_general_32.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 1) {
//        cudaFFTProcessorTestTest_general_1.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 8) {
//        cudaFFTProcessorTestTest_general_8.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 4) {
//        cudaFFTProcessorTestTest_general_4.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 2) {
//        cudaFFTProcessorTestTest_general_2.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 48) {
//        cudaFFTProcessorTestTest_general_48.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 24) {
//        cudaFFTProcessorTestTest_general_24.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 12) {
//        cudaFFTProcessorTestTest_general_12.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 6) {
//        cudaFFTProcessorTestTest_general_6.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 3) {
//        cudaFFTProcessorTestTest_general_3.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 64) {
//        cudaFFTProcessorTestTest_general_64.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 80) {
//        cudaFFTProcessorTestTest_general_80.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 128) {
//        cudaFFTProcessorTestTest_general_128.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 256) {
//        cudaFFTProcessorTestTest_general_256.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 512) {
//        cudaFFTProcessorTestTest_general_512.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 1024) {
//        cudaFFTProcessorTestTest_general_1024.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else if(bitSize == 2048) {
//        cudaFFTProcessorTestTest_general_2048.execute_direct_Torus32_gpu(result->coefsT, source);
//    } else {
        cout << "TorusPolynomial_fft_gpu: " << bitSize << endl;
//    }
//    cudaDeviceSynchronize();
}






EXPORT void TorusPolynomial_fft_gpu_2(Torus32 *result, cufftDoubleComplex *source, int nOutputs, int bitSize,
                                      int N, int Ns2) {
//    cout << "bitSize: " << bitSize << endl;

//    if (nOutputs == 2) {
//        if (bitSize == 16) {
//            cudaFFTProcessorTestTest_general_coal_2_16.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 32) {
//            cudaFFTProcessorTestTest_general_coal_2_32.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 8) {
//            cudaFFTProcessorTestTest_general_coal_2_8.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_1.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_4.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 24) {
//            cudaFFTProcessorTestTest_general_coal_2_24.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_2.execute_direct_Torus32_gpu(result, source);
//        } else if (bitSize == 6) {
//            cudaFFTProcessorTestTest_general_coal_2_6.execute_direct_Torus32_gpu(result, source);
//        } else {
//            cout << " TorusPolynomial_fft_gpu_2: in" << bitSize << " " << nOutputs << endl;
//        }
//    } else {
        cout << " TorusPolynomial_fft_gpu_2: out" << bitSize << " " << nOutputs << endl;
//    }
//    cudaDeviceSynchronize();
}




EXPORT void TorusPolynomial_fft_gpu_16_2_Coalesce_vector(TorusPolynomial* result, cufftDoubleComplex *source,
                                                         int vLength, int nOutputs, int bitSize, int N, int Ns2) {

//    if(bitSize == 16) {
//        if (vLength == 8) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_8.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_2.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 32) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_32.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 16) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_16.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 256) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_256.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else {
//            cout << "TorusPolynomial_fft_gpu_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//        }
//    } else if (bitSize == 8) {
//        if (vLength == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4_8.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_2_8.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_1_8.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else {
//            cout << "TorusPolynomial_fft_gpu_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//        }
//    } else if (bitSize == 1) {
//        if (vLength == 16) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_16_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 8) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_8_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 4) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 2) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_2_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 1) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_1_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 24) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_24_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 12) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_12_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 6) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_6_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 3) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_3_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 32) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_32_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 64) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_64_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 128) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_128_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 256) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_256_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 512) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_512_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else if (vLength == 4096) {
//            cudaFFTProcessorTestTest_general_coal_2_16_vector_4096_1.execute_direct_Torus32_gpu(result->coefsT, source);
//        } else {
//            cout << "TorusPolynomial_fft_gpu_16_2_Coalesce_vector: " << bitSize << " " << vLength << endl;
//        }
//    } else {
        cout << "Outer: TorusPolynomial_fft_gpu_16_2_Coalesce_vector: " << bitSize << " " << vLength << " " << nOutputs << endl;
//    }
//    cudaDeviceSynchronize();
}