#ifndef TFHE_TEST_ENVIRONMENT

#include <cstdlib>
#include <iostream>
#include <random>
#include <cassert>
#include "tfhe_core.h"
#include "numeric_functions.h"
#include "lweparams.h"
#include "lwekey.h"
#include "lwesamples.h"
#include "lwekeyswitch.h"
#include "lwe-functions.h"
#include "lwebootstrappingkey.h"
#include "tfhe.h"
#include <fstream>


using namespace std;

#else
#undef EXPORT
#define EXPORT static
#endif


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

//*//*****************************************
// zones on the torus -> to see
//*//*****************************************


/*
 * Homomorphic bootstrapped NAND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsNAND(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,1/8) - ca - cb
    static const Torus32 NandConst = modSwitchToTorus32(1, 8);
    lweNoiselessTrivial(temp_result, NandConst, in_out_params);
    lweSubTo(temp_result, ca, in_out_params);
    lweSubTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped OR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,1/8) + ca + cb
    static const Torus32 OrConst = modSwitchToTorus32(1, 8);
    lweNoiselessTrivial(temp_result, OrConst, in_out_params);
    lweAddTo(temp_result, ca, in_out_params);
    lweAddTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped AND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsAND(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result, AndConst, in_out_params);

    lweAddTo(temp_result, ca, in_out_params);
    lweAddTo(temp_result, cb, in_out_params);

//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << temp_result->a[i] << " ";
//    }
//    cout << endl;

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

//    cout << "old: ";
//    for (int i = 0; i < 10; ++i) {
//        cout << result->a[i] << " ";
//    }
//    cout << result->b;
//    cout << endl;

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped XOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsXOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    lweNoiselessTrivial(temp_result, XorConst, in_out_params);
    lweAddMulTo(temp_result, 2, ca, in_out_params);
    lweAddMulTo(temp_result, 2, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped XNOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsXNOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/4) + 2*(-ca-cb)
    static const Torus32 XnorConst = modSwitchToTorus32(-1, 4);
    lweNoiselessTrivial(temp_result, XnorConst, in_out_params);
    lweSubMulTo(temp_result, 2, ca, in_out_params);
    lweSubMulTo(temp_result, 2, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped NOT gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void bootsNOT(LweSample *result, const LweSample *ca, const TFheGateBootstrappingCloudKeySet *bk) {
    const LweParams *in_out_params = bk->params->in_out_params;
    lweNegate(result, ca, in_out_params);
}


/*
 * Homomorphic bootstrapped COPY gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void bootsCOPY(LweSample *result, const LweSample *ca, const TFheGateBootstrappingCloudKeySet *bk) {
    const LweParams *in_out_params = bk->params->in_out_params;
    lweCopy(result, ca, in_out_params);
}

/*
 * Homomorphic Trivial Constant gate (doesn't need to be bootstrapped)
 * Takes a boolean value)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void bootsCONSTANT(LweSample *result, int value, const TFheGateBootstrappingCloudKeySet *bk) {
    const LweParams *in_out_params = bk->params->in_out_params;
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    lweNoiselessTrivial(result, value ? MU : -MU, in_out_params);
}


/*
 * Homomorphic bootstrapped NOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsNOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/8) - ca - cb
    static const Torus32 NorConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result, NorConst, in_out_params);
    lweSubTo(temp_result, ca, in_out_params);
    lweSubTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped AndNY Gate: not(a) and b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsANDNY(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/8) - ca + cb
    static const Torus32 AndNYConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result, AndNYConst, in_out_params);
    lweSubTo(temp_result, ca, in_out_params);
    lweAddTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped AndYN Gate: a and not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsANDYN(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/8) + ca - cb
    static const Torus32 AndYNConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result, AndYNConst, in_out_params);
    lweAddTo(temp_result, ca, in_out_params);
    lweSubTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped OrNY Gate: not(a) or b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsORNY(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,1/8) - ca + cb
    static const Torus32 OrNYConst = modSwitchToTorus32(1, 8);
    lweNoiselessTrivial(temp_result, OrNYConst, in_out_params);
    lweSubTo(temp_result, ca, in_out_params);
    lweAddTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}


/*
 * Homomorphic bootstrapped OrYN Gate: a or not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void
bootsORYN(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,1/8) + ca - cb
    static const Torus32 OrYNConst = modSwitchToTorus32(1, 8);
    lweNoiselessTrivial(temp_result, OrYNConst, in_out_params);
    lweAddTo(temp_result, ca, in_out_params);
    lweSubTo(temp_result, cb, in_out_params);

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);

    delete_LweSample(temp_result);
}




/*
 * Homomorphic bootstrapped Mux(a,b,c) = a?b:c = a*b + not(a)*c
 * Takes in input 3 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
EXPORT void bootsMUX(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *c,
                     const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;
    const LweParams *extracted_params = &bk->params->tgsw_params->tlwe_params->extracted_lweparams;

    LweSample *temp_result = new_LweSample(in_out_params);
    LweSample *temp_result1 = new_LweSample(extracted_params);
    LweSample *u1 = new_LweSample(extracted_params);
    LweSample *u2 = new_LweSample(extracted_params);


    //compute "AND(a,b)": (0,-1/8) + a + b
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result, AndConst, in_out_params);
    lweAddTo(temp_result, a, in_out_params);
    lweAddTo(temp_result, b, in_out_params);
    // Bootstrap without KeySwitch
    tfhe_bootstrap_woKS_FFT(u1, bk->bkFFT, MU, temp_result);


    //compute "AND(not(a),c)": (0,-1/8) - a + c
    lweNoiselessTrivial(temp_result, AndConst, in_out_params);
    lweSubTo(temp_result, a, in_out_params);
    lweAddTo(temp_result, c, in_out_params);
    // Bootstrap without KeySwitch
    tfhe_bootstrap_woKS_FFT(u2, bk->bkFFT, MU, temp_result);

    // Add u1=u1+u2
    static const Torus32 MuxConst = modSwitchToTorus32(1, 8);
    lweNoiselessTrivial(temp_result1, MuxConst, extracted_params);
    lweAddTo(temp_result1, u1, extracted_params);
    lweAddTo(temp_result1, u2, extracted_params);
    // Key switching
    lweKeySwitch(result, bk->bkFFT->ks, temp_result1);


    delete_LweSample(u2);
    delete_LweSample(u1);
    delete_LweSample(temp_result1);
    delete_LweSample(temp_result);
}

/////new for gpu
EXPORT LweSample_16* convertBitToNumberZero(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    int polySize = bk->params->in_out_params->n;
    LweSample_16* temp = (LweSample_16 *)malloc(sizeof(LweSample_16));

    temp->a = (int*)calloc(bitSize*polySize, sizeof(int));
    temp->b = (int*)calloc(bitSize, sizeof(int));
    temp->current_variance = (double*)calloc(bitSize, sizeof(double));

    return temp;
}

EXPORT LweSample_16 *convertBitToNumberZero_GPU(int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    int polySize = bk->params->in_out_params->n;
    LweSample_16 *temp = (LweSample_16 *) malloc(sizeof(LweSample_16));

    cudaMalloc(&(temp->a), bitSize * polySize * sizeof(int));
    temp->b = (int *) calloc(bitSize, sizeof(int));
    //testing start
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    for (int i = 0; i < bitSize; ++i) {
        temp->b[i] = -MU;
    }
    // testing end
    temp->current_variance = (double *) calloc(bitSize, sizeof(double));
    return temp;
}

EXPORT LweSample_16 *convertBitToNumberZero_GPU_2(int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    int polySize = bk->params->in_out_params->n;
    LweSample_16 *temp = (LweSample_16 *) malloc(sizeof(LweSample_16));

    cudaMalloc(&(temp->a), nOutputs * bitSize * polySize * sizeof(int));
    temp->b = (int *) calloc(nOutputs * bitSize, sizeof(int));
    temp->current_variance = (double *) calloc(nOutputs * bitSize, sizeof(double));
    return temp;
}


EXPORT LweSample_16 *
newLweSample_16(int bitSize, const LweParams *params) {
    int polySize = params->n;
    LweSample_16 *temp = (LweSample_16 *) malloc(sizeof(LweSample_16));

    temp->a = (int *) calloc(bitSize * polySize, sizeof(int));
    temp->b = (int *) calloc(bitSize, sizeof(int));
    temp->current_variance = (double *) calloc(bitSize, sizeof(double));

    return temp;
}

EXPORT LweSample_16 *
newLweSample_16_2(int nOutputs, int bitSize, const LweParams *params) {
    int polySize = params->n;
    LweSample_16 *temp = (LweSample_16 *) malloc(sizeof(LweSample_16));

    temp->a = (int *) calloc(nOutputs * bitSize * polySize, sizeof(int));
    temp->b = (int *) calloc(nOutputs * bitSize, sizeof(int));
    temp->current_variance = (double *) calloc(nOutputs * bitSize, sizeof(double));

    return temp;
}

EXPORT LweSample_16* convertBitToNumber(const LweSample* input, int bitSize,
                                        const TFheGateBootstrappingCloudKeySet *bk) {
    int polySize = bk->params->in_out_params->n;

    LweSample_16* temp = (LweSample_16 *)malloc(sizeof(LweSample_16));

    temp->a = (int*)malloc(sizeof(int)*bitSize*polySize);
    temp->b = (int*)malloc(sizeof(int)*bitSize);
    temp->current_variance = (double*)malloc(sizeof(double)*bitSize);

    for (int i = 0; i < bitSize; ++i) {
        for (int j = 0; j < polySize; ++j) {
            temp->a[i * polySize + j] = (int)input[i].a[j];
        }
        temp->b[i] = input[i].b;
        temp->current_variance[i] = input[i].current_variance;
    }

    return temp;
}

EXPORT LweSample*
convertNumberToBits(LweSample_16* number, int bitSize, const TFheGateBootstrappingCloudKeySet *bk) {
    LweSample *tempCiphertext = new_gate_bootstrapping_ciphertext_array(bitSize, bk->params);
    const int n = bk->params->in_out_params->n;

    for (int i = 0; i < bitSize; ++i) {
        int startIndex = i * n;
        for (int j = 0; j < n; ++j) {
            tempCiphertext[i].a[j] = number->a[startIndex + j];
        }
        tempCiphertext[i].b = number->b[i];
        tempCiphertext[i].current_variance = number->current_variance[i];
    }
    return tempCiphertext;
}

EXPORT void
freeLweSample_16(LweSample_16* input) {
    free(input->a);
    free(input->b);
    free(input->current_variance);
    free(input);
}

int* allocateAndCopyIntVectorFromHostToDevice(int *source, int len) {
    int *d_temp;
    int bytes = len * sizeof(int);
    cudaMalloc(&d_temp, bytes);
    cudaMemcpy(d_temp, source, bytes, cudaMemcpyHostToDevice);
    return d_temp;
}

int* allocateAndCopyIntVectorFromDeviceToHost(int *d_source, int len) {
    int bytes = len * sizeof(int);
    int *temp = (int*)malloc(bytes);
    cudaMemcpy(temp, d_source, bytes, cudaMemcpyDeviceToHost);
    return temp;
}

__global__ void vecAdd(int *result, int *a, int *b, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < length) {
        result[id] = a[id] + b[id];
    }
}

__global__ void vecAddMulTo(int *result, int mulVal, int *a, int *b, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < length) {
        result[id] = (mulVal * (a[id] + b[id]));
    }
}

void sendLweSmaple_16_a_ToGPU(LweSample_16 *sample, int bitSize, int polySize) {
    int *temp = sample->a;
    int byteLength = bitSize * polySize * sizeof(int);
    cudaMalloc(&(sample->a), byteLength);
    cudaMemcpy(sample->a, temp, byteLength, cudaMemcpyHostToDevice);
}


EXPORT void bootsAND_16(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                        Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                        int ***ks_b_gpu, double ***ks_cv_gpu, Torus32* ks_a_gpu_extendedPtr,
                        Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    int BLOCKSIZE = in_out_params->n;
    int gridSize = (int) ceil((float) (in_out_params->n * bitSize) / BLOCKSIZE);

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    LweSample_16 *temp_result = convertBitToNumberZero_GPU(bitSize, bk);


    for (int i = 0; i < bitSize; ++i) {
        temp_result->b[i] = AndConst;
    }

    vecAdd<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca->a, cb->a, in_out_params->n * bitSize);
//    cudaDeviceSynchronize();
//    cudaCheckErrors("kernel fail");

    for (int i = 0; i < bitSize; ++i) {
        temp_result->b[i] += (ca->b[i] + cb->b[i]);
//        cout << temp_result->b[i] << " ";
        temp_result->current_variance[i] += (ca->current_variance[i] + cb->current_variance[i]);
    }

    //test start
//            cout << "Inside AND:" << endl;
//    int *tempaa = new int[in_out_params->n * bitSize];
////////    int *tempba = new int[in_out_params->n * bitSize];
//    cudaMemcpy(tempaa, temp_result->a, in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//////////    cudaMemcpy(tempba, cb->a, in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * in_out_params->n;
////        cout << "ca: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << tempaa[sI + j] << " ";
//        }
//        cout << endl;
////        cout << "cb: ";
////        for (int j = 0; j < 10; ++j) {
////            cout << tempba[sI + j] << " ";
////        }
//        cout << temp_result->b[i] << " ";
//        cout << endl;
//    }
//    cout << endl;
    //test end

    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8

    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
                          ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);



//    assert(bitSize%2 == 0);

//    tfhe_bootstrap_FFT_16_2(result, bk->bkFFT, MU, 1, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
//                            ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                            ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
//    int *temp = new int[in_out_params->n * bitSize];
//    cudaMemcpy(temp, result->a, in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * in_out_params->n;
////        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
////        cout << result->b[i];
////        cout << endl;
//    }

    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}


EXPORT void bootsXOR_16(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int bitSize,
                        const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                        Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                        Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    int BLOCKSIZE = in_out_params->n;
    int gridSize = (int) ceil((float) (in_out_params->n * bitSize) / BLOCKSIZE);


    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);

    LweSample_16 *temp_result = convertBitToNumberZero_GPU(bitSize, bk);
    for (int i = 0; i < bitSize; ++i) {
        temp_result->b[i] = XorConst;
    }

    int mulVal = 2;
    vecAddMulTo<<<gridSize, BLOCKSIZE>>>(temp_result->a, mulVal, ca->a, cb->a, in_out_params->n * bitSize);
    for (int i = 0; i < bitSize; ++i) {
        temp_result->b[i] += (mulVal * (ca->b[i] + cb->b[i]));
        temp_result->current_variance[i] += ((mulVal * mulVal) * (ca->current_variance[i] + cb->current_variance[i]));
    }
    //test start
//            cout << "Inside xor: " << endl;
//    int *tempaa = new int[in_out_params->n * bitSize];
//    cudaMemcpy(tempaa, temp_result->a, in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
// //    cudaMemcpy(tempba, cb->a, in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * in_out_params->n;
//        cout << "a: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << tempaa[sI + j] << " ";
//        }
////        cout << temp_result->b[i] << " ";
//        cout << endl;
//    }
//    cout << endl;
//    cout << endl;
    //test end





    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                          ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    int *temp = new int[in_out_params->n * bitSize];
//    cudaMemcpy(temp, result->a, in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * in_out_params->n;
//        cout << "new: ";
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << result->b[i];
//        cout << endl;
//    }

    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}

__global__ void ANDXORvecMulAllto(int *destination, int *ca, int *cb, int n, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int mulVal = (id / (n * bitSize)) + 1;
        destination[id] = (mulVal * (ca[id % (n * bitSize)] + cb[id % (n * bitSize)]));
    }
}

EXPORT void bootsANDXOR_16(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nOutputs,
                           int bitSize, const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                           cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                           int ***ks_b_gpu, double ***ks_cv_gpu,
                           Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    static const int mulValXor = 2;

    LweSample_16 *temp_result = convertBitToNumberZero_GPU_2(nOutputs, bitSize, bk);

    //compute temp_result->a
    int BLOCKSIZE = in_out_params->n;
    int length = in_out_params->n * bitSize * nOutputs;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "gridSize " << gridSize << endl;
    ANDXORvecMulAllto<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca->a, cb->a, in_out_params->n, bitSize, length);
    //compute temp_result->b
    for (int i = 0; i < bitSize; ++i) {
        temp_result->b[i] = ca->b[i] + cb->b[i] + AndConst; //for and
        temp_result->b[i + bitSize] = mulValXor * (ca->b[i] + cb->b[i]) + XorConst;// for xor
        temp_result->current_variance[i] = ca->current_variance[i] + cb->current_variance[i]; //for and
        temp_result->current_variance[i + bitSize] = (mulValXor * mulValXor) * (ca->current_variance[i] + cb->current_variance[i]);// for xor
    }

    /*//test start
//    cout << "Inside AND:" << endl;
//    int *tempaa = new int[in_out_params->n * bitSize * nOutputs];
//    cudaMemcpy(tempaa, temp_result->a, nOutputs * in_out_params->n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = i * in_out_params->n ;
//        for (int j = 0; j < 10; ++j) {
//            cout << tempaa[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << "Inside XOR:" << endl;
//    for (int i = 0; i < bitSize; ++i) {
//        int sI = (bitSize + i) * in_out_params->n ;
//        for (int j = 0; j < 10; ++j) {
//            cout << tempaa[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;*/

//    cout << "compute temp_result->b" << endl;
//    cout << "total: " << nOutputs * bitSize << endl;

    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize * nOutputs, temp_result, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                          ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//    tfhe_bootstrap_FFT_16_2(result, bk->bkFFT, MU, nOutputs, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
//                            ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                            ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    int *temp = new int[length];
//    cudaMemcpy(temp, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nOutputs * bitSize; ++i) {
//        int sI = i * 500;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
////        cout << result->b[i] << " " << result->current_variance[i] << endl;
//    }
//    cout << endl;
//    cout << "I am inside the function" << endl;
    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}


__global__ void XORXORvecMulAllto(int *destination, int *ca, int *cb, int n, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int mulVal = 2;
        destination[id] = (mulVal * (ca[id % (n * bitSize)] + cb[id % (n * bitSize)]));
    }
}
EXPORT void bootsXORXOR_16(LweSample_16 *result,
                           const LweSample_16 *ca1, const LweSample_16 *ca2,
                           const LweSample_16 *cb1, const LweSample_16 *cb2,
                           int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                           cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                           Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                           Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    static const int mulValXor = 2, n = in_out_params->n;

    //compute temp_result->a
    int BLOCKSIZE = n;
    int length = n * bitSize;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "bitSize: " << bitSize<< endl;
//    cout << "length: " << length << endl;
//    cout << "nOut: " << nOutputs << endl;
//    cout << "gridSize: " << gridSize << endl;

    LweSample_16 *temp_result = convertBitToNumberZero_GPU_2(nOutputs, bitSize, bk);
    //compute temp_result->a
    XORXORvecMulAllto<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca1->a, ca2->a, n, bitSize, length);
    XORXORvecMulAllto<<<gridSize, BLOCKSIZE>>>(temp_result->a + n, cb1->a, cb2->a, n, bitSize, length);
    //compute temp_result->b
    for (int i = 0; i < bitSize; ++i) {
        temp_result->b[i] = mulValXor * (ca1->b[i] + ca2->b[i]) + XorConst; //for and
        temp_result->b[i + bitSize] = mulValXor * (cb1->b[i] + cb2->b[i]) + XorConst;// for xor

        temp_result->current_variance[i] = (mulValXor * mulValXor) * (ca1->current_variance[i] + ca2->current_variance[i]); //for and
        temp_result->current_variance[i + bitSize] = (mulValXor * mulValXor) * (cb1->current_variance[i] + cb2->current_variance[i]);// for xor
    }

//    tfhe_bootstrap_FFT_16_2(result, bk->bkFFT, MU, nOutputs, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
//                            ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                            ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize * nOutputs, temp_result, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                          ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}


__global__ void XORXORvecMulAllto_vector(int *destination, int *ca, int *cb, int n, int bitSize, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int mulVal = 2;
        destination[id] = (mulVal * (ca[id] + cb[id]));
    }
}

EXPORT void bootsXORXOR_16_vector(LweSample_16 *result,
                           const LweSample_16 *ca1, const LweSample_16 *ca2,
                           const LweSample_16 *cb1, const LweSample_16 *cb2,
                           int vLength, int nOutputs, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                           cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                           Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                           Torus32* ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    static const int mulValXor = 2, n = in_out_params->n;

    int totalBitSize = vLength * bitSize;
    //compute temp_result->a
    int BLOCKSIZE = n;
    int length = n * totalBitSize;//svLength * bitSize;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
//    cout << "vLen: " << vLength << endl;
//    cout << "bitSize: " << bitSize<< endl;
//    cout << "length: " << length << endl;
//    cout << "nOut: " << nOutputs << endl;
//    cout << "gridSize: " << gridSize << endl;

    LweSample_16 *temp_result = convertBitToNumberZero_GPU_2(nOutputs * vLength, bitSize, bk);
    //compute temp_result->a
    XORXORvecMulAllto_vector<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca1->a, ca2->a, n, bitSize, length);
    XORXORvecMulAllto_vector<<<gridSize, BLOCKSIZE>>>(temp_result->a + n * vLength, cb1->a, cb2->a, n, bitSize, length);
    //compute temp_result->b
    for (int i = 0; i < totalBitSize; ++i) {
        temp_result->b[i] = mulValXor * (ca1->b[i] + ca2->b[i]) + XorConst; //for and
        temp_result->b[i + totalBitSize] = mulValXor * (cb1->b[i] + cb2->b[i]) + XorConst;// for xor

        temp_result->current_variance[i] = (mulValXor * mulValXor) * (ca1->current_variance[i] + ca2->current_variance[i]); //for and
        temp_result->current_variance[i + totalBitSize] = (mulValXor * mulValXor) * (cb1->current_variance[i] + cb2->current_variance[i]);// for xor
    }
//    cout << "HEREZZZZZZZZ----" << endl;

//    tfhe_bootstrap_FFT_16_2(result, bk->bkFFT, MU, nOutputs, bitSize * vLength, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
//                            ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                            ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize * nOutputs * vLength, temp_result, cudaBkFFT, cudaBkFFTCoalesce, ks_a_gpu,
                          ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT,
//                                   cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}




__global__ void ANDXORvecMulAllto_vector(int *destination, int *ca, int *cb, int vLength, int bitSize, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        int mulVal = (id / (vLength * bitSize * n)) + 1;
        destination[id] = (mulVal * (ca[id % (vLength * bitSize * n)] + cb[id % (vLength * bitSize * n)]));
    }
}

EXPORT void bootsANDXOR_16_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nOutputs,
                                  int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                                  cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                                  int ***ks_b_gpu, double ***ks_cv_gpu,
                                  Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                  double *ks_cv_gpu_extendedPtr) {

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;
    const int n = in_out_params->n;//500

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    static const int mulValXor = 2;

    LweSample_16 *temp_result = convertBitToNumberZero_GPU_2(nOutputs, vLength * bitSize, bk);
    int BLOCKSIZE = 1024;
    int length = vLength * bitSize * nOutputs * n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    ANDXORvecMulAllto_vector<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca->a, cb->a, vLength, bitSize, n, length);

    //compute temp_result->b
    int totalBitSize = vLength * bitSize;
    for (int i = 0; i < totalBitSize; ++i) {
        temp_result->b[i] = ca->b[i] + cb->b[i] + AndConst; //for and
        temp_result->b[i + totalBitSize] = mulValXor * (ca->b[i] + cb->b[i]) + XorConst;// for xor
        temp_result->current_variance[i] = ca->current_variance[i] + cb->current_variance[i]; //for and
        temp_result->current_variance[i + totalBitSize] = (mulValXor * mulValXor) * (ca->current_variance[i] + cb->current_variance[i]);// for xor
    }

    //test start
//    cout << "Inside AND:" << endl;
//    int *tempaa = new int[n * bitSize * nOutputs * vLength];
//    cudaMemcpy(tempaa, temp_result->a, vLength * nOutputs * n * bitSize * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < bitSize * vLength; ++i) {
//        int sI = i * n ;
//        for (int j = 0; j < 10; ++j) {
//            cout << tempaa[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;
//    cout << "Inside XOR:" << endl;
//    for (int i = 0; i < bitSize * vLength; ++i) {
//        int sI = (bitSize * vLength + i) * n ;
//        for (int j = 0; j < 10; ++j) {
//            cout << tempaa[sI + j] << " ";
//        }
//        cout << endl;
//    }
//    cout << endl;


//    cout << "HEREZZZZZZZZZZZ" << endl;

    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, nOutputs * vLength * bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
                          ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
//    tfhe_bootstrap_FFT_16_2(result, bk->bkFFT, MU, nOutputs, vLength * bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
//                            ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                            ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    cout << "HEREZZZZZZZZZZZ--" << endl;

//    tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT,
//                                   cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    int *temp = new int[length];
//    cudaMemcpy(temp, result->a, length * sizeof(int), cudaMemcpyDeviceToHost);
//    cout << "AND PART" << endl;
//    for (int i = 0; i < 16 * bitSize; ++i) {
//        int sI = i * 500;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
////        cout << result->b[i] << " " << result->current_variance[i] << endl;
//    }
//    cout << endl;
//    cout << endl << "XOR PART" << endl;
//    for (int i = 0; i < 16 * bitSize; ++i) {
//        int sIB = bitSize * vLength ;
//        int sI = i * 500 + bitSize * vLength * 500;
//        for (int j = 0; j < 10; ++j) {
//            cout << temp[sI + j] << " ";
//        }
//        cout << endl;
////        cout << result->b[sIB + i] << " " << result->current_variance[sI + i] << endl;
//    }
//    cout << endl;



    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}












































//only used for multiplication
__global__ void vecAdd_MULT(int *result, int *a, int *b, int bAStart, int n, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < length) {
        result[id] = a[id] + b[(id % n) + bAStart];
    }
}



EXPORT void bootsAND_MULT(LweSample_16 *result,
                          const LweSample_16 *ca, const LweSample_16 *cb,
                          int resBitSize, int bitSize_A, int bIndex,
                          const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                          cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                          int ***ks_b_gpu, double ***ks_cv_gpu,
                          Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    assert(bitSize_A == resBitSize);
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    int n = in_out_params->n;
    int BLOCKSIZE = 1024;

    int gridSize = (int) ceil((float) (in_out_params->n * bitSize_A) / BLOCKSIZE);
    int bAstartIndex = bIndex * n;

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    LweSample_16 *temp_result = convertBitToNumberZero_GPU(bitSize_A, bk);

    for (int i = 0; i < bitSize_A; ++i) {
        temp_result->b[i] = AndConst;
    }

    vecAdd_MULT<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca->a, cb->a, bAstartIndex, n, n * bitSize_A);

    for (int i = 0; i < bitSize_A; ++i) {
        temp_result->b[i] += (ca->b[i] + cb->b[bIndex]);
        temp_result->current_variance[i] += (ca->current_variance[i] + cb->current_variance[bIndex]);
    }

    int bitSize = bitSize_A;
    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
                          ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    //dhor tokta mar perek
    //find out later on
//    cudaMemset(result->a + (n * bitSize_A), 0, n * (resBitSize - bitSize_A) * sizeof(int));
//    for (int i = bitSize_A; i < resBitSize; ++i) {
//        cout << result->b[i] << " ";
//    }
//    cout << endl;

    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}


EXPORT void bootsAND_MULT_con(LweSample_16 *result,
                              LweSample_16 **ca, LweSample_16 **cb,
                              int nConMul, int resBitSize, int bitSize_A, int bIndex,
                              const TFheGateBootstrappingCloudKeySet *bk, cufftDoubleComplex ****cudaBkFFT,
                              cufftDoubleComplex ***cudaBkFFTCoalesce, Torus32 ****ks_a_gpu,
                              Torus32 ****ks_a_gpu_extended, int ***ks_b_gpu, double ***ks_cv_gpu,
                              Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                              double *ks_cv_gpu_extendedPtr) {

    assert(bitSize_A == resBitSize);
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    int n = in_out_params->n;
    int BLOCKSIZE = n;

    int gridSize = (int) ceil((float) (n * bitSize_A) / BLOCKSIZE);
    int bAstartIndex = bIndex * n;

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    LweSample_16 *temp_result = convertBitToNumberZero_GPU(bitSize_A * nConMul, bk);

//    for (int i = 0; i < bitSize_A; ++i) {
//        temp_result->b[i] = AndConst;
//    }

    for (int i = 0; i < nConMul; ++i) {
        vecAdd_MULT<<<gridSize, BLOCKSIZE>>>(temp_result->a + i * bitSize_A * n, ca[i]->a, cb[i]->a, bAstartIndex, n, n * bitSize_A);
    }
    for (int j = 0; j < nConMul; ++j) {
        int sI = j * bitSize_A;
        for (int i = 0; i < bitSize_A; ++i) {
            int sI2 = sI + i;
            temp_result->b[sI2] = (ca[j]->b[i] + cb[j]->b[bIndex]) + AndConst;
            temp_result->current_variance[sI2] = (ca[j]->b[i] + cb[j]->b[bIndex]);
        }
    }

    int toalBitSize = bitSize_A * nConMul;
//    cout << "totalBitSize:" << toalBitSize << endl;
//    int nOutputs = 2;
//    int vLength = nConMul/2;
//    int bitSize = bitSize_A;
    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, toalBitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
                          ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//        cout << "bootsAND_MULT_con: bitSize: " << bitSize << " vLen: " << vLength << endl;
//        if (nConMul % 2 == 1) {
//            cout << "ERROR: Provide even number of vector" << endl;
//            exit(1);
//        }
//    tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT,
//                                   cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}






















//(a xor b) and c

EXPORT void bootsXOR_AND(LweSample *result, const LweSample *ca, const LweSample *cb, const LweSample *cc, const TFheGateBootstrappingCloudKeySet *bk) {

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);
    LweSample *temp_result1 = new_LweSample(in_out_params);
    LweSample *temp_result2 = new_LweSample(in_out_params);

    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);

    lweNoiselessTrivial(temp_result, XorConst, in_out_params);

    lweAddMulTo(temp_result, 2, ca, in_out_params);
    lweAddMulTo(temp_result, 2, cb, in_out_params);

    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);


    lweNoiselessTrivial(temp_result2, AndConst, in_out_params);

    lweAddTo(temp_result2, cc, in_out_params);
    lweAddTo(temp_result2, result, in_out_params);

//    static const Torus32 MU = modSwitchToTorus32(1, 8);
//    const LweParams *in_out_params = bk->params->in_out_params;

//    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/8) + ca + cb


//    lweAddTo(temp_result, cb, in_out_params);

    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result2);

    delete_LweSample(temp_result);
}

__global__ void reverseLweSample(int *dest, int *source, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        dest[id] = -source[id];
    }
}

void bootsNOT_16(LweSample_16 *output, LweSample_16 *input, int bitSize, int params_n) {
    int length = bitSize * params_n, BLOCKSIZE = 1024, gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    reverseLweSample<<<gridSize, BLOCKSIZE>>>(output->a, input->a, length);
    for (int i = 0; i < bitSize; ++i) {
        output->b[i] = -input->b[i];
        output->current_variance[i] = input->current_variance[i];
    }
}







//add vector




__global__ void ANDvec_vector(int *destination, int *ca, int *cb, int vLength, int bitSize, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = ca[id] + cb[id];
    }
}

EXPORT void bootsAND_16_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nOutputs,
                                  int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                                  cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                                  Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                                  int ***ks_b_gpu, double ***ks_cv_gpu,
                                  Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                  double *ks_cv_gpu_extendedPtr) {
    assert(nOutputs == 1);

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;
    const int n = in_out_params->n;//500

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);


    LweSample_16 *temp_result = convertBitToNumberZero_GPU_2(nOutputs, vLength * bitSize, bk);
    int BLOCKSIZE = 1024;
    int length = vLength * bitSize * nOutputs * n;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    ANDvec_vector<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca->a, cb->a, vLength, bitSize, n, length);

    //compute temp_result->b
    int totalBitSize = vLength * bitSize;
    for (int i = 0; i < totalBitSize; ++i) {
        temp_result->b[i] = ca->b[i] + cb->b[i] + AndConst; //for and
        temp_result->current_variance[i] = ca->current_variance[i] + cb->current_variance[i]; //for and
    }



//    cout << "xxxxxxxxxxxxxxxxxxxx" << endl;
//    cout << nOutputs << endl;
    tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, vLength * bitSize * nOutputs, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
                          ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
                          ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    if (vLength % 2 == 1 && vLength < 2) {
////        cout <<  "vLen: " << vLength << " bitSize: " << bitSize << endl;
//        tfhe_bootstrap_FFT_16(result, bk->bkFFT, MU, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce,
//                              ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                              ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
////        bitSize = bitSize/2;
////        tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, 2, 2, bitSize/4, temp_result, cudaBkFFT,
////                                       cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
////                                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//    } else {
//        nOutputs = 2;
//        vLength = vLength / 2;
//        tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT,
//                                       cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                                       ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//    }



    cudaFree(temp_result->a);
    temp_result->a = NULL;
    freeLweSample_16(temp_result);
}




__global__ void SUBvec_vector(int *destination, int *ca, int *cc, int vLength, int bitSize, int n, int length) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        destination[id] = cc[id] - ca[id];
    }
}


EXPORT void bootsMUX_16_vector(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb,
                               const LweSample_16 *cc, int vLength, int bitSize, const TFheGateBootstrappingCloudKeySet *bk,
                               cufftDoubleComplex ****cudaBkFFT, cufftDoubleComplex ***cudaBkFFTCoalesce,
                               Torus32 ****ks_a_gpu, Torus32 ****ks_a_gpu_extended,
                               int ***ks_b_gpu, double ***ks_cv_gpu,
                               Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                               double *ks_cv_gpu_extendedPtr) {

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;
    const LweParams *extracted_params = &bk->params->tgsw_params->tlwe_params->extracted_lweparams;
    const int n = in_out_params->n;//500
    const int extracted_n = extracted_params->n;//1024
    int nOutputs = 2;
    //for now vLength = 1
    assert(vLength == 1);
//    cout << "n: " << n << endl;
//    cout << "nOutputs: " << nOutputs << endl;
//    cout << "vLength: " << vLength << endl;
//    cout << "extracted_n: " << extracted_n << endl;

    int ex_length = vLength * bitSize * extracted_n;//ex_length does not include nOutputs
    int length = vLength * bitSize * n;//length does not include nOutputs
    int BLOCKSIZE = 1024;

    LweSample_16 *temp_result = convertBitToNumberZero_GPU_2(nOutputs, vLength * bitSize, bk);
    LweSample_16 *u = newLweSample_16_2(nOutputs, vLength * bitSize, extracted_params);
    LweSample_16 *ex_temp_result = newLweSample_16_2(1, vLength * bitSize, extracted_params);
    free(u->a);
    free(ex_temp_result->a);
    cudaMalloc(&(u->a), ex_length * nOutputs * sizeof(int));
    cudaMalloc(&(ex_temp_result->a), ex_length * sizeof(int));

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    static const Torus32 MuxConst = modSwitchToTorus32(1, 8);

    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    ANDvec_vector<<<gridSize, BLOCKSIZE>>>(temp_result->a, ca->a, cb->a, vLength, bitSize, n, length);
    SUBvec_vector<<<gridSize, BLOCKSIZE>>>(temp_result->a + length, ca->a, cc->a, vLength, bitSize, n, length);

    //compute temp_result->b
    int totalBitSize = vLength * bitSize;
    for (int i = 0; i < totalBitSize; ++i) {
        temp_result->b[i] = ca->b[i] + cb->b[i] + AndConst;
        temp_result->current_variance[i] = ca->current_variance[i] + cb->current_variance[i];

        temp_result->b[i + totalBitSize] = - ca->b[i] + cc->b[i] + AndConst;
        temp_result->current_variance[i + totalBitSize] = - ca->current_variance[i] + cc->current_variance[i]; //for and
    }

    tfhe_bootstrap_woKS_FFT_16_2_vector(u, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT, cudaBkFFTCoalesce);

    gridSize = (int) ceil((float) (ex_length) / BLOCKSIZE);
    ANDvec_vector<<<gridSize, BLOCKSIZE>>>(ex_temp_result->a, u->a, u->a + ex_length,
                                            vLength, bitSize, extracted_n, ex_length);

    for (int i = 0; i < vLength * bitSize; ++i) {
        ex_temp_result->b[i] = u->b[i] + u->b[i + vLength * bitSize] + MuxConst;
        ex_temp_result->current_variance[i] = u->current_variance[i] + u->current_variance[i + vLength * bitSize];
    }

//    lweKeySwitch_16(result, bk->ks, u, bitSize, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                    ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    lweKeySwitch_16_2_vector(result, bk->bkFFT->ks, ex_temp_result, vLength, 1, bitSize, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu,
                             ks_cv_gpu, ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

//    tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT,
//                                   cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);


//    if (vLength % 2 == 1 && vLength < 2) {
//        cout <<  "Odd number in bootsAND_16_vector" << endl;
//    }
//    nOutputs = 2;
//    vLength = vLength/2;
//    tfhe_bootstrap_FFT_16_2_vector(result, bk->bkFFT, MU, vLength, nOutputs, bitSize, temp_result, cudaBkFFT,
//                                   cudaBkFFTCoalesce, ks_a_gpu, ks_a_gpu_extended, ks_b_gpu, ks_cv_gpu,
//                                   ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);
//
//
//
//    cudaFree(temp_result->a);
//    temp_result->a = NULL;
//    freeLweSample_16(temp_result);
}


__device__ int modSwitchFromTorus32_GPU_device(Torus32 phase, int Msize){
    uint64_t interv = ((UINT64_C(1)<<63)/Msize)*2; // width of each intervall
    uint64_t half_interval = interv/2; // begin of the first intervall
    uint64_t phase64 = (uint64_t(phase)<<32) + half_interval;
    //floor to the nearest multiples of interv
    return phase64/interv;
}



__global__ void bootstrappingUptoBlindRotate_OneBit(int *accum_a_b, int *temp_accum_a_b, int *bara_g, Torus32 MU, int *temp_res_a, int temp_res_b, double temp_res_cv,
                                      cufftDoubleComplex *cudaBkFFTCoalesceExt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < 1024) {
        //x is temp_res
        register int n = 500, N = 1024, _2N = 2048, Ns2 = 512, Nx2 = 2048;
        //tfhe_bootstrap_FFT_16--> u
//        __shared__ int u_a[1024], u_b;//N
//        __shared__ double u_cv;
        //tfhe_bootstrap_woKS_FFT_16
//        __shared__ int bara[1024];//N//torusPolyTestvect_coef[1024],
        register int barb;

        bara_g[id] = 0;
//        torusPolyTestvect_coef[id] = MU;
        if (id < n) {//500
            bara_g[id] = modSwitchFromTorus32_GPU_device(temp_res_a[id], Nx2);
        }
        __syncthreads();

        barb = modSwitchFromTorus32_GPU_device(temp_res_b, Nx2);
        //tfhe_blindRotateAndExtract_FFT_16 -> here v = torusPolyTestvect_coef
        __shared__ int testvectbis[1024];//N
        //torusPolynomialMulByXai_16 -> res ->testvectbis, v-> torusPolyTestvect_coef
        register int a = _2N - barb;

        if (a < N) {//1024
            if (id < a) {
                testvectbis[id] = -MU;//torusPolyTestvect_coef[id - a + N];
            } else {
                testvectbis[id] = MU;//torusPolyTestvect_coef[id - a];
            }
        } else {
            register int aa = a - N;
            if (id < aa) {
                testvectbis[id] = MU;//torusPolyTestvect_coef[id - aa + N];
            } else {
                testvectbis[id] = -MU;//torusPolyTestvect_coef[id - aa];
            }
        }
        __syncthreads();
        accum_a_b[id] = 0;//accum_a
        accum_a_b[1024 + id] = testvectbis[id];

        temp_accum_a_b[id] = 0;//accum_a
        temp_accum_a_b[1024 + id] = 0;

//        bara_g[id] = bara[id];
    }
}

__global__ void prepareForiFFT_1_Bit(int *des, int *decaCoalesce, cufftDoubleReal *d_rev_in,
                                                     int *bara, int baraIndex, int *source) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int N = 1024, _2N = 2048, Ns2 = 512;

    register int tIndex = id % N;
    register int a = bara[baraIndex];
    register int aa = a - N;

    register bool l1 = a < N, l2 = tIndex < a, l3 = tIndex < aa;

    int des_id = l1 * (l2 * (-source[id - a + N] - source[id]) + (!l2) * (source[id - a] - source[id]))
                  + (!l1) * (l3 * (source[id - aa + N] - source[id])
                             + (!l3) * (-source[id - aa] - source[id]));

    register uint32_t halfBg = 512, maskMod = 1023, Bgbit = 10;
//    register uint32_t offset = 2149580800;


    register int p = 0;
    register int decal = (32 - (p + 1) * Bgbit);
    register uint32_t temp1 = (((uint32_t)(des_id + 2149580800)) >> decal) & maskMod;//offset

    register int xxxxx1 = (temp1 - halfBg);
//    decaCoalesce[((id / (N)) * (N)) + id] =
//            (middleBlock) * xxxxx1 + (!middleBlock) * (decaCoalesce[((id / (N)) * (N)) + id]);

    p = 1;
    decal = (32 - (p + 1) * Bgbit);
    temp1 = (((uint32_t)(des_id + 2149580800)) >> decal) & maskMod;//offset
    register int xxxxx2 = temp1 - halfBg;

//    decaCoalesce[((id / (N)) * (N)) + id + (N)] = middleBlock * xxxxx2 + (!middleBlock) * decaCoalesce[((id / (N)) * (N)) + id + (N)];


    register int bIndex = id / N;

    int destTod_rev_in = bIndex * _2N + tIndex + (bIndex >= 1) * N * 2;

    d_rev_in[destTod_rev_in] = xxxxx1/2.;
    d_rev_in[destTod_rev_in + 1024] = -xxxxx1/2.;

    destTod_rev_in += 2 * 1024;
    d_rev_in[destTod_rev_in] = xxxxx2/2.;
    d_rev_in[destTod_rev_in + 1024] = -xxxxx2/2.;
}


__global__  void prepareForFFT_1_Bit(cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex *tmpa_gpuCoal,
                                                 cufftDoubleComplex *d_in, cufftDoubleComplex *d_rev_out,
                                                 cufftDoubleComplex *bki,  int keyIndex,
                                                 int N, int Ns2, int length) {

    register int id = blockIdx.x*blockDim.x+threadIdx.x;
    register int k = 1, kpl = 4, keySI = keyIndex * (k + 1) * kpl * Ns2, aID, bID, offset;



//    if(id < 512) {

        int tempId = id;
        int bitIndex = tempId / Ns2;
        register cufftDoubleComplex v0 = d_rev_out[2 * tempId + 1 + bitIndex];//d_rev_out[2 * id + 1 + bitIndex];
//        cuDecaFFTCoalesce[tempId] = v0;

        tempId = tempId + (Ns2);
        bitIndex = (tempId) / Ns2;
        register cufftDoubleComplex v1 = d_rev_out[2 * tempId + 1 + bitIndex];
//        cuDecaFFTCoalesce[tempId] = v1;

        tempId = tempId + (Ns2);
        bitIndex = (tempId) / Ns2;
        register cufftDoubleComplex v2 = d_rev_out[2 * tempId + 1 + bitIndex];
//        cuDecaFFTCoalesce[tempId] = v2;

        tempId = tempId + (Ns2);
        bitIndex = (tempId) / Ns2;
        register cufftDoubleComplex v3 = d_rev_out[2 * tempId + 1 + bitIndex];
//        cuDecaFFTCoalesce[tempId] = v3;



        int i = 0;
        offset = i * Ns2;
        aID = keySI + offset + id % Ns2;
        bID = keySI + offset + id % Ns2 + Ns2 * kpl;
        cufftDoubleComplex temp_a0 = cuCmul(v0, bki[aID]);
        cufftDoubleComplex temp_b0 = cuCmul(v0, bki[bID]);

        i = 1;
        offset = i * Ns2;
        aID = keySI + offset + id % Ns2;
        bID = keySI + offset + id % Ns2 + Ns2 * kpl;
        cufftDoubleComplex temp_a1 = cuCmul(v1, bki[aID]);
        cufftDoubleComplex temp_b1 = cuCmul(v1, bki[bID]);

        i = 2;
        offset = i * Ns2;
        aID = keySI + offset + id % Ns2;
        bID = keySI + offset + id % Ns2 + Ns2 * kpl;
        cufftDoubleComplex temp_a2 = cuCmul(v2, bki[aID]);
        cufftDoubleComplex temp_b2 = cuCmul(v2, bki[bID]);

        i = 3;
        offset = i * Ns2;
        aID = keySI + offset + id % Ns2;
        bID = keySI + offset + id % Ns2 + Ns2 * kpl;
        cufftDoubleComplex temp_a3 = cuCmul(v3, bki[aID]);
        cufftDoubleComplex temp_b3 = cuCmul(v3, bki[bID]);

        cufftDoubleComplex tmpa_gpuCoal0;
        tmpa_gpuCoal0.x = temp_a0.x + temp_a1.x + temp_a2.x + temp_a3.x;
        tmpa_gpuCoal0.y = temp_a0.y + temp_a1.y + temp_a2.y + temp_a3.y;
//        tmpa_gpuCoal[id] = tmpa_gpuCoal0;

        cufftDoubleComplex tmpa_gpuCoal1;
        tmpa_gpuCoal1.x = temp_b0.x + temp_b1.x + temp_b2.x + temp_b3.x;
        tmpa_gpuCoal1.y = temp_b0.y + temp_b1.y + temp_b2.y + temp_b3.y;
//        tmpa_gpuCoal[id + Ns2] = tmpa_gpuCoal1;

        register int largeSI = (id / Ns2) * (N + 1);
        register int tid = id % Ns2;
        d_in[largeSI + 2 * tid + 1] = tmpa_gpuCoal0;

        largeSI = (id / Ns2 + 1) * (N + 1);
        d_in[largeSI + 2 * tid + 1] = tmpa_gpuCoal1;




        //init with 0
//        tmpa_gpuCoal[id].x = 0;
//        tmpa_gpuCoal[id].y = 0;
//        tmpa_gpuCoal[Ns2 + id].x = 0;
//        tmpa_gpuCoal[Ns2 + id].y = 0;
//#pragma unroll
//        for (int i = 0; i < kpl; ++i) {//kpl
//            offset = i * Ns2;
//            aID = keySI + offset + id;
//            bID = keySI + offset + id + Ns2 * kpl;
//
//            cufftDoubleComplex temp_a = cuCmul(cuDecaFFTCoalesce[offset + id], bki[aID]);
//            tmpa_gpuCoal[id].x += temp_a.x;
//            tmpa_gpuCoal[id].y += temp_a.y;
//
//            cufftDoubleComplex temp_b = cuCmul(cuDecaFFTCoalesce[offset + id], bki[bID]);
//            tmpa_gpuCoal[Ns2 + id].x += temp_b.x;
//            tmpa_gpuCoal[Ns2 + id].y += temp_b.y;
//
//        }
//    }
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();

//    if (id < 1024) {
//        register int largeSI = (id / Ns2) * (N + 1);
//        register int tid = id % Ns2;
//        d_in[largeSI + 2 * tid + 1] = tmpa_gpuCoal[id];
////        d_in[largeSI + 2 * tid + 1].y = 1;//tmpa_gpuCoal[id];
//    }

}


__global__ void finishUpFFT_n_Bit(int *temp2, cufftDoubleReal *d_out, int *temp3) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    register int N = 1024, _2N = 2048;
    register double _2p32 = double(INT64_C(1) << 32);
    register double _1sN = double(1) / double(N);
    register int bitIndex = id / N;
    register int tIndex = id % N;
    register int startIndexLarge = bitIndex * _2N;
    temp2[id] = Torus32(int64_t(d_out[startIndexLarge + tIndex] * _1sN * _2p32)) + temp3[id];//

}




__global__ void extractionAndKeySwitch_1_Bit(int *result_a, int *result_b,
                                       uint32_t *coal_d_aibar, uint32_t  *coal_d_aij,
                                       int *accum_a_b,
                                       Torus32 *ks_a_gpu_extendedPtr, Torus32 *ks_b_gpu_extendedPtr,
                                       double *ks_cv_gpu_extendedPtr) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    register int N = 1024, _2N = 2048, basebit = 2, base = 1 << basebit, mask = base - 1, t =8;
    register int32_t prec_offset = 1 << (32 - (1 + basebit * t));


    register int index = 0;
    register int bitIndex = id / N;
    register int tIndex = id % N;//corresponding to j
    register int startIndex = bitIndex * N;

    __shared__ uint32_t s_coal_d_aibar[1024];
//    __shared__ uint32_t coal_d_aij[1024 * 8];


    bool multipleOfN = id % N == 0;
    s_coal_d_aibar[id] = (multipleOfN) * (accum_a_b[index - tIndex + startIndex] + prec_offset)
            + (!multipleOfN) * (-accum_a_b[index - tIndex + startIndex + N] + prec_offset);


//    if (id % N == 0) {
//        coal_d_aibar[id] = accum_a_b[index - tIndex + startIndex] + prec_offset;
//    } else {
//        coal_d_aibar[id] = -accum_a_b[index - tIndex + startIndex + N] + prec_offset;
//    }


    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    __syncthreads();
//    if(id < 1024) {//t
        register int tempID = id;
        register int i = tempID / t;
        register int j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;

        tempID += 1024;
        i = tempID / t;
        j = tempID % t;
        coal_d_aij[tempID] = (s_coal_d_aibar[i] >> (32 - (j + 1) * basebit)) & mask;
//    }

//    __syncthreads();
    int subFromB = 0;
    int bi;
    if (id < 500) {
        result_a[id] = 0;
        register int A = 1024, B = t, C = base, D = 500, ks_t = 8;
#pragma unroll 0
        for (int i = 0; i < 1024; ++i) {
            int sI =  i * ks_t;
#pragma unroll 0
            for (int j = 0; j < 8; ++j) {//ks_t
                int sI2 = sI + j;
                int aij = coal_d_aij[sI2];
                if (aij != 0) {
                    result_a[id] -= ks_a_gpu_extendedPtr[i * B * C * D + j * C * D + aij * D + (id % D)];//sourceA[(i * B * C * D + j * C * D+ aij * params_n +  id)];//source[aij][id];
                }
//                if(id < 1) {
                    bi = coal_d_aij[sI2 + id];
                    subFromB += ks_b_gpu_extendedPtr[i * B * C + j * C + bi];
//                }
            }
        }
    }

    if (id < 1) {
        result_b[0] = accum_a_b[N] - subFromB;
    }


}


void bootstrapping_gull_gpu_1_bit_wise(LweSample_16 *result, int *temp_res_a, int *temp_res_b, int nBits,
                                       cufftDoubleComplex *cudaBkFFTCoalesceExt,
                                       Torus32 *ks_a_gpu_extendedPtr,
                                       Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    //bootstrapping woks uptoFFT
    int nThreads = 1024, BLOCKSIZE = 1024, k = 1, N = 1024, kpl = 4, Ns2 = 512, _2N = 2048;
    static const Torus32 MU = modSwitchToTorus32(1, 8);

    int gridSize = (int) ceil((float) (nThreads) / BLOCKSIZE);//1

    int *accum_a_b, *bara, *temp_accum_a_b;//accum a and accum b together; bara; tempaccum for mux rotate
    cudaMalloc(&accum_a_b, nBits * 1024 * (k + 1) * sizeof(int));
    cudaMalloc(&temp_accum_a_b, nBits * 1024 * (k + 1) * sizeof(int));
    cudaMalloc(&bara, nBits * 1024 * sizeof(int));


    cudaDeviceProp cProfile;
    cudaGetDeviceProperties(&cProfile, 0);
    int nSM = cProfile.multiProcessorCount;
    cout << "#SM: " << nSM << endl; //20
    cudaStream_t streams[nSM];

#pragma unroll
    for (int i = 0; i < 20; ++i) {//nSM
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    for (int bIndex = 0; bIndex < nBits; ++bIndex) {
        int accumStart = bIndex * (1024 * (k + 1));
        int baraStart = bIndex * 1024;
        int temp_res_aStart = bIndex * 500;

        bootstrappingUptoBlindRotate_OneBit<<<gridSize, BLOCKSIZE, 0, streams[bIndex % nSM]>>>
                                                                      (accum_a_b + accumStart,
                                                                              temp_accum_a_b + accumStart,
                                                                              bara + baraStart,
                                                                              MU,
                                                                              temp_res_a + temp_res_aStart,
                                                                              temp_res_b[bIndex], NULL,
                                                                              cudaBkFFTCoalesceExt);
    }
    cudaDeviceSynchronize();


    //after blind rotate
    int *decaCoalesce;
    cudaMalloc(&decaCoalesce, nBits * N * kpl * sizeof(int));//1024*4

    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, nBits * kpl * Ns2 * sizeof(cufftDoubleComplex));//512*4

    cufftDoubleComplex *tmpa_gpuCoal;
    cudaMalloc(&tmpa_gpuCoal, nBits * Ns2 * sizeof(cufftDoubleComplex) * (k + 1));

    //fft variables
    cufftDoubleReal* d_rev_in;
    cufftDoubleComplex *d_rev_out;
    cufftDoubleComplex *d_in;
    cufftDoubleReal *d_out;
    int batch = kpl;
    int dParts = 4;
    //fft plans
    cufftHandle p;
    cufftHandle rev_p;
    //fft variables allocation
    cudaMalloc(&d_rev_in, nBits * sizeof(cufftDoubleReal) * _2N * batch);
    cudaMalloc(&d_rev_out, nBits * sizeof(cufftDoubleComplex) * (N + 1) * batch);
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, nBits * batch);//(nBits * batch)/dParts);// (batch - (batch/dParts)));

    batch = 2;//batch/dParts;//a and b together
    cudaMalloc(&d_in, nBits * sizeof(cufftDoubleComplex) * (N + 1) * batch);//batch
    cudaMalloc(&d_out, nBits * sizeof(cufftDoubleReal) * _2N * batch);
    cufftPlan1d(&p, _2N, CUFFT_Z2D, nBits * batch);

    int *temp2 = temp_accum_a_b;
    int *temp3 = accum_a_b;


//    assert(nBits == 1);

    //call tfhe_MuxRotate_FFT_16
#pragma unroll
    for (int j = 0; j < 500; ++j) {//500
        gridSize = 2;//2;//as accum is of (k + 1)

        for (int bIndex = 0; bIndex < nBits; ++bIndex) {
            //find starting indices
            int accumStart = bIndex * 1024 * (k + 1);
            int decaCoalesceStart = bIndex * 1024 * kpl;
            int d_rev_inStart = bIndex * _2N * kpl;
            int baraStart = bIndex * N;
            prepareForiFFT_1_Bit<<<gridSize, BLOCKSIZE, 0, streams[bIndex % nSM]>>>
                                                           (temp2 + accumStart,
                                                                   decaCoalesce + decaCoalesceStart,
                                                                   d_rev_in + d_rev_inStart,
                                                                   bara + baraStart,
                                                                   j,
                                                                   temp3 + accumStart);
        }
        cudaDeviceSynchronize();

        cufftExecD2Z(rev_p, d_rev_in, d_rev_out);
        cudaDeviceSynchronize();

        int length = kpl * Ns2;//4 * 512 = 2048
        gridSize = 1;//(int) ceil((float) (length) / BLOCKSIZE); //2
        for (int bIndex = 0; bIndex < nBits; ++bIndex) {
            int cuDecaFFTCoalesceStart = bIndex * kpl * Ns2;
            int tmpa_gpuCoalStart = bIndex * (k + 1) * Ns2;
            int d_inStart = bIndex * (N + 1) * (k + 1);
            int d_rev_outStart = bIndex *(N + 1) * kpl;
            prepareForFFT_1_Bit<<<gridSize, 512, 0, streams[bIndex % nSM]>>>
                                                    (cuDecaFFTCoalesce + cuDecaFFTCoalesceStart,
                                                            tmpa_gpuCoal + tmpa_gpuCoalStart,
                                                            d_in + d_inStart,
                                                            d_rev_out + d_rev_outStart,
                                                            cudaBkFFTCoalesceExt, j, N, Ns2, length);
        }
        cudaDeviceSynchronize();



        cufftExecZ2D(p, d_in, d_out);
        cudaDeviceSynchronize();


        //after fft
        length = N * 2;
        gridSize = (int) ceil((float) (length) / BLOCKSIZE); //2
        for (int bIndex = 0; bIndex < nBits; ++bIndex) {
            int accumStart = bIndex * 1024 * (k + 1);
            int d_outStart = bIndex * _2N * (k + 1);
            finishUpFFT_n_Bit<<<gridSize, BLOCKSIZE, 0, streams[bIndex % nSM]>>>
                                                        (temp2 + accumStart,
                                                                d_out + d_outStart,
                                                                temp3 + accumStart);
        }
        cudaDeviceSynchronize();

        swap(temp2, temp3);
    }




    //output is in temp3
    //extract and ks
    //intermediate variables to test u (delete afterwards)
    int *result_b;
    double *result_cv = NULL;
    cudaMalloc(&result_b, nBits * sizeof(int));
//    cudaMalloc(&result_cv, 1 * sizeof(double));


    uint32_t *coal_d_aibar;
    cudaMalloc(&coal_d_aibar, nBits * N * sizeof(uint32_t));

    int coal_d_aijSize = nBits * N * 8;//t
    uint32_t  *coal_d_aij;
    cudaMalloc(&coal_d_aij, coal_d_aijSize * sizeof(uint32_t));

//    int length = N * 8;//t
    gridSize = 1;//(int) ceil((float) (length) / BLOCKSIZE);
    for (int bIndex = 0; bIndex < nBits; ++bIndex) {

        int result_aStart = bIndex * 500;
        int result_bStart = bIndex;
        int coal_d_aibarStart = bIndex * N;
        int coal_d_aijStart = bIndex * N * 8;
        int accumStart = bIndex * (k + 1) * 1024;


        extractionAndKeySwitch_1_Bit<<<gridSize, BLOCKSIZE, 0, streams[bIndex % nSM]>>>
                                                               (result->a + result_aStart,
                                                                       result_b + result_bStart,
                                                                       coal_d_aibar + coal_d_aibarStart,
                                                                       coal_d_aij + coal_d_aijStart,
                                                                       temp3 + accumStart,
                                                                       ks_a_gpu_extendedPtr,
                                                                       ks_b_gpu_extendedPtr,
                                                                       ks_cv_gpu_extendedPtr);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(result->b, result_b, nBits * sizeof(int), cudaMemcpyDeviceToHost);


//    int *temp = new int[500];
//    cudaMemcpy(temp, result->a, 500 * sizeof(int), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 500; ++i) {
//        cout << temp[i] << " ";
//    }
//    cout << endl;
//    cout << result->b[0] << endl;

//    assert(nBits == 1);

#pragma unroll
    for (int i = 0; i < 20; ++i) { //nSM
        cudaStreamDestroy(streams[i]);

    }

    cudaFree(temp_res_a);
    cudaFree(accum_a_b);
    cudaFree(temp_accum_a_b);
    cudaFree(bara);
    cudaFree(decaCoalesce);//1024*4
    cudaFree(cuDecaFFTCoalesce);//512*4
    cudaFree(tmpa_gpuCoal);
    cudaFree(d_rev_in);
    cudaFree(d_rev_out);
    cudaFree(d_in);//batch
    cudaFree(d_out);
    cudaFree(result_b);
    cudaFree(coal_d_aibar);
    cudaFree(coal_d_aij);

    cufftDestroy(rev_p);
    cufftDestroy(p);
}






EXPORT void bootsAND_fullGPU_OneBit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                    cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                    Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    const int n = 500, BLOCKSIZE = 1024, N = 1024, _2N = 2048, Ns2 = 512, k = 1, kpl = 4, l = 2, offset = 2149580800,
            halfBg = 512, maskMod = 1023;

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);

    int *temp_res_a, *temp_res_b;
    cudaMalloc(&temp_res_a, n * nBits * sizeof(Torus32));
    temp_res_b = new int[nBits];

    int gridSize = (int) ceil((float) (n * nBits) / BLOCKSIZE);

    vecAdd<<<gridSize, BLOCKSIZE>>>(temp_res_a, ca->a, cb->a, n * nBits);
    for (int i = 0; i < nBits; ++i) {
        temp_res_b[i] = ca->b[i] + cb->b[i] + AndConst;
    }

    bootstrapping_gull_gpu_1_bit_wise(result, temp_res_a, temp_res_b, nBits, cudaBkFFTCoalesceExt,
                                      ks_a_gpu_extendedPtr, ks_b_gpu_extendedPtr, ks_cv_gpu_extendedPtr);

    cudaFree(temp_res_a);
    delete [] temp_res_b;
}





























__global__ void bootstrappingUptoBlindRotate_n_Bit(int *accum_a_b, int *temp_accum_a_b, int *bara, int *testvectbis,
                                                   Torus32 MU, int nBits,
                                                   int *temp_res_a, int *barb) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int n = 500, N = 1024, _2N = 2048, Ns2 = 512, Nx2 = 2048;

    register int bIndex = id / N;
    register int baraIndex = id % N;
    register int barbi = barb[bIndex];
    //torusPolynomialMulByXai_16 -> res ->testvectbis, v-> torusPolyTestvect_coef
    register int a = _2N - barbi;
    register int aa = a - N;
    testvectbis[id] = MU;

    if (a < N) {//1024
        if (baraIndex < a) {
            testvectbis[id] = -MU;
        }
    } else {
        if (baraIndex >= aa) {
            testvectbis[id] = -MU;
        }
    }
    for (int i = 0; i < nBits; ++i) {
        __syncthreads();
    }
    accum_a_b[id] = 0;//accum_a
    accum_a_b[1024 * nBits + id] = testvectbis[id];

    temp_accum_a_b[id] = 0;//accum_a
    temp_accum_a_b[1024 * nBits + id] = 0;

    bara[id] = 0;

    if (id < nBits * 500) {//500
        bIndex = id / 500;
        register  int destinationIndex = bIndex * N + id % n;
        bara[destinationIndex] = modSwitchFromTorus32_GPU_device(temp_res_a[id], Nx2);
    }
}



__global__ void prepareForiFFT_n_Bit(int *des, int *decaCoalesce, cufftDoubleReal *d_rev_in,
                                     int nBits, int nGrid, int *bara, int baraIndex, int *source) {
    register int id = blockIdx.x * blockDim.x + threadIdx.x;
    register int N = 1024, _2N = 2048, Ns2 = 512;

//    bool outerBlock = id < nBits * 2 * 1024;
//    if (id < nBits * 2 * 1024) {//nBits * (k + 1) * 1024
        register int bitIndex = (id / N) % nBits;
        register int threadIdModN = id % N;

        register int a = bara[bitIndex * N + baraIndex];
        register int aa = a - N;

        register bool l1 = a < N, l2 = threadIdModN < a, l3 = threadIdModN < aa;

//    des[id] = (!outerBlock) * des[id]
//              + outerBlock * (l1 * (l2 * (-source[id - a + N] - source[id])
//                                    + (!l2) * (source[id - a] - source[id]))
//                              + (!l1) * (l3 * (source[id - aa + N] - source[id])
//                                         + (!l3) * (-source[id - aa] - source[id])));

    int des_id = (l1 * (l2 * (-source[id - a + N] - source[id])
                                       + (!l2) * (source[id - a] - source[id]))
                                 + (!l1) * (l3 * (source[id - aa + N] - source[id])
                                            + (!l3) * (-source[id - aa] - source[id])));

//        if (a < N) {
//            if (threadIdModN < a) {
//                des[id] = -source[id - a + N] - source[id];
//            } else {
//                des[id] = source[id - a] - source[id];
//            }
//        } else {
//            if (threadIdModN < aa) {
//                des[id] = source[id - aa + N] - source[id];
//            } else {
//                des[id] = -source[id - aa] - source[id];
//            }
//        }
//    }
//    for (int i = 0; i < nGrid; ++i) {
//        __syncthreads();
//    }

    register uint32_t halfBg = 512, maskMod = 1023, Bgbit = 10, kpl = 4, l = 2;
    register uint32_t offset = 2149580800;

    bool middleBlock = id < nBits * 2 * 1024;//4//kpl

//    if(id < nBits * 4 * 1024) {//nBits * kpl * 1024
//        register int p = (id/(N * nBits)) % l;//0 1 0 1
//        register int index = (id/(l * N * nBits)) % l;//0 1

//    register int sI = index * N * nBits;
//    register int tid2 = id % (N * nBits);

//        register uint32_t val = ((uint32_t)(des[sI + tid2] + offset));



//        decaCoalesce[id] = middleBlock * (id) + (!middleBlock) * (decaCoalesce[id]);// middleBlock * (temp1 - halfBg) + (!middleBlock) * decaCoalesce[id];
    register int p = 0;
    register int decal = (32 - (p + 1) * Bgbit);
    register uint32_t val = ((uint32_t)(des_id + offset));
    register uint32_t temp1 = (val >> decal) & maskMod;

    register int xxxxx1 = (temp1 - halfBg);// + (!middleBlock) * (decaCoalesce[((id / (N * nBits)) * (N * nBits)) + id]);

    decaCoalesce[((id / (N * nBits)) * (N * nBits)) + id] = xxxxx1;
//            middleBlock * (temp1 - halfBg) +
//            (!middleBlock) * (decaCoalesce[((id / (N * nBits)) * (N * nBits)) + id]);


    p = 1;
    decal = (32 - (p + 1) * Bgbit);
    val = ((uint32_t)(des_id + offset));
    temp1 = (val >> decal) & maskMod;

    register int xxxxx2 = temp1 - halfBg;// +

    decaCoalesce[((id / (N * nBits)) * (N * nBits)) + id + (N * nBits)] = xxxxx2;
//            middleBlock * (temp1 - halfBg) +
//            (!middleBlock) * decaCoalesce[((id / (N * nBits)) * (N * nBits)) + id + (N *
//                                                                                     nBits)];//(temp1 - halfBg) + (!middleBlock) * decaCoalesce[id];//middleBlock * id;//middleBlock * id; //1;//middleBlock * (id) + (!middleBlock) * (decaCoalesce[id]);// middleBlock * (temp1 - halfBg) + (!middleBlock) * decaCoalesce[id];
                          //(!middleBlock) * decaCoalesce[((id / (N * nBits)) * (N * nBits)) + id + (N * nBits)];
//        decaCoalesce[(nBits * N) + id] = middleBlock * id;//middleBlock * id; //1;//middleBlock * (id) + (!middleBlock) * (decaCoalesce[id]);// middleBlock * (temp1 - halfBg) + (!middleBlock) * decaCoalesce[id];
//        decaCoalesce[(nBits * N) + id] = middleBlock * id;//middleBlock * id; //1;//middleBlock * (id) + (!middleBlock) * (decaCoalesce[id]);// middleBlock * (temp1 - halfBg) + (!middleBlock) * decaCoalesce[id];
//    }

//    for (int i = 0; i < nGrid; ++i) {
//        __syncthreads();
//    }
    register int bIndex = id / _2N, tIndex = id % _2N, startIndexSmall = bIndex * N;

//    middleBlock = tIndex < N;

//    d_rev_in[id] = middleBlock * (decaCoalesce[startIndexSmall + tIndex] / 2.)
//            + (!middleBlock) * (d_rev_in[id] = -decaCoalesce[startIndexSmall + tIndex - N] / 2.);


//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (tIndex >= N) * 1024 * bIndex] = id;//middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
        bIndex = id / N;
        tIndex = id % N;
        int destTod_rev_in = bIndex * _2N + tIndex + (bIndex >= nBits) * nBits * N * 2;
        d_rev_in[destTod_rev_in] = xxxxx1/2.;//id;//
        d_rev_in[destTod_rev_in + 1024] = -xxxxx1/2.;//id;//

        destTod_rev_in += nBits * 2 * 1024;
        d_rev_in[destTod_rev_in] = xxxxx2/2.;//id;
        d_rev_in[destTod_rev_in + 1024] = -xxxxx2/2.;//id;
//        d_rev_in[bIndex * _2N + tIndex + nBits * 4 * 1024] = 3;//xxxxx2/2.;
//        d_rev_in[bIndex * _2N + tIndex + 1024 + nBits * 4 * 1024] = 4;//-xxxxx2/2.;


//        d_rev_in[id + (N * nBits * 2) * 2] = id;

//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (tIndex < N) * 1024] = id;//middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (tIndex < N) * 1024] = id;//middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (N * 2 * nBits) + (tIndex >= N) * 1024] = id;
//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (N * 2 * nBits * 2) + (tIndex >= N) * 1024] = id; //middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (N * 2 * nBits * 3) + (tIndex >= N) * 1024] = id + (N * 2 * nBits); //middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (N * 2 * nBits * 3)] = id; //middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
//    d_rev_in[((id / (N * nBits)) * (N * nBits)) + id + (N * nBits)] = id; //middleBlock * (1) + (!middleBlock) * (d_rev_in[id]);
//    d_rev_in[nBits * N + id] = middleBlock * (2) + (!middleBlock) * (d_rev_in[id]);

//    if (tIndex < N) {
//        d_rev_in[id] = decaCoalesce[startIndexSmall + tIndex] / 2.;
//    } else {
//        d_rev_in[id] = -decaCoalesce[startIndexSmall + tIndex - N] / 2.;
//    }

}




__global__  void prepareForFFT_n_Bit(cufftDoubleComplex *cuDecaFFTCoalesce, cufftDoubleComplex *tmpa_gpuCoal,
                                     cufftDoubleComplex *d_in, cufftDoubleComplex *d_rev_out,
                                     cufftDoubleComplex *bki,  int keyIndex, int nBits, int nGrid) {
    register int id = blockIdx.x*blockDim.x+threadIdx.x;
    register int Ns2 = 512;
//    if (id < nBits * 4 * 512) {//nBits * kpl * Ns2
    int tempId = id;
    int bitIndex = tempId/Ns2;
    register cufftDoubleComplex v0 = d_rev_out[2 * tempId + 1 + bitIndex];//d_rev_out[2 * id + 1 + bitIndex];
//    cuDecaFFTCoalesce[tempId] = v0;

    tempId = tempId + (Ns2 * nBits);
    bitIndex = (tempId)/Ns2;
    register cufftDoubleComplex v1 = d_rev_out[2 * tempId + 1 + bitIndex];
//    cuDecaFFTCoalesce[tempId] = v1;

    tempId = tempId + (Ns2 * nBits);
    bitIndex = (tempId)/Ns2;
    register cufftDoubleComplex v2 = d_rev_out[2 * tempId + 1 + bitIndex];
//    cuDecaFFTCoalesce[tempId] = v2;

    tempId = tempId + (Ns2 * nBits);
    bitIndex = (tempId)/Ns2;
    register cufftDoubleComplex v3 = d_rev_out[2 * tempId + 1 + bitIndex];
//    cuDecaFFTCoalesce[tempId] = v3;

//    }
    register int k = 1, kpl = 4, keySI = keyIndex * (k + 1) * kpl * Ns2, aID, bID, offset;
//
//    if (id < nBits * 512) {
//        tmpa_gpuCoal[id].x = 0;
//        tmpa_gpuCoal[id].y = 0;
//        tmpa_gpuCoal[nBits * Ns2 + id].x = 0;
//        tmpa_gpuCoal[nBits * Ns2 + id].y = 0;


    int i = 0;
    offset = i * Ns2;
    aID = keySI + offset + id % Ns2;
    bID = keySI + offset + id % Ns2 + Ns2 * kpl;
    cufftDoubleComplex temp_a0 = cuCmul(v0, bki[aID]);
    cufftDoubleComplex temp_b0 = cuCmul(v0, bki[bID]);

    i = 1;
    offset = i * Ns2;
    aID = keySI + offset + id % Ns2;
    bID = keySI + offset + id % Ns2 + Ns2 * kpl;
    cufftDoubleComplex temp_a1 = cuCmul(v1, bki[aID]);
    cufftDoubleComplex temp_b1 = cuCmul(v1, bki[bID]);

    i = 2;
    offset = i * Ns2;
    aID = keySI + offset + id % Ns2;
    bID = keySI + offset + id % Ns2 + Ns2 * kpl;
    cufftDoubleComplex temp_a2 = cuCmul(v2, bki[aID]);
    cufftDoubleComplex temp_b2 = cuCmul(v2, bki[bID]);


    i = 3;
    offset = i * Ns2;
    aID = keySI + offset + id % Ns2;
    bID = keySI + offset + id % Ns2 + Ns2 * kpl;
    cufftDoubleComplex temp_a3 = cuCmul(v3, bki[aID]);
    cufftDoubleComplex temp_b3 = cuCmul(v3, bki[bID]);




//    tmpa_gpuCoal[id] = temp_a3;
//    tmpa_gpuCoal[nBits * Ns2 + id] = temp_b3;



    cufftDoubleComplex tmpa_gpuCoal0;
    tmpa_gpuCoal0.x = temp_a0.x + temp_a1.x +temp_a2.x +temp_a3.x;
    tmpa_gpuCoal0.y = temp_a0.y + temp_a1.y +temp_a2.y +temp_a3.y;
//    tmpa_gpuCoal[id] = tmpa_gpuCoal0;

    cufftDoubleComplex tmpa_gpuCoal1;
    tmpa_gpuCoal1.x = temp_b0.x + temp_b1.x +temp_b2.x +temp_b3.x;
    tmpa_gpuCoal1.y = temp_b0.y + temp_b1.y +temp_b2.y +temp_b3.y;
//    tmpa_gpuCoal[nBits * Ns2 + id] = tmpa_gpuCoal1;


//    cufftDoubleComplex temp_a = cuCmul(cuDecaFFTCoalesce[i * (Ns2 * nBits) + id], bki[aID]);
//    cufftDoubleComplex temp_b = cuCmul(cuDecaFFTCoalesce[i * (Ns2 * nBits) + id], bki[bID]);

//
//    for (int i = 0; i < nGrid * 4; ++i) {
//        __syncthreads();
//    }
//    if (id < nBits * 2 * Ns2) {//nBits * (k + 1) * Ns2
        register int N = 1024, largeSI = (id / Ns2) * (N + 1);
        register int tid = id % Ns2;
        d_in[largeSI + 2 * tid + 1] = tmpa_gpuCoal0;

    largeSI = (id / Ns2 + nBits) * (N + 1);
        d_in[largeSI + 2 * tid + 1] = tmpa_gpuCoal1;

//        d_in[largeSI + 2 * tid + 1].y = tmpa_gpuCoal[id].y;
//    }
        __syncthreads();

}

__global__ void finishUpFFT_n_Bit(int *temp2, cufftDoubleReal *d_out, int *temp3, int nBits) {

    register int id = blockIdx.x*blockDim.x+threadIdx.x;
    register int N = 1024, _2N = 2048;
    register double _2p32 = double(INT64_C(1) << 32);
    register double _1sN = double(1) / double(N);
    register int bitIndex = id / N;
    register int tIndex = id % N;
    register int startIndexLarge = bitIndex * _2N;
    temp2[id] = Torus32(int64_t(d_out[startIndexLarge + tIndex] * _1sN * _2p32)) + temp3[id];
//    if(id < nBits * 2048) {
//    }
        __syncthreads();
}


EXPORT void bootsAND_fullGPU_n_Bit(LweSample_16 *result, const LweSample_16 *ca, const LweSample_16 *cb, int nBits,
                                    cufftDoubleComplex *cudaBkFFTCoalesceExt, Torus32 *ks_a_gpu_extendedPtr,
                                    Torus32 *ks_b_gpu_extendedPtr, double *ks_cv_gpu_extendedPtr) {

    register int n = 500, BLOCKSIZE = 1024, N = 1024, _2N = 2048, Ns2 = 512, k = 1, kpl = 4, l = 2, offset = 2149580800,
            halfBg = 512, maskMod = 1023;
    cout << "bBits: " << endl;
    cout << "here" << endl;

    static const Torus32 MU = modSwitchToTorus32(1, 8);
    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);

    int *temp_res_a, *temp_res_b;
    cudaMalloc(&temp_res_a, n * nBits * sizeof(Torus32));
    temp_res_b = new Torus32[nBits];


    register int length = 500 * nBits;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);

    vecAdd<<<gridSize, BLOCKSIZE>>>(temp_res_a, ca->a, cb->a, length);
    for (int i = 0; i < nBits; ++i) {
        temp_res_b[i] = ca->b[i] + cb->b[i] + AndConst;
        temp_res_b[i] = modSwitchFromTorus32(temp_res_b[i], _2N);
    }
//    temp_res_cv += ca->current_variance[0] + cb->current_variance[0];

    //bootstrapping woks uptoFFT
    int *accum_a_b, *bara, *temp_accum_a_b, *barb, *testvectbis;//accum a and accum b together; bara; tempaccum for mux rotate
    cudaMalloc(&accum_a_b, nBits * N * (k + 1) * sizeof(int));
    cudaMalloc(&temp_accum_a_b, nBits * N * (k + 1) * sizeof(int));
    cudaMalloc(&bara, nBits * N * sizeof(int));
    cudaMalloc(&barb, nBits * sizeof(int));
    cudaMalloc(&testvectbis, nBits * N * sizeof(int));
    cudaMemcpy(barb, temp_res_b, nBits * sizeof(int), cudaMemcpyHostToDevice);

    gridSize = nBits;
    bootstrappingUptoBlindRotate_n_Bit<<<gridSize, BLOCKSIZE>>>(accum_a_b, temp_accum_a_b, bara, testvectbis, MU, nBits, temp_res_a, barb);

    int *decaCoalesce;
    cudaMalloc(&decaCoalesce, nBits * N * kpl * sizeof(int));//1024*4

    cufftDoubleComplex *cuDecaFFTCoalesce;
    cudaMalloc(&cuDecaFFTCoalesce, nBits * kpl * Ns2 * sizeof(cufftDoubleComplex));//512*4

    cufftDoubleComplex *tmpa_gpuCoal;
    cudaMalloc(&tmpa_gpuCoal, nBits * Ns2 * sizeof(cufftDoubleComplex) * (k + 1));//512*2

    //fft variables
    int iFFTBatch = nBits * kpl;
    int FFTBatch = nBits * (k + 1);
    int dParts = 4;
    //cufft helper variables
    cufftDoubleReal* d_rev_in;
    cufftDoubleComplex *d_rev_out;
    cufftDoubleComplex *d_in;
    cufftDoubleReal *d_out;
    //cufft plans
    cufftHandle p;
    cufftHandle rev_p;
    //ifft variables allocation
    cudaMalloc(&d_rev_in, iFFTBatch * _2N * sizeof(cufftDoubleReal));
    cudaMalloc(&d_rev_out, iFFTBatch * (N + 1) * sizeof(cufftDoubleComplex));
    cufftPlan1d(&rev_p, _2N, CUFFT_D2Z, iFFTBatch);// - (iFFTBatch / dParts));
    //fft variables allocation
    cudaMalloc(&d_in, FFTBatch * (N + 1) * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_out, FFTBatch * _2N * sizeof(cufftDoubleReal));
    cufftPlan1d(&p, _2N, CUFFT_Z2D, FFTBatch);
    cudaMemset(d_in, 0, FFTBatch * (N + 1) * sizeof(cufftDoubleComplex));

    int *temp2 = temp_accum_a_b;
    int *temp3 = accum_a_b;

//    int *exp = new int[nBits * N];
//    for (int i = 0; i < nBits * N; ++i) {
//        exp[i] = i << 5 + 1 << 20;
//    }
//    cudaMemcpy(temp3, exp, nBits * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(temp2, exp, nBits * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(temp3 + nBits * N, exp, nBits * N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(temp2 + nBits * N, exp, nBits * N * sizeof(int), cudaMemcpyHostToDevice);

        //create streams
    cudaDeviceProp cProfile;
    cudaGetDeviceProperties(&cProfile, 0);
    int nSM = cProfile.multiProcessorCount;
    cout << "#SM: " << nSM << endl; //20
    cudaStream_t streams[nSM];

#pragma unroll
    for (int i = 0; i < 20; ++i) {//nSM
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    ofstream myfile;
    myfile.open ("fullGPU.txt", ios::out);
    for (int j = 0; j < 1; ++j) {

        gridSize = nBits * (k + 1);//as accum is of (k + 1) * 1024;
//        prepareForiFFT_n_Bit<<<gridSize, BLOCKSIZE>>>(temp2, decaCoalesce, d_rev_in, nBits, gridSize, bara, j, temp3);

//
        for (int bIndex = 0; bIndex < nBits; ++bIndex) {
            int temp2Start = bIndex * N * (k + 1);
            int decaCoalesceStart = bIndex * 1024 * kpl;
            int d_rev_inStart = bIndex * 2048 * kpl;
            int baraStart = bIndex * N;
            int nUnitBit = 1;
            gridSize = nUnitBit * (k + 1);//1 * 2
            prepareForiFFT_n_Bit<<<gridSize, BLOCKSIZE>>>(temp2 + temp2Start, decaCoalesce + decaCoalesceStart,
                    d_rev_in + d_rev_inStart, nUnitBit, gridSize, bara + baraStart, j, temp3 + temp2Start);
        }

        cudaDeviceSynchronize();

        cufftExecD2Z(rev_p, d_rev_in, d_rev_out);
        cudaDeviceSynchronize();

        int length = nBits * 1 * Ns2;//kpl
        gridSize = (int) ceil((float) (length) / BLOCKSIZE);
        int bkKeyIndex = j * (k + 1) * kpl * Ns2;
        prepareForFFT_n_Bit<<<gridSize, BLOCKSIZE>>>(cuDecaFFTCoalesce, tmpa_gpuCoal, d_in, d_rev_out, cudaBkFFTCoalesceExt, j, nBits, gridSize);

        cufftExecZ2D(p, d_in, d_out);
        cudaDeviceSynchronize();

        length = nBits * N * 2;
        gridSize = (int) ceil((float) (length) / BLOCKSIZE); //2*nBits
        finishUpFFT_n_Bit<<<gridSize, BLOCKSIZE>>>(temp2, d_out, temp3, nBits);

        cudaDeviceSynchronize();

//        myfile << "j: " << j << " input: ";
//        length = nBits * N * (k + 1);//nBits * (N + 1) * (k + 1);//iFFTBatch * Ns2;
//        int *temp = new int[length];
//        cudaMemcpy(temp, temp3, length * sizeof(int), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < nBits * (k + 1); ++i) {
//            int sI = i * N;//(N + s1);
//            for (int j = 0; j < 10; ++j) {
//                myfile << temp[sI + j] << " ";
//            }
////        cout << endl;
//        }
//        myfile << endl;

//        length = nBits * _2N * kpl;
////        myfile << "j: " << j << " output: ";
//        cufftDoubleReal *temp = new cufftDoubleReal[length];
//        cudaMemcpy(temp, d_rev_in, length * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < nBits * kpl; ++i) {
//            int sI = i * (_2N);
//            for (int j = 0; j < 10; ++j) {
////                myfile << temp[sI + j] << " ";
//                cout << temp[sI + j] << " ";
//            }
//            cout << endl;
//        }
////        myfile << endl;
//        cout << endl;

        swap(temp2, temp3);


//        length = FFTBatch * (N + 1);//nBits * kpl * (N + 1);
//        cufftDoubleComplex *tempxx = new cufftDoubleComplex[length];
//        cudaMemcpy(tempxx, d_in, length * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < FFTBatch; ++i) {
//            int sI = i * (N + 1);//(N + 1);//(N + s1);
//            for (int j = 0; j < (N + 1); ++j) {
//                cout << "(" << tempxx[sI + j].x << "," <<  tempxx[sI + j].y << ") ";
//            }
//            cout << endl;
//        }

//make main && ./main 10 2 > test1.txt  && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt  && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt && ./main 10 2 >> test1.txt

    }
//    myfile << endl;
    myfile.close();




    length = nBits * N * (k + 1);//nBits * (N + 1) * (k + 1);//iFFTBatch * Ns2;
    int *temp = new int[length];
    cudaMemcpy(temp, temp3, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nBits * 2; ++i) {
        int sI = i * N;//(N + 1);
        for (int j = 0; j < 10; ++j) {
            cout << temp[sI + j] << " ";
//                cout << "(" << temp[sI + j].x << "," <<  temp[sI + j].y << ") ";
        }
        cout << endl;
    }





#pragma unroll
    for (int i = 0; i < 20; ++i) { //nSM
        cudaStreamDestroy(streams[i]);

    }



    delete [] temp_res_b;
    cudaFree(temp_res_a);
    cudaFree(accum_a_b);
    cudaFree(temp_accum_a_b);
    cudaFree(bara);
    cudaFree(barb);
    cudaFree(testvectbis);

}