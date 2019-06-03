#ifndef TFHE_TEST_ENVIRONMENT

#include <cstdlib>
#include <iostream>
#include <random>
#include <cassert>
#include "tfhe_core.h"
#include "numeric_functions.h"
#include "tlwe_functions.h"
#include "tgsw_functions.h"
#include "polynomials_arithmetic.h"
#include "lagrangehalfc_arithmetic.h"

#define INCLUDE_ALL
#else
#undef EXPORT
#define EXPORT
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_INIT
#undef INCLUDE_TGSW_INIT
//initialize the sample structure
//(equivalent of the C++ constructor)
EXPORT void init_TGswSample(TGswSample *obj, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int l = params->l;
    TLweSample *all_sample = new_TLweSample_array((k + 1) * l,
                                                  params->tlwe_params); // tous les samples comme un vecteur ligne
    TLweSample **bloc_sample = new TLweSample *[k + 1]; // blocs horizontaux (l lignes) de la matrice TGsw

    for (int p = 0; p < k + 1; ++p)
        bloc_sample[p] = all_sample + p * l;

    new(obj) TGswSample(all_sample, bloc_sample, k, l);
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_DESTROY
#undef INCLUDE_TGSW_DESTROY
//destroys the TGswSample structure
//(equivalent of the C++ destructor)
EXPORT void destroy_TGswSample(TGswSample *obj) {
    const int k = obj->k;
    const int l = obj->l;
    delete_TLweSample_array((k + 1) * l, obj->all_sample);
    delete[] obj->bloc_sample;
    obj->~TGswSample();
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_KEYGEN
#undef INCLUDE_TGSW_KEYGEN
// TGsw
/** generate a tgsw key (in fact, a tlwe key) */
EXPORT void tGswKeyGen(TGswKey *result) {
    tLweKeyGen(&result->tlwe_key);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_CLEAR
#undef INCLUDE_TGSW_CLEAR
// support Functions for TGsw
// Result = 0
EXPORT void tGswClear(TGswSample *result, const TGswParams *params) {
    const int kpl = params->kpl;

    for (int p = 0; p < kpl; ++p)
        tLweClear(&result->all_sample[p], params->tlwe_params);
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_ADD_H
#undef INCLUDE_TGSW_ADD_H
// Result += H
EXPORT void tGswAddH(TGswSample *result, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int l = params->l;
    const Torus32 *h = params->h;

    // compute result += H
    for (int bloc = 0; bloc <= k; ++bloc)
        for (int i = 0; i < l; i++)
            result->bloc_sample[bloc][i].a[bloc].coefsT[0] += h[i];
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_ADD_MU_H
#undef INCLUDE_TGSW_ADD_MU_H
// Result += mu*H
EXPORT void tGswAddMuH(TGswSample *result, const IntPolynomial *message, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int N = params->tlwe_params->N;
    const int l = params->l;
    const Torus32 *h = params->h;
    const int *mu = message->coefs;

    // compute result += H
    for (int bloc = 0; bloc <= k; ++bloc)
        for (int i = 0; i < l; i++) {
            Torus32 *target =
                    result->bloc_sample[bloc][i].a[bloc].coefsT;
            const Torus32 hi = h[i];
            for (int j = 0; j < N; j++) {
                target[j] += mu[j] * hi;
            }
        }
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_ADD_MU_INT_H
#undef INCLUDE_TGSW_ADD_MU_INT_H
// Result += mu*H, mu integer
EXPORT void tGswAddMuIntH(TGswSample *result, const int message, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int l = params->l;
    const Torus32 *h = params->h;

    // compute result += H
    for (int bloc = 0; bloc <= k; ++bloc)
        for (int i = 0; i < l; i++)
            result->bloc_sample[bloc][i].a[bloc].coefsT[0] += message * h[i];
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_ENCRYPT_ZERO
#undef INCLUDE_TGSW_ENCRYPT_ZERO
// Result = tGsw(0)
EXPORT void tGswEncryptZero(TGswSample *result, double alpha, const TGswKey *key) {
    const TLweKey *rlkey = &key->tlwe_key;
    const int kpl = key->params->kpl;

    for (int p = 0; p < kpl; ++p) {
        tLweSymEncryptZero(&result->all_sample[p], alpha, rlkey);
    }
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_MUL_BY_XAI_MINUS_ONE
#undef INCLUDE_TGSW_MUL_BY_XAI_MINUS_ONE
//mult externe de X^{a_i} par bki
EXPORT void tGswMulByXaiMinusOne(TGswSample *result, int ai, const TGswSample *bk, const TGswParams *params) {
    const TLweParams *par = params->tlwe_params;
    const int kpl = params->kpl;
    for (int i = 0; i < kpl; i++)
        tLweMulByXaiMinusOne(&result->all_sample[i], ai, &bk->all_sample[i], par);
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_EXTERN_MUL_TO_TLWE
#undef INCLUDE_TGSW_EXTERN_MUL_TO_TLWE
//Update l'accumulateur ligne 5 de l'algo toujours
//void tGswTLweDecompH(IntPolynomial* result, const TLweSample* sample,const TGswParams* params);
//accum *= sample
EXPORT void tGswExternMulToTLwe(TLweSample *accum, const TGswSample *sample, const TGswParams *params) {
    const TLweParams *par = params->tlwe_params;
    const int N = par->N;
    const int kpl = params->kpl;
    //TODO: improve this new/delete
    IntPolynomial *dec = new_IntPolynomial_array(kpl, N);

    tGswTLweDecompH(dec, accum, params);
    tLweClear(accum, par);
    for (int i = 0; i < kpl; i++) {
        tLweAddMulRTo(accum, &dec[i], &sample->all_sample[i], par);
    }

    delete_IntPolynomial_array(kpl, dec);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_SYM_ENCRYPT
#undef INCLUDE_TGSW_SYM_ENCRYPT
/**
 * encrypts a poly message
 */
EXPORT void tGswSymEncrypt(TGswSample *result, const IntPolynomial *message, double alpha, const TGswKey *key) {
    tGswEncryptZero(result, alpha, key);
    tGswAddMuH(result, message, key->params);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_SYM_ENCRYPT_INT
#undef INCLUDE_TGSW_SYM_ENCRYPT_INT
/**
 * encrypts a constant message
 */
EXPORT void tGswSymEncryptInt(TGswSample *result, const int message, double alpha, const TGswKey *key) {
    tGswEncryptZero(result, alpha, key);
    tGswAddMuIntH(result, message, key->params);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_ENCRYPT_B
#undef INCLUDE_TGSW_ENCRYPT_B
/**
 * encrypts a message = 0 ou 1
 */
EXPORT void tGswEncryptB(TGswSample *result, const int message, double alpha, const TGswKey *key) {
    tGswEncryptZero(result, alpha, key);
    if (message == 1)
        tGswAddH(result, key->params);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_DECRYPT
#undef INCLUDE_TGSW_DECRYPT
// à revoir
EXPORT void tGswSymDecrypt(IntPolynomial *result, const TGswSample *sample, const TGswKey *key, const int Msize) {
    const TGswParams *params = key->params;
    const TLweParams *rlwe_params = params->tlwe_params;
    const int N = rlwe_params->N;
    const int l = params->l;
    const int k = rlwe_params->k;
    TorusPolynomial *testvec = new_TorusPolynomial(N);
    TorusPolynomial *tmp = new_TorusPolynomial(N);
    IntPolynomial *decomp = new_IntPolynomial_array(l, N);

    const Torus32 indic = modSwitchToTorus32(1, Msize);
    torusPolynomialClear(testvec);
    testvec->coefsT[0] = indic;
    tGswTorus32PolynomialDecompH(decomp, testvec, params);

    torusPolynomialClear(testvec);
    for (int i = 0; i < l; i++) {
        for (int j = 1; j < N; j++) assert(decomp[i].coefs[j] == 0);
        tLwePhase(tmp, &sample->bloc_sample[k][i], &key->tlwe_key);
        torusPolynomialAddMulR(testvec, decomp + i, tmp);
    }
    for (int i = 0; i < N; i++)
        result->coefs[i] = modSwitchFromTorus32(testvec->coefsT[i], Msize);

    delete_TorusPolynomial(testvec);
    delete_TorusPolynomial(tmp);
    delete_IntPolynomial_array(l, decomp);
}
#endif

/*
// à revoir
EXPORT int tGswSymDecryptInt(const TGswSample* sample, const TGswKey* key){
    TorusPolynomial* phase = new_TorusPolynomial(key->params->tlwe_params->N);

    tGswPhase(phase, sample, key);
    int result = modSwitchFromTorus32(phase->coefsT[0], Msize);

    delete_TorusPolynomial(phase);
    return result;
}
*/
//do we really decrypt Gsw samples?
// EXPORT void tGswMulByXaiMinusOne(Gsw* result, int ai, const Gsw* bk);
// EXPORT void tLweExternMulRLweTo(RLwe* accum, Gsw* a); //  accum = a \odot accum


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_TLWE_DECOMP_H
#undef INCLUDE_TGSW_TLWE_DECOMP_H
//fonction de decomposition
EXPORT void tGswTLweDecompH(IntPolynomial *result, const TLweSample *sample, const TGswParams *params) {
    const int k = params->tlwe_params->k;
    const int l = params->l;

    for (int i = 0; i <= k; ++i) // b=a[k]
        tGswTorus32PolynomialDecompH(result + (i * l), &sample->a[i], params);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_TORUS32POLYNOMIAL_DECOMP_H_OLD
#undef INCLUDE_TGSW_TORUS32POLYNOMIAL_DECOMP_H_OLD
EXPORT void
Torus32PolynomialDecompH_old(IntPolynomial *result, const TorusPolynomial *sample, const TGswParams *params) {
    const int N = params->tlwe_params->N;
    const int l = params->l;
    const int Bgbit = params->Bgbit;
    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;

    for (int j = 0; j < N; ++j) {
        uint32_t temp0 = sample->coefsT[j] + offset;
        for (int p = 0; p < l; ++p) {
            uint32_t temp1 = (temp0 >> (32 - (p + 1) * Bgbit)) & maskMod; // doute
            result[p].coefs[j] = temp1 - halfBg;
        }
    }
}
#endif

#if defined INCLUDE_ALL || defined INCLUDE_TGSW_TORUS32POLYNOMIAL_DECOMP_H
#undef INCLUDE_TGSW_TORUS32POLYNOMIAL_DECOMP_H

#include <iostream>
using namespace std;
EXPORT void
tGswTorus32PolynomialDecompH(IntPolynomial *result, const TorusPolynomial *sample, const TGswParams *params) {
    const int N = params->tlwe_params->N;
    const int l = params->l;
    const int Bgbit = params->Bgbit;
    uint32_t *buf = (uint32_t *) sample->coefsT;

//#define __AVX2__ //(to test)
#ifndef __AVX2__
    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
#else
    const uint32_t* maskMod_addr = &params->maskMod;
    const int32_t* halfBg_addr = &params->halfBg;
    const uint32_t* offset_addr = &params->offset;
    //const uint32_t offset = params->offset;
    //const uint32_t maskMod = params->maskMod;
    //const int32_t halfBg = params->halfBg;
#endif

    //First, add offset to everyone
#ifndef __AVX2__
    for (int j = 0; j < N; ++j) buf[j] += offset;
#else
    {
    const uint32_t* sit = buf;
    const uint32_t* send = buf+N;
    __asm__ __volatile__ (
        "vpbroadcastd (%2),%%ymm0\n"
        "0:\n"
        "vmovdqu (%0),%%ymm3\n"
        "vpaddd %%ymm0,%%ymm3,%%ymm3\n" // add offset
        "vmovdqu %%ymm3,(%0)\n"
        "addq $32,%0\n"
        "cmpq %1,%0\n"
        "jb 0b\n"
        : "=r"(sit),"=r"(send),"=r"(offset_addr)
        :  "0"(sit), "1"(send), "2"(offset_addr)
        : "%ymm0","%ymm3","memory"
        );
    }
#endif

    //then, do the decomposition (in parallel)
    for (int p = 0; p < l; ++p) {
        const int decal = (32 - (p + 1) * Bgbit);
#ifndef __AVX2__
        int32_t *res_p = result[p].coefs;
        for (int j = 0; j < N; ++j) {
            uint32_t temp1 = (buf[j] >> decal) & maskMod;
            res_p[j] = temp1 - halfBg;
        }
#else
        int32_t* dst = result[p].coefs;
        const uint32_t* sit = buf;
        const uint32_t* send = buf+N;
        const int32_t* decal_addr = &decal;
        __asm__ __volatile__ (
            "vpbroadcastd (%4),%%ymm0\n"
            "vpbroadcastd (%5),%%ymm1\n"
            "vmovd (%3),%%xmm2\n"
            "1:\n"
            "vmovdqu (%1),%%ymm3\n"
            "VPSRLD %%xmm2,%%ymm3,%%ymm3\n" // shift by decal
            "VPAND %%ymm1,%%ymm3,%%ymm3\n"  // and maskMod
            "VPSUBD %%ymm0,%%ymm3,%%ymm3\n" // sub halfBg
            "vmovdqu %%ymm3,(%0)\n"
            "addq $32,%0\n"
            "addq $32,%1\n"
            "cmpq %2,%1\n"
            "jb 1b\n"
            : "=r"(dst),"=r"(sit),"=r"(send),"=r"(decal_addr),"=r"(halfBg_addr),"=r"(maskMod_addr)
            :  "0"(dst), "1"(sit), "2"(send), "3"(decal_addr), "4"(halfBg_addr) ,"5"(maskMod_addr)
            : "%ymm0","%ymm1","%ymm2","%ymm3","memory"
            );
        /* // verify that the assembly block was ok
        int32_t* res_p = result[p].coefs;
        for (int j = 0; j < N; ++j)
        {
            uint32_t temp1 = (buf[j] >> decal) & maskMod;
            if (res_p[j] != int32_t(temp1 - halfBg)) {
            fprintf(stderr, "j=%d,buf[j]=%u,decal=%u,mask=%u,halfbg=%d,res_p[j]=%d\n",j,buf[j],decal,maskMod,halfBg,res_p[j]);
            abort();
            }
        }*/

#endif
    }

    //finally, remove offset to everyone
#ifndef __AVX2__
    for (int j = 0; j < N; ++j) buf[j] -= offset;
#else
    {
    const uint32_t* sit = buf;
    const uint32_t* send = buf+N;
    __asm__ __volatile__ (
        "vpbroadcastd (%2),%%ymm0\n"
        "2:\n"
        "vmovdqu (%0),%%ymm3\n"
        "vpsubd %%ymm0,%%ymm3,%%ymm3\n" // add offset
        "vmovdqu %%ymm3,(%0)\n"
        "addq $32,%0\n"
        "cmpq %1,%0\n"
        "jb 2b\n"
        "vzeroall\n"
        : "=r"(sit),"=r"(send),"=r"(offset_addr)
        :  "0"(sit), "1"(send), "2"(offset_addr)
        : "%ymm0","%ymm3","memory"
        );
    }
#endif
}


__global__ void addToVector(uint32_t *destination, int val, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        destination[id] = destination[id] + val;
    }
}

__global__ void decomposeParallel(int32_t *destination, int decal, uint32_t maskMod, int halfBg, uint32_t *source, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        uint32_t temp1 = (source[id] >> decal) & maskMod;
        destination[id] = temp1 - halfBg;
    }
}

EXPORT void
tGswTorus32PolynomialDecompH_16(IntPolynomial *result, const TorusPolynomial *sample, int bitSize, const TGswParams *params) {
    const int N = params->tlwe_params->N;
    const int l = params->l;
    const int Bgbit = params->Bgbit;
    uint32_t *buf = (uint32_t *) sample->coefsT;


    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
    int bigN = N * bitSize;

    int BLOCKSIZE = 1024;
    int gridSize = (int)ceil((float)(N*bitSize)/BLOCKSIZE);

//    cout << "sample->N: " << sample->N << endl; //16*1024
//    cout << "result->N: " << result->N << endl; //16*1024
//    cout << "N: " << N << endl; //1024
//    cout << "l: " << l << endl; //2

//    First, add offset to everyone
    addToVector<<<gridSize, BLOCKSIZE>>>(buf, offset, bigN);
    //gpu testing code
//    uint32_t *temp = new uint32_t[bigN];
//    cudaMemcpy(temp, buf, bigN* sizeof(int), cudaMemcpyDeviceToHost);

//    for (int bi = 0; bi < bitSize; ++bi) {
//    int bi = 0;
//        int sI = bi * N;
//        cout << "new: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << temp[i + sI] << " ";
//        }
//        cout << endl;
//    }
    //then, do the decomposition (in parallel)
    for (int p = 0; p < l; ++p) {
        const int decal = (32 - (p + 1) * Bgbit);
        int32_t *res_p = result[p].coefs;

        decomposeParallel<<<gridSize, BLOCKSIZE>>>(res_p, decal, maskMod, halfBg, buf, bigN);
//        for (int j = 0; j < N; ++j) {
//            uint32_t temp1 = (buf[j] >> decal) & maskMod;
//            res_p[j] = temp1 - halfBg;
//        }
    }


//
    //finally, remove offset to everyone
    addToVector<<<gridSize, BLOCKSIZE>>>(buf, -offset, bigN);
//    for (int j = 0; j < N; ++j) buf[startIndex + j] -= offset;
}

__global__ void decomposeParallel_Coalesce(int32_t *destination, int Bgbit, uint32_t maskMod, int halfBg,
                                           int N, int bitSize, int kpl, int l, uint32_t *source1, uint32_t *source2,
                                           int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        int p = (id/(N * bitSize)) % l;
        int index = (id/(l * N * bitSize)) % l;
        int decal = (32 - (p + 1) * Bgbit);
        uint32_t *targetSource = index == 0 ? source1 : source2;
//        uint32_t temp1 = (source[index][id % (N * bitSize)] >> decal) & maskMod;
        uint32_t temp1 = (targetSource[id % (N * bitSize)] >> decal) & maskMod;
        destination[id] = temp1 - halfBg;//(uint32_t)(targetSource[id % (N * bitSize)]- 2149580800);//temp1;//
    }
}

EXPORT void
tGswTorus32PolynomialDecompH_16_Coalesce(int *result, TLweSample *sample, int bitSize, const TGswParams *params) {
    const int N = params->tlwe_params->N;//1024
    const int l = params->l;//2
    int k = params->tlwe_params->k;//1
    const int Bgbit = params->Bgbit;
    int kpl = params->kpl;
    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
    int bigN = N * bitSize;
    int BLOCKSIZE = 1024;
    int gridSize = (int) ceil((float) (bigN) / BLOCKSIZE);
//    cout << "sample->N: " << sample->N << endl; //16*1024
//    cout << "result->N: " << result->N << endl; //16*1024
//        cout << "params->offset: " << (uint32_t) params->offset << endl;
//        cout << "params->halfBg: " << params->halfBg << endl;
//        cout << "params->maskMod: " << params->maskMod << endl;
//        cout << "params->Bgbit: " << params->Bgbit << endl;

    //add offset
    for (int i = 0; i < l; ++i) {
        addToVector<<<bigN/BLOCKSIZE, BLOCKSIZE>>>((uint32_t *)(sample->a + i)->coefsT, offset, bigN);
    }
    gridSize = (int) ceil((float) (kpl * bigN) / BLOCKSIZE);
    //then, do the decomposition (in parallel)
    decomposeParallel_Coalesce<<<gridSize, BLOCKSIZE>>>
                                           (result, Bgbit, maskMod, halfBg, N, bitSize, kpl, l,
                                                   (uint32_t *)(sample->a + 0)->coefsT,
                                                   (uint32_t *)(sample->a + 1)->coefsT, bigN * kpl);
//    for (int p = 0; p < l; ++p) {
//        const int decal = (32 - (p + 1) * Bgbit);
//        int32_t *res_p = result[p].coefs;
//
//        decomposeParallel<<<gridSize, BLOCKSIZE>>>(res_p, decal, maskMod, halfBg, buf, bigN);
////        for (int j = 0; j < N; ++j) {
////            uint32_t temp1 = (buf[j] >> decal) & maskMod;
////            res_p[j] = temp1 - halfBg;
////        }
//    }
//    decomposeParallel<<<gridSize, BLOCKSIZE>>>(res_p, decal, maskMod, halfBg, buf, bigN);

    //finally, remove offset to everyone
    gridSize = (int) ceil((float) (bigN) / BLOCKSIZE);
    for (int i = 0; i < l; ++i) {
        addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sample->a + i)->coefsT, -offset, bigN);
    }
}

__global__ void decomposeParallel_Coalesce_2(int32_t *destination, int Bgbit, uint32_t maskMod, int halfBg,
                                           int N, int nOutputs, int bitSize, int kpl, int l, uint32_t *source1,
                                             uint32_t *source2, int length) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < length) {
        int index = (id / (N * bitSize)) / kpl;
        int p = (id / (l * N * bitSize)) % l;
//        int index = (id / (l * N * bitSize * nOutputs)) % l;
        int decal = (32 - (p + 1) * Bgbit);
        uint32_t *targetSource = index == 0 ? source1 : source2;
//        uint32_t temp1 = (source[index][id % (N * bitSize)] >> decal) & maskMod;
        uint32_t temp1 = (targetSource[id % (N * bitSize * nOutputs)] >> decal) & maskMod;
        destination[id] = temp1 - halfBg;
    }
    __syncthreads();
}


EXPORT void tGswTorus32PolynomialDecompH_16_2_Coalesce(int *result, const TLweSample *sample, int nOutputs,
                                           int bitSize, const TGswParams *params) {

    const int N = params->tlwe_params->N;//1024
    const int l = params->l;//2
    const int Bgbit = params->Bgbit;//10
//    uint32_t *buf = (uint32_t *) sample->coefsT;
    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
    const int k = params->tlwe_params->k;
    const int kpl = params->kpl;

    int length = nOutputs * bitSize * N;

    int BLOCKSIZE = 1024;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    //add offset
    for (int i = 0; i <= k; ++i) {
        addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sample->a + i)->coefsT, offset, length);
    }
    gridSize = (int) ceil((float) (kpl * length) / BLOCKSIZE);
    decomposeParallel_Coalesce_2<<<gridSize, BLOCKSIZE>>>
                                           (result, Bgbit, maskMod, halfBg, N, nOutputs, bitSize, kpl, l,
                                                   (uint32_t *)(sample->a + 0)->coefsT,
                                                   (uint32_t *)(sample->a + 1)->coefsT, kpl * length);
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    //remove offset
    for (int i = 0; i <= k; ++i) {
        addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sample->a + i)->coefsT, -offset, length);
    }
}

EXPORT void tGswTorus32PolynomialDecompH_16_2_CoalesceV2(int *result, const TLweSample *sampleV2, int nOutputs,
                                                       int bitSize, const TGswParams *params) {

    const int N = params->tlwe_params->N;//1024
    const int l = params->l;//2
    const int Bgbit = params->Bgbit;//10
//    uint32_t *buf = (uint32_t *) sample->coefsT;
    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
    const int k = params->tlwe_params->k;
    const int kpl = params->kpl;

    int length = nOutputs * bitSize * N * (k + 1);

    int BLOCKSIZE = 1024;
    //add offset
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sampleV2->a)->coefsT, offset, length);//both of a are concatenated

    gridSize = (int) ceil((float) (kpl * length) / BLOCKSIZE);
    decomposeParallel_Coalesce_2<<<gridSize, BLOCKSIZE>>>
                                             (result, Bgbit, maskMod, halfBg, N, nOutputs, bitSize, kpl, l,
                                                     (uint32_t *)(sampleV2->a + 0)->coefsT,
                                                     (uint32_t *)(sampleV2->a + 1)->coefsT, kpl * length);
    //remove offset
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sampleV2->a)->coefsT, -offset, length);

}








EXPORT void tGswTorus32PolynomialDecompH_16_2_Coalesce_vector(int *result, const TLweSample *sample, int vLength,
                                                              int nOutputs, int bitSize, const TGswParams *params) {

    const int N = params->tlwe_params->N;//1024
    const int l = params->l;//2
    const int Bgbit = params->Bgbit;//10

    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
    const int k = params->tlwe_params->k;
    const int kpl = params->kpl;

    int length = vLength * nOutputs * bitSize * N;

    int BLOCKSIZE = 1024;
    int gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    //add offset
    for (int i = 0; i <= k; ++i) {
        addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sample->a + i)->coefsT, offset, length);
    }
    gridSize = (int) ceil((float) (kpl * length) / BLOCKSIZE);
    int totalBitSize = bitSize * vLength;
    decomposeParallel_Coalesce_2<<<gridSize, BLOCKSIZE>>>
                                             (result, Bgbit, maskMod, halfBg, N, nOutputs, totalBitSize, kpl, l,
                                                     (uint32_t *)(sample->a + 0)->coefsT,
                                                     (uint32_t *)(sample->a + 1)->coefsT, kpl * length);
    gridSize = (int) ceil((float) (length) / BLOCKSIZE);
    //remove offset
    for (int i = 0; i <= k; ++i) {
        addToVector<<<gridSize, BLOCKSIZE>>>((uint32_t *)(sample->a + i)->coefsT, -offset, length);
    }
}









EXPORT void
tGswTorus32PolynomialDecompH_16_2(IntPolynomial *result, const TorusPolynomial *sample, int nOutputs,
                                  int bitSize, const TGswParams *params) {
    const int N = params->tlwe_params->N;//1024
    const int l = params->l;//2
    const int Bgbit = params->Bgbit;//10
    uint32_t *buf = (uint32_t *) sample->coefsT;


    const uint32_t maskMod = params->maskMod;
    const int32_t halfBg = params->halfBg;
    const uint32_t offset = params->offset;
    int length = nOutputs * bitSize * N;

    int BLOCKSIZE = 1024;
    int gridSize = (int)ceil((float)(length)/BLOCKSIZE);//21
//    cout << "sample->N: " << sample->N << endl; //16*1024
//    cout << "result->N: " << result->N << endl; //16*1024
//    cout << "N: " << N << endl; //1024
//    cout << "l: " << l << endl; //2
//    cout << "Bgbit: " << Bgbit << endl;
//    cout << "gridSize: " << gridSize << endl;


//    First, add offset to everyone
    addToVector<<<gridSize, BLOCKSIZE>>>(buf, offset, length);
//    //gpu testing code
////    uint32_t *temp = new uint32_t[bigN];
////    cudaMemcpy(temp, buf, bigN* sizeof(int), cudaMemcpyDeviceToHost);
//
////    for (int bi = 0; bi < bitSize; ++bi) {
////    int bi = 0;
////        int sI = bi * N;
////        cout << "new: ";
////        for (int i = 0; i < 10; ++i) {
////            cout << temp[i + sI] << " ";
////        }
////        cout << endl;
////    }
//    //then, do the decomposition (in parallel)
    for (int p = 0; p < l; ++p) {
        const int decal = (32 - (p + 1) * Bgbit);
        int32_t *res_p = result[p].coefs;

        decomposeParallel<<<gridSize, BLOCKSIZE>>>(res_p, decal, maskMod, halfBg, buf, length);
//        for (int j = 0; j < N; ++j) {
//            uint32_t temp1 = (buf[j] >> decal) & maskMod;
//            res_p[j] = temp1 - halfBg;
//        }
    }
//
//
////
//    //finally, remove offset to everyone
    addToVector<<<gridSize, BLOCKSIZE>>>(buf, -offset, length);
////    for (int j = 0; j < N; ++j) buf[startIndex + j] -= offset;
}

#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_EXTERN_PRODUCT
#undef INCLUDE_TGSW_EXTERN_PRODUCT
//result = a*b
EXPORT void tGswExternProduct(TLweSample *result, const TGswSample *a, const TLweSample *b, const TGswParams *params) {
    const TLweParams *parlwe = params->tlwe_params;
    const int N = parlwe->N;
    const int kpl = params->kpl;
    IntPolynomial *dec = new_IntPolynomial_array(kpl, N);

    tGswTLweDecompH(dec, b, params);

    tLweClear(result, parlwe);
    for (int i = 0; i < kpl; i++)
        tLweAddMulRTo(result, &dec[i], &a->all_sample[i], parlwe);

    result->current_variance += b->current_variance; //todo + the error term?

    delete_IntPolynomial_array(kpl, dec);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TGSW_NOISELESS_TRIVIAL
#undef INCLUDE_TGSW_NOISELESS_TRIVIAL
/**
 * result = (0,mu)
 */
EXPORT void tGswNoiselessTrivial(TGswSample *result, const IntPolynomial *mu, const TGswParams *params) {
    tGswClear(result, params);
    tGswAddMuH(result, mu, params);
}
#endif






//Autogenerated templates for allocation/construction/initialization...
//allocate memory space for a TGswSample
USE_DEFAULT_CONSTRUCTOR_DESTRUCTOR_IMPLEMENTATIONS1(TGswSample, TGswParams);


#undef INCLUDE_ALL
