//
// Created by morshed on 5/15/2018.
//

#include <stdio.h>
#include <iostream>
//#include <time.h>

#include <omp.h>
#include <assert.h>
#include<stdint.h>
#include <inttypes.h>
#include <time.h>
#include <ctime>
#include "Cipher.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


#define DEBUG
using namespace std;

//customizedLweSample
struct cLweSample {
    Torus32* a; //-- the n coefs of the mask
    Torus32 b[1000];  //
    double current_variance; //-- average noise of the sample
};


int decryptCheck(Cipher cipher, TFheGateBootstrappingSecretKeySet *key) {
    int int_answer = 0;
    for (int i = 0; i < cipher.numberOfBits; i++) {
        int ai = bootsSymDecrypt(&cipher.data[i], key);
#ifdef DEBUG
        printf("%d ",ai);
#endif
        int_answer |= (ai << i);
    }
#ifdef DEBUG
    printf("\n");
#endif
    if (int_answer > pow(2, cipher.numberOfBits - 1)) {
        int_answer = -1 * (pow(2, cipher.numberOfBits) - int_answer);

    }
//    cout << "decrypt check: number of bits:" << cipher.numberOfBits << endl;
    return int_answer;
}

void analyze(int a, int b, int carryin, int s, int carryout, int ts, int tcarryout) {
    std::cout << a << " " << b << " " << carryin << " | " << s << " " << carryout << endl;
    assert(s == ts);
    assert(carryout == tcarryout);
}

void lweSymEncryptFixedA(LweSample* result, Torus32 message, double alpha, const LweKey* key){
    const int n = key->params->n;

    result->b = gaussian32(message, alpha);
    for (int i = 0; i < n; ++i)
    {
//        result->a[i] = uniformTorus32_distrib(generator);//uniform_int_distribution<Torus32> uniformTorus32_distrib(INT32_MIN, INT32_MAX);
        result->b += result->a[i]*key->key[i];
    }

    result->current_variance = alpha*alpha;
}

void bootsSymEncryptFixedA(LweSample *result, int message, const TFheGateBootstrappingSecretKeySet *key) {
    Torus32 _1s8 = modSwitchToTorus32(1, 8);
    Torus32 mu = message ? _1s8 : -_1s8;
    double alpha = key->params->in_out_params->alpha_min; //TODO: specify noise
    lweSymEncryptFixedA(result, mu, alpha, key->lwe_key);
}



int main() {

    //reads the cloud key from file
    FILE *cloud_key = fopen("cloud.key", "rb");
    TFheGateBootstrappingCloudKeySet *bk = new_tfheGateBootstrappingCloudKeySet_fromFile(cloud_key);
    fclose(cloud_key);
    FILE *secret_key = fopen("secret.key", "rb");
    TFheGateBootstrappingSecretKeySet *secretKeySet = new_tfheGateBootstrappingSecretKeySet_fromFile(secret_key);
    fclose(secret_key);

    //if necessary, the params are inside the key
    const TFheGateBootstrappingParameterSet *params = bk->params;
    int bitsize=16;
//    LweSample *a = new_gate_bootstrapping_ciphertext_array(bitsize, params);
//    LweSample *b = new_gate_bootstrapping_ciphertext_array(bitsize, params);
//    LweSample *c = new_gate_bootstrapping_ciphertext_array(bitsize, params);
//    LweSample *result = new_gate_bootstrapping_ciphertext_array(2, params);
//
//    int tcarryout [8] = {0,0,0,1,0,1,1,1};
//    int ts [8] = {0,1,1,0,1,0,0,1};
//
//    bootsCONSTANT(&a[0], 0, Cipher::bk);
//    bootsCONSTANT(&b[0], 0, Cipher::bk);
//    bootsCONSTANT(&c[0], 0, Cipher::bk);
//    int counter = 0;
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 2; ++j) {
//            for (int k = 0; k < 2; ++k) {
////                cout << i << " " << j << " " << k << endl;
//                bootsCONSTANT(&a[0], i, Cipher::bk);
//                bootsCONSTANT(&b[0], j, Cipher::bk);
//                bootsCONSTANT(&c[0], k, Cipher::bk);
//                Cipher::addBits(result, a, b, c);
//                int s = Cipher::decryptBitCheck(&result[0], secretKeySet);
//                int carryout = Cipher::decryptBitCheck(&result[1], secretKeySet);
//                analyze(i, j, k, s, carryout, ts[counter], tcarryout[counter]);
//                counter++;
//            }
//        }
//    }

//    LweSample *res= new_gate_bootstrapping_ciphertext_array(bitsize, params);

//    Cipher::bootsANDXOR(res,a,b,c,Cipher::bk);
//    cout<<Cipher::decryptBitCheck(&res[0],secretKeySet)<<endl;
//
    //read the 2x16 ciphertexts
//    int bitSize = 16;

//    cout << "number: " << bk->params->in_out_params->n << endl;
    LweSample *ciphertext1 = new_gate_bootstrapping_ciphertext_array(bitsize, params);
    LweSample *ciphertext2 = new_gate_bootstrapping_ciphertext_array(bitsize, params);
    LweSample *ciphertext3 = new_gate_bootstrapping_ciphertext_array(bitsize, params);
    LweSample *ciphertext4 = new_gate_bootstrapping_ciphertext_array(bitsize, params);
    //reads the 2x16 ciphertexts from the cloud file
    FILE *cloud_data = fopen("cloud.data", "rb");
    for (int i = 0; i < bitsize; i++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertext1[i], params);
    for (int i = 0; i < bitsize; i++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertext2[i], params);
    fclose(cloud_data);

    cout << "starting  here" << endl;

    for (int i = 0; i < 1; ++i) {
        bootsAND(&ciphertext3[i], &ciphertext1[i], &ciphertext2[i], Cipher::bk);
        cout << Cipher::decryptBitCheck(&ciphertext3[i], secretKeySet);
    }

    int i = 0;
    cout << "a[" << i <<"]: " << Cipher::decryptBitCheck(&ciphertext1[0], secretKeySet) << " b[" << i <<"]: " << Cipher::decryptBitCheck(&ciphertext2[00], secretKeySet);
    cout << " c[" << i <<"]: " << Cipher::decryptBitCheck(&ciphertext3[0], secretKeySet) << endl;
    cout << endl;
    cout << "ending" << endl;

    cout << "Cuda mem allocation" << endl;
    TFheGateBootstrappingCloudKeySet *d_bk;
//    cudaMallocManaged(&d_bk, sizeof(TFheGateBootstrappingCloudKeySet));
    cudaMallocManaged(&d_bk, sizeof(Cipher::bk));
    cudaError_t error = cudaMemcpy(d_bk, Cipher::bk, sizeof(Cipher::bk), cudaMemcpyHostToDevice);
    cout << error << endl;
    cout << "sizeof(TFheGateBootstrappingCloudKeySet): " << sizeof(TFheGateBootstrappingCloudKeySet) << endl;
    cout << "sizeof(Cipher::bk): " << sizeof(Cipher::bk) << endl;
    cout << "mem allocation ends" << endl;
    cout << "n in host: " << Cipher::bk->bk->bk->all_sample->a->N << endl;
    cout << "n in device: " << d_bk->bk->bk->all_sample->a->N<< endl;


//    cout << "size of lwe sample: " << sizeof(ciphertext1) << endl;
//    cout << "starting" << endl;
//
//    clock_t begin = clock();
//
//    bootsAND(&ciphertext3[0], &ciphertext1[0], &ciphertext2[0], Cipher::bk);
//
//    clock_t end = clock();
//    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    cout << "ending" << endl;
//    cout << "Output: " << Cipher::decryptBitCheck(&ciphertext3[0], secretKeySet) << " " << "time: " << elapsed_secs << endl;
//    cout << "I am clear" << endl;
//    srand(time(0));
//    int NELEM = 1000;//secretKeySet->lwe_key->params->n;
//    cout << "NELEM: " << NELEM << endl;
//    int testBinary[NELEM];
//    LweSample *lweSampleArray[NELEM];

//    testBinary[0] = 0;
    //get one LWE sample
//    LweSample *test1 = new_gate_bootstrapping_ciphertext_array(1, params);
//    bootsSymEncrypt(test1, 0, secretKeySet);

//    for (int j = 1; j < NELEM; ++j) {
//        testBinary[j] = rand() % 2;
//    }


//    for (int i = 0; i < NELEM; ++i) {
//        lweSampleArray[i] = new_gate_bootstrapping_ciphertext_array(1, params);
//        lweSampleArray[i]->a = test1->a;
//        if (i == 0) {
//            lweSampleArray[i]->b = test1->b;
//        } else {
//            bootsSymEncryptFixedA(lweSampleArray[i], testBinary[i], secretKeySet);
//        }
//    }
//    for (int i = 0; i < NELEM; ++i) {
//        printf("%d\t", testBinary[i]);
//        printf("%" PRId32 "\n", lweSampleArray[i]->b);
//    }


//    int flag = 1;
//    for (int i = 0; i < NELEM; ++i) {
//        if (testBinary[i] != Cipher::decryptBitCheck(lweSampleArray[i], secretKeySet)) {
//            flag = 0;
//            break;
//        }
//    }
//    assert(flag == 1);









//    printf("%" PRId32 "\n", test1->b);


//    Cipher a, b, c, d;
//    a.setNumberOfBitsAndData(bitsize, ciphertext1);
//    b.setNumberOfBitsAndData(bitsize, ciphertext2);
//
//    clock_t begin = clock();
//    c = Cipher::cipherAND(a, b, secretKeySet);
//    clock_t end = clock();
//    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    cout << "output: " << decryptCheck(a, secretKeySet) << endl;
//    cout << "output: " << decryptCheck(b, secretKeySet)  << endl;
//    cout << "output: " << decryptCheck(c, secretKeySet) << " Time: " << elapsed_secs << endl;
//    c = a + b;
//    cout << "time: " << omp_get_wtime() - s_time << endl;
//
//    cout << "a: " << decryptCheck(a, secretKeySet) << endl;
//    cout << "b: " << decryptCheck(b, secretKeySet) << endl;
//    cout << "c: " << decryptCheck(c, secretKeySet) << endl;
//
//    //do some operations on the ciphertexts: here, we will compute the
//    //minimum of the two
//    LweSample *result = new_gate_bootstrapping_ciphertext_array(32, params);
//
//    //export the 32 ciphertexts to a file (for the cloud)
//    FILE *answer_data = fopen("answer.data", "wb");
//    for (int i = 0; i < 16; i++) export_gate_bootstrapping_ciphertext_toFile(answer_data, &result[i], params);
//    fclose(answer_data);

    //clean up all pointers
//    delete_gate_bootstrapping_ciphertext_array(bitsize, res);
//    delete_gate_bootstrapping_ciphertext_array(bitsize, b);
//    delete_gate_bootstrapping_ciphertext_array(bitsize, a);
//    delete_gate_bootstrapping_ciphertext_array(bitsize, c);
    delete_gate_bootstrapping_cloud_keyset(bk);

    return 0;
}