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

#include "Cipher.h"

#define nEXP 1


//#define DEBUG
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

void karatSubaMultiplication(Cipher a, Cipher b, int bitSize, TFheGateBootstrappingSecretKeySet *key) {
    double sT = omp_get_wtime();
    int h_bS = bitSize/2;
    Cipher Xl(h_bS);
    Cipher Xr(h_bS);
    Cipher Yl(h_bS);
    Cipher Yr(h_bS);
    Cipher result(bitSize);

    for (int i = 0; i < h_bS; ++i) {
        bootsCOPY(&Xl.data[i], &a.data[i], Cipher::bk);
        bootsCOPY(&Xr.data[i], &a.data[i + h_bS], Cipher::bk);
        bootsCOPY(&Yl.data[i], &b.data[i], Cipher::bk);
        bootsCOPY(&Yr.data[i], &b.data[i + h_bS], Cipher::bk);
    }
//    cout << "output: " << decryptCheck(Xl, key) <<endl;
//    cout << "output: " << decryptCheck(Xr, key) <<endl;
//    cout << "output: " << decryptCheck(Yl, key) <<endl;
//    cout << "output: " << decryptCheck(Yr, key) <<endl;
    Cipher XlpXr = Xl + Xr;
    Cipher YlpYr = Yl + Yr;
    Cipher XlmYl, XrmYr, P;

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task
            XlmYl = Xl * Yl;
#pragma omp task
            XrmYr = Xr * Yr;
#pragma omp task
            P = XlpXr * YlpYr;
#pragma omp taskwait
        }
    }
    cout << "I am here" << endl;
    cout << "XlpXr: " << decryptCheck(XlpXr, key) <<endl;//25
    cout << "YlpYr: " << decryptCheck(YlpYr, key) <<endl;//21
    cout << "XlmYl: " << decryptCheck(XlmYl, key) <<endl;//480
    cout << "XrmYr: " << decryptCheck(XrmYr, key) <<endl;//1
    cout << "P: " << decryptCheck(P, key) <<endl;//525
    Cipher E_add = XrmYr + XlmYl;
    Cipher E = P - E_add;
    cout << "E_add: " << decryptCheck(E_add, key) <<endl;
    cout << "E: " << decryptCheck(E, key) <<endl;

    for (int i = 0; i < h_bS; ++i) {
        bootsCOPY(&result.data[i + h_bS], &E.data[i], Cipher::bk);
        bootsCOPY(&result.data[i], &XlmYl.data[i], Cipher::bk);
    }
    cout << "Res: " << decryptCheck(result, key) << " bitSize: " << result.numberOfBits <<endl;
    cout << "Time taken: " << omp_get_wtime() - sT << " seconds." << endl;

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
//
    //read the 2x16 ciphertexts
    int bitSize = 16;

//    cout << "number: " << bk->params->in_out_params->n << endl;
    LweSample *ciphertext1 = new_gate_bootstrapping_ciphertext_array(bitSize, params);
    LweSample *ciphertext2 = new_gate_bootstrapping_ciphertext_array(bitSize, params);
    LweSample *ciphertext3 = new_gate_bootstrapping_ciphertext_array(bitSize, params);

    //reads the 2x16 ciphertexts from the cloud file
    FILE *cloud_data = fopen("cloud.data", "rb");
    for (int i = 0; i < bitSize; i++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertext1[i], params);
    for (int i = 0; i < bitSize; i++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertext2[i], params);
    fclose(cloud_data);
    Cipher a(bitSize);
    Cipher b(bitSize);
    Cipher c(bitSize);
    a.data = ciphertext1;
    b.data = ciphertext2;


/*
 * BIT computation
 * *//*
    for (int i = 0; i < nEXP; ++i) {
        double st = omp_get_wtime();
//        omp_set_num_threads(4);
//#pragma omp parallel for
        for (int j = 0; j < 32; ++j) {
            bootsAND(&ciphertext3[j], &ciphertext1[j], &ciphertext2[j], Cipher::bk);
        }
        cout << "time taken for " << 1 << " bit AND: \t" << omp_get_wtime() - st << endl;
    }
    cout << endl;
    */


    /*
    * Addition
    */
    /*
    for (int i = 0; i <nEXP; ++i) {
        double start_time = omp_get_wtime();
        c = a + b;
        double time = omp_get_wtime() - start_time;
        cout << "--------OpenMP Add time taken:" << time << " --------" << endl;
        cout << "output: " << decryptCheck(c, secretKeySet) << " time: " << time <<endl;
    }
     */

    /*
    * Multiplication
    */
    /*
    for (int i = 0; i <nEXP; ++i) {
        double start_time = omp_get_wtime();
        c = a * b;
        double time = omp_get_wtime() - start_time;
        cout << "--------OpenMP Multiplication time taken:" << time << " --------" << endl;
        cout << "output: " << decryptCheck(c, secretKeySet) << " time: " << time <<endl;
    }
    */
    /*
     * Vector op
     *
     */
    //prepare vector data
            /*
    int vecLen = 64;
    LweSample **ciphertextArray = new LweSample*[vecLen * 2];
    Cipher *aa = new Cipher[vecLen * 2];
    Cipher *cc = new Cipher[vecLen];


    cout << "Preparing Data..." << endl;
    for (int j = 0; j < vecLen; ++j) {
        ciphertextArray[j] = new_gate_bootstrapping_ciphertext_array(bitSize, params);
        aa[j].numberOfBits = bitSize;
        ciphertextArray[j + vecLen] = new_gate_bootstrapping_ciphertext_array(bitSize, params);
        aa[j + vecLen].numberOfBits = bitSize;
        FILE *cloud_data = fopen("cloud.data", "rb");
        for (int i = 0; i < bitSize; i++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertextArray[j][i], params);
        aa[j].data = ciphertextArray[j];
//        cout << "index: " << j << " : " << decryptCheck(aa[j], secretKeySet) << endl;
        for (int i = 0; i < bitSize; i++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertextArray[j + vecLen][i], params);
        aa[j + vecLen].data = ciphertextArray[j + vecLen];
//        cout << "index: " << j + vecLen << " : " << decryptCheck(aa[j + vecLen], secretKeySet) << endl;
        fclose(cloud_data);
    }
    cout << "16 sample created" << endl;*/

    //VECTOR SUM TEST
/*
    for (int k = 0; k < nEXP; ++k) {
        vecLen = 1;
        double start_time = omp_get_wtime();
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
//            cout << "bitSize:" << cc[i].numberOfBits << endl;
            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 1:" << time << " --------" << endl;
    }
    cout << endl;


    for (int k = 0; k < nEXP; ++k) {
        vecLen = 2;
        double start_time = omp_get_wtime();
//        omp_set_num_threads(2);
//#pragma omp parallel for
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 2:" << time << " --------" << endl;
    }
    cout << endl;

    for (int k = 0; k < nEXP; ++k) {
        vecLen = 4;
        double start_time = omp_get_wtime();
//        omp_set_num_threads(4);
//#pragma omp parallel for
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 4:" << time << " --------" << endl;
    }
    cout << endl;

    for (int k = 0; k < nEXP; ++k) {
        vecLen = 8;
        double start_time = omp_get_wtime();
//        omp_set_num_threads(4);
//#pragma omp parallel for
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
//            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 8:" << time << " --------" << endl;
    }
    cout << endl;




    for (int k = 0; k < nEXP; ++k) {
        vecLen = 16;
        double start_time = omp_get_wtime();
//        omp_set_num_threads(4);
//#pragma omp parallel for
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
//            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 16:" << time << " --------" << endl;
    }
    cout << endl;





    for (int k = 0; k < nEXP; ++k) {
        vecLen = 32;
        double start_time = omp_get_wtime();
//        omp_set_num_threads(4);
//#pragma omp parallel for
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
//            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 32:" << time << " --------" << endl;
    }
    cout << endl;



    for (int k = 0; k < nEXP; ++k) {
        vecLen = 64;
        cout << "vLen: " << vecLen <<  endl;
        double start_time = omp_get_wtime();
        omp_set_num_threads(4);
#pragma omp parallel for
        for (int i = 0; i < vecLen; ++i) {
            cc[i] = aa[i] * aa[i + vecLen];
//            cout << "index: " << i << " : " << decryptCheck(cc[i], secretKeySet) << endl;
        }
        double time = omp_get_wtime() - start_time;
        cout << "--------Sequential Add time taken: 64:" << time << " --------" << endl;
    }
    cout << endl;
  */
    //kartSuba Computation

//    karatSubaMultiplication(a, b, bitSize, secretKeySet);


    //matrix multiplication
    cout << "creating matrix" << endl;
    int row = 16, col = 16;
    LweSample ***ciphertextArray1 = new LweSample**[row];
    LweSample ***ciphertextArray2 = new LweSample**[row];
    Cipher **matA = new Cipher*[row];
    Cipher **matB = new Cipher*[row];
    Cipher **matC = new Cipher*[row];
    Cipher temp(bitSize * 2);
    Cipher temp2(bitSize * 2);
    for (int i = 0; i < row; ++i) {
        ciphertextArray1[i] = new LweSample*[col];
        ciphertextArray2[i] = new LweSample*[col];
        matA[i] = new Cipher[col];
        matB[i] = new Cipher[col];
        matC[i] = new Cipher[col];
        for (int j = 0; j < col; ++j) {
            ciphertextArray1[i][j] = new_gate_bootstrapping_ciphertext_array(bitSize, params);
            ciphertextArray2[i][j] = new_gate_bootstrapping_ciphertext_array(bitSize, params);
            matA[i][j].numberOfBits = bitSize;
            matB[i][j].numberOfBits = bitSize;

            FILE *cloud_data = fopen("cloud.data", "rb");
            for (int k = 0; k < bitSize; k++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertextArray1[i][j][k], params);
            matA[i][j].data = ciphertextArray1[i][j];
            for (int k = 0; k < bitSize; k++) import_gate_bootstrapping_ciphertext_fromFile(cloud_data, &ciphertextArray2[i][j][k], params);
            matB[i][j].data = ciphertextArray2[i][j];
            fclose(cloud_data);
            Cipher xtemp(bitSize * 2);
            matC[i][j] = xtemp;
        }
    }
    cout << "matrix creation DONE" << endl;


    double sT = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < row; ++i) {
#pragma omp parallel for
        for (int j = 0; j < col; ++j) {
            cout << i << " " << j << endl;
            for (int k = 0; k < row; ++k) {
                    temp =  matA[i][k] * matB[k][j];
                    cout << "temp =  matA[i][k] * matB[k][j];" << endl;
                    cout << temp.numberOfBits << endl;
                    cout << matC[i][j].numberOfBits  << endl;
                    temp2 = matC[i][j] + temp;
                    cout << "matC[i][j] = matC[i][j] + temp;" << endl;
                    cout << temp2.numberOfBits << endl;
                    matC[i][j] = temp2;
                    cout << "matC[i][j] = temp2;" << endl;
//                }
            }
        }
    }
    cout << "Time taken for dim " << row << ": "  << omp_get_wtime() - sT << endl;


//    cout << "printing C" << endl;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            cout << "index: " << i * col + j << " : " << decryptCheck(matC[i][j], secretKeySet) << endl;
        }
    }
//
//    cout << "printing B" << endl;
//    for (int i = 0; i < row; ++i) {
//        for (int j = 0; j < col; ++j) {
//            cout << "index: " << i * col + j << " : " << decryptCheck(matB[i][j], secretKeySet) << endl;
//        }
//    }
//
//    cout << "Matrix created" << endl;





/*
    //compiling time for sequential bootstrapping and
    LweSample *ca = &a.data[0];
    LweSample *cb = &b.data[0];
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;

    LweSample *temp_result = new_LweSample(in_out_params);

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result, AndConst, in_out_params);
    lweAddTo(temp_result, ca, in_out_params);
    lweAddTo(temp_result, cb, in_out_params);

    tfhe_bootstrap_FFT(&c.data[0], bk->bkFFT, MU, temp_result);

    LweSample *u = new_LweSample(&bk->bkFFT->accum_params->extracted_lweparams);
    for (int i = 0; i < nEXP; ++i) {
        double sT = omp_get_wtime();
        tfhe_bootstrap_woKS_FFT(u, bk->bkFFT, MU, temp_result);
        cout << "BS time taken:" << omp_get_wtime()-sT << endl;
    }
    cout << endl;

    for (int i = 0; i < nEXP; ++i) {
        double sT = omp_get_wtime();
        lweKeySwitch(&c.data[0], bk->bkFFT->ks, u);
        cout << "kS time taken:" << omp_get_wtime()-sT << endl;
    }
*/



    delete_gate_bootstrapping_cloud_keyset(bk);

    return 0;
}