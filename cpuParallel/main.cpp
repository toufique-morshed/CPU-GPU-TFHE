#include <stdio.h>
#include <iostream>
#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#include <stdlib.h>
#include<stdint.h>
#include<inttypes.h>

using namespace std;

int main(int argc, char* argv[]) {
    if(argc != 3) {
        cout << "Usage: <binary_file_name> <first_number> <second_number>" << endl;
        return 1;
    }
    const int minimum_lambda = 110;
    TFheGateBootstrappingParameterSet* params = new_default_gate_bootstrapping_parameters(minimum_lambda);


    //generate a random key
    uint32_t seed[] = { 314, 1592, 657 };
    tfhe_random_generator_setSeed(seed,3);
    TFheGateBootstrappingSecretKeySet* key = new_random_gate_bootstrapping_secret_keyset(params);

    //export the secret key to file for later use
    FILE* secret_key = fopen("secret.key","wb");
    export_tfheGateBootstrappingSecretKeySet_toFile(secret_key, key);
    fclose(secret_key);

    //export the cloud key to a file (for the cloud)
    FILE* cloud_key = fopen("cloud.key","wb");
    export_tfheGateBootstrappingCloudKeySet_toFile(cloud_key, &key->cloud);
    fclose(cloud_key);

    //you can put additional instructions here!!
    //...

    //generate encrypt the 16 bits of 2017
    int plaintext1 = atoi(argv[1]);
    int bitSize = 16;
    LweSample* ciphertext1 = new_gate_bootstrapping_ciphertext_array(bitSize, params);
    for (int i=0; i<bitSize; i++) {
        bootsSymEncrypt(&ciphertext1[i], (plaintext1>>i)&1, key);
    }

    //generate encrypt the 16 bits of 42
    int plaintext2 = atoi(argv[2]);
    LweSample* ciphertext2 = new_gate_bootstrapping_ciphertext_array(bitSize, params);
    for (int i=0; i<bitSize; i++) {
        bootsSymEncrypt(&ciphertext2[i], (plaintext2>>i)&1, key);
    }
//    cout << "test values of b and a" << endl;
//    printf("%" PRId32 "\n", ciphertext2->b);
//    for (int j = 0; j < 10; ++j) {
//        printf("%" PRId32 " ", ciphertext2->a[j]);
//    }
//    cout << endl;
//    for (int j = 0; j < 10; ++j) {
//        printf("%" PRId32 " ", key->lwe_key->key[j]);
//    }
//    cout << endl;

    printf("Hi there! Today, I will ask the cloud what is the minimum between %d and %d in %d bits\n",plaintext1, plaintext2, bitSize);

    //export the 2x16 ciphertexts to a file (for the cloud)
    FILE* cloud_data = fopen("cloud.data","wb");
    for (int i=0; i<bitSize; i++)
        export_gate_bootstrapping_ciphertext_toFile(cloud_data, &ciphertext1[i], params);
    for (int i=0; i<bitSize; i++)
        export_gate_bootstrapping_ciphertext_toFile(cloud_data, &ciphertext2[i], params);
    fclose(cloud_data);

    //clean up all pointers
    delete_gate_bootstrapping_ciphertext_array(bitSize, ciphertext2);
    delete_gate_bootstrapping_ciphertext_array(bitSize, ciphertext1);

    //clean up all pointers
    delete_gate_bootstrapping_secret_keyset(key);
    delete_gate_bootstrapping_parameters(params);

    return 0;
}

