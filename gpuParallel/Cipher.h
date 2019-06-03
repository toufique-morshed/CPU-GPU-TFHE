//
// Created by morshed on 5/16/2018.
//

//#include <tfhe/tfhe.h>
#include "tfhe.h"
#include "tfhe_io.h"
//#include <tfhe/tfhe_io.h>

class Cipher {
public:
    int numberOfBits;
    LweSample *data;
    static TFheGateBootstrappingCloudKeySet *bk;
    static int nThreads;
public:
    Cipher();
    Cipher(int numberOfBits);
    Cipher(int numberOfBits, LweSample *data);
    void init(int nbits);
    void initWithVal(int nbits, int val);
    int getNumberOfBits();
    LweSample *getData();
    void setNumberOfBits(int numberOfBits);
    void setData(LweSample *data);
    void setNumberOfBitsAndData(int numberOfBits, LweSample *data);
    void setAllTo(int val);
//    void freeMemory();
    friend Cipher operator+(const Cipher a, const Cipher b);
    friend Cipher operator-(const Cipher a, const Cipher b);
    friend Cipher operator*(const Cipher a, const Cipher b);
    Cipher leftShift(const int nbit);
    void innerLeftShift(const int nbit);
    void addNumberToSelf(Cipher a);
    Cipher rightShift(const int nbit);
    Cipher operator++(int);
    static void addBits(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *carry);
    static void mulBinary(LweSample *result, const LweSample *number, const LweSample *bit, const int nb_bits);
    friend Cipher twosComplement(Cipher a);
    friend Cipher minimum(Cipher a, Cipher b); //this works only for positive numbers
    static void compare_bit(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *lsb_carry,
                            LweSample *tmp, const TFheGateBootstrappingCloudKeySet *bk);
    friend Cipher mul(Cipher a, Cipher b);
    friend void freeMemory(Cipher c);
    static void add(LweSample *result, const LweSample *a, const LweSample *b,
                    const int nb_bits, const LweSample *carry);
    friend Cipher addNaiveVersion(Cipher a, Cipher b);
    static void addBitsNaiveVersion(LweSample *result, LweSample *a, LweSample *b, LweSample *cin);//result has to be two bit
    void changeBitValTo(int position, int val);
    void setAllBitsToVal(int val);
    friend Cipher getNewCopyOf(Cipher a);
    Cipher& operator+=(const Cipher &rhs);
    void innerRightShift(const int nbit);
    friend Cipher absolute(Cipher &a);
    friend Cipher operator/(Cipher &a, Cipher &b);//, TFheGateBootstrappingSecretKeySet *secretKeySet);
    static Cipher divInternal(Cipher &a, Cipher &b);//, TFheGateBootstrappingSecretKeySet *secretKeySet);
    friend Cipher mux(Cipher &a, Cipher &b);
    friend int decryptCheck(Cipher cipher, TFheGateBootstrappingSecretKeySet *key);
    friend Cipher addSign(Cipher a, LweSample *bit);//, TFheGateBootstrappingSecretKeySet *key);
    friend int decryptBitCheck(LweSample* bit, TFheGateBootstrappingSecretKeySet *key);
    static int decryptBitCheck(LweSample* bit, TFheGateBootstrappingSecretKeySet *key);
//    friend Cipher greaterThan(Cipher &x, Cipher &y, TFheGateBootstrappingSecretKeySet *secretKeySet);
    Cipher operator>(Cipher &b);
    Cipher operator<=(Cipher &b);
    static void compareBit_g(LweSample* result, LweSample* x, LweSample* y, LweSample* cin);//, TFheGateBootstrappingSecretKeySet *secretKeySet);
    Cipher operator==(Cipher &b);
    static void bootsANDXOR(LweSample *result, const LweSample *ca, const LweSample *cb, const LweSample *cc,  const TFheGateBootstrappingCloudKeySet *bk);
    static void addBitsRaw(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *carry);
    static Cipher cipherAND(Cipher a, Cipher b, TFheGateBootstrappingSecretKeySet* secretKeySet);
};
