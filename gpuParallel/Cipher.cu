//
// Created by morshed on 5/16/2018.
//

/*convention
 * MSB is sign bit
*/

#include <iostream>
#include <omp.h>
#include "Cipher.h"

#define PARALLEL

using namespace std;

TFheGateBootstrappingCloudKeySet *initFunction() {
    FILE *cloud_key = fopen("cloud.key", "rb");
    TFheGateBootstrappingCloudKeySet *bk = new_tfheGateBootstrappingCloudKeySet_fromFile(cloud_key);
    fclose(cloud_key);
    return bk;
}

TFheGateBootstrappingCloudKeySet *Cipher::bk = initFunction();
int Cipher::nThreads = 4;


Cipher::Cipher() {}

Cipher::Cipher(int numberOfBits) {
    this->numberOfBits = numberOfBits;
    this->data = new_gate_bootstrapping_ciphertext_array(this->numberOfBits, Cipher::bk->params);
    this->setAllTo(0);
}

Cipher::Cipher(int numberOfBits, LweSample *data) {
    this->numberOfBits = numberOfBits;
    this->data = data;
}

int Cipher::getNumberOfBits() {
    return this->numberOfBits;
}

void Cipher::init(int nbits) {
    this->numberOfBits = nbits;
    this->data = new_gate_bootstrapping_ciphertext_array(this->numberOfBits, Cipher::bk->params);
}

void Cipher::initWithVal(int nbits, int val) {
    this->numberOfBits = nbits;
    this->data = new_gate_bootstrapping_ciphertext_array(this->numberOfBits, Cipher::bk->params);
    this->setAllTo(val);
}

LweSample *Cipher::getData() {
    return this->data;
}

void Cipher::setNumberOfBits(int numberOfBits) {
    this->numberOfBits = numberOfBits;
}

void Cipher::setData(LweSample *data) {
    this->data = data;
}

void Cipher::setNumberOfBitsAndData(int numberOfBits, LweSample *data) {
    this->numberOfBits = numberOfBits;
    this->data = data;
}

void Cipher::setAllTo(int val) {
#ifdef PARALLEL
    omp_set_num_threads(Cipher::nThreads);
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < this->numberOfBits; i++) {
        bootsCONSTANT(&data[i], val, bk);
    }
}

Cipher operator*(Cipher a, Cipher b) {
    //assumes both has same bit number
    int nb_bits = a.numberOfBits;// + b.numberOfBits;
    Cipher out(nb_bits);

#ifdef PARALLEL
#pragma omp declare reduction \
  (OMP_CIPHER_SUM:Cipher:omp_out += omp_in) \
  initializer(omp_priv=Cipher(16))
    omp_set_num_threads(Cipher::nThreads);
#pragma omp parallel for schedule(static) reduction(OMP_CIPHER_SUM:out)
#endif
    for (int i = 0; i < b.numberOfBits; ++i) {
        LweSample *tmpBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
        Cipher sum(nb_bits);
        bootsCOPY(&tmpBit[0], &b.data[i], Cipher::bk);
        Cipher::mulBinary(sum.data, a.data, tmpBit, a.numberOfBits);
        sum.innerLeftShift(i);
        sum.numberOfBits = nb_bits;
        out += sum;

        delete_gate_bootstrapping_ciphertext_array(1, tmpBit);
    }
    //assumes both has same bit number
    out.numberOfBits = a.numberOfBits;
    return out;
}


Cipher mul(Cipher a, Cipher b) {
    //assumes both has same bit number
    int nb_bits_a = a.numberOfBits;
    int nb_bits_b = b.numberOfBits;

    int nb_bits = nb_bits_a + nb_bits_b - 1;
    Cipher out(nb_bits);

    LweSample *toAdd = new_gate_bootstrapping_ciphertext_array(nb_bits_a, Cipher::bk->params);
    LweSample *zeros = new_gate_bootstrapping_ciphertext_array(nb_bits_a, Cipher::bk->params);

    for (int i = 0; i < nb_bits_a; i++) {
        bootsCONSTANT(&zeros[i], 0, Cipher::bk);
        bootsCONSTANT(&toAdd[i], 0, Cipher::bk);
    }

    for (int i = 0; i < nb_bits_a; i++) {
        bootsMUX(&toAdd[i], &b.data[0], &a.data[i], &zeros[i], Cipher::bk);

        bootsCOPY(&out.data[i], &toAdd[i], Cipher::bk);
    }

    for (int i = 1; i < nb_bits_b; ++i) {

        for (int j = i; j < i + nb_bits_a; j++) {
            bootsCOPY(&toAdd[j - i], &out.data[j], Cipher::bk);
        }

        LweSample *tmp = new_gate_bootstrapping_ciphertext_array(nb_bits_a, Cipher::bk->params);

        for (int j = 0; j < nb_bits_a; j++) {
            bootsMUX(&tmp[j], &b.data[i], &a.data[j], &zeros[j], Cipher::bk);
        }

        LweSample *carry = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
        bootsCONSTANT(&carry[0], 0, Cipher::bk);

        Cipher::add(toAdd, tmp, toAdd, nb_bits_a, carry);

        for (int j = 0; j < nb_bits_a; j++) {
            bootsCOPY(&out.data[i + j], &toAdd[j], Cipher::bk);
        }

        delete_gate_bootstrapping_ciphertext_array(nb_bits_a, tmp);
    }


    delete_gate_bootstrapping_ciphertext_array(nb_bits_a, zeros);
    out.numberOfBits = a.numberOfBits; //assumes both a and b have same number of bits
    return out;
}

///////////////////////////////////////////////
void Cipher::add(LweSample *result, const LweSample *a, const LweSample *b, const int nb_bits, const LweSample *carry) {
    LweSample *new_res = new_gate_bootstrapping_ciphertext_array(2, Cipher::bk->params);

    Cipher::addBits(new_res, &a[0], &b[0], carry);
    bootsCOPY(&result[0], &new_res[0], Cipher::bk);

    //LweSample* t = new_gate_bootstrapping_ciphertext_array(2, bk->params);
    for (unsigned i = 0; i < nb_bits - 1; i++) {
        Cipher::addBits(new_res, &a[i + 1], &b[i + 1], &new_res[1]);
        bootsCOPY(&result[i + 1], &new_res[0], Cipher::bk);
    }

    bootsCOPY(&result[nb_bits - 1], &new_res[1], Cipher::bk);
    //printf("%d\n",bootsSymDecrypt(&result[nb_bits],bk))
    //delete_gate_bootstrapping_ciphertext_array(2,t);
    delete_gate_bootstrapping_ciphertext_array(2, new_res);
}

///////////////////////////////////////////////
void Cipher::addNumberToSelf(Cipher a) {
    //asumes both self and cipher a has same bit length
    //create carries
    LweSample *carry = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *bitResult = new_gate_bootstrapping_ciphertext_array(2,
                                                                   Cipher::bk->params); //0th bit->output 1st bit->cout

    //set carries to 0
    bootsCONSTANT(&carry[0], 0, Cipher::bk);
    //add first bit
    Cipher::addBits(bitResult, &data[0], &a.data[0], carry); //T[] t = add(x[0], y[0], env.newT(cin));
    //copy first bit to output at 0 position
    bootsCOPY(&data[0], &bitResult[0], Cipher::bk); //res[0] = t[S];

    for (int i = 0; i < this->numberOfBits - 1; ++i) { //for (int i = 0; i < x.length - 1; i++) {
        Cipher::addBits(bitResult, &data[i + 1], &a.data[i + 1],
                        &bitResult[1]); //t = add(x[i + 1], y[i + 1], t[COUT]);
        //copy result to the output array
        bootsCOPY(&data[i + 1], &bitResult[0], Cipher::bk);
    }
    //bootsCOPY(&output.data[output.numberOfBits - 1], &bitResult[1], Cipher::bk); is generally used to check overflow
    // However as we are overlooking overflow
    //we are copying the bit previous bit. This is to keep room for future work.
//    bootsCOPY(&output.data[output.numberOfBits - 1], &output.data[output.numberOfBits - 2], Cipher::bk);
    //free used memory
    delete_gate_bootstrapping_ciphertext_array(2, bitResult);
    delete_gate_bootstrapping_ciphertext_array(1, carry);

}


void Cipher::innerLeftShift(const int nbit) {
    //assumptions: unsigned
    //set first bit to zero
    if (nbit == 0) {
        return;
    }
    for (int i = this->numberOfBits - 1; i >= 0; i--) {
        if (i >= nbit) {
            bootsCOPY(&data[i], &data[i - nbit], Cipher::bk);
        } else {
            bootsCONSTANT(&data[i], 0, Cipher::bk);
        }
    }
}

void Cipher::mulBinary(LweSample *result, const LweSample *number, const LweSample *bit, const int nb_bits) {

    for (int i = 0; i < nb_bits; i++) {
        bootsAND(&result[i], &bit[0], &number[i], Cipher::bk);
    }
}

Cipher Cipher::rightShift(const int nbit) {
    Cipher out(this->numberOfBits);
    LweSample *signBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCOPY(&signBit[0], &data[this->numberOfBits - 1], Cipher::bk);
    for (int i = 0; i < this->numberOfBits - nbit; ++i) {
        bootsCOPY(&out.data[i], &data[i + nbit], Cipher::bk);
    }
    for (int i = 0; i < nbit; ++i) {
        bootsCOPY(&out.data[this->numberOfBits - 1 - i], &signBit[0], Cipher::bk);
    }
    //check this for keeping the negative numbers like positive numbers
    //it takes more time for negative number precision
    Cipher toAdd(this->numberOfBits);

    LweSample *muxBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *oneBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCONSTANT(&oneBit[0], 1, Cipher::bk);
    LweSample *zeroBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCONSTANT(&zeroBit[0], 0, Cipher::bk);

    bootsMUX(&muxBit[0], &data[this->numberOfBits - 1], &oneBit[0], &zeroBit[0], Cipher::bk);

    bootsCOPY(&toAdd.data[0], muxBit, Cipher::bk);
    out.addNumberToSelf(toAdd);
    return out;
}

Cipher Cipher::leftShift(const int nbit) {
    //assumptions: Signed
    Cipher out(this->numberOfBits);
    //as we are assuming this is signed we will start computing number of bits-1
    //at first copy the signed bit
    bootsCOPY(&out.data[this->numberOfBits - 1], &this->data[this->numberOfBits - 1], Cipher::bk);

    for (int i = 0, j = nbit; j < this->numberOfBits - 1; ++i, ++j) {
        bootsCOPY(&out.data[j], &this->data[i], Cipher::bk);
    }
    return out;
}

//post-increment
Cipher Cipher::operator++(int) {
    Cipher one(this->numberOfBits);
    bootsCONSTANT(&one.data[0], 1, Cipher::bk);
    Cipher out = *this + one;
    *this = out;
    return *this;
}

Cipher twosComplement(Cipher a) {
    Cipher output(a.numberOfBits);
    LweSample *reachOne = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCONSTANT(&reachOne[0], 0, Cipher::bk);
    for (int i = 0; i < a.numberOfBits; i++) {
        bootsXOR(&output.data[i], &a.data[i], &reachOne[0], Cipher::bk);
        bootsOR(&reachOne[0], &reachOne[0], &a.data[i], Cipher::bk);
    }
    //free memory
    delete_gate_bootstrapping_ciphertext_array(1, reachOne);
    return output;
}

//get minimum between 2 number
Cipher minimum(Cipher a, Cipher b) {
    //assumes lengths of a and b are same and both are positive numbers
    Cipher output(a.numberOfBits);
    LweSample *tmps = new_gate_bootstrapping_ciphertext_array(2, Cipher::bk->params);
    //initialize the carry to 0
    bootsCONSTANT(&tmps[0], 0, Cipher::bk);
    //run the elementary comparator gate n times
    for (int i = 0; i < output.numberOfBits; i++) {
        Cipher::compare_bit(&tmps[0], &a.data[i], &b.data[i], &tmps[0], &tmps[1], Cipher::bk);
    }
    //tmps[0] is the result of the comparaison: 0 if a is larger, 1 if b is larger
    //select the max and copy it to the result
    for (int i = 0; i < output.numberOfBits; i++) {
        bootsMUX(&output.data[i], &tmps[0], &b.data[i], &a.data[i], Cipher::bk);
    }

    delete_gate_bootstrapping_ciphertext_array(2, tmps);
    return output;

}

void Cipher::compare_bit(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *lsb_carry,
                         LweSample *tmp, const TFheGateBootstrappingCloudKeySet *bk) {
    bootsXNOR(tmp, a, b, bk);
    bootsMUX(result, tmp, lsb_carry, a, bk);
}

//substract two number
Cipher operator-(Cipher a, Cipher b) {
    Cipher out = a + twosComplement(b);
    return out;
}

//add two numbers
Cipher operator+(Cipher a, Cipher b) {
    //assuming both a and b has same bit leangth
    //create output
    Cipher output(a.getNumberOfBits() + 1); //T[] res = env.newTArray(x.length + 1);
    //create carries
    LweSample *carry = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *bitResult = new_gate_bootstrapping_ciphertext_array(2,
                                                                   Cipher::bk->params); //0th bit->output 1st bit->cout
    //set carries to 0
    bootsCONSTANT(&carry[0], 0, Cipher::bk);
    //add first bit
    Cipher::addBits(bitResult, &a.data[0], &b.data[0], carry); //T[] t = add(x[0], y[0], env.newT(cin));
    //copy first bit to output at 0 position
    bootsCOPY(&output.data[0], &bitResult[0], Cipher::bk); //res[0] = t[S];

    for (int i = 0; i < a.numberOfBits - 1; ++i) { //for (int i = 0; i < x.length - 1; i++) {
        Cipher::addBits(bitResult, &a.data[i + 1], &b.data[i + 1],
                        &bitResult[1]); //t = add(x[i + 1], y[i + 1], t[COUT]);
        //copy result to the output array
        bootsCOPY(&output.data[i + 1], &bitResult[0], Cipher::bk);
    }
    //bootsCOPY(&output.data[output.numberOfBits - 1], &bitResult[1], Cipher::bk); is generally used to check overflow
    // However as we are overlooking overflow
    //we are copying the bit previous bit. This is to keep room for future work.
    bootsCOPY(&output.data[output.numberOfBits - 1], &output.data[output.numberOfBits - 2], Cipher::bk);
    //make the bit number same as a and b
    output.numberOfBits = a.numberOfBits;
    //free used memory
    delete_gate_bootstrapping_ciphertext_array(2, bitResult);
    delete_gate_bootstrapping_ciphertext_array(1, carry);
    return output;
}

void Cipher::addBits(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *carry) {
    LweSample *t1 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *t2 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsXOR(t1, a, carry, bk);
    bootsXOR(t2, b, carry, Cipher::bk);
    bootsXOR(&result[0], a, t2, Cipher::bk);//not bootstrapped
    bootsAND(t1, t1, t2, bk);
    bootsXOR(&result[1], carry, t1, Cipher::bk);
    //free memory
    delete_gate_bootstrapping_ciphertext_array(1, t1);
    delete_gate_bootstrapping_ciphertext_array(1, t2);
}

//void Cipher::freeMemory() {
//    //clean mem
//    delete_gate_bootstrapping_ciphertext_array(this->numberOfBits, this->data);
//}

//addition naive version
void Cipher::addBitsNaiveVersion(LweSample *result, LweSample *a, LweSample *b, LweSample *cin) {
    //assumes result is of 2 bit. first bit (o) is for the 's' and second is for 'cout'
    //cin is one bit, a is one bit and b is one bit
    //first part calculate s
    LweSample *tmpBit1 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *tmpBit2 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *tmpBit3 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsXOR(&tmpBit1[0], &a[0], &b[0], Cipher::bk);
    bootsXOR(&result[0], &tmpBit1[0], &cin[0], Cipher::bk);
    //second part calulate 'cout'
    bootsAND(&tmpBit2[0], &tmpBit1[0], &cin[0], Cipher::bk);
    bootsAND(&tmpBit3[0], &a[0], &b[0], Cipher::bk);
    bootsOR(&result[1], &tmpBit2[0], &tmpBit3[0], Cipher::bk);
}

Cipher addNaiveVersion(Cipher a, Cipher b) {
    //assumes both have same number of bits
    Cipher out(a.numberOfBits);
    LweSample *result = new_gate_bootstrapping_ciphertext_array(2, Cipher::bk->params);
    bootsCONSTANT(&result[0], 0, Cipher::bk);
    bootsCONSTANT(&result[1], 0, Cipher::bk);
    for (int i = 0; i < out.numberOfBits; ++i) {
        Cipher::addBitsNaiveVersion(result, &a.data[i], &b.data[i], &result[1]);
        bootsCOPY(&out.data[i], &result[0], Cipher::bk);
    }
    return out;
}

void freeMemory(Cipher c) {
    delete_gate_bootstrapping_ciphertext_array(c.numberOfBits, c.data);
}

void Cipher::changeBitValTo(int position, int val) {
    bootsCONSTANT(&data[position], val, bk);
}

void Cipher::setAllBitsToVal(int val) {
    for (int i = 0; i < this->numberOfBits; ++i) {
        bootsCONSTANT(&data[i], val, Cipher::bk);
    }
}

Cipher getNewCopyOf(Cipher a) {
    Cipher out(a.numberOfBits);
    for (int i = 0; i < a.numberOfBits; ++i) {
        bootsCOPY(&out.data[i], &a.data[i], Cipher::bk);
    }
    return out;
}

Cipher &Cipher::operator+=(const Cipher &rhs) {
    this->addNumberToSelf(rhs);
    return *this;
}

void Cipher::innerRightShift(const int nbit) {
    if (nbit == 0) {
        return;
    }
    LweSample *signBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCOPY(&signBit[0], &data[this->numberOfBits - 1], Cipher::bk);
    for (int i = 0; i < this->numberOfBits - nbit; ++i) {
        bootsCOPY(&data[i], &data[i + nbit], Cipher::bk);
    }
    for (int i = 0; i < nbit; ++i) {
        bootsCOPY(&data[this->numberOfBits - 1 - i], &signBit[0], Cipher::bk);
    }

    //check this for keeping the negative numbers like positive numbers
    //it takes more time for negative number precision
    Cipher toAdd(this->numberOfBits);
    LweSample *muxBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *oneBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCONSTANT(&oneBit[0], 1, Cipher::bk);
    LweSample *zeroBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsCONSTANT(&zeroBit[0], 0, Cipher::bk);

    bootsMUX(&muxBit[0], &data[this->numberOfBits - 1], &oneBit[0], &zeroBit[0], Cipher::bk);

    bootsCOPY(&toAdd.data[0], muxBit, Cipher::bk);
    this->addNumberToSelf(toAdd);
}

Cipher absolute(Cipher &a) {
//    double s_time = omp_get_wtime();
    Cipher output(a.numberOfBits);
    Cipher mask(a.numberOfBits);
#ifdef PARALLEL
    omp_set_num_threads(Cipher::nThreads);
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < a.numberOfBits; ++i) {
        bootsCOPY(&mask.data[i], &a.data[a.numberOfBits - 1], Cipher::bk);
    }
    Cipher res = mask + a;
#ifdef PARALLEL
    omp_set_num_threads(Cipher::nThreads);
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < mask.numberOfBits; ++i) {
        bootsXOR(&output.data[i], &res.data[i], &mask.data[0], Cipher::bk);
    }
//    double e_time = omp_get_wtime();
//    cout << "time to abs " << e_time - s_time << endl;
    return output;
}

Cipher operator/(Cipher &a, Cipher &b) {//, TFheGateBootstrappingSecretKeySet *secretKeySet) {
    //Assumption: same number of bits
//    double s_time = omp_get_wtime();
    Cipher absA = absolute(a);
    Cipher absB = absolute(b);
    Cipher PA = Cipher::divInternal(absA, absB);//, secretKeySet);
    Cipher quotient(a.numberOfBits), remainder(a.numberOfBits);
    for (int i = 0; i < a.numberOfBits; ++i) {
        bootsCOPY(&quotient.data[i], &PA.data[i], Cipher::bk);
        bootsCOPY(&remainder.data[i], &PA.data[i + a.numberOfBits], Cipher::bk);
    }
    LweSample *signBit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsXOR(&signBit[0], &a.data[a.numberOfBits - 1], &b.data[b.numberOfBits - 1], Cipher::bk);
    Cipher output = addSign(quotient, &signBit[0]);//, secretKeySet);
//    double e_time = omp_get_wtime();
//    cout << "time to divide " << e_time - s_time << endl;
    return output;
}

Cipher Cipher::divInternal(Cipher &a, Cipher &b) {//, TFheGateBootstrappingSecretKeySet *secretKeySet) {
    Cipher PA(a.numberOfBits + b.numberOfBits);
    for (int i = 0; i < a.numberOfBits; ++i) {
        bootsCOPY(&PA.data[i], &a.data[i], Cipher::bk);
    }
    for (int i = 0; i < a.numberOfBits; ++i) {
        //PA = leftShift(PA);
        PA.innerLeftShift(1);
        //T[] tempP = sub(Arrays.copyOfRange(PA, x.length, PA.length), B);
        Cipher temp(a.numberOfBits);
        for (int j = a.numberOfBits; j < PA.numberOfBits; ++j) {
            bootsCOPY(&temp.data[j - a.numberOfBits], &PA.data[j], Cipher::bk);
        }
        Cipher tempP = temp - b;
        //PA[0] = not(tempP[tempP.length - 1]);
        LweSample *bit = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
        bootsNOT(&bit[0], &tempP.data[tempP.numberOfBits - 1], Cipher::bk);
        bootsCOPY(&PA.data[0], &bit[0], Cipher::bk);
        Cipher aTemp(a.numberOfBits);
        for (int j = 0; j < a.numberOfBits; ++j) {
            bootsCOPY(&aTemp.data[j], &PA.data[j + a.numberOfBits], Cipher::bk);
        }
        Cipher tempMux(temp.numberOfBits);
        for (int j = 0; j < tempMux.numberOfBits; ++j) {
            bootsMUX(&tempMux.data[j], &tempP.data[tempP.numberOfBits - 1], &aTemp.data[j], &tempP.data[j], Cipher::bk);
        }
        for (int j = 0; j < tempMux.numberOfBits; ++j) {
            bootsCOPY(&PA.data[j + tempMux.numberOfBits], &tempMux.data[j], Cipher::bk);
        }
        delete_gate_bootstrapping_ciphertext_array(1, bit);
    }
    return PA;
}

Cipher addSign(Cipher x, LweSample *sign) {//, TFheGateBootstrappingSecretKeySet *key) {
//    T[] reachedOneSignal = zeros(x.length);
    Cipher reachedOneSignal(x.numberOfBits);
//    T[] result = env.newTArray(x.length);
    Cipher result(x.numberOfBits);
    for (int i = 0; i < x.numberOfBits - 1; ++i) {
        bootsOR(&reachedOneSignal.data[i + 1], &reachedOneSignal.data[i], &x.data[i], Cipher::bk);
        bootsXOR(&result.data[i], &x.data[i], &reachedOneSignal.data[i], Cipher::bk);
    }

    bootsXOR(&result.data[result.numberOfBits - 1], &x.data[x.numberOfBits - 1],
             &reachedOneSignal.data[reachedOneSignal.numberOfBits - 1], Cipher::bk);
    Cipher out(x.numberOfBits);
    for (int i = 0; i < x.numberOfBits; ++i) {
        bootsMUX(&out.data[i], sign, &result.data[i], &x.data[i], Cipher::bk);
    }
    return out;
}

int Cipher::decryptBitCheck(LweSample *bit, TFheGateBootstrappingSecretKeySet *key) {
    int ai = bootsSymDecrypt(bit, key);
    return ai;
}

//Cipher greaterThan(Cipher &x, Cipher &y, TFheGateBootstrappingSecretKeySet *secretKeySet) {
//    //assumptions: both with same bit length
//    Cipher out(1);
//    for (int i = 0; i < x.numberOfBits; ++i) {
//        Cipher::compareBit_g(&out.data[0], &x.data[i], &y.data[i], &out.data[0], secretKeySet);
//    }
//    LweSample* sign = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
//    bootsXOR(sign, &x.data[x.numberOfBits - 1], &y.data[y.numberOfBits - 1], Cipher::bk);
//    bootsXOR(&out.data[0], sign, &out.data[0], Cipher::bk);
//
//    delete_gate_bootstrapping_ciphertext_array(1, sign);
//    return out;
//}
Cipher Cipher::operator>(Cipher &y) {
    Cipher out(1);
    for (int i = 0; i < this->numberOfBits; ++i) {
        Cipher::compareBit_g(&out.data[0], &data[i], &y.data[i], &out.data[0]);
    }
    LweSample* sign = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    bootsXOR(sign, &data[numberOfBits - 1], &y.data[y.numberOfBits - 1], Cipher::bk);
    bootsXOR(&out.data[0], sign, &out.data[0], Cipher::bk);

    delete_gate_bootstrapping_ciphertext_array(1, sign);
    return out;
}

Cipher Cipher::operator<=(Cipher &b) {
    Cipher out = *this > b;
    bootsNOT(&out.data[0], &out.data[0], Cipher::bk);
    return out;
}

void Cipher::compareBit_g(LweSample* result, LweSample* x, LweSample* y, LweSample* cin) {//}, TFheGateBootstrappingSecretKeySet *secretKeySet) {
    LweSample *t1 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    LweSample *t2 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
            bootsXOR(t1, x, cin, Cipher::bk);
    bootsXOR(t2, y, cin, Cipher::bk);
    bootsAND(t1, t1, t2, Cipher::bk);
    bootsXOR(result, x, t1, Cipher:: bk);

    delete_gate_bootstrapping_ciphertext_array(1, t1);
    delete_gate_bootstrapping_ciphertext_array(1, t2);
}

Cipher Cipher::operator==(Cipher &b) {
    //assumptions: same bit length
    Cipher out(1);
    Cipher XOROfNumbers(this->numberOfBits);
#ifdef PARALLEL
    omp_set_num_threads(Cipher::nThreads);
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < this->numberOfBits; ++i) {
        bootsXOR(&XOROfNumbers.data[i], &data[i], &b.data[i], Cipher::bk);
    }
    for (int i = 0; i < this->numberOfBits; ++i) {
        bootsOR(&out.data[0], &out.data[0], &XOROfNumbers.data[i], Cipher::bk);
    }
    bootsNOT(&out.data[0], &out.data[0], Cipher::bk);
    return out;
}


void Cipher::addBitsRaw(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *carry) {

    const LweParams *extracted_params = &bk->params->tgsw_params->tlwe_params->extracted_lweparams;
    LweSample *u1 = new_LweSample(extracted_params);
    LweSample *u2 = new_LweSample(extracted_params);
//    LweSample *u1 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
//    LweSample *u2 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);

//    LweSample *t1 = new_gate_bootstrapping_ciphertext_array(1, Cipher::bk->params);
    ////bootsXOR(t1, a, carry, bk);
    ////XOR
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = Cipher::bk->params->in_out_params;
    LweSample *temp_result = new_LweSample(in_out_params);
    //compute: (0,1/4) + 2*(ca + cb)
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
    lweNoiselessTrivial(temp_result, XorConst, in_out_params);
    lweAddMulTo(temp_result, 2, a, in_out_params);
    lweAddMulTo(temp_result, 2, carry, in_out_params);
    tfhe_bootstrap_woKS_FFT(u1, Cipher::bk->bkFFT, MU, temp_result);
//    tfhe_bootstrap_FFT(u1, Cipher::bk->bkFFT, MU, temp_result);

    ////bootsXOR(t2, b, carry, Cipher::bk);
    ////XOR
    LweSample *temp_result2 = new_LweSample(in_out_params);
    //compute: (0,1/4) + 2*(ca + cb)
    lweNoiselessTrivial(temp_result2, XorConst, in_out_params);
    lweAddMulTo(temp_result2, 2, b, in_out_params);
    lweAddMulTo(temp_result2, 2, carry, in_out_params);
    tfhe_bootstrap_woKS_FFT(u2, Cipher::bk->bkFFT, MU, temp_result2);
//    tfhe_bootstrap_FFT(u2, Cipher::bk->bkFFT, MU, temp_result2);


    ////bootsXOR(&result[0], a, t2, Cipher::bk);
    ////XOR
    LweSample *temp_result3 = new_LweSample(in_out_params);
    lweNoiselessTrivial(temp_result3, XorConst, in_out_params);
    lweAddMulTo(temp_result3, 2, a, in_out_params);
    lweAddMulTo(temp_result3, 2, u2, in_out_params);
//    tfhe_bootstrap_woKS_FFT(&result[0], Cipher::bk->bkFFT, MU, temp_result3);
//    lweKeySwitch(&result[0], Cipher::bk->bkFFT->ks, temp_result3);
    tfhe_bootstrap_FFT(&result[0], Cipher::bk->bkFFT, MU, temp_result3);
//
//    //////bootsAND(t1, t1, t2, bk);
//    ////AND
    static const Torus32 MUand = modSwitchToTorus32(1, 8);
    LweSample *temp_result4 = new_LweSample(in_out_params);
    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);
    lweNoiselessTrivial(temp_result4, AndConst, in_out_params);
    lweAddTo(temp_result4, u1, in_out_params);
    lweAddTo(temp_result4, u2, in_out_params);
    tfhe_bootstrap_woKS_FFT(u1, Cipher::bk->bkFFT, MUand, temp_result4);
//
//
//    ////bootsXOR(&result[1], carry, t1, Cipher::bk);
//    //XOR
    LweSample *temp_result5 = new_LweSample(in_out_params);
    lweNoiselessTrivial(temp_result5, XorConst, in_out_params);
    lweAddMulTo(temp_result5, 2, carry, in_out_params);
    lweAddMulTo(temp_result5, 2, u1, in_out_params);
    tfhe_bootstrap_FFT(&result[1], Cipher::bk->bkFFT, MU, temp_result5);
//    lweKeySwitch(&result[1], Cipher::bk->bkFFT->ks,  temp_result5);


//    //free memory
//    delete_gate_bootstrapping_ciphertext_array(1, t1);
//    delete_gate_bootstrapping_ciphertext_array(1, t2);
}


/*
 * Homomorphic bootstrapped AND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
void Cipher::bootsANDXOR(LweSample *result, const LweSample *ca, const LweSample *cb, const LweSample *cc,  const TFheGateBootstrappingCloudKeySet *bk) {
    static const Torus32 MU = modSwitchToTorus32(1, 8);
    const LweParams *in_out_params = bk->params->in_out_params;
    const LweParams *extracted_params = &bk->params->tgsw_params->tlwe_params->extracted_lweparams;

    LweSample *temp_result = new_LweSample(in_out_params);
    LweSample *temp_result2 = new_LweSample(in_out_params);
    LweSample *u2 = new_LweSample(extracted_params);

    //compute: (0,-1/8) + ca + cb
    static const Torus32 AndConst = modSwitchToTorus32(-1, 8);


    //compute: (0,1/4) + 2*(ca + cb)
    //compute: [(0,1/4) + 2*(ca + cb)]+(0,-1/8) + cc
    static const Torus32 XorConst = modSwitchToTorus32(1, 4);


    lweNoiselessTrivial(temp_result, XorConst, in_out_params);
    lweAddMulTo(temp_result, 2, ca, in_out_params);
    lweAddMulTo(temp_result, 2, cb, in_out_params);
//    tfhe_bootstrap_FFT(u2, bk->bkFFT, MU, temp_result);
//    tfhe_bootstrap_woKS_FFT(u2, bk->bkFFT, MU, temp_result);


    lweNoiselessTrivial(temp_result2, AndConst, in_out_params);
    lweAddTo(temp_result2, u2, in_out_params);
    lweAddTo(temp_result2, cc, in_out_params);


//    lweNoiselessTrivial(temp_result2, AndConst, in_out_params);
//    lweAddTo(temp_result2, temp_result, in_out_params);
//    lweAddTo(temp_result2, cc, in_out_params);



    //if the phase is positive, the result is 1/8
    //if the phase is positive, else the result is -1/8
//    lweKeySwitch(result, bk->bkFFT->ks,  temp_result2);
    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result2);

    delete_LweSample(temp_result);
    delete_LweSample(temp_result2);
}


/*
 * Homomorphic bootstrapped XOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
*/
//EXPORT void
//bootsXOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk) {
//    static const Torus32 MU = modSwitchToTorus32(1, 8);
//    const LweParams *in_out_params = bk->params->in_out_params;
//
//    LweSample *temp_result = new_LweSample(in_out_params);
//
//    //compute: (0,1/4) + 2*(ca + cb)
//    static const Torus32 XorConst = modSwitchToTorus32(1, 4);
//    lweNoiselessTrivial(temp_result, XorConst, in_out_params);
//    lweAddMulTo(temp_result, 2, ca, in_out_params);
//    lweAddMulTo(temp_result, 2, cb, in_out_params);
//
//    //if the phase is positive, the result is 1/8
//    //if the phase is positive, else the result is -1/8
//    tfhe_bootstrap_FFT(result, bk->bkFFT, MU, temp_result);
//
//    delete_LweSample(temp_result);
//}

Cipher Cipher::cipherAND(Cipher a, Cipher b, TFheGateBootstrappingSecretKeySet* secretKeySet) {
    Cipher out(a.numberOfBits);
    for (int i = 0; i < 16; ++i) {
        bootsAND(&out.data[i], &a.data[i], &b.data[i], Cipher::bk);
        cout << "a[" << i <<"]: " << Cipher::decryptBitCheck(&a.data[i], secretKeySet) << " b[" << i <<"]: " << Cipher::decryptBitCheck(&b.data[i], secretKeySet);
        cout << " c[" << i <<"]: " << Cipher::decryptBitCheck(&out.data[i], secretKeySet) << endl;
    }
    return out;
}
