/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>
using namespace cufhe;

#include <iostream>
using namespace std;

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

void OrCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) > 0;
}

void AndCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = in0.message_ * in1.message_;
}

void XorCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) & 0x1;
}

void addBits(Ctxt *r, Ctxt &a, Ctxt &b, Ctxt *carry) {
	Ctxt *t1 = new Ctxt[1];
    Ctxt *t2 = new Ctxt[1];
	Xor(t1[0], a, carry[0]);
    Xor(t2[0], b, carry[0]);
	Synchronize();
	Xor(r[0], a, t2[0]);
	And(t1[0], t1[0], t2[0]);
	Synchronize();
	Xor(r[1], carry[0], t1[0]);
	Synchronize();
	delete [] t1;
	delete [] t2;
}


void addNumbers(Ctxt *ctRes, Ctxt *ctA, Ctxt *ctB, int nBits) {
	
	Ctxt *carry = new Ctxt[1];
    Ctxt *bitResult = new Ctxt[2];

	Xor(ctRes[0], ctA[0], ctB[0]);
	And(carry[0], ctA[0], ctB[0]);
	Synchronize();
	for(int i = 1; i < nBits; i++) {
		addBits(bitResult, ctA[i], ctB[i], carry);
		Copy(ctRes[i], bitResult[0]);
		Copy(carry[0], bitResult[1]);
		Synchronize();
	}
	delete [] carry;
	delete [] bitResult;
}

void mulNumbers(Ctxt *ctRes, Ctxt *ctA, Ctxt *ctB, int iBits, int oBits, PriKey pri_key) {
	
	Ctxt *tempSum = new Ctxt[oBits];
	Ctxt *andRes = new Ctxt[iBits];
	Ptxt* pt = new Ptxt[0];
	
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	uint32_t kNumSMs = prop.multiProcessorCount;
	Stream* st = new Stream[kNumSMs];
	for (int i = 0; i < kNumSMs; i ++) {
		st[i].Create();
	}
	
	//cout << endl;
	//cout << endl;
	//cout << "***" << endl;
	for(int i = 0; i < iBits; i ++) {
		Ctxt *andResLeft = new Ctxt[oBits];
		
		/*Decrypt(pt[0], ctB[i], pri_key);
		cout << pt[0].message_ << endl;*/
		
		for(int j = 0; j < iBits; j++) {
			And(andRes[j], ctA[j], ctB[i], st[j % kNumSMs]);
		}	
		Synchronize();
		for(int j = 0; j < iBits; j++) {
			Copy(andResLeft[j + i], andRes[j]);
		}
		Synchronize();		
		/*for(int j = 0; j < oBits; j++) {			
			Decrypt(pt[0], andResLeft[j], pri_key);
			cout << pt[0].message_;
		}
		cout << endl;*/

		addNumbers(tempSum, andResLeft, tempSum, oBits);
		/*Synchronize();
		cout << "sum: ";
		for(int j = 0; j < oBits; j++) {
			Decrypt(pt[0], tempSum[j], pri_key);
			Synchronize();
			cout << pt[0].message_;
		}
		cout << endl;
		cout << endl;*/	
		delete [] andResLeft;		
	}
	
	for (int i = 0; i < kNumSMs; i ++)
		st[i].Destroy();
	delete [] st;
  
	//cout << "***" << endl;
	for(int i=0; i < oBits; i ++) {
		Copy(ctRes[i], tempSum[i]);
	}
	Synchronize();
	delete [] tempSum;
	delete [] andRes;
}

void testAND(Ctxt *ctRes, Ctxt *ctA, Ctxt *ctB, int nBits, int nStreams) {
	Stream* st = new Stream[nStreams];
	for (int i = 0; i < nStreams; i ++)
		st[i].Create();
	
	for (int i = 0; i < nBits; i ++) {
		And(ctRes[i], ctA[i], ctB[i], st[i % nStreams]);
	}
	Synchronize();
	
	for (int i = 0; i < nStreams; i ++)
		st[i].Destroy();
	delete [] st;
}

int main(int argc, char** argv) {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t kNumSMs = prop.multiProcessorCount;
  uint32_t kNumTests = kNumSMs * 32;// * 8;
  uint32_t kNumLevels = 4;
  uint32_t val1 = 1;
  uint32_t val2 = 2;
  int nBits = 16;

  SetSeed(); // set random seed

  PriKey pri_key; // private key
  PubKey pub_key; // public key
  Ptxt* pt = new Ptxt[nBits];
  Ctxt* ct = new Ctxt[nBits];
  Ptxt* pt1 = new Ptxt[nBits];
  Ctxt* ct1 = new Ctxt[nBits];
  Ptxt* ptRes = new Ptxt[nBits * 2];
  Ctxt* ctRes = new Ctxt[nBits * 2];
  
  cout<< "------ Key Generation ------" <<endl;
  KeyGen(pub_key, pri_key);

  
  for (int i = 0; i < nBits; i ++) {
    pt[i].message_ = (val1 >> i) & 1;//rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[i], pt[i], pri_key);
    Decrypt(pt[i], ct[i], pri_key);
    cout << pt[i].message_;
  }
  cout << endl;  
  
  for (int i = 0; i < nBits; i ++) {
    pt1[i].message_ = (val2 >> i) & 1;//rand() % Ptxt::kPtxtSpace;
    Encrypt(ct1[i], pt1[i], pri_key);
    Decrypt(pt1[i], ct1[i], pri_key);
    cout << pt1[i].message_;
  }
  cout << endl;
  
  Initialize(pub_key); // essential for GPU computing
  
  for(int i = 1; i <= 8; i++) {
	float et;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//addNumbers(ctRes, ct, ct1, nBits);
	//mulNumbers(ctRes, ct, ct1, nBits, nBits * 2, pri_key);
	//AND test

	int nBitsAndTest = i * 4;
	Ctxt* ctAND0 = new Ctxt[nBitsAndTest];
	Ctxt* ctAND1 = new Ctxt[nBitsAndTest];
	Ctxt* ctANDRes = new Ctxt[nBitsAndTest];
	testAND(ctANDRes, ctAND0, ctAND1, nBitsAndTest, kNumSMs);



	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);

	cout << "nBits: " << nBitsAndTest << " op et: " << et << endl;
  }
  
  for (int i = 0; i < nBits * 2; i ++) {
	  Decrypt(ptRes[i], ctRes[i], pri_key);
		cout << ptRes[i].message_;
  }
  cout << endl;
  



  //CleanUp(); // essential to clean and deallocate data
  delete [] ct;
  delete [] pt;
  return 0;
}