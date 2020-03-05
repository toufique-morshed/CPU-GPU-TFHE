## Compile and Run instruction
### GPU parallel

 - Uncomment anyline from [2773](https://github.com/tmp1370/tmpProject/blob/master/gpuParallel/main.cu#L2773)-[2787](https://github.com/tmp1370/tmpProject/blob/master/gpuParallel/main.cu#L2787)
 - **Compile:** make main
 - **Run:** ./main \<number of bits> \<first number> \<second number>
### CPU parallel
- Install thread safe version of [TFHE](https://github.com/tfhe/tfhe)
- **Compile:** g++ main.cpp -I./include/ -o main.o -ltfhe-spqlios-avx -lgomp -std=c++11 
- **Run:** ./main.o \<first number> \<second number>
- **Compile:** g++ cloud.cpp Cipher.cpp -I./include/ -o cloud.o -ltfhe-spqlios-avx -std=c++11 -fopenmp
- **Run:** ./cloud.o


# CPU and GPU Accelerated Fully Homomorphic Encryption
The framework ports [TFHE](https://tfhe.github.io/tfhe/) implementation to GPU, and extends the gates computation to algebraic, vector, and matrix operations.

Owners: Toufique Morshed, Md Momin Al Aziz, and Noman Mohammed

#### Publication (To Appear):
Toufique Morshed, Md Momin Al Aziz, and Noman Mohammed. "CPU and GPU Accelerated Fully Homomorphic Encryption." 2020 IEEE International Symposium on Hardware Oriented Security and Trust, HOST 2020.


#### Disclaimer:
The software is provided as-is with no warranty or support. We do not take any responsibility for any damage, loss of income, or any problems you might experience from using our software. If you have questions, you are encouraged to consult the paper and the source code. If you find our software useful, please cite our paper above.
