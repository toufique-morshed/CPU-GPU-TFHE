## Compile and Run instruction
### GPU parallel

 - Uncomment anyline from [3449](https://github.com/tmp1370/tmpProject/blob/master/gpuParallel/main.cu#L3449)-[3464](https://github.com/tmp1370/tmpProject/blob/master/gpuParallel/main.cu#L3464)
 - **Compile:** make main
 - **Run:** ./main \<number of bits> \<first number> \<second number>
### CPU parallel
- Install thread safe version of [TFHE](https://github.com/tfhe/tfhe)
- **Compile:** g++ main.cpp -I./include/ -o main.o -ltfhe-spqlios-avx -lgomp -std=c++11 
- **Run:** ./main.o \<first number> \<second number>
- **Compile:** g++ cloud.cpp Cipher.cpp -I./include/ -o cloud.o -ltfhe-spqlios-avx -std=c++11 -fopenmp
- **Run:** ./cloud.o
