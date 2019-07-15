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
