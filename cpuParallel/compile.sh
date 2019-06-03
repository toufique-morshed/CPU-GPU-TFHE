g++ main.cpp -I./include/ -o main.o -ltfhe-spqlios-avx -lgomp -std=c++11 && ./main.o 10 2
g++ cloud.cpp Cipher.cpp -I./include/ -o cloud.o -ltfhe-spqlios-avx -std=c++11 -fopenmp && ./cloud.o
