#oregon version
g++ main.cpp -o main.o -ltfhe-spqlios-fma -std=c++11 && ./main.o
g++ cloud.cpp Cipher.cpp -o cloud.o -ltfhe-spqlios-avx -std=c++11 -fopenmp && ./cloud.o




g++ verify.cpp -o verify.o -ltfhe-spqlios-avx -std=c++11

g++ data_owner.cpp -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o boston.txt boston_encrypted.data 506 13 1000


g++ cloudComputation.cpp Cipher.cpp -o cloudComputation.o -ltfhe-spqlios-avx -std=c++11 -fopenmp && ./cloudComputation.o boston_encrypted.data boston_linear_model 506 13 10 10 1

#edin.txt
g++ data_owner.cpp -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o edin.txt edin_encrypted.data 1253 9 1000
g++ cloudComputation.cpp Cipher.cpp -o cloudComputation.o -ltfhe-spqlios-avx -std=c++11 -fopenmp && ./cloudComputation.o edin_encrypted.data edin_linear_model 1253 9 4 10 1 3

#fertility data
g++ data_owner.cpp -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o fertility.txt fertility_encrypted.data 100 9 1000
#bank note data
g++ data_owner.cpp -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o banknote.txt banknote_encrypted.data 1372 4 1000
#lbwdata
g++ data_owner.cpp -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o lbw.txt lbw_encrypted.data 189 9 1000
#cancerData
g++ data_owner.cpp -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o cancer.txt cancer_encrypted.data 1421 18 1000

#danta version
g++ main.cpp -I./include/ -o main.o -ltfhe-spqlios-avx -std=c++11 && ./main.o
g++ main.cpp -I./include/ -o main.o -ltfhe-spqlios-avx -lgomp -std=c++11 && ./main.o 10 2
g++ cloud.cpp Cipher.cpp -I./include/ -o cloud.o -ltfhe-spqlios-avx -std=c++11 -fopenmp && ./cloud.o
#edin
g++ data_owner.cpp -I./include/ -o data_owner.o -ltfhe-spqlios-avx -std=c++11 && ./data_owner.o edin.txt edin_encrypted.data 1253 9 1000