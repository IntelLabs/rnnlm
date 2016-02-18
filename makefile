CC = icpc
CFLAGS = -std=c++11 -O3 -openmp -lboost_program_options -Wall -Wno-sign-compare -mkl -xhost #-fimf-precision=low
OPT_DEF = 

all: rnnlm

rnnlm: rnnlm.cpp rnnlmlib.cpp rnnlmlib.hpp parameter.hpp
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlm.cpp rnnlmlib.cpp -o rnnlm

clean:
	rm -rf rnnlm
