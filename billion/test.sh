#!/bin/bash

path=../data
ncores=28

cp model model2

../rnnlm --rnnlm model2 --test $path/billion.te --numcores $ncores --test-prob test-prob
