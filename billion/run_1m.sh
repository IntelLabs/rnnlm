#!/bin/bash

path=../data
ncores=28 # please set it to number of physical cores
hidden=2048

# rnnlm training
export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
numactl --interleave=all ../rnnlm --train $path/billion.tr --valid $path/billion.te --rnnlm model --hidden $hidden --random-seed 1 --min-count 1 --max-vocab-size 1000000 --batch-size 128 --bptt-block 20 --momentum 0.9995 --lr 0.0005 --lr-decay 0.8 --rmsprop-damping 1e-6 --max-iter 30 --gradient-cutoff 1 --numcores $ncores --algo blackout --alpha 0.1 --sample-k 2000

# rnnlm test
../rnnlm --rnnlm model --test $path/billion.te --numcores $ncores --test-prob test-prob
