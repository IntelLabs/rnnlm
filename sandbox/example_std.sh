#!/bin/bash

path=../data

# rnnlm training
export KMP_AFFINITY=explicit,proclist=[0-3],granularity=fine
numactl --interleave=all ../rnnlm --train $path/train --valid $path/valid --rnnlm model --hidden 16 --random-seed 1 --min-count 1 --max-vocab-size 3720 --batch-size 8 --momentum 0.999 --bptt-block 15 --lr 0.01 --lr-decay 0.999 --rmsprop-damping 1e-3 --max-iter 50 --gradient-cutoff 1 --numcores 4 --algo std

# rnnlm test
../rnnlm --rnnlm model --test $path/test --numcores 4 --test-prob test-prob

