#!/bin/bash

echo "Fetch the 1 billion word language modeling benchmark"

if [ ! -e 1-billion-word-language-modeling-benchmark-r13output.tar.gz ]; then
  echo "Downloading ..."
  wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
fi

if [ ! -d 1-billion-word-language-modeling-benchmark-r13output ]; then
  echo "Unzipping ..."
  tar xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
fi

cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* > billion.tr
cat 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050 > billion.te
