#!/bin/bash

echo "Fetch the small dataset from rnnlm.org"

if [ ! -e rnnlm-0.3e.tgz ]; then
  echo "Downloading ..."
  wget http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-0.3e.tgz
fi

if [ ! -d rnnlm-0.3e ]; then
  echo "Unzipping ..."
  mkdir -p rnnlm-0.3e && tar xvzf rnnlm-0.3e.tgz -C rnnlm-0.3e
fi

cp rnnlm-0.3e/train .
cp rnnlm-0.3e/valid .
cp rnnlm-0.3e/test  .
