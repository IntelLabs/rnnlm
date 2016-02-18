#!/bin/bash

echo "Fetch the small dataset from rnnlm.org"

if [ ! -e rnnlm-0.4b.tgz ]; then
  echo "Downloading ..."
  wget https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U/rnnlm-0.4b.tgz
fi

if [ ! -d rnnlm-0.4b ]; then
  echo "Unzipping ..."
  tar xvzf rnnlm-0.4b.tgz
fi

cp rnnlm-0.4b/train .
cp rnnlm-0.4b/valid .
cp rnnlm-0.4b/test  .
