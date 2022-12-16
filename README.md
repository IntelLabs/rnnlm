# RNNLM Toolkit

> :warning: **DISCONTINUATION OF PROJECT** - 
> *This project will no longer be maintained by Intel.
> Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.*
> **Intel no longer accepts patches to this project.**
> *If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.*


This is a C++ implementation of RNNLM toolkit that supports three algorithms: standard RNNLM, NCE and BlackOut as described in "[Blackout: Speeding up recurrent neural network language models with very large vocabularies](http://arxiv.org/abs/1511.06909), ICLR 2016".

## License
All source code files in the package are under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Prerequisites
The code is developed and tested on UNIX-based systems with the following software dependencies:

- [Intel Compiler](https://software.intel.com/en-us/qualify-for-free-software) (The code is optimized on Intel CPUs)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has much better support for tall-skinny sgemms)
- Boost library (at least 1.49)
- Numactl package (for multi-socket NUMA systems)

## Environment Setup
* Install Intel C++ development environment (i.e., Intel compiler, OpenMP, MKL "16.0.0 or higher". [free copies](https://software.intel.com/en-us/qualify-for-free-software) are available for some users)
* Enable Intel C++ development environment
```
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64 (pointing to the path of your installation)
```
* Install Boost library
```
sudo yum install boost-devel (on RedHat/Centos)
sudo apt-get install libboost-all-dev (on Ubuntu)
```
* Install numactl package
```
sudo yum install numactl (on RedHat/Centos)
sudo apt-get install numactl (on Ubuntu)
```

## Quick Start
1. Download the code: ```git clone https://github.com/IntelLabs/rnnlm```
2. Compile the code: ```make clean all```
3. Download the data: ```cd data; .\getsmall.sh or .\get1billion.sh```
4. Run the demo script: ```cd sandbox; ./example_blackout.sh```
5. Run the code on the 1-billion-word-benchmark: ```cd billion; ./run_64k.sh or ./run_800k.sh or ./run_1m.sh (please set the ncores=number of physical cores of your machine)```

## Reference
Shihao Ji, S. V. N. Vishwanathan, Nadathur Satish, Michael J. Anderson, Pradeep Dubey, [Blackout: Speeding up recurrent neural network language models with very large vocabularies](http://arxiv.org/abs/1511.06909), in International Conference on Learning Representations (ICLR'16), 2016.

For questions and bug reports, you can reach me at https://cs.gsu.edu/~sji/
