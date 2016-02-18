/*
 * Copyright 2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Shihao Ji (Intel)
 */
#ifndef __RNNLM_PARAMETER_H
#define __RNNLM_PARAMETER_H

#include <string>
#include <boost/program_options.hpp>

using std::string;
using std::cerr;
using std::endl;

class parameter {
public:
    string train_file;
    string valid_file;
    string test_file;
    string rnnlm_file;
    string init_rnnlm_file;
    string test_prob_file;
    string algo_name;

    int sample_k;
    int random_seed;
    float alpha;
    int numcores;
    float lr;
    float lr_decay;
    float rmsprop_damping;
    float momentum;
    float Z;
    int batch_size;
    int hidden_size;
    int bptt_block;
    int min_count;
    int max_vocab_size;
    int max_iter;
    float gradient_cutoff;
    bool eval_unk;

    bool train_mode;
    bool test_mode;

public:
    parameter() {
        train_mode = false;
        test_mode = false;
    }

    friend std::ostream& operator<<(std::ostream& stream, const parameter& param) {
        stream << "-----------------------------\n";
        if (param.train_mode) {
            stream << "training file: " << param.train_file << "\n"
                   << "validation file: " << param.valid_file << "\n"
                   << "algorithm: " << param.algo_name << "\n"
                   << "init rnnlm file: " << param.init_rnnlm_file << "\n"
                   << "sample k: " << param.sample_k << "\n"
                   << "alpha: " << param.alpha << "\n"
                   << "lr: " << param.lr << "\n"
                   << "lr decay: " << param.lr_decay << "\n"
                   << "rmsprop damping: " << param.rmsprop_damping << "\n"
                   << "momentum: " << param.momentum << "\n"
                   << "Z of NCE: " << param.Z << "\n"
                   << "batch size: " << param.batch_size << "\n"
                   << "hidden size: " << param.hidden_size << "\n"
                   << "bptt block: " << param.bptt_block << "\n"
                   << "minimal count: " << param.min_count << "\n"
                   << "maximal vocabuary size: " << param.max_vocab_size << "\n"
                   << "maximal number of iters: " << param.max_iter << "\n"
                   << "gradient cutoff: " << param.gradient_cutoff << "\n"
                   << "random seed: " << param.random_seed << "\n";
        }
        if (param.test_mode) {
            stream << "test file: " << param.test_file << "\n"
                   << "test prob file: " << param.test_prob_file << "\n";
        }
        stream << "rnnlm file: " << param.rnnlm_file << "\n"
               << "number of cores: " << param.numcores << "\n"
               << "evaluate <unk>: " << param.eval_unk << "\n";
        stream << "-----------------------------\n";
        return stream;
    }

    int read_arguments(int argc, char **argv) {

        namespace po = boost::program_options;
        po::options_description desc("rnnlm options:");
        desc.add_options()
                ("help,h", "produce help message")
                ("train", po::value<string>(&train_file)->default_value(""), "Training data file")
                ("valid", po::value<string>(&valid_file)->default_value(""), "Validation data file")
                ("test", po::value<string>(&test_file)->default_value(""), "Test data file")
                ("rnnlm", po::value<string>(&rnnlm_file)->default_value(""), "Use this name to store trained rnnlm")
                ("init-rnnlm", po::value<string>(&init_rnnlm_file)->default_value(""), "Use this file to initialize rnnlm")
                ("algo", po::value<string>(&algo_name)->default_value("std"), "The algorithm (std, nce or blackout) used to train rnnlm")
                ("sample-k", po::value<int>(&sample_k)->default_value(0), "Number of samples for nce/blackout training")
                ("random-seed", po::value<int>(&random_seed)->default_value(1), "Random seed of random number generator")
                ("Z", po::value<float>(&Z)->default_value(1.f), "NCE's normalization parameter Z")
                ("alpha", po::value<float>(&alpha)->default_value(0.f), "The power of the proposal distribution")
                ("numcores", po::value<int>(&numcores)->default_value(4), "Number of cores")
                ("lr", po::value<float>(&lr)->default_value(0.1f), "The initial learning rate")
                ("lr-decay", po::value<float>(&lr_decay)->default_value(1.f), "The decay rate of learning rate")
                ("rmsprop-damping", po::value<float>(&rmsprop_damping)->default_value(0.01f), "Rmsprop's damping factor")
                ("momentum", po::value<float>(&momentum)->default_value(0.f), "Momentum for SGD")
                ("batch-size", po::value<int>(&batch_size)->default_value(1), "Mini-batch size for SGD")
                ("hidden", po::value<int>(&hidden_size)->default_value(16), "Size of hidden layer")
                ("bptt-block", po::value<int>(&bptt_block)->default_value(10), "Number of time steps over which the error is backpropagated")
                ("min-count", po::value<int>(&min_count)->default_value(1), "When building vocabulary, discard words that appear less than this number of times")
                ("max-vocab-size", po::value<int>(&max_vocab_size)->default_value(INT_MAX), "maximum vocabulary size (including </s> and <unk>)")
                ("max-iter", po::value<int>(&max_iter)->default_value(50), "Maximum number of iterations")
                ("gradient-cutoff", po::value<float>(&gradient_cutoff)->default_value(5.f), "Maximal norm of gradient for gradient clipping")
                ("eval-unk", po::bool_switch(&eval_unk)->default_value(false), "Evaluate <unk>'s when evaluating rnnlm on validation and test data")
                ("test-prob", po::value<string>(&test_prob_file)->default_value("test-prob"), "Save the probability of each token from test data to this file");

        bool flagHelp = false;
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help") || argc == 1) {
            flagHelp = true;
        }

        if (flagHelp == true) {
            cerr << desc << endl;
            return 1;
        }

        if (train_file.length() > 0) {
            train_mode = true;
        }

        if (test_file.length() > 0) {
            test_mode = true;
        }

        if (train_mode && valid_file.length() == 0) {
            cerr << "Validation data file isn't specified for training!" << endl;
            return 1;
        }

        if (train_mode && rnnlm_file.length() == 0) {
            cerr << "Rnnlm file isn't specified for training!" << endl;
            return 1;
        }

        if (test_mode && rnnlm_file.length() == 0) {
            cerr << "Rnnlm file isn't specified for testing!" << endl;
            return 1;
        }

        if (!train_mode && !test_mode) {
            cerr << "Neither training file nor test file is specified!" << endl;
            return 1;
        }

        return 0;
    }

};

#endif
