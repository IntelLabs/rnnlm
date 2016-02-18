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
#ifndef __RNNLM_RNNLMLIB_H_
#define __RNNLM_RNNLMLIB_H_

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <climits>
#include <numeric>
#include <random>
#include "mkl.h"
#include "parameter.hpp"

using namespace std;

typedef float scalar;
typedef mt19937_64 rng_type;

enum class algo_type {
    std, nce, blackout
};

class rnnlm {
protected:

    string train_file;
    string valid_file;
    string test_file;
    string rnnlm_file;
    string init_rnnlm_file;
    string test_prob_file;
    algo_type algo;

    int sample_k;
    int random_seed;
    scalar Z;
    scalar alpha;
    int numcores;
    scalar lr;
    scalar lr_decay;
    scalar rmsprop_damping;
    scalar momentum;
    int batch_size;
    int hidden_size;
    int bptt_block;
    int min_count;
    int max_vocab_size;
    int max_iter;
    scalar gradient_cutoff;
    bool eval_unk;


    int train_words;
    unordered_map<string, unsigned int> vocab;
    unsigned int vocab_size;
    unsigned int nEnds;
    unsigned int nUnkOccs;
    unsigned int nUnkTokens;
    unsigned int oov;
    int input_size;
    int output_size;

    vector<vector<int>> trainStreams;
    vector<vector<int>> validateData;
    vector<vector<int>> testData;
    vector<scalar> proposal_PMF;
    vector<scalar> proposal_CMF;
    vector<scalar> input_moPow;
    vector<scalar> output_moPow;
    vector<vector<unsigned int>> vocab_idx;
    scalar *hidden;
    scalar *output;
    scalar *errorH;
    scalar *errorH2;
    scalar *errorH3;
    scalar *Wih; //weights between input and hidden layer
    scalar *Who; //weights between hidden and output layer
    scalar *Wr;  //weights of recurrent matrix
    scalar *dWih; // gradients
    scalar *dWho;
    scalar *dWr;
    scalar *vWih; // for rmsprop
    scalar *vWho;
    scalar *vWr;

    // data for approximate algo
    scalar *output_samples;
    scalar *Who_samples;
    scalar *dWho_samples;
    unsigned int *sample_indices;
    unsigned int *target_indices;

    rng_type* rngs;
    uniform_real_distribution<scalar> uniform_real_dist;

public:

    rnnlm(parameter& param) {

        train_file = param.train_file;
        valid_file = param.valid_file;
        test_file = param.test_file;
        rnnlm_file = param.rnnlm_file;
        init_rnnlm_file = param.init_rnnlm_file;
        test_prob_file = param.test_prob_file;

        string algo_name = param.algo_name;
        if (algo_name == "std") {
            algo = algo_type::std;
        } else if (algo_name == "nce") {
            algo = algo_type::nce;
        } else if (algo_name == "blackout") {
            algo = algo_type::blackout;
        } else {
            cerr << "unsupported algorithm: " << algo_name << endl;
            exit(1);
        }

        sample_k = param.sample_k;
        random_seed = param.random_seed;
        Z = param.Z;
        alpha = param.alpha;
        numcores = param.numcores;
        lr = param.lr;
        lr_decay = param.lr_decay;
        rmsprop_damping = param.rmsprop_damping;
        momentum = param.momentum;
        batch_size = param.batch_size;
        hidden_size = param.hidden_size;
        bptt_block = param.bptt_block;
        min_count = param.min_count;
        max_vocab_size = param.max_vocab_size;
        max_iter = param.max_iter;
        gradient_cutoff = param.gradient_cutoff;
        eval_unk = param.eval_unk;

        hidden = NULL;
        output = NULL;
        errorH = NULL;
        errorH2 = NULL;
        errorH3 = NULL;
        Wih = NULL;
        Who = NULL;
        Wr = NULL;
        dWih = NULL;
        dWho = NULL;
        dWr = NULL;
        vWih = NULL;
        vWho = NULL;
        vWr = NULL;
        output_samples = NULL;
        Who_samples = NULL;
        dWho_samples = NULL;
        sample_indices = NULL;
        target_indices = NULL;
        rngs = NULL;

        mkl_set_num_threads(numcores);
    }

    ~rnnlm()
    {
        if (hidden != NULL) _mm_free(hidden);
        if (output != NULL) _mm_free(output);
        if (errorH != NULL) _mm_free(errorH);
        if (errorH2 != NULL) _mm_free(errorH2);
        if (errorH3 != NULL) _mm_free(errorH3);
        //
        if (Wih != NULL) _mm_free(Wih);
        if (Who != NULL) _mm_free(Who);
        if (Wr != NULL) _mm_free(Wr);
        if (dWih != NULL) _mm_free(dWih);
        if (dWho != NULL) _mm_free(dWho);
        if (dWr != NULL) _mm_free(dWr);
        if (vWih != NULL) _mm_free(vWih);
        if (vWho != NULL) _mm_free(vWho);
        if (vWr != NULL) _mm_free(vWr);
        //
        if (output_samples != NULL) _mm_free(output_samples);
        if (Who_samples != NULL) _mm_free(Who_samples);
        if (dWho_samples != NULL) _mm_free(dWho_samples);
        if (sample_indices != NULL) _mm_free(sample_indices);
        if (target_indices != NULL) _mm_free(target_indices);
        if (rngs != NULL) delete[] rngs;
    }

    inline scalar gradient_clipping(scalar v) {
        if (v > gradient_cutoff)
            v = gradient_cutoff;
        else if (v < -gradient_cutoff)
            v = -gradient_cutoff;
        return v;
    }

    void learnVocabFromFile(string filename);
    void loadTrainingStreams(string filename, vector<vector<int>>& streams);
    //void marshalTrainingStreams(const vector<vector<int>>& data, vector<vector<int>>& stream);
    void loadVTSentences(string filename, vector<vector<int>>& data);
    string trim(const string& str, const string& whitespace = " \t");

    void initNet();
    void initFromRandom();
    void saveNet(string filename);
    void loadNet(string filename);
    void pruneVocab();
    void buildMapPMFCMF();
    void samplingFromCMF(vector<scalar>& cdf, unsigned int k, vector<vector<unsigned int>>& samples);
    void indicesTobeUpdated(vector<vector<int>>& stream, unsigned int sliceOffset, vector<vector<unsigned int>>& samples,
            vector<unsigned int>& input_indices, vector<unsigned int>& output_indices);

    scalar forwardPropagation(vector<vector<int>>& stream, unsigned int sliceOffset);
    void backPropagation(vector<vector<int>>& stream, unsigned int sliceOffset);
    void fullModelUpdate(unsigned int numWordsInBatch);
    void subModelUpdate(vector<unsigned int>& input_indices, vector<unsigned int>& output_indices, unsigned int numWordsInBatch);
    void inference(const int *input, const int sentence_length, vector<scalar>& logl);
    scalar nce_forwardPropagation(vector<vector<int>>& stream, unsigned int sliceOffset, vector<vector<unsigned int>>& samples);
    scalar blackout_forwardPropagation(vector<vector<int>>& stream, unsigned int sliceOffset, vector<vector<unsigned int>>& samples);
    void train();
    void test();
    void saveTestProb(string filename, vector<vector<scalar>>& prob);
};

#endif
