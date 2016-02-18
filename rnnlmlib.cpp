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
#include <cstring>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include "rnnlmlib.hpp"

void rnnlm::pruneVocab() {

    unsigned int thres = 0, offset = 0;
    if (max_vocab_size < vocab.size() + 2) {
        vector<unsigned int> occurrences;
        for (unordered_map<string, unsigned int>::iterator it = vocab.begin(); it != vocab.end(); it++) {
            occurrences.push_back(it->second);
        }
        nth_element(occurrences.begin(), occurrences.begin() + max_vocab_size - 3, occurrences.end(),
                [](unsigned int a, unsigned int b) {return a > b;});
        thres = occurrences[max_vocab_size - 3];
        int nGreats = count_if(occurrences.begin(), occurrences.begin() + max_vocab_size - 3,
                [&thres](unsigned int a)->bool {return a > thres;});
        offset = max_vocab_size - 2 - nGreats;
    }

    bool flag = true;
    if (min_count > thres) {
        thres = min_count;
        flag = false;
    }

    nUnkOccs = 0;
    nUnkTokens = 0;
    for (unordered_map<string, unsigned int>::iterator it = vocab.begin(); it != vocab.end();) {
        if (it->second < thres || (flag && it->second == thres && offset <= 0)) {
            nUnkOccs += it->second;
            nUnkTokens++;
            it = vocab.erase(it);
        } else {
            if (flag && it->second == thres && offset > 0) {
                offset--;
            }
            it++;
        }
    }

    vocab_size = vocab.size() + 2; // add </s> and unk
}

void rnnlm::buildMapPMFCMF()
{
    proposal_PMF.clear();
    proposal_CMF.clear();
    input_moPow.clear();
    output_moPow.clear();
    proposal_PMF.resize(vocab_size);
    proposal_CMF.resize(vocab_size);
    input_moPow.resize(vocab_size);
    output_moPow.resize(vocab_size);

    vector<scalar> unigram_PMF(vocab_size);
    unigram_PMF[0] = nEnds; // for </s>
    double norm = nEnds;
    unsigned int index = 1;
    for (unordered_map<string, unsigned int>::iterator it = vocab.begin(); it != vocab.end();) {
        unigram_PMF[index] = it->second;
        norm += it->second;
        it->second = index;
        index++;
        it++;
    }
    vocab["</s>"] = 0;
    unigram_PMF[index] = nUnkOccs; // for unk
    norm += nUnkOccs;

    // normalization 1
    double norm2 = 0.;
    const double normr = 1.0 / norm;
    #pragma omp parallel for simd num_threads(numcores) reduction(+: norm2)
    for (int i = 0; i < vocab_size; i++) {
        scalar p = unigram_PMF[i] * normr;
        unigram_PMF[i] = p;
        p = powf(p, alpha);
        proposal_PMF[i] = p;
        norm2 += p;
    }

    // normalization 2
    const double norm2r = 1.0 / norm2;
    scalar p = proposal_PMF[0] * norm2r;
    proposal_PMF[0] = p;
    proposal_CMF[0] = p;
    for (int i = 1; i < vocab_size; i++) {
        p = proposal_PMF[i] * norm2r;
        proposal_PMF[i] = p;
        proposal_CMF[i] = proposal_CMF[i - 1] + p;
    }

    #pragma omp parallel for simd num_threads(numcores)
    for (int i = 0; i < vocab_size; i++) {
        input_moPow[i] = powf(momentum, 1.f / (unigram_PMF[i] * batch_size * bptt_block));
        output_moPow[i] = powf(momentum, 1.f / (unigram_PMF[i] * batch_size * bptt_block + proposal_PMF[i] * sample_k * bptt_block));
    }
    //cout << "max proposal cmf: " << proposal_CMF[vocab_size - 1] << endl;
    //cout << "max proposal pmf: " << *max_element(proposal_PMF.begin(), proposal_PMF.end()) << endl;
    //cout << "min proposal pmf: " << *min_element(proposal_PMF.begin(), proposal_PMF.end()) << endl;
}


void rnnlm::learnVocabFromFile(string filename)
{
    // assume the text is tokenized already
    cout << "construct vocabulary from file: " << filename << endl << flush;

    ifstream file;
    file.open(filename, ios_base::in);

    if (!file.is_open()) {
        cerr << "failed to open file " << filename << endl;
        exit(1);
    }

    vocab.clear();

    int train_wcn = 0;
    int nlines = 0;
    string line;
    while (!file.eof()) {
        getline(file, line);
        //line = trim(line);
        if (line.size()) {
            size_t cur = 0, prev = 0;
            while (cur != string::npos) {
                cur = line.find(' ', prev);
                string word = line.substr(prev, cur - prev);
                if (vocab.find(word) == vocab.end())
                    vocab[word] = 1;
                else
                    vocab[word]++;
                prev = cur + 1;
                train_wcn++;
            }
            nlines++;
            train_wcn++; // add </s> for each line
        }
    }
    file.close();

    nEnds = nlines;
    pruneVocab();

    oov = vocab_size - 1;
    cout << "-->vocab size (including </s> and <unk>): " << vocab_size << endl;
    cout << "-->words in the file (including </s> and <unk>): " << train_wcn << endl << flush;

    train_words = train_wcn;
}


void rnnlm::loadTrainingStreams(string filename, vector<vector<int>>& streams)
{
    cout << "load streams from file: " << filename << endl << flush;

    ifstream file;
    file.open(filename, ios_base::in);

    if (!file.is_open()) {
        cerr << "failed to open file " << filename << endl;
        exit(1);
    }

    streams.clear();
    streams.resize(batch_size, vector<int>(1, 0));
    vector<unsigned int> streamLength(batch_size, 1);

    unsigned int numSentences = 0;
    vector<int> sentence;
    string line;

    while (!file.eof()) {
        getline(file, line);
        //line = trim(line);
        if (line.size()) {
            sentence.clear();
            size_t cur = 0, prev = 0;
            while (cur != string::npos) {
                cur = line.find(' ', prev);
                string word = line.substr(prev, cur - prev);
                if (vocab.find(word) == vocab.end())
                    sentence.push_back(oov);
                else
                    sentence.push_back(vocab[word]);
                prev = cur + 1;
            }
            sentence.push_back(0); // add </s> for each line
            numSentences++;
            // find a shortest stream and append the sentence to it
            int shortestStreamIdx = distance(streamLength.begin(), min_element(streamLength.begin(), streamLength.end()));
            vector<int> &s = streams[shortestStreamIdx];
            s.insert(s.end(), sentence.begin(), sentence.end());
            streamLength[shortestStreamIdx] += sentence.size();
        }
    }
    file.close();

    cout << "-->number of non-empty sentences from training file: " << numSentences << endl;
}


//void rnnlm::marshalTrainingStreams(const vector<vector<int>>& sentences, vector<vector<int>>& streams)
//{
//    streams.clear();
//    streams.resize(batch_size, vector<int>(1, 0));
//    vector<unsigned int> streamLength(batch_size, 1);
//
//    // shuffle sentences
//    random_shuffle(sentenceIndex.begin(), sentenceIndex.end());
//
//    for (int i = 0; i < sentences.size(); i++) {
//        int shortestStreamIdx = distance(streamLength.begin(), min_element(streamLength.begin(), streamLength.end()));
//        vector<int> &s = streams[shortestStreamIdx];
//        int idx = sentenceIndex[i];
//        s.insert(s.end(), sentences[idx].begin(), sentences[idx].end());
//        streamLength[shortestStreamIdx] += sentences[idx].size();
//    }
//}


void rnnlm::loadVTSentences(string filename, vector<vector<int>>& data)
{
    cout << "load sentences from file: " << filename << endl << flush;

    ifstream file;
    file.open(filename, ios_base::in);

    if (!file.is_open()) {
        cerr << "failed to open file " << filename << endl;
        exit(1);
    }

    data.clear();
    string line;
    vector<int> sentence;

    while (!file.eof()) {
        getline(file, line);
        //line = trim(line);
        if (line.size()) {
            sentence.clear();
            sentence.push_back(0);
            size_t cur = 0, prev = 0;
            while (cur != string::npos) {
                cur = line.find(' ', prev);
                string word = line.substr(prev, cur - prev);
                if (vocab.find(word) == vocab.end())
                    sentence.push_back(oov);
                else
                    sentence.push_back(vocab[word]);
                prev = cur + 1;
            }
            sentence.push_back(0); // add </s> for each line
            data.push_back(sentence);
        }
    }
    file.close();

    cout << "-->number of non-empty sentences from the file: " << data.size() << endl;
}

string rnnlm::trim(const string& str, const string& whitespace)
{
    const size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == string::npos)
        return ""; // no content

    const size_t strEnd = str.find_last_not_of(whitespace);
    const size_t strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

void rnnlm::initNet() {
    if (init_rnnlm_file.length() > 0) {
        loadNet(init_rnnlm_file);
    } else {
        initFromRandom();
    }
}

void rnnlm::initFromRandom() {

    input_size = vocab_size;
    output_size = vocab_size;
    int output_sample_size = sample_k + batch_size;

    hidden = (scalar *) _mm_malloc((bptt_block + 1) * batch_size * hidden_size * sizeof(scalar), 64);
    errorH = (scalar *) _mm_malloc(bptt_block * batch_size * hidden_size * sizeof(scalar), 64);
    errorH2 = (scalar *) _mm_malloc(batch_size * hidden_size * sizeof(scalar), 64);
    errorH3 = (scalar *) _mm_malloc(batch_size * hidden_size * sizeof(scalar), 64);
    output = (scalar *) _mm_malloc(batch_size * output_size * sizeof(scalar), 64);
    Wih = (scalar *) _mm_malloc(input_size * hidden_size * sizeof(scalar), 64);
    Who = (scalar *) _mm_malloc(hidden_size * output_size * sizeof(scalar), 64);
    Wr = (scalar *) _mm_malloc(hidden_size * hidden_size * sizeof(scalar), 64);
    dWih = (scalar *) _mm_malloc(input_size * hidden_size * sizeof(scalar), 64);
    dWho = (scalar *) _mm_malloc(hidden_size * output_size * sizeof(scalar), 64);
    dWr = (scalar *) _mm_malloc(hidden_size * hidden_size * sizeof(scalar), 64);
    vWih = (scalar *) _mm_malloc(input_size * hidden_size * sizeof(scalar), 64);
    vWho = (scalar *) _mm_malloc(hidden_size * output_size * sizeof(scalar), 64);
    vWr = (scalar *) _mm_malloc(hidden_size * hidden_size * sizeof(scalar), 64);
    //
    sample_indices = (unsigned int *) _mm_malloc(output_sample_size * bptt_block * sizeof(unsigned int), 64);
    target_indices = (unsigned int *) _mm_malloc(batch_size * bptt_block * sizeof(unsigned int), 64);
    output_samples = (scalar *) _mm_malloc(batch_size * output_sample_size * sizeof(scalar), 64);
    Who_samples = (scalar *) _mm_malloc(hidden_size * output_sample_size * sizeof(scalar), 64);
    dWho_samples = (scalar *) _mm_malloc(hidden_size * output_sample_size * sizeof(scalar), 64);

    if (!hidden || !errorH || !errorH2 || !errorH3 || !output || !Wih || !Who ||
            !Wr || !dWih || !dWho || !dWr || !vWih || !vWho || !vWr || !output_samples ||
            !Who_samples || !dWho_samples || !sample_indices || !target_indices) {
        cout << "memory allocation failed" << endl;
        exit(1);
    }

    scalar r = 0.08f;
    rng_type rng = rng_type(random_seed);
    uniform_real_distribution<scalar> dist(-r, r);

    for (int x = 0; x < input_size * hidden_size; x++) {
        Wih[x] = dist(rng);
    }

    for (int x = 0; x < output_size * hidden_size; x++) {
        Who[x] = dist(rng);
    }

    for (int x = 0; x < hidden_size * hidden_size; x++) {
        Wr[x] = dist(rng);
    }

    memset(vWih, 0.f, input_size * hidden_size * sizeof(scalar));
    memset(vWho, 0.f, hidden_size * output_size * sizeof(scalar));
    memset(vWr, 0.f, hidden_size * hidden_size * sizeof(scalar));
}


void rnnlm::samplingFromCMF(vector<scalar>& cmf, unsigned int k, vector<vector<unsigned int>>& samples)
{
    // always include <unk> in training and make sure the first 2 samples are different
    #pragma omp parallel for num_threads(numcores)
    for (int t = 0; t < bptt_block; t++) {
        // 1st sample
        samples[t][0] = oov;
        // 2nd sample
        while (1) {
            scalar v = uniform_real_dist(rngs[t]);
            unsigned int token = upper_bound(cmf.begin(), cmf.end(), v) - cmf.begin();
            if (token != oov) {
                samples[t][1] = token;
                break;
            }
        }
        // the rest of samples
        for (int i = 2; i < k; i++) {
            scalar v = uniform_real_dist(rngs[t]);
            unsigned int token = upper_bound(cmf.begin(), cmf.end(), v) - cmf.begin();
            samples[t][i] = token;
        }
    }
}


void rnnlm::indicesTobeUpdated(vector<vector<int>>& stream, unsigned int sliceOffset, vector<vector<unsigned int>>& samples,
        vector<unsigned int>& input_indices, vector<unsigned int>& output_indices)
{
    static vector<unsigned int> input(batch_size * bptt_block);
    static vector<unsigned int> output((batch_size + sample_k) * bptt_block);

    unsigned int bptt_block_bytes = bptt_block * sizeof(unsigned int);
    unsigned int sample_k_bytes = sample_k * sizeof(unsigned int);

    #pragma omp parallel for num_threads(numcores)
    for (unsigned int b = 0; b < batch_size; b++) {
        memcpy(&input[b * bptt_block], &stream[b][sliceOffset], bptt_block_bytes);
        memcpy(&output[b * bptt_block], &stream[b][sliceOffset + 1], bptt_block_bytes);
    }
    #pragma omp parallel for num_threads(numcores)
    for (int t = 0; t < bptt_block; t++) {
        memcpy(&output[batch_size * bptt_block + t * sample_k], &samples[t][0], sample_k_bytes);
    }

    #pragma omp parallel sections num_threads(numcores)
    {
        #pragma omp section
        {
            unordered_set<unsigned int> input_set(input.begin(), input.end());
            input_indices.clear();
            input_indices.insert(input_indices.end(), input_set.begin(), input_set.end());
        }
        #pragma omp section
        {
            unordered_set<unsigned int> output_set(output.begin(), output.end());
            output_indices.clear();
            output_indices.insert(output_indices.end(), output_set.begin(), output_set.end());
        }
    }
}


void rnnlm::saveNet(string filename)
{
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Can't create file " << filename << endl;
        exit(1);
    }

    // meta data
    file << "input layer size: " << input_size << endl;
    file << "hidden layer size: " << hidden_size << endl;
    file << "output layer size: " << output_size << endl;
    file << "vocabulary size (including </s> and <unk>): " << vocab_size << endl;

    // vocabulary
    file << "\nVocabulary:" << endl;
    file << setw(10) << 0 << "\t" << "</s>" << endl;
    for (unordered_map<string, unsigned int>::iterator it = vocab.begin(); it != vocab.end(); it++) {
        if (it->first != "</s>")
            file << setw(10) << it->second << "\t" << it->first << endl;
    }

    // weights
    file << "\nWin:" << endl;
    for (int a = 0; a < input_size; a++) {
        for (int b = 0; b < hidden_size; b++) {
            file << Wih[a * hidden_size + b] << " ";
        }
        file << endl;
    }

    file << "\nWout:" << endl;
    for (int b = 0; b < output_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file << Who[a + b * hidden_size] << " ";
        }
        file << endl;
    }

    file << "\nWr:" << endl;
    for (int b = 0; b < hidden_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file << Wr[a + b * hidden_size] << " ";
        }
        file << endl;
    }

    // save rmsprop status
    file << "\nRmsProps-Win:" << endl;
    for (int a = 0; a < input_size; a++) {
        for (int b = 0; b < hidden_size; b++) {
            file << vWih[a * hidden_size + b] << " ";
        }
        file << endl;
    }

    file << "\nRmsProps-Wout:" << endl;
    for (int b = 0; b < output_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file << vWho[a + b * hidden_size] << " ";
        }
        file << endl;
    }

    file << "\nRmsProps-Wr:" << endl;
    for (int b = 0; b < hidden_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file << vWr[a + b * hidden_size] << " ";
        }
        file << endl;
    }

    file.close();
}


void rnnlm::loadNet(string filename) {
    ifstream file;
    file.open(filename, ios_base::in);

    if (!file.is_open()) {
        cerr << "Can't open model file " << filename << endl;
        exit(1);
    }

    string line, word;
    size_t pos = 0;

    // get meta data
    getline(file, line);
    pos = line.find(':', 0);
    input_size = atoi(line.substr(pos + 1, line.size()).c_str());

    getline(file, line);
    pos = line.find(':', 0);
    hidden_size = atoi(line.substr(pos + 1, line.size()).c_str());

    getline(file, line);
    pos = line.find(':', 0);
    output_size = atoi(line.substr(pos + 1, line.size()).c_str());

    getline(file, line);
    pos = line.find(':', 0);
    vocab_size = atoi(line.substr(pos + 1, line.size()).c_str());

    //read vocabulary
    int index;
    file >> word;
    for (int i = 0; i < vocab_size - 1; i++) {
        file >> word;
        index = atoi(word.c_str());
        file >> word;
        vocab[word] = index;
    }
    oov = vocab_size - 1;

    initFromRandom();       //memory allocation here

    // load weights
    file >> word;
    for (int a = 0; a < input_size; a++) {
        for (int b = 0; b < hidden_size; b++) {
            file >> word;
            Wih[a * hidden_size + b] = atof(word.c_str());
        }
    }

    file >> word;
    for (int b = 0; b < output_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file >> word;
            Who[a + b * hidden_size] = atof(word.c_str());
        }
    }

    file >> word;
    for (int b = 0; b < hidden_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file >> word;
            Wr[a + b * hidden_size] = atof(word.c_str());
        }
    }

    // load rmpprop status
    file >> word;
    for (int a = 0; a < input_size; a++) {
        for (int b = 0; b < hidden_size; b++) {
            file >> word;
            vWih[a * hidden_size + b] = atof(word.c_str());
        }
    }

    file >> word;
    for (int b = 0; b < output_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file >> word;
            vWho[a + b * hidden_size] = atof(word.c_str());
        }
    }

    file >> word;
    for (int b = 0; b < hidden_size; b++) {
        for (int a = 0; a < hidden_size; a++) {
            file >> word;
            vWr[a + b * hidden_size] = atof(word.c_str());
        }
    }

    file.close();
}


scalar rnnlm::forwardPropagation(vector<vector<int>>& stream, unsigned int sliceOffset)
{
    scalar ll = 0.f;

    for (int t = 0; t < bptt_block; t++) {

        int offset_hidden = t * hidden_size * batch_size;
        int offset_previous_hidden = t ? (t - 1) * hidden_size * batch_size : bptt_block * hidden_size * batch_size;
        scalar* hidden_step = hidden + offset_hidden;
        scalar* errorH_step = errorH + offset_hidden;

        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * hidden_size;
            int input_word = stream[b][sliceOffset + t];
            memcpy(hidden_step + offset_batch, Wih + input_word * hidden_size, hidden_size * sizeof(scalar));
        }

        // input + wr * hidden:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hidden_size, hidden_size, 1.0f,
                hidden + offset_previous_hidden, hidden_size, Wr, hidden_size, 1.0f, hidden_step, hidden_size);

        // set hidden = Wih if input_word = 0
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int input_word = stream[b][sliceOffset + t];
            if (!input_word) {
                memcpy(hidden_step + b * hidden_size, Wih, hidden_size * sizeof(scalar));
            }
        }

        // sigmoid
        #pragma omp parallel for simd num_threads(numcores)
        for (unsigned int x = 0; x < hidden_size * batch_size; x++) {
            scalar h = hidden_step[x];
            if (h > 50.f)
                h = 50.f;
            else if (h < -50.f)
                h = -50.f;
            hidden_step[x] = 1.f / (1.f + expf(-h));
            //hidden_step[x] = (1 - exp(-2 * h)) / (1 + exp(-2 * h));
        }

        // output layer
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_size, hidden_size, 1.0f,
                hidden_step, hidden_size, Who, hidden_size, 0.0f, output, output_size);

        //softmax:
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_size;
            scalar* output_batch = output + offset_batch;

            scalar maxAc = -FLT_MAX;
            #pragma simd
            for (int x = 0; x < output_size; x++) {
                if (output_batch[x] > maxAc)
                    maxAc = output_batch[x];
            }

            scalar denom = 0.f;
            #pragma simd
            for (int x = 0; x < output_size; x++) {
                output_batch[x] = max(expf(output_batch[x] - maxAc), 1e-7f);
                denom += output_batch[x];
            }

            scalar denomr = 1.f / denom;
            // normalized the prediction
            #pragma simd
            for (int x = 0; x < output_size; x++)
                output_batch[x] *= denomr;
        }

        // calculate log likelihood
        #pragma omp parallel for num_threads(numcores) reduction(+: ll)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_size;
            int output_word = stream[b][sliceOffset + t + 1];
            ll += logf(output[offset_batch + output_word]);
        }
        if ((ll != ll) || (std::isinf(ll))) {
            cerr << "\nnumerical error in computing log-likelihood" << endl;
            exit(1);
        }

        // calculate dWho and errorH early here to save memory
        #pragma omp parallel for simd num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch_output = b * output_size;
            int output_word = stream[b][sliceOffset + t + 1];
            output[offset_batch_output + output_word] -= 1;
        }

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_size, hidden_size, batch_size, -1.0f, output,
                output_size, hidden_step, hidden_size, 1.0f, dWho, hidden_size);

        // computing the error on the hidden layer regarding to the output
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, hidden_size, output_size, -1.0f, output,
                output_size, Who, hidden_size, 0.0f, errorH_step, hidden_size);
    }

    return ll;
}


scalar rnnlm::nce_forwardPropagation(vector<vector<int>>& stream, unsigned int sliceOffset, vector<vector<unsigned int>>& samples)
{
    int output_sample_sizes[bptt_block];
    vector<vector<unsigned int>> sample_counts;
    sample_counts.clear();
    sample_counts.resize(bptt_block);

    unsigned int hidden_layer_bytes = hidden_size * sizeof(scalar);

    scalar ll = 0;

    // update sample_indices and target_indices
    #pragma omp parallel for num_threads(numcores)
    for (int t = 0; t < bptt_block; t++) {
        unsigned int* sample_indices_offset = sample_indices + t * (sample_k + batch_size);
        unsigned int* target_indices_offset = target_indices + t * batch_size;

        int output_sample_size = 0;
        for (int k = 0; k < sample_k; k++) {
            unsigned int word = samples[t][k];
            unsigned int* p = find(sample_indices_offset, sample_indices_offset + output_sample_size, word);
            if (p == sample_indices_offset + output_sample_size) {
                sample_indices_offset[output_sample_size] = word;
                sample_counts[t].push_back(1);
                output_sample_size++;
            } else {
                sample_counts[t][p - sample_indices_offset]++;
            }
        }

        for (unsigned int b = 0; b < batch_size; b++) {
            int output_word = stream[b][sliceOffset + t + 1];
            unsigned int* p = find(sample_indices_offset, sample_indices_offset + output_sample_size, output_word);
            if (p == sample_indices_offset + output_sample_size) {
                sample_indices_offset[output_sample_size] = output_word;
                target_indices_offset[b] = output_sample_size;
                output_sample_size++;
            } else {
                target_indices_offset[b] = p - sample_indices_offset;
            }
        }
        output_sample_sizes[t] = output_sample_size;
    }

    for (int t = 0; t < bptt_block; t++) {

        unsigned int* sample_indices_offset = sample_indices + t * (sample_k + batch_size);
        unsigned int* target_indices_offset = target_indices + t * batch_size;
        int output_sample_size = output_sample_sizes[t];
        int sample_count = sample_counts[t].size();

        // collect Who_samples
        #pragma omp parallel for num_threads(numcores)
        for (int k = 0; k < output_sample_size; k++) {
            memcpy(Who_samples + k * hidden_size, Who + sample_indices_offset[k] * hidden_size, hidden_layer_bytes);
        }

        // now back to normal
        int offset_hidden = t * hidden_size * batch_size;
        int offset_previous_hidden = t ? (t - 1) * hidden_size * batch_size : bptt_block * hidden_size * batch_size;
        scalar* hidden_step = hidden + offset_hidden;
        scalar* errorH_step = errorH + offset_hidden;

        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int input_word = stream[b][sliceOffset + t];
            memcpy(hidden_step + b * hidden_size, Wih + input_word * hidden_size, hidden_layer_bytes);
        }

        // input + wr * hidden
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hidden_size, hidden_size, 1.0f,
                hidden + offset_previous_hidden, hidden_size, Wr, hidden_size, 1.0f, hidden_step, hidden_size);

        // set hidden = Wih if input_word = 0
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int input_word = stream[b][sliceOffset + t];
            if (!input_word) {
                memcpy(hidden_step + b * hidden_size, Wih, hidden_layer_bytes);
            }
        }

        // sigmoid
        #pragma omp parallel for simd num_threads(numcores)
        for (unsigned int x = 0; x < hidden_size * batch_size; x++) {
            scalar h = hidden_step[x];
            if (h > 50.f)
                h = 50.f;
            else if (h < -50.f)
                h = -50.f;
            hidden_step[x] = 1.f / (1.f + expf(-h));
            //hidden_step[x] = (1 - exp(-2 * h)) / (1 + exp(-2 * h));
        }

        // output layer
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_sample_size, hidden_size, 1.0f,
                hidden_step, hidden_size, Who_samples, hidden_size, 0.0f, output_samples, output_sample_size);

        // softmax:
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_sample_size;
            scalar* output_batch = output_samples + offset_batch;
            int output_word = target_indices_offset[b];
            bool outside = output_word >= sample_count;

            #pragma simd
            for (int x = 0; x < sample_count; x++) output_batch[x] = expf(output_batch[x]) / Z;
            if (outside) {
                scalar tv = expf(output_batch[output_word]) / Z;
                memset(output_batch + sample_count, 0.f, (output_sample_size - sample_count) * sizeof(scalar));
                output_batch[output_word] = tv;
            } else {
                memset(output_batch + sample_count, 0.f, (output_sample_size - sample_count) * sizeof(scalar));
            }
        }

        // calculate log likelihood
        #pragma omp parallel for num_threads(numcores) reduction(+: ll)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_sample_size;
            scalar* output_batch = output_samples + offset_batch;
            int output_word = target_indices_offset[b];
            int real_output_word = sample_indices_offset[output_word];
            scalar loglikelihood = logf(output_batch[output_word] / (output_batch[output_word] + sample_k * proposal_PMF[real_output_word]) + 1e-7f);

            #pragma simd
            for (int k = 0; k < sample_count; k++) {
                int word = sample_indices_offset[k];
                loglikelihood += sample_counts[t][k] * logf(sample_k * proposal_PMF[word] / (output_batch[k] + sample_k * proposal_PMF[word]) + 1e-7f);
            }
            ll += loglikelihood;
        }
        if ((ll != ll) || (std::isinf(ll))) {
            cerr << "\nnumerical error in computing log-likelihood" << endl;
            exit(1);
        }

        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_sample_size;
            scalar* output_batch = output_samples + offset_batch;
            int output_word = target_indices_offset[b];
            int real_output_word = sample_indices_offset[output_word];
            bool outside = output_word >= sample_count;

            scalar ui = output_batch[output_word];

            #pragma simd
            for (int k = 0; k < sample_count; k++) {
                int word = sample_indices_offset[k];
                output_batch[k] = sample_counts[t][k] * output_batch[k] / (output_batch[k] + sample_k * proposal_PMF[word]);
            }
            if (outside) output_batch[output_word] = 0.f;
            output_batch[output_word] += -sample_k * proposal_PMF[real_output_word] / (ui + sample_k * proposal_PMF[real_output_word]);
        }

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_sample_size, hidden_size, batch_size, -1.0f, output_samples, output_sample_size,
                hidden_step, hidden_size, 0.0f, dWho_samples, hidden_size);

        // update dWho
        #pragma omp parallel for num_threads(numcores)
        for (int k = 0; k < output_sample_size; k++) {
            scalar *src = dWho_samples + k * hidden_size;
            scalar *des = dWho + sample_indices_offset[k] * hidden_size;
            #pragma simd
            for (int x = 0; x < hidden_size; x++) {
                des[x] += src[x];
            }
        }

        // computing the error on the hidden layer regarding to the output
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, hidden_size, output_sample_size, -1.0f, output_samples,
                output_sample_size, Who_samples, hidden_size, 0.0f, errorH_step, hidden_size);

    }

    return ll;
}


scalar rnnlm::blackout_forwardPropagation(vector<vector<int>>& stream, unsigned int sliceOffset, vector<vector<unsigned int>>& samples)
{
    int output_sample_sizes[bptt_block];
    vector<vector<unsigned int>> sample_counts;
    sample_counts.clear();
    sample_counts.resize(bptt_block);

    unsigned int hidden_layer_bytes = hidden_size * sizeof(scalar);

    scalar ll = 0.f;

    // update sample_indices and target_indices
    #pragma omp parallel for num_threads(numcores)
    for (int t = 0; t < bptt_block; t++) {
        unsigned int* sample_indices_offset = sample_indices + t * (sample_k + batch_size);
        unsigned int* target_indices_offset = target_indices + t * batch_size;

        int output_sample_size = 0;
        for (int k = 0; k < sample_k; k++) {
            unsigned int word = samples[t][k];
            unsigned int* p = find(sample_indices_offset, sample_indices_offset + output_sample_size, word);
            if (p == sample_indices_offset + output_sample_size) {
                sample_indices_offset[output_sample_size] = word;
                sample_counts[t].push_back(1);
                output_sample_size++;
            } else {
                sample_counts[t][p - sample_indices_offset]++;
            }
        }

        for (unsigned int b = 0; b < batch_size; b++) {
            int output_word = stream[b][sliceOffset + t + 1];
            unsigned int* p = find(sample_indices_offset, sample_indices_offset + output_sample_size, output_word);
            if (p == sample_indices_offset + output_sample_size) {
                sample_indices_offset[output_sample_size] = output_word;
                target_indices_offset[b] = output_sample_size;
                output_sample_size++;
            } else {
                target_indices_offset[b] = p - sample_indices_offset;
            }
        }
        output_sample_sizes[t] = output_sample_size;
    }

    for (int t = 0; t < bptt_block; t++) {

        unsigned int* sample_indices_offset = sample_indices + t * (sample_k + batch_size);
        unsigned int* target_indices_offset = target_indices + t * batch_size;
        int output_sample_size = output_sample_sizes[t];
        int sample_count = sample_counts[t].size();

        // collect Who_samples
        #pragma omp parallel for num_threads(numcores)
        for (int k = 0; k < output_sample_size; k++) {
            memcpy(Who_samples + k * hidden_size, Who + sample_indices_offset[k] * hidden_size, hidden_layer_bytes);
        }

        // now back to normal
        int offset_hidden = t * hidden_size * batch_size;
        int offset_previous_hidden = t ? (t - 1) * hidden_size * batch_size : bptt_block * hidden_size * batch_size;
        scalar* hidden_step = hidden + offset_hidden;
        scalar* errorH_step = errorH + offset_hidden;

        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int input_word = stream[b][sliceOffset + t];
            memcpy(hidden_step + b * hidden_size, Wih + input_word * hidden_size, hidden_layer_bytes);
        }

        // input + wr * hidden
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hidden_size, hidden_size, 1.0f,
                hidden + offset_previous_hidden, hidden_size, Wr, hidden_size, 1.0f, hidden_step, hidden_size);

        // set hidden = Wih if input_word = 0
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int input_word = stream[b][sliceOffset + t];
            if (!input_word) {
                memcpy(hidden_step + b * hidden_size, Wih, hidden_layer_bytes);
            }
        }

        // sigmoid
        #pragma omp parallel for simd num_threads(numcores)
        for (unsigned int x = 0; x < hidden_size * batch_size; x++) {
            scalar h = hidden_step[x];
            if (h > 50.f)
                h = 50.f;
            else if (h < -50.f)
                h = -50.f;
            hidden_step[x] = 1.f / (1.f + expf(-h));
            //hidden_step[x] = (1 - exp(-2 * h)) / (1 + exp(-2 * h));
        }

        // output layer
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, output_sample_size, hidden_size, 1.0f,
                hidden_step, hidden_size, Who_samples, hidden_size, 0.0f, output_samples, output_sample_size);

        // softmax
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_sample_size;
            scalar* output_batch = output_samples + offset_batch;
            int output_word = target_indices_offset[b];
            scalar wi = proposal_PMF[sample_indices_offset[output_word]];
            bool outside = output_word >= sample_count;

            scalar maxAc = -FLT_MAX;
            #pragma simd
            for (int x = 0; x < sample_count; x++) {
                if (output_batch[x] > maxAc)
                    maxAc = output_batch[x];
            }
            if (outside) {
                if (output_batch[output_word] > maxAc)
                    maxAc = output_batch[output_word];
            }

            scalar denom = 0.f;
            #pragma simd
            for (int x = 0; x < sample_count; x++) {
                scalar wx = wi / proposal_PMF[sample_indices_offset[x]];
                output_batch[x] = max(expf(output_batch[x] - maxAc), 1e-7f);
                denom += sample_counts[t][x] * wx * output_batch[x];
            }
            if (outside) {
                output_batch[output_word] = max(expf(output_batch[output_word] - maxAc), 1e-7f);
                denom += output_batch[output_word];
            } else {
                denom -= (sample_counts[t][output_word] - 1.f) * output_batch[output_word];
            }

            scalar denomr = 1.f / denom;
            // normalized the prediction
            #pragma simd
            for (int x = 0; x < sample_count; x++) {
                scalar wx = wi / proposal_PMF[sample_indices_offset[x]];
                output_batch[x] *= wx * denomr;
            }
            if (outside) {
                scalar tv = output_batch[output_word] * denomr;
                memset(output_batch + sample_count, 0.f, (output_sample_size - sample_count) * sizeof(scalar));
                output_batch[output_word] = tv;
            } else {
                memset(output_batch + sample_count, 0.f, (output_sample_size - sample_count) * sizeof(scalar));
            }
        }

        // calculate log likelihood
        #pragma omp parallel for num_threads(numcores) reduction(+: ll)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_sample_size;
            int output_word = target_indices_offset[b];
            ll += logf(output_samples[offset_batch + output_word]);
        }
        if ((ll != ll) || (std::isinf(ll))) {
            cerr << "\nnumerical error in computing log-likelihood" << endl;
            exit(1);
        }

        // update error at output layer
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++) {
            int offset_batch = b * output_sample_size;
            scalar* output_batch = output_samples + offset_batch;
            int output_word = target_indices_offset[b];
            bool outside = output_word >= sample_count;
            const scalar nl = sample_k - (outside ? 0 : sample_counts[t][output_word]);

            scalar pi = output_batch[output_word];

            scalar sum = 0.f;
            #pragma simd
            for (int j = 0; j < sample_count; j++) {
                scalar pj = output_batch[j];
                sum += sample_counts[t][j] / (1.f - pj);
            }
            if (!outside) {
                sum -= sample_counts[t][output_word] / (1.f - pi);
            }

            #pragma simd
            for (int l = 0; l < sample_count; l++) {
                scalar pl = output_batch[l];
                output_batch[l] = sample_counts[t][l] * pl * (1.f + nl - (sum - 1.f / (1.f - pl)));
            }
            output_batch[output_word] = pi * (1.f + nl - sum) - 1.f;
        }

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_sample_size, hidden_size, batch_size, -1.0f, output_samples, output_sample_size,
                hidden_step, hidden_size, 0.0f, dWho_samples, hidden_size);

        // update dWho
        #pragma omp parallel for num_threads(numcores)
        for (int k = 0; k < output_sample_size; k++) {
            scalar *src = dWho_samples + k * hidden_size;
            scalar *des = dWho + sample_indices_offset[k] * hidden_size;
            #pragma simd
            for (int x = 0; x < hidden_size; x++) {
                des[x] += src[x];
            }
        }

        // computing the error on the hidden layer regarding to the output
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, hidden_size, output_sample_size, -1.0f, output_samples,
                output_sample_size, Who_samples, hidden_size, 0.0f, errorH_step, hidden_size);
    }

    return ll;
}


void rnnlm::backPropagation(vector<vector<int>>& stream, unsigned int sliceOffset)
{
    unsigned int hidden_layer_bytes = hidden_size * sizeof(scalar);

    // set initial errorH3
    #pragma omp parallel for num_threads(numcores)
    for (unsigned int b = 0; b < batch_size; b++) {
        int offset_hidden = (bptt_block - 1) * hidden_size * batch_size;
        int offset_batch = b * hidden_size;
        memcpy(errorH3 + offset_batch, errorH + offset_hidden + offset_batch, hidden_layer_bytes);
    }

    for (int t = bptt_block - 1; t >= 0; t--) {

        int offset_hidden = t * hidden_size * batch_size;
        int offset_next_hidden = (t - 1) * hidden_size * batch_size;
        scalar* hidden_step = hidden + offset_hidden;

        // gradient on sigmoid
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int x = 0; x < batch_size; x++) {
            int offset = x * hidden_size;
            scalar* errorH2_offset = errorH2 + offset;
            scalar* errorH3_offset = errorH3 + offset;
            scalar* hidden_step_offset = hidden_step + offset;
            #pragma simd
            for (unsigned int y = 0; y < hidden_size; y++) {
                errorH2_offset[y] = errorH3_offset[y] * hidden_step_offset[y] * (1 - hidden_step_offset[y]);
                //errorH2_offset[y] = errorH3_offset[y] * (1 - hidden_step_offset[y] * hidden_step_offset[y]);
            }
        }

        // compute the gradient of Wih
        //#pragma omp parallel for num_threads(numcores) // hogwild???
        for (unsigned int b = 0; b < batch_size; b++) {
            scalar* dWih_offset = dWih + stream[b][sliceOffset + t] * hidden_size;
            scalar* errorH2_batch = errorH2 + b * hidden_size;
            #pragma simd
            for (int x = 0; x < hidden_size; x++)
                dWih_offset[x] += errorH2_batch[x];
        }

        if (t > 0) {

            // set errorH2 = 0 if input_word = 0
            #pragma omp parallel for num_threads(numcores)
            for (unsigned int b = 0; b < batch_size; b++) {
                int input_word = stream[b][sliceOffset + t];
                if (!input_word) {
                    memset(errorH2 + b * hidden_size, 0, hidden_layer_bytes);
                }
            }

            // compute the gradient of Wr
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hidden_size, hidden_size, batch_size, 1.0f,
                    errorH2, hidden_size, hidden + offset_next_hidden, hidden_size, 1.0f, dWr, hidden_size);

            // propagate the error through time
            #pragma omp parallel for num_threads(numcores)
            for (unsigned int b = 0; b < batch_size; b++) {
                memcpy(errorH3 + b * hidden_size, errorH + offset_next_hidden + b * hidden_size, hidden_layer_bytes);
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, hidden_size, hidden_size, 1.0f, errorH2,
                    hidden_size, Wr, hidden_size, 1.0f, errorH3, hidden_size);
        }

    }

}


void rnnlm::inference(const int *input, const int length, vector<scalar>& prob)
{

    for (int a = 0; a < length; a++) {

        int offset_hidden = (a % 2) * hidden_size;
        int offset_previous_hidden = ((a - 1) % 2) * hidden_size;
        int input_word = input[a];
        int output_word = input[a + 1];

        memcpy(hidden + offset_hidden, Wih + input_word * hidden_size, hidden_size * sizeof(scalar));

        // input + wr * hidden
        if (a > 0) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, hidden_size, hidden_size, 1.0f, Wr, hidden_size,
                    hidden + offset_previous_hidden, 1, 1.0f, hidden + offset_hidden, 1);
        }

        // sigmoid
        #pragma omp parallel for simd num_threads(numcores)
        for (int x = 0; x < hidden_size; x++) {
            scalar h = hidden[offset_hidden + x];
            if (h > 50.f)
                h = 50.f;
            else if (h < -50.f)
                h = -50.f;
            hidden[offset_hidden + x] = 1.f / (1.f + expf(-h));
            //hidden[offset_hidden + x] = (1 - exp(-2 * h)) / (1 + exp(-2 * h));
        }

        // output layer
        cblas_sgemv(CblasRowMajor, CblasNoTrans, output_size, hidden_size, 1.0f, Who, hidden_size,
                hidden + offset_hidden, 1, 0.0f, output, 1);

        // softmax
        scalar maxAc = -FLT_MAX;
        #pragma omp parallel for simd num_threads(numcores) reduction(max: maxAc)
        for (int x = 0; x < output_size; x++) {
            if (output[x] > maxAc)
                maxAc = output[x];
        }
        scalar denom = 0.f;
        #pragma omp parallel for simd num_threads(numcores) reduction(+: denom)
        for (int x = 0; x < output_size; x++) {
            output[x] = max(expf(output[x] - maxAc), 1e-7f);
            denom += output[x];
        }

        scalar denomr = 1.f / denom;
        // normalized the prediction
        #pragma omp parallel for simd num_threads(numcores)
        for (int x = 0; x < output_size; x++)
            output[x] *= denomr;

        // calucate prob
        scalar p = 0.f;
        if (eval_unk || output_word != oov)
            p = output[output_word];
        prob.push_back(p);
    }
}


void rnnlm::fullModelUpdate(unsigned int numWordsInBatch)
{
    /* Gradient Clipping: calculate norm */
    scalar norm2 = 0.f;
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < input_size; x++) {
        scalar* temp = dWih + x * hidden_size;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) {
            temp[y] = temp[y] / numWordsInBatch;
            norm2 += temp[y] * temp[y];
        }
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < hidden_size; x++) {
        scalar* temp = dWr + x * hidden_size;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) {
            temp[y] = temp[y] / numWordsInBatch;
            norm2 += temp[y] * temp[y];
        }
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < output_size; x++) {
        scalar* temp = dWho + x * hidden_size;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) {
            temp[y] = temp[y] / numWordsInBatch;
            norm2 += temp[y] * temp[y];
        }
    }
    norm2 = sqrtf(norm2);
    /* Gradient Clipping: normalization */
    if (norm2 > gradient_cutoff) {
        const scalar norm = gradient_cutoff / norm2;
        #pragma omp parallel for num_threads(numcores)
        for (int x = 0; x < input_size; x++) {
            scalar* temp = dWih + x * hidden_size;
            #pragma simd
            for (int y = 0; y < hidden_size; y++) temp[y] *= norm;
        }
        #pragma omp parallel for num_threads(numcores)
        for (int x = 0; x < hidden_size; x++) {
            scalar* temp = dWr + x * hidden_size;
            #pragma simd
            for (int y = 0; y < hidden_size; y++) temp[y] *= norm;
        }
        #pragma omp parallel for num_threads(numcores)
        for (int x = 0; x < output_size; x++) {
            scalar* temp = dWho + x * hidden_size;
            #pragma simd
            for (int y = 0; y < hidden_size; y++) temp[y] *= norm;
        }
    }
    /* RMSProp velocity */
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < input_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = vWih + offset;
        scalar* temp2 = dWih + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) temp[y] = momentum * temp[y] + (1.f - momentum) * temp2[y] * temp2[y];
    }

    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < hidden_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = vWr + offset;
        scalar* temp2 = dWr + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) temp[y]  = momentum * temp[y] + (1.f - momentum) * temp2[y] * temp2[y];
    }

    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < output_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = vWho + offset;
        scalar* temp2 = dWho + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) temp[y] = momentum * temp[y] + (1.f - momentum) * temp2[y] * temp2[y];
    }

    /* RMSProp update */
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < input_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = Wih + offset;
        scalar* temp2 = dWih + offset;
        scalar* temp3 = vWih + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++)
            temp[y] += lr * temp2[y] / (sqrtf(temp3[y]) + rmsprop_damping);
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < hidden_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = Wr + offset;
        scalar* temp2 = dWr + offset;
        scalar* temp3 = vWr + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++)
            temp[y] += lr * temp2[y] / (sqrtf(temp3[y]) + rmsprop_damping);
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < output_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = Who + offset;
        scalar* temp2 = dWho + offset;
        scalar* temp3 = vWho + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++)
            temp[y] += lr * temp2[y] / (sqrtf(temp3[y]) + rmsprop_damping);
    }
}


void rnnlm::subModelUpdate(vector<unsigned int>& input_indices, vector<unsigned int>& output_indices, unsigned int numWordsInBatch)
{
    int input_indices_size = input_indices.size();
    int output_indices_size = output_indices.size();

    /* Gradient Clipping: calculate norm */
    scalar norm2 = 0.f;
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < input_indices_size; x++) {
        scalar* temp = dWih + input_indices[x] * hidden_size;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) {
            temp[y] = temp[y] / numWordsInBatch;
            norm2 += temp[y] * temp[y];
        }
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < hidden_size; x++) {
        scalar* temp = dWr + x * hidden_size;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) {
            temp[y] = temp[y] / numWordsInBatch;
            norm2 += temp[y] * temp[y];
        }
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < output_indices_size; x++) {
        scalar* temp = dWho + output_indices[x] * hidden_size;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) {
            temp[y] = temp[y] / numWordsInBatch;
            norm2 += temp[y] * temp[y];
        }
    }
    norm2 = sqrtf(norm2);
    /* Gradient Clipping: normalization */
    if (norm2 > gradient_cutoff) {
        const scalar norm = gradient_cutoff / norm2;
        #pragma omp parallel for num_threads(numcores)
        for (int x = 0; x < input_indices_size; x++) {
            scalar* temp = dWih + input_indices[x] * hidden_size;
            #pragma simd
            for (int y = 0; y < hidden_size; y++) temp[y] *= norm;
        }
        #pragma omp parallel for num_threads(numcores)
        for (int x = 0; x < hidden_size; x++) {
            scalar* temp = dWr + x * hidden_size;
            #pragma simd
            for (int y = 0; y < hidden_size; y++) temp[y] *= norm;
        }
        #pragma omp parallel for num_threads(numcores)
        for (int x = 0; x < output_indices_size; x++) {
            scalar* temp = dWho + output_indices[x] * hidden_size;
            #pragma simd
            for (int y = 0; y < hidden_size; y++) temp[y] *= norm;
        }
    }
    /* RMSProp velocity */
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < input_indices_size; x++) {
        int idx = input_indices[x];
        int offset = idx * hidden_size;
        scalar* temp  = vWih + offset;
        scalar* temp2 = dWih + offset;
        scalar mo2 = input_moPow[idx];
        #pragma simd
        for (int y = 0; y < hidden_size; y++) temp[y] = mo2 * temp[y] + (1.f - momentum) * temp2[y] * temp2[y];
    }

    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < hidden_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = vWr + offset;
        scalar* temp2 = dWr + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++) temp[y]  = momentum * temp[y] + (1.f - momentum) * temp2[y] * temp2[y];
    }

    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < output_indices_size; x++) {
        int idx = output_indices[x];
        int offset = idx * hidden_size;
        scalar* temp  = vWho + offset;
        scalar* temp2 = dWho + offset;
        scalar mo2 = output_moPow[idx];
        #pragma simd
        for (int y = 0; y < hidden_size; y++) temp[y] = mo2 * temp[y] + (1.f - momentum) * temp2[y] * temp2[y];
    }

    /* RMSProp update */
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < input_indices_size; x++) {
        int offset = input_indices[x] * hidden_size;
        scalar* temp  = Wih + offset;
        scalar* temp2 = dWih + offset;
        scalar* temp3 = vWih + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++)
            temp[y] += lr * temp2[y] / (sqrtf(temp3[y]) + rmsprop_damping);
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < hidden_size; x++) {
        int offset = x * hidden_size;
        scalar* temp  = Wr + offset;
        scalar* temp2 = dWr + offset;
        scalar* temp3 = vWr + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++)
            temp[y] += lr * temp2[y] / (sqrtf(temp3[y]) + rmsprop_damping);
    }
    #pragma omp parallel for num_threads(numcores)
    for (int x = 0; x < output_indices_size; x++) {
        int offset = output_indices[x] * hidden_size;
        scalar* temp  = Who + offset;
        scalar* temp2 = dWho + offset;
        scalar* temp3 = vWho + offset;
        #pragma simd
        for (int y = 0; y < hidden_size; y++)
            temp[y] += lr * temp2[y] / (sqrtf(temp3[y]) + rmsprop_damping);
    }
}


void rnnlm::saveTestProb(string filename, vector<vector<scalar>>& prob)
{
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Can't create file " << filename << endl;
        exit(1);
    }

    file << uppercase << scientific;
    for (int i = 0; i < prob.size(); i++)
        for (int j = 0; j < prob[i].size(); j++)
            file << prob[i][j] << endl;

    file.close();
}


void rnnlm::train() {

    // construct vocabulary
    learnVocabFromFile(train_file);

    // build token->index map and compute CDF for sampling
    buildMapPMFCMF();

    // load training streams
    loadTrainingStreams(train_file, trainStreams);

    // load validation data
    loadVTSentences(valid_file, validateData);
    unsigned int numValidates = validateData.size();

    initNet();

    int iter = 0;

    double logp = 0, last_logp = -FLT_MAX;

    cout << "start training rnnlm ...\n" << endl;

    unsigned int numWordsInBatch = bptt_block * batch_size;
    vector<vector<unsigned int>> samples(bptt_block, vector<unsigned int>(sample_k, -1));

    // create thread-specific RNGs
    rngs = new rng_type[bptt_block];
    for (int t = 0; t < bptt_block; t++) {
        rngs[t] = rng_type(random_seed + 15791 * t);
    }
    scalar prox1 = proposal_CMF[vocab_size - 1];
    uniform_real_dist = uniform_real_distribution<scalar>(0.f, prox1 - prox1 / vocab_size);

    int hidden_layer_bytes = hidden_size * sizeof(scalar);
    vector<unsigned int> input_indices;
    vector<unsigned int> output_indices;

    //marshalTrainingStreams(trainData, trainStreams);

    size_t shortestStreamLength = INT_MAX;
    for (int i = 0; i < batch_size; i++) {
        if (shortestStreamLength > trainStreams[i].size())
            shortestStreamLength = trainStreams[i].size();
    }
    unsigned int numSlices = (shortestStreamLength - 1) / bptt_block;

    vocab_idx.clear();
    vocab_idx.resize(bptt_block, vector<unsigned int>(vocab_size - 1));

    while (iter < max_iter) {

        printf("iter: %3d ", iter);
        fflush(stdout);

        /* training */
        double start = omp_get_wtime();
        int numTrainedWords = 0;
        logp = 0.;

        // initialize history of hidden states to zeros
        #pragma omp parallel for num_threads(numcores)
        for (unsigned int b = 0; b < batch_size; b++)
            memset(hidden + bptt_block * batch_size * hidden_size + b * hidden_size, 0, hidden_layer_bytes);

        for (unsigned int i = 0; i < numSlices; i++) {

            unsigned int sliceOffset = i * bptt_block;

            unsigned int input_indices_size, output_indices_size;

            switch (algo) {
            case algo_type::std:
                // initialize full-nets
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < input_size; x++) memset(dWih + x * hidden_size, 0.f, hidden_layer_bytes);
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < hidden_size; x++) memset(dWr + x * hidden_size, 0.f, hidden_layer_bytes);
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < output_size; x++) memset(dWho + x * hidden_size, 0.f, hidden_layer_bytes);

                logp += forwardPropagation(trainStreams, sliceOffset);
                break;
            case algo_type::nce:
                // sample an output layer
                samplingFromCMF(proposal_CMF, sample_k, samples);
                // identify input and output nodes to be updated in this batch
                indicesTobeUpdated(trainStreams, sliceOffset, samples, input_indices, output_indices);
                input_indices_size  = input_indices.size();
                output_indices_size = output_indices.size();
                // initialize sub-nets
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < input_indices_size; x++) memset(dWih + input_indices[x] * hidden_size, 0.f, hidden_layer_bytes);
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < hidden_size; x++) memset(dWr + x * hidden_size, 0.f, hidden_layer_bytes);
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < output_indices_size; x++) memset(dWho + output_indices[x] * hidden_size, 0.f, hidden_layer_bytes);

                logp += nce_forwardPropagation(trainStreams, sliceOffset, samples);
                break;
            case algo_type::blackout:
                // sample an output layer
                samplingFromCMF(proposal_CMF, sample_k, samples);
                // identify input and output nodes to be updated in this batch
                indicesTobeUpdated(trainStreams, sliceOffset, samples, input_indices, output_indices);
                input_indices_size  = input_indices.size();
                output_indices_size = output_indices.size();
                // initialize sub-nets
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < input_indices_size; x++) memset(dWih + input_indices[x] * hidden_size, 0.f, hidden_layer_bytes);
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < hidden_size; x++) memset(dWr + x * hidden_size, 0.f, hidden_layer_bytes);
                #pragma omp parallel for num_threads(numcores)
                for (int x = 0; x < output_indices_size; x++) memset(dWho + output_indices[x] * hidden_size, 0.f, hidden_layer_bytes);

                logp += blackout_forwardPropagation(trainStreams, sliceOffset, samples);
                break;
            default:
                cerr << "Un-supported algorithm!" << endl;
                exit(-1);
            }
            backPropagation(trainStreams, sliceOffset);

            for (unsigned int b = 0; b < batch_size; b++) {
                memcpy(hidden + bptt_block * batch_size * hidden_size + b * hidden_size,
                        hidden + (bptt_block - 1) * batch_size * hidden_size + b * hidden_size, hidden_layer_bytes);
            }

            numTrainedWords += numWordsInBatch;

            if (i > 0 && (i % 100) == 0) {
                double elapsed_secs = omp_get_wtime() - start;
                printf("%citer: %3d lr: %.6f train loss: %.4f progress: %.2f%% words/sec: %.1f ", 13,
                        iter, lr, min(exp(-logp / numTrainedWords), 10000.0), numTrainedWords / (scalar) train_words * 100,
                        numTrainedWords / elapsed_secs);
                fflush(stdout);
            }

            /* model update */
            if (algo == algo_type::std) {
                fullModelUpdate(numWordsInBatch);
            } else {
                subModelUpdate(input_indices, output_indices, numWordsInBatch);
            }

        }

        double elapsed_secs = omp_get_wtime() - start;
        printf("%citer: %3d lr: %.6f train loss: %.4f words/sec: %.1f ", 13, iter, lr,
                exp(-logp / numTrainedWords), numTrainedWords / elapsed_secs);

        // save net first
        saveNet(rnnlm_file);


        /* validation */
        logp = 0.;
        int numValidatedWords = 0;

        scalar lp = 0.f;
        vector<scalar> prob;
        for (unsigned int s = 0; s < numValidates; s++) {
            int *input = &validateData[s][0];
            int length = validateData[s].size() - 1;

            // forward propagation
            prob.clear();
            inference(input, length, prob);

            lp = 0.f;
            #pragma simd
            for (int i = 0; i < length; i++) {
                scalar p = prob[i];
                if (p > 0.f) {
                    lp += logf(p);
                    numValidatedWords++;
                }
            }
            logp += lp;
        }

        printf("validate ppl: %.4f\n", exp(-logp / numValidatedWords));

        if (logp < last_logp) {
            lr *= 0.7f;
        } else {
            lr *= lr_decay;
        }

        iter++;
        last_logp = logp;
    }
}


void rnnlm::test() {

    loadNet(rnnlm_file);

    loadVTSentences(test_file, testData);

    int numTests = testData.size();
    vector<vector<scalar>> testProb;
    testProb.reserve(numTests);

    double logp = 0.;
    int numRNNWords = 0;
    vector<scalar> prob;
    for (int s = 0; s < numTests; s++) {
        int *input = &testData[s][0];
        int length = testData[s].size() - 1;

        // forward propagation
        prob.clear();
        inference(input, length, prob);

        testProb.push_back(prob);

        scalar lp = 0.f;
        #pragma simd
        for (int i = 0; i < length; i++) {
            scalar prob_rnn = prob[i];
            if (prob_rnn > 0.f) {
                lp += logf(prob_rnn);
                numRNNWords++;
            }
        }
        logp += lp;
    }

    cout << "\nPPL: " << exp(-logp / numRNNWords) << endl;

    saveTestProb(test_prob_file, testProb);
}
