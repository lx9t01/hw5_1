#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"
#include <windows.h> // For Timing via "QueryPerformanceCounter"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}


// Global variables to assist in timing
double PCFreq = 0.0;
__int64 CounterStart = 0;

// Initialize Windows-specific precise timing 
void initTiming()
{
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed! Timing routines won't work. \n");
    
    PCFreq = double(li.QuadPart)/1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}

// Get precise time
double preciseClock()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart)/PCFreq;
}

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size; i++) {
        output[i] = distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[stride * component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
    // TODO: randomly initialize weights. allocate and initialize buffers on
    //       host & device. allocate and initialize streams

    // main loop to process input lines (each line corresponds to a review)
    int review_idx = 0;
    for (string review_str; getline(in_stream, review_str); review_idx++) {
        // TODO: process review_str with readLSAReview

        // TODO: if you have filled up a batch, copy H->D, call kernel and copy
        //      D->H all in a stream
    }

    // TODO: print out weights
    // TODO: free all memory
}

int main(int argc, char** argv) {
    // Init timing
    initTiming();
	double time_initial, time_final, elapsed_ms;
	
    int batch_size = 2048;
    if (argc == 1) {
        classify(cin, batch_size);
    } else if (argc == 2) {
        ifstream ifs(argv[1]);
        stringstream buffer;
        buffer << ifs.rdbuf();
        classify(buffer, batch_size);
    }
}
