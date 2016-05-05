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
#include "ta_utilities.hpp"

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

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
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
    //assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
    // TODO ok: randomly initialize weights. allocate and initialize buffers on
    //       host & device. allocate and initialize streams
    // host weights vector
    /*
    float* weights = (float*) malloc(REVIEW_DIM * sizeof(float));
    // random initialize
    gaussianFill(weights, REVIEW_DIM);

    // device weights vector
    float* dev_weights;
    gpuErrChk(cudaMalloc((void**) &dev_weights, REVIEW_DIM * sizeof(float)));
    gpuErrChk(cudaMemcpy(dev_weights, weights, REVIEW_DIM * sizeof(float), cudaMemcpyHostToDevice));
    */
    // the buffer is 2*batch_size of data points
    float *host_buffer = (float*) malloc(2 * batch_size * (REVIEW_DIM + 1) * sizeof(float)); 
    /*
    // device data memory for 2 streams, batch_size of data each
    float *dev_data_0, *dev_data_1; 
    gpuErrChk(cudaMalloc((void**) &dev_data_0, (REVIEW_DIM+1) * batch_size * sizeof(float)));
    gpuErrChk(cudaMalloc((void**) &dev_data_1, (REVIEW_DIM+1) * batch_size * sizeof(float)));

    // streams
    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    printf("original weights: \n");
    for(int i = 0; i < REVIEW_DIM; ++i){
        printf("%f, ", weights[i]);
    }
    printf("\n");
    
    // main loop to process input lines (each line corresponds to a review)
    float acm_gpu_mps = -1.0;
    float gpu_mps = -1.0;
    float step_size = 2.0;
    float error_0 = 0;
    float error_1 = 0;
    */
    int review_idx = 0;
    for(string review_str; getline(in_stream, review_str); review_idx++){
        // TODO ok: process review_str with readLSAReview
        readLSAReview(review_str, host_buffer + (REVIEW_DIM+1) * review_idx, 1);
        /*
        // TODO ok: if you have filled up a batch, copy H->D, call kernel and copy
        //      D->H all in a stream
        if(review_idx == 2 * batch_size - 1){ // if the buffer is full, copy data to 2 streams
            review_idx = 0;
            START_TIMER(); 
            gpuErrChk(cudaMemcpyAsync(dev_data_0, host_buffer, (REVIEW_DIM+1) * batch_size * sizeof(float), cudaMemcpyHostToDevice, s[0]));
            error_0 = cudaClassify(dev_data_0, batch_size, step_size, dev_weights, s[0]);
            printf("the error 0 is %f \n", error_0);
            gpuErrChk(cudaMemcpyAsync(dev_data_1, host_buffer + (REVIEW_DIM+1) * batch_size, (REVIEW_DIM+1) * batch_size * sizeof(float), cudaMemcpyHostToDevice, s[1]));
            error_1 = cudaClassify(dev_data_1, batch_size, step_size, dev_weights, s[1]);
            printf("the error 1 is %f \n", error_1);
            STOP_RECORD_TIMER(gpu_mps);
            acm_gpu_mps += gpu_mps;
            */
        }
    }
    /*
    cout << "the gpu time is " << acm_gpu_mps << endl;
    for(int i = 0; i < 2; ++i){
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }
    cudaMemcpy(weights, dev_weights, REVIEW_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: print out weights
    cudaFree(dev_weights);
    cudaFree(dev_data_0);
    cudaFree(dev_data_1);

    printf("final weights: \n");
    for(int i = 0; i < REVIEW_DIM; ++i){
        printf("%f, ", weights[i]);
    }
    printf("\n");
    // TODO: free all memory
    free(host_buffer);
    free(weights);
    */

}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("./classify <path to datafile>\n");
        return -1;
    } 
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 100;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    
    // Init timing
    float time_initial, time_final;
    
    int batch_size = 2048;
    
    // begin timer
    time_initial = clock();
    
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
    
    // End timer
    time_final = clock();
    printf("Total time to run classify: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);
    

}
