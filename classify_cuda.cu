#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    int step_size,
	float *weights,
    float *errors)
{
    // TODO: write me
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float gradient[1024];
    float temp[50];

    while (thread_index < batch_size) {
        float wx = 0;
        for (int i = 0; i < REVIEW_DIM; ++i) {
            wx += weights[i] * data[thread_index*(REVIEW_DIM+1)+i];
        }
        if (wx * data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] < 0) {
            atomicAdd(errors, 1);
        }

        float denom = (1 + exp(data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] * wx));

        for (int i = 0; i < REVIEW_DIM; ++i) {
            float grad_elem = 0.0;
            gradient[threadIdx.x] = (-1.0/batch_size * data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] * data[thread_index*(REVIEW_DIM+1)+i])/denom;
            
            int l = blockDim.x;
            while (l > 1) {
                l /= 2;
                if (threadIdx.x < l) {
                    gradient[threadIdx.x] += gradient[threadIdx.x + l];
                }    
                __syncthreads();
            }
            // if (threadIdx.x == 0) {
            //     temp[i] += gradient[0];
            // }
        }
        thread_index += blockDim.x * gridDim.x;
    }
    

}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data,
    int batch_size, 
    float step_size,
    float *weights, 
    cudaStream_t stream)
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = 0;

    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}
