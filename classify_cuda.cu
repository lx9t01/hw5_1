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
 * weight_temp: a temporary global memory for weights, storing weight vector for 
 *              shared memory result in a batch
 */
__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    float step_size,
	float *weights,
    float *errors,
    float *weight_temp)
{
    // TODO: write me
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float gradient[1024];

    while (thread_index < batch_size) {
        float wx = 0;
        // calculate w^T * x
        for (int i = 0; i < REVIEW_DIM; ++i) {
            wx += weights[i] * data[thread_index*(REVIEW_DIM+1)+i];
        }
        // w^T * x is the prediction result of previous w
        if (wx * data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] < 0) { // if the prediction is not right
            atomicAdd(errors, 1);
        }
        // calclate the denominator
        float denom = (1 + exp(data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] * wx));

        for (int i = 0; i < REVIEW_DIM; ++i) { 
        // for each dimension in a data point
            gradient[threadIdx.x] = (-1.0/batch_size * \
                data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] * \
                data[thread_index*(REVIEW_DIM+1)+i])/denom;
            int l = blockDim.x;
            while (l > 1) { // reduction loop for the accumulation
                l /= 2;
                if (threadIdx.x < l) {
                    gradient[threadIdx.x] += gradient[threadIdx.x + l];
                }    
                __syncthreads();
            }
            // printf("%f\n", gradient[0]);
            weight_temp[i] = gradient[0]; // the sum is stored in the 0th element
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int i = 0; i < REVIEW_DIM; ++i) {
                // the addition of two shared memory result in a batch
                atomicAdd(&weights[i], -step_size * weight_temp[i]);
                // weights[i] = 0;
                // weights[i] = weights[i] - step_size * weight_temp[i];
            }
        }
        if (thread_index == batch_size - 1) {
            // calculate error rate, using just (random) one threadIdx 
            *errors /= batch_size;
        }
        // __syncthreads();
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
    // int block_size = 512;
    // int grid_size = 65535;
    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    // int grid_size = 512;
    printf("%d %d \n", block_size, grid_size);
    int shmem_bytes = 0;

    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));

    float *weight_temp;
    cudaMalloc(&weight_temp, REVIEW_DIM * sizeof(float));
    cudaMemset(weight_temp, 0, REVIEW_DIM * sizeof(float));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors,
        weight_temp);
    // cudaDeviceSynchronize();
    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    cudaFree(weight_temp);



    return h_errors;
}
