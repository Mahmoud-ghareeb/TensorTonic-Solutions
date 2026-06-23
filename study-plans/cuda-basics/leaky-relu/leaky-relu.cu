#include <cuda_runtime.h>

__device__ float leaky_relu(float x, float alpha){

    if (x >= 0) return x;
    
    return alpha * x;
}

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) output[i] = leaky_relu(input[i], alpha);
    
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}