#include <cuda_runtime.h>
#include <math.h>

__device__ float _tanh(float x){
    
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x)); 
}

__global__ void tanh_kernel(const float* input, float* output, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) output[i] = _tanh(input[i]);
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}