#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;                  
    
    float m = -INFINITY;
    for (int j = 0; j < N; j++)
        m = fmaxf(m, input[j]);

    float sum = 0.0f;
    for (int j = 0; j < N; j++)
        sum += expf(input[j] - m);

    output[i] = expf(input[i] - m) / sum;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}