// Comparison line counts: Wukong=29, C=28, CUDA=36.
// Physical lines include comments and blanks; update all peer headers together.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void gemm(int m, int n, int k, const float *__restrict__ a,
                     const float *__restrict__ b, float *__restrict__ c) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int p = 0; p < k; ++p) sum += a[row * k + p] * b[p * n + col];
    c[row * n + col] = sum;
}

int main() {
    const float ha[6] = {1, 2, 3, 4, 5, 6};
    const float hb[6] = {1, 0, 0, 1, 1, 1};
    float hc[4], *a, *b, *c;
    cudaMalloc(&a, sizeof(ha));
    cudaMalloc(&b, sizeof(hb));
    cudaMalloc(&c, sizeof(hc));
    cudaMemcpy(a, ha, sizeof(ha), cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, sizeof(hb), cudaMemcpyHostToDevice);

    const dim3 block(16, 16);
    const dim3 grid((2 + block.x - 1) / block.x, (2 + block.y - 1) / block.y);
    gemm<<<grid, block>>>(2, 2, 3, a, b, c);
    cudaMemcpy(hc, c, sizeof(hc), cudaMemcpyDeviceToHost);
    for (float value : hc) std::printf("%.0f\n", value);

    cudaFree(c);
    cudaFree(b);
    cudaFree(a);
    return 0;
}
