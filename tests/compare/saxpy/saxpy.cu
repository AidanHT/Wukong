// Comparison line counts: Wukong=22, C=23, CUDA=31.
// Physical lines include comments and blanks; update all peer headers together.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void saxpy(int n, float a, const float *__restrict__ x,
                      const float *__restrict__ y, float *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main() {
    constexpr int n = 8;
    const float hx[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float hy[n] = {10, 20, 30, 40, 50, 60, 70, 80};
    float hout[n], *x, *y, *out;
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    cudaMalloc(&out, n * sizeof(float));
    cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, n * sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, x, y, out);
    cudaMemcpy(hout, out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (float value : hout) std::printf("%.0f\n", value);

    cudaFree(out);
    cudaFree(y);
    cudaFree(x);
    return 0;
}
