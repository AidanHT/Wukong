// Comparison line counts: Wukong=23, C=21, CUDA=32.
// Physical lines include comments and blanks; update all peer headers together.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void stencil(int n, const float *__restrict__ x,
                        float *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0 || i == n - 1) {
        out[i] = x[i];
    } else {
        out[i] = x[i - 1] + 2.0f * x[i] + x[i + 1];
    }
}

int main() {
    constexpr int n = 8;
    const float hx[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    float hout[n], *x, *out;
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&out, n * sizeof(float));
    cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice);

    stencil<<<(n + 255) / 256, 256>>>(n, x, out);
    cudaMemcpy(hout, out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (float value : hout) std::printf("%.0f\n", value);

    cudaFree(out);
    cudaFree(x);
    return 0;
}
