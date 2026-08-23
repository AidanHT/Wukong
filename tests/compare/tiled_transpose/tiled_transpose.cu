// Comparison line counts: Wukong=63, C=41, CUDA=58.
// Physical lines include comments and blanks; update all peer headers together.
// COMMITTED, NOT BUILT: this box has the CUDA driver and an RTX 4050 but no `nvcc`, the same
// standing as saxpy.cu / naive_gemm.cu / unknown_op.cu. Nothing claims a measured number from it.
// Build line it is written against: nvcc -O3 -arch=sm_89 tiled_transpose.cu -o tiled_transpose
#include <cuda_runtime.h>
#include <cstdio>

// The canonical coalesced __shared__ transpose: a TILE_DIM x TILE_DIM staging tile padded by one
// column so the transposed read hits 32 distinct banks instead of one, and BLOCK_ROWS < TILE_DIM so
// each thread row walks TILE_DIM/BLOCK_ROWS strips -- which is what `@tile(2)` spells in the .wk.
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeCoalesced(int width, int height,
                                   const float *__restrict__ in,
                                   float *__restrict__ out) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }
    __syncthreads();

    // Re-index against the TRANSPOSED block origin so the store is coalesced too.
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    constexpr int n = 8;
    float ha[n * n], hb[n * n], *a, *b;
    for (int k = 0; k < n * n; ++k) ha[k] = static_cast<float>(k + 1);
    cudaMalloc(&a, sizeof(ha));
    cudaMalloc(&b, sizeof(hb));
    cudaMemcpy(a, ha, sizeof(ha), cudaMemcpyHostToDevice);

    const dim3 block(TILE_DIM, BLOCK_ROWS);
    const dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);
    transposeCoalesced<<<grid, block>>>(n, n, a, b);
    cudaMemcpy(hb, b, sizeof(hb), cudaMemcpyDeviceToHost);

    // Every one of the 64 lanes, from the scalar semantics b[r*8+c] = a[c*8+r].
    for (float value : hb) std::printf("%.0f\n", value);

    cudaFree(b);
    cudaFree(a);
    return 0;
}
