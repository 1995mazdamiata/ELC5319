/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SZ 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float A_s [TILE_SZ][TILE_SZ];
    __shared__ float B_s [TILE_SZ][TILE_SZ];

    int c = threadIdx.x;
    int r = threadIdx.y;
    int col = blockIdx.x * TILE_SZ + c;
    int row = blockIdx.y * TILE_SZ + r;
    int tiles = (k+TILE_SZ-1)/TILE_SZ;

    float sum = 0.0f;
    for(int tile = 0; tile < tiles; tile++) {

        // load current tile in shared memory, each thread loads one element
        if ((row<m) && ((tile*TILE_SZ + c)<k))
            A_s[r][c] = A[row*k + tile*TILE_SZ + c];
        else A_s[r][c] = 0.0f;
        
        if (((tile*TILE_SZ + r)<k) && (col<n))
            B_s[r][c] = B[(tile*TILE_SZ + r)*n + col];
        else B_s[r][c] = 0.0f;
        __syncthreads();

        // sum up elements of the tile
        for(int element = 0; element < TILE_SZ; element++) {
            sum += A_s[r][element] * B_s[element][c];
        }
        __syncthreads();
    }

    if (row<m && col<n) C[row*n+col] = sum;
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE
    int blocksx = (n+TILE_SZ-1)/TILE_SZ;
    int blocksy = (m+TILE_SZ-1)/TILE_SZ;
    dim3 dimGrid(blocksx, blocksy, 1);
    dim3 dimBlock(TILE_SZ, TILE_SZ, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);

}
