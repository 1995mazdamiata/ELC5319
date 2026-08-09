/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D 16

/**
 * luColUpdateKernel
 * 
 * Compute multipliers for column k and store them in lower triangular part. This becomes
 * the columns of L. One thread per row of the column.
 */
__global__ void luColUpdateKernel_V1(float *out, unsigned int size, unsigned int k) {
	int i = k + 1 + blockIdx.x*blockDim.x + threadIdx.x;

	if(i < size) {
		out[i*size + k] /= out[k*size + k];
	}
}

/**
 * luTrailingUpdateKernel
 * 
 * Update the trailing submatrix after column k has been filled. One thread for every
 * element of the output.
 */
__global__ void luTrailingUpdateKernel_V1(float *out, unsigned int size, unsigned int k) {
	int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
	int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size && j < size) {
		out[i*size + j] -= out[i*size + k] * out[k*size + j];
	}
}

// Right-looking naive approach
void luFactorization_V1(float *out, float *in, unsigned int in_size) {
	cudaMemcpy(out, in, in_size*in_size*sizeof(float), cudaMemcpyDeviceToDevice);

	for(int k = 0; k < in_size - 1; ++k) {
		int remaining = in_size - (k + 1);

		if (remaining > 0) {
			// compute multipliers for column k (columns of L)
			int gridDim1D = (remaining + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
			luColUpdateKernel_V1<<<gridDim1D, BLOCK_SIZE_1D>>>(out, in_size, k);

			// update trailing submatrix (rows of U)
			dim3 blockDim2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
			dim3 gridDim2D((remaining + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 
		                   (remaining + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
			luTrailingUpdateKernel_V1<<<gridDim2D, blockDim2D>>>(out, in_size, k);

			if(cudaDeviceSynchronize() != cudaSuccess) FATAL("Error in kernel execution");
		}
	}
}