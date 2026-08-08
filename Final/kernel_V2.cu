/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/


#define TILE 16
#define PGRID_R 4
#define PGRID_C 4

/**
 * luDiagBlockKernel
 * 
 * LU factor diagonal block A[k][k]. Diagonal block intentionally has same
 * dimensions as thread block, so only use 1 thread block for this part
 */
__global__ void luDiagBlockKernel_V2(float *out, unsigned int size, unsigned int k) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = k*TILE;
	int col = k*TILE;
	int bs = min(TILE, size - row);

	// LU factorization of A[k][k]
	for (unsigned int p = 0; p < bs; ++p) {
		// Multipliers
		if(tx == p && ty > p && ty < bs) {
			out[(row+ty)*size + (col+p)] = out[(row+ty)*size + (col+p)] / out[(row+p)*size + (col+p)];
		}
		__syncthreads();

		// in place update
		if(ty > p && ty < bs && tx > p && tx < bs) {
			out[(row+ty)*size + (col+tx)] -= out[(row+ty)*size + (col+p)] * out[(row+p)*size + (col+tx)];
		}
		__syncthreads();
	}
}

/**
 * luRowUpdateKernel
 * 
 * Fill out the rest of the columns of the rows corresponding to A[k][k].
 * Diagonal block A[k][k] has already been LU factored, so this finishes
 * U for the rest of the columns in the rows of A[k][k].
 */
__global__ void luRowUpdateKernel_V2(float *out, unsigned int size, unsigned int k, unsigned int num_blocks) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int pc = blockIdx.x;
	int row = k*TILE;
	int bsk = min(TILE, size - row);

	// First column > k assigned to this thread block
	int j0 = (k + 1) + ((pc - (k + 1)%PGRID_C)%PGRID_C + PGRID_C)%PGRID_C;

	// iterate over columns assigned to this thread block
	for (unsigned int j = j0; j < num_blocks; j += PGRID_C) {
		int col = j*TILE;
		int bsj = min(TILE, size - col);

		// Forward substitution: solve L*X = A
		// Finding U for the rest of the columns in the rows of A[k][k]
		if(ty == 0 && tx < bsj) {
			for (int p = 0; p < bsk; ++p) {
				float sum = 0.0f;

				for(int t = 0; t < p; ++t) {
					//sum += Lkk_s[p][t] * Akj_s[t][tx];
					sum += out[(row+p)*size + (row+t)] * out[(row+t)*size + (col+tx)];
				}
				//Akj_s[p][tx] -= sum;
				out[(row+p)*size + (col+tx)] -= sum;
			}
		}
		__syncthreads();
	}
}

/**
 * luColUpdateKernel
 * 
 * Fill out the rest of the rows of the columns corresponding to A[k][k].
 * Diagonal block A[k][k] has already been LU factored, so this finishes
 * L for the rest of the rows in the columns of A[k][k].
 */
__global__ void luColUpdateKernel_V2(float *out, unsigned int size, unsigned int k, unsigned int num_blocks) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int pr = blockIdx.x;
	int row = k*TILE;
	int bsk = min(TILE, size - row);

	// First row > k assigned to this thread block
	int i0 = (k + 1) + ((pr - (k + 1)%PGRID_R)%PGRID_R + PGRID_R)%PGRID_R;

	// iterate over rows assigned to this thread block
	for (int i = i0; i < num_blocks; i += PGRID_R) {
		int rowi = i*TILE;
		int bsi = min(TILE, size - rowi);

		// Backward substitution: solve X*U = A
		// Finding L for the rest of the rows in the columns of A[k][k]
		if(tx == 0 && ty < bsi) {
			for(int c = 0; c < bsk; ++c) {
				float sum = 0.0f;

				for(int r = 0; r < c; ++r) {
					sum += out[(rowi+ty)*size + (row+r)] * out[(row+r)*size + (row+c)];
				}
				out[(rowi+ty)*size + (row+c)] = (out[(rowi+ty)*size + (row+c)] - sum)/out[(row+c)*size + (row+c)];
			}
		}
		__syncthreads();
	}
}

__global__ void luTrailingUpdateKernel_V2(float *out, unsigned int size, unsigned int k, unsigned int num_blocks) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int pc = blockIdx.x;
	int pr = blockIdx.y;

	int rowk = k*TILE;
	int bsk = min(TILE, size - rowk);

	// first row > k assigned to this thread block
	int i0 = (k + 1) + ((pr - (k + 1)%PGRID_R)%PGRID_R + PGRID_R)%PGRID_R;

	// iterate over rows assigned to this thread block
	for(int i = i0; i < num_blocks; i += PGRID_R) {
		int rowi = i*TILE;
		int bsi = min(TILE, size - rowi);

		// first column > k assigned to this thread block
		int j0 = (k + 1) + ((pc - (k + 1)%PGRID_C)%PGRID_C + PGRID_C)%PGRID_C;

		// iterate over columns assigned to this thread block
		for (int j = j0; j < num_blocks; j += PGRID_C) {
			int colj = j*TILE;
			int bsj = min(TILE, size - colj);

			// Fill out rest of A[i][j]
			if (ty < bsi && tx < bsj) {
				float sum = 0.0f;

				for(int t = 0; t < bsk; ++t) {
					sum += out[(rowi+ty)*size + (rowk+t)] * out[(rowk+t)*size + (colj+tx)];
				}
				out[(rowi + ty)*size + (colj + tx)] -= sum;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

// Block cyclic approach
void luFactorization_V2(float *out, float *in, unsigned int in_size) {
	cudaMemcpy(out, in, in_size*in_size*sizeof(float), cudaMemcpyDeviceToDevice);

	int num_blocks = (in_size + TILE - 1)/ TILE;
	dim3 blockDim(TILE, TILE, 1);

	for(int k = 0; k < num_blocks; ++k) {
		// step 1: factor diagonal block of input matrix
		luDiagBlockKernel_V2<<<1, blockDim>>>(out, in_size, k);

		if (k + 1 < num_blocks) {
			// step 2: row update
			luRowUpdateKernel_V2<<<dim3(PGRID_C, 1, 1), blockDim>>>(out, in_size, k, num_blocks);

			// step 3: column update
			luColUpdateKernel_V2<<<dim3(PGRID_R, 1, 1), blockDim>>>(out, in_size, k, num_blocks);

			// step 4: trailing submatrix update
			luTrailingUpdateKernel_V2<<<dim3(PGRID_C, PGRID_R, 1), blockDim>>>(out, in_size, k, num_blocks);
		}
	}
}