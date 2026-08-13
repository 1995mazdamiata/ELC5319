/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/


#define TILE 16
#define PGRID_R 8
#define PGRID_C 8

/**
 * luDiagBlockKernel
 * 
 * LU factor diagonal block A[k][k]. Diagonal block intentionally has same
 * dimensions as thread block, so only use 1 thread block for this part
 */
__global__ void luDiagBlockKernel_V3(float *out, unsigned int size, unsigned int k) {
	__shared__ float Akk_s[TILE][TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = k*TILE;
	int col = k*TILE;
	int bs = min(TILE, size - row);

	// Load A[k][k], diagonal block into shared memory
	if (ty < bs && tx < bs) {
		Akk_s[ty][tx] = out[(row + ty)*size + (col + tx)];
	}
	__syncthreads();

	// LU factorization of A[k][k]
	for (unsigned int p = 0; p < bs; ++p) {
		// Multipliers
		if(tx == p && ty > p && ty < bs) {
			Akk_s[ty][p] = Akk_s[ty][p] / Akk_s[p][p];
		}
		__syncthreads();

		// in place update
		if(ty > p && ty < bs && tx > p && tx < bs) {
			Akk_s[ty][tx] -= Akk_s[ty][p] * Akk_s[p][tx];
		}
		__syncthreads();
	}

	// write out LU factored A[k][k]
	if(ty < bs && tx < bs) {
		out[(row + ty)*size + (col + tx)] = Akk_s[ty][tx];
	}
}

/**
 * luRowUpdateKernel
 * 
 * Fill out the rest of the columns of the rows corresponding to A[k][k].
 * Diagonal block A[k][k] has already been LU factored, so this finishes
 * U for the rest of the columns in the rows of A[k][k].
 */
__global__ void luRowUpdateKernel_V3(float *out, unsigned int size, unsigned int k, unsigned int num_blocks) {
	
	__shared__ float Lkk_s[TILE][TILE];
	__shared__ float Akj_s[TILE][TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int pc = blockIdx.x;
	int row = k*TILE;
	int bsk = min(TILE, size - row);

	// Load L[k][k], L portion of LU factored A[k][k]
	// Assumes LU factored A[k][k] already written to out
	if(ty < bsk && tx < bsk) {
		Lkk_s[ty][tx] = out[(row + ty)*size + (row + tx)];
	}
	__syncthreads();

	// First column > k assigned to this thread block
	int j0 = (k + 1) + ((pc - (k + 1)%PGRID_C)%PGRID_C + PGRID_C)%PGRID_C;

	// iterate over columns assigned to this thread block
	for (unsigned int j = j0; j < num_blocks; j += PGRID_C) {
		int col = j*TILE;
		int bsj = min(TILE, size - col);

		// Load A[k][j], the rest of the columns in rows of A[k][k]
		if(ty < bsk && tx < bsj) {
			Akj_s[ty][tx] = out[(row + ty)*size + (col + tx)];
		}
		__syncthreads();

		// Forward substitution: solve L*X = A
		// Finding U for the rest of the columns in the rows of A[k][k]
		if(ty == 0 && tx < bsj) {
			for (int p = 0; p < bsk; ++p) {
				float sum = 0.0f;

				for(int t = 0; t < p; ++t) {
					sum += Lkk_s[p][t] * Akj_s[t][tx];
				}
				Akj_s[p][tx] -= sum;
			}
		}
		__syncthreads();

		// write out U[k][j]
		if (ty < bsk && tx < bsj) {
			out[(row + ty)*size + (col + tx)] = Akj_s[ty][tx];
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
__global__ void luColUpdateKernel_V3(float *out, unsigned int size, unsigned int k, unsigned int num_blocks) {

	__shared__ float Ukk_s[TILE][TILE];
	__shared__ float Aik_s[TILE][TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int pr = blockIdx.x;
	int row = k*TILE;
	int bsk = min(TILE, size - row);

	// Load U[k][k], U portion of LU factored A[k][k]
	// Assumes LU factored A[k][k] alread written to out
	if (ty < bsk && tx < bsk) {
		Ukk_s[ty][tx] = out[(row + ty)*size + (row + tx)];
	}
	__syncthreads();

	// First row > k assigned to this thread block
	int i0 = (k + 1) + ((pr - (k + 1)%PGRID_R)%PGRID_R + PGRID_R)%PGRID_R;

	// iterate over rows assigned to this thread block
	for (int i = i0; i < num_blocks; i += PGRID_R) {
		int rowi = i*TILE;
		int bsi = min(TILE, size - rowi);

		// Load A[i][k], the rest of the rows in columns of A[k][k]
		if(ty < bsi && tx < bsk) {
			Aik_s[ty][tx] = out[(rowi + ty)*size + (row + tx)];
		}
		__syncthreads();

		// Backward substitution: solve X*U = A
		// Finding L for the rest of the rows in the columns of A[k][k]
		if(tx == 0 && ty < bsi) {
			for(int c = 0; c < bsk; ++c) {
				float sum = 0.0f;

				for(int r = 0; r < c; ++r) {
					sum += Aik_s[ty][r] * Ukk_s[r][c];
				}
				Aik_s[ty][c] = (Aik_s[ty][c] - sum)/Ukk_s[c][c];
			}
		}
		__syncthreads();

		// write out L[i][k]
		if(ty < bsi && tx < bsk) {
			out[(rowi + ty)*size + (row + tx)] = Aik_s[ty][tx];
		}
		__syncthreads();
	}
}

/**
 * luTrailingUpdateKernel
 * 
 * Update the ramining active trailing submatrix with the Schur complement:
 * A[i][j] = A[i][j] - L[i][k]*U[k][j]. Uses parallel block matrix multiplication.
 */
__global__ void luTrailingUpdateKernel_V3(float *out, unsigned int size, unsigned int k, unsigned int num_blocks) {
	
	__shared__ float Lik_s[TILE][TILE];
	__shared__ float Ukj_s[TILE][TILE];

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

		// load L[i][k], already computed
		if(ty < bsi && tx < bsk) {
			Lik_s[ty][tx] = out[(rowi + ty)*size + (rowk + tx)];
		}
		__syncthreads();

		// first column > k assigned to this thread block
		int j0 = (k + 1) + ((pc - (k + 1)%PGRID_C)%PGRID_C + PGRID_C)%PGRID_C;

		// iterate over columns assigned to this thread block
		for (int j = j0; j < num_blocks; j += PGRID_C) {
			int colj = j*TILE;
			int bsj = min(TILE, size - colj);

			// load U[k][j], already computed
			if(ty < bsk && tx < bsj) {
				Ukj_s[ty][tx] = out[(rowk + ty)*size + (colj + tx)];
			}
			__syncthreads();

			// Fill out rest of trailing A[i][j] with Schur complement
			if (ty < bsi && tx < bsj) {
				float sum = 0.0f;

				// Parallel matrix multiplication
				for(int t = 0; t < bsk; ++t) {
					sum += Lik_s[ty][t] * Ukj_s[t][tx];
				}
				out[(rowi + ty)*size + (colj + tx)] -= sum;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

// Right-looking block cyclic approach with shared memory
void luFactorization_V3(float *out, float *in, unsigned int in_size) {
	cudaMemcpy(out, in, in_size*in_size*sizeof(float), cudaMemcpyDeviceToDevice);

	int num_blocks = (in_size + TILE - 1)/ TILE;
	dim3 blockDim(TILE, TILE, 1);

	for(int k = 0; k < num_blocks; ++k) {
		// step 1: factor diagonal block of input matrix
		luDiagBlockKernel_V3<<<1, blockDim>>>(out, in_size, k);

		if (k + 1 < num_blocks) {
			// step 2: row update
			luRowUpdateKernel_V3<<<dim3(PGRID_C, 1, 1), blockDim>>>(out, in_size, k, num_blocks);

			// step 3: column update
			luColUpdateKernel_V3<<<dim3(PGRID_R, 1, 1), blockDim>>>(out, in_size, k, num_blocks);

			// step 4: trailing submatrix update
			luTrailingUpdateKernel_V3<<<dim3(PGRID_C, PGRID_R, 1), blockDim>>>(out, in_size, k, num_blocks);
		}
	}
}