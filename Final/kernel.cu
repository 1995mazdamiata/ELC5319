/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define C_FACTOR 4
#define SECTION 2048

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void preScanKernel(float *out, float *in, unsigned size, float *sum)
{
	__shared__ float in_s[SECTION];
	unsigned int idx = blockIdx.x*SECTION + threadIdx.x;

	// phase 0: load data into shared memory
	for(int i = 0; i < SECTION; i += blockDim.x) {
		if(idx == 0 && i == 0) {
			in_s[threadIdx.x] = 0.0f;
		}
		else if(idx + i < size) {
			in_s[threadIdx.x + i] = in[idx + i - 1];
		}
		else {
			in_s[threadIdx.x + i] = 0.0f;
		}
	}

	// phase 1: sequential scan
	int n = C_FACTOR;
	int offset = threadIdx.x * n;
	for (int i = 0; i < n - 1; i++) {
		__syncthreads();
		if (i + 1 + offset < SECTION) {
			in_s[i + 1 + offset] += in_s[i + offset];
		}
	}

	// phase 2: iterative Brent-Kung scan
	// BK part 1
	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*2*stride*n - 1;
		if(index < SECTION) {
			in_s[index] += in_s[index - stride*n];
		}
	}

	// BK part 2
	for (unsigned int stride = SECTION/4; stride > 0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*2*stride*n - 1;
		if(index + stride*n < SECTION) {
			in_s[index + stride*n] += in_s[index];
		}
	}

	// phase 3: add offsets
	__syncthreads();
	offset += n;
	for (int i = 0; i < n - 1; i++) {
		if (i + offset < SECTION) {
			in_s[i + offset] += in_s[offset - 1];
		}
	}

	// write output
	__syncthreads();
	for(int i = 0; i < SECTION; i += blockDim.x) {
		if(idx + i < size) {
			out[idx + i] = in_s[threadIdx.x + i];
		}
	}
	if (sum && (threadIdx.x == blockDim.x - 1)) {
		sum[blockIdx.x] = in_s[SECTION - 1];
	}
}

__global__ void addKernel(float *out, float *sum, unsigned size)
{
    // INSERT CODE HERE
	int blk = blockIdx.x*SECTION;
	float Bias = sum[blockIdx.x];
	for (int idx = threadIdx.x; idx < SECTION; idx += blockDim.x) {
		if (idx + blk < size) {
			out[idx + blk] += Bias;
		}
	}
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
	float *sum;
	unsigned num_blocks;
	cudaError_t cuda_ret;
	dim3 dim_grid, dim_block;

	num_blocks = in_size/SECTION;
	if(in_size%SECTION !=0) num_blocks++;

	dim_block.x = SECTION/C_FACTOR; dim_block.y = 1; dim_block.z = 1;
	dim_grid.x = num_blocks; dim_grid.y = 1; dim_grid.z = 1;

	if(num_blocks > 1) {
		cuda_ret = cudaMalloc((void**)&sum, num_blocks*sizeof(float));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, sum);
		preScan(sum, sum, num_blocks);
		addKernel<<<dim_grid, dim_block>>>(out, sum, in_size);

		cudaFree(sum);
	}
	else
		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, NULL);
}

#define TILE 16
#define PGRID_R 4
#define PGRID_C 4

/**
 * luDiagBlockKernel
 * 
 * LU factor diagonal block A[k][k]. Diagonal block intentionally has same
 * dimensions as thread block, so only use 1 thread block for this part
 */
__global__ void luDiagBlockKernel(float *out, float *in, unsigned int size, unsigned int k) {
	__shared__ float Akk_s[TILE][TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = k*TILE;
	int col = k*TILE;
	int bs = min(TILE, size - row);

	// Load A[k][k], diagonal block into shared memory
	if (ty < bs && tx < bs) {
		Akk_s[ty][tx] = in[(row + ty)*size + (col + tx)];
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
__global__ void luRowUpdateKernel(float *out, float *in, unsigned int size, 
	                                     unsigned int k, unsigned int num_blocks) {
	
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
			Akj_s[ty][tx] = in[(row + ty)*size + (col + tx)];
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
__global__ void luColUpdateKernel(float *out, float *in, unsigned int size,
                                         unsigned int k, unsigned int num_blocks) {

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
			Aik_s[ty][tx] = in[(rowi + ty)*size + (row + tx)];
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
	}
}

void luFactorization(float *out, float *in, unsigned int in_size) {

}