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