/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

    //INSERT KERNEL CODE HERE
	int r = (FILTER_SIZE-1)/2;
	int col = blockIdx.x*TILE_SIZE + threadIdx.x - r;
	int row = blockIdx.y*TILE_SIZE + threadIdx.y - r;

	__shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];
	if(row>=0 && row<N.height && col>=0 && col<N.width) {
		N_s[threadIdx.y][threadIdx.x] = N.elements[row*N.width + col];
	}
	else {
		N_s[threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();

	int tileCol = threadIdx.x - r;
	int tileRow = threadIdx.y - r;
	if(col>=0 && col<N.width && row>=0 && row<N.height) {
		if(tileCol>=0 && tileCol<TILE_SIZE && tileRow>=0 && tileRow<TILE_SIZE) {
			float sum = 0.0f;
			for(int fRow = 0; fRow < FILTER_SIZE; fRow++) {
				for(int fCol = 0; fCol < FILTER_SIZE; fCol++) {
					sum += M_c[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
				}
			}
			P.elements[row*P.width + col] = sum;
		}
	}
}
