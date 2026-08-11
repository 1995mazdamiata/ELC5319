/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// naive implementation
__global__ void histogram_kernel_V1(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {

    
    // INSERT CODE HERE
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ((i < num_elements) && (input[i] < num_bins)) {
        atomicAdd(&(bins[input[i]]), 1);
    }
}

/*
 * Privatization using global memory
 * 
 *   - Assumes enough device memory has been allocated for bins array to hold
 *     all private copies of the histogram
 *   - Each block computes its own private histogram, then they are merged
 */
__global__ void histogram_kernel_V2(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {
    
    // make private histogram in global memory
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ((i < num_elements) && (input[i] < num_bins)) {
        atomicAdd(&(bins[blockIdx.x*num_bins + input[i]]), 1);
    }

    // merge private histograms to main one
    if(blockIdx.x > 0) {
        __syncthreads();
        for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
            unsigned int binValue = bins[blockIdx.x*num_bins + bin];
            if (binValue > 0) {
                atomicAdd(&(bins[bin]), binValue);
            }
        }
    }
}

/*
 * Privatization using shared memory
 *
 *   - Each block computes its own private histogram, then they are merged
 *   - Each private copy is stored in shared memory
 */
__global__ void histogram_kernel_V3(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {
    
    // initialize private histogram bins in shared memory
    extern __shared__ unsigned int bins_s[];
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        bins_s[bin] = 0u;
    }
    __syncthreads();

    // compute private histogram
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if((i < num_elements) && (input[i] < num_bins)) {
        atomicAdd(&(bins_s[input[i]]), 1);
    }
    __syncthreads();

    // combine private histograms and store in global memory
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        unsigned int binValue = bins_s[bin];
        if(binValue > 0) {
            atomicAdd(&(bins[bin]), binValue);
        }
    }
}

/*
 * Thread coarsening with contiguous partitioning
 *
 *   - Each block still computes private copy, and private copy stored in
 *     shared memory
 *   - Input partitioned into contiguous segments, and each segment assigned
 *     to a thread
 *   - Each thread computes a few elements of the histogram
 */
__global__ void histogram_kernel_V4(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {

    // coarsening factor
    unsigned int cfactor = 16;
    
    // inititialize private histogram bins in shared memory
    extern __shared__ unsigned int bins_s[];
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        bins_s[bin] = 0u;
    }
    __syncthreads();

    // compute private histogram
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i = id*cfactor; i < min((id+1)*cfactor, num_elements); ++i) {
        if(input[i] < num_bins) {
            atomicAdd(&(bins_s[input[i]]), 1);
        }
    }
    __syncthreads();

    // combine private histograms and store in global memory
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        unsigned int binValue = bins_s[bin];
        if(binValue > 0) {
            atomicAdd(&(bins[bin]), binValue);
        }
    }
}

/*
 * Thread coarsening with interleaved partitioning
 *
 *   - Instead of partitioning input into contiguous segments and assigning
 *     each thread a segment, the elements used by each thread are interleaved
 *   - Threads load elements that are next to each other in memory
 *   - Because they are next to each other, they can be loaded at the same
 *     time, reducing the total number of memory accesses. 
 */
__global__ void histogram_kernel_V5(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {

    // inititialize private histogram bins in shared memory
    extern __shared__ unsigned int bins_s[];
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        bins_s[bin] = 0u;
    }
    __syncthreads();

    // compute private histogram
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i = id; i < num_elements; i += blockDim.x*gridDim.x) {
        if(input[i] < num_bins) {
            atomicAdd(&(bins_s[input[i]]), 1);
        }
    }
    __syncthreads();

    // combine private histograms and store in global memory
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        unsigned int binValue = bins_s[bin];
        if(binValue > 0) {
            atomicAdd(&(bins[bin]), binValue);
        }
    }
}

/*
 * Aggregation
 *
 *   - Aggregate consecutive updates into a single update
 *   - If consecutive values are the same, accumulate count until new value
 *     reached
 *   - Only update histogram when a new value is reached
 */
__global__ void histogram_kernel_V6(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {
    
    // inititialize private histogram bins in shared memory
    extern __shared__ unsigned int bins_s[];
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        bins_s[bin] = 0u;
    }
    __syncthreads();

    // compute private histogram
    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i = id; i < num_elements; i += blockDim.x*gridDim.x) {
        if(input[i] < num_bins) {
            int bin = input[i];
            if(bin == prevBinIdx) {
                ++accumulator;
            }
            else {
                if(accumulator > 0) {
                    atomicAdd(&(bins_s[prevBinIdx]), accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    if(accumulator > 0) {
        atomicAdd(&(bins_s[prevBinIdx]), accumulator);
    }
    __syncthreads();

    // combine private histograms and store in global memory
    for(unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        unsigned int binValue = bins_s[bin];
        if(binValue > 0) {
            atomicAdd(&(bins[bin]), binValue);
        }
    }
}

__global__ void convert_kernel(unsigned int *bins32, uint8_t *bins8,
    unsigned int num_bins) {

    // INSERT CODE HERE
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < num_bins) {
        bins8[i] = min(bins32[i], 255u);
    }
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // setup grid dimensions
    dim3 dim_grid, dim_block;
    dim_block.x = 512; 
    dim_block.y = dim_block.z = 1;
    dim_grid.x = (num_elements - 1)/dim_block.x + 1; 
    dim_grid.y = dim_grid.z = 1;
    
    /* 
    // Naive Implementation ------------------------------------------------------------
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));
    histogram_kernel_V1<<<dim_grid, dim_block>>>(input, bins32, num_elements, num_bins);
    //----------------------------------------------------------------------------------
    */
    
    /*
    // Privatization with global memory -----------------------------------------------
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, dim_grid.x * num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, dim_grid.x * num_bins * sizeof(unsigned int));
    histogram_kernel_V2<<<dim_grid, dim_block>>>(input, bins32, num_elements, num_bins);
    //----------------------------------------------------------------------------------
    */
    
    /*
    // Privatization with shared memory -----------------------------------------------
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));
    histogram_kernel_V3<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
        (input, bins32, num_elements, num_bins);
    //----------------------------------------------------------------------------------
    */
    
    /*
    // Thread coarsening with contiguous partitioning-----------------------------------
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));
    histogram_kernel_V4<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
        (input, bins32, num_elements, num_bins);
    //----------------------------------------------------------------------------------
    */
    
    /*
    // Thread coarsening with interleaved partitioning----------------------------------
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));
    histogram_kernel_V5<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
        (input, bins32, num_elements, num_bins);
    //----------------------------------------------------------------------------------
    */

    // Aggregation----------------------------------------------------------------------
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));
    histogram_kernel_V6<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
        (input, bins32, num_elements, num_bins);
    //----------------------------------------------------------------------------------

    // Convert 32-bit bins into 8-bit bins
    dim_block.x = 512;
    dim_grid.x = (num_bins - 1)/dim_block.x + 1;
    convert_kernel<<<dim_grid, dim_block>>>(bins32, bins, num_bins);

    // Free allocated device memory
    cudaFree(bins32);

}