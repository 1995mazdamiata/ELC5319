/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel_V1.cu"
#include "kernel_V2.cu"
#include "kernel_V3.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

	float *in_h, *out_h;
	float *in_d, *out_d;
    unsigned size;
    unsigned version;
	cudaError_t cuda_ret;

	/* Allocate and initialize input vector */
    if(argc == 1) {
        size = 70;
        version = 1;
    } else if(argc == 2) {
        size = atoi(argv[1]);
        version = 1;
    } else if (argc == 3) {
        size = atoi(argv[1]);
        version = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./Final         # Input of size 70 is used"
           "\n    Usage: ./Final <m>     # Input of size m x m is used"
           "\n    Usage: ./Final <m> <v> # Input of size m x m, kernel version v"
           "\n");
        exit(0);
    }
    initMatrix(&in_h, size);

	/* Allocate and initialize output vector */
	out_h = (float*)calloc(size*size, sizeof(float));
	if(out_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", size);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

	cuda_ret = cudaMalloc((void**)&in_d, size*size*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
	cuda_ret = cudaMalloc((void**)&out_d, size*size*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, size*size*sizeof(float),
        cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

	cuda_ret = cudaMemset(out_d, 0, size*size*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel "); fflush(stdout);
    startTime(&timer);

    switch (version) {
        case 1:
            printf("V%d...", version);
            luFactorization_V1(out_d, in_d, size);
            break;

        case 2:
            printf("V%d...", version);
            luFactorization_V2(out_d, in_d, size);
            break;

        case 3:
            printf("V%d...", version);
            luFactorization_V3(out_d, in_d, size);
            break;

        default:
            printf("V%d...", 1);
            luFactorization_V1(out_d, in_d, size);
    }
    
    //gpuErrChk(cudaDeviceSynchronize());
	cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out_h, out_d, size*size*sizeof(float),
        cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, out_h, size);

    // Free memory ------------------------------------------------------------

	cudaFree(in_d); cudaFree(out_d);
	free(in_h); free(out_h);

	return 0;
}

