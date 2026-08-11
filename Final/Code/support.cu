/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initMatrix(float **mat_h, unsigned size)
{
    *mat_h = (float*)malloc(size*size*sizeof(float));

    if(*mat_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i = 0; i < size; ++i) {
        float rowsum = 0.0f;

        for (unsigned int j = 0; j < size; ++j) {
            float v = (float)(rand() % 1000)/100.0f - 5.0f;
            (*mat_h)[i*size + j] = v;

            if (i != j) rowsum += fabs(v);
        }

        (*mat_h)[i*size + i] = (float)(rowsum + 10.0); //ensure diagonals are strong
    }
}

// Sequentially compute LU and compare with GPU results 
void verify(float* input, float* output, unsigned size) {

    const float relTol = 5e-4*size;

    float *A = (float*)calloc(size*size, sizeof(float));
    if(A == NULL) FATAL("Unable to allocate host");

    // Right-looking Doolittle algorithm for sequential LU factorization
    for(int i = 0; i < size; i++) {

        // U
        for(int k = i; k < size; k++) {
            float sum = 0.0f;
            for(int j = 0; j < i; j++) {
                sum += A[i*size + j] * A[j*size + k];
            }

            A[i*size + k] = input[i*size + k] - sum;
        }

        // L
        for (int k = i; k < size; k++) {
            if(i != k) {
                float sum = 0.0f;
                for(int j = 0; j < i; j++) {
                    sum += A[k*size + j] * A[j*size + i];
                }

                A[k*size + i] = (input[k*size + i] - sum) / A[i*size + i];
            }
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float relErr = (A[i*size + j] - output[i*size + j]) / A[i*size + j];

            if(relErr > relTol || relErr < -relTol) {
                printf("TEST FAILED at i = %d, j = %d, cpu = %0.5f, gpu = %0.5f, relErr = %0.5f\n\n", 
                       i, j, A[i*size + j], output[i*size + j], relErr);
                exit(0);
            }
        }
    }
    printf("TEST PASSED\n\n");
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

