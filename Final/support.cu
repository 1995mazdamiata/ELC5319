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

// reconstruct L*U = A and compare to A 
void verify(float* input, float* output, unsigned size) {

  const double relativeTolerance = 2e-5;

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
        double sum = 0.0;
        int kmax = (i < j) ? i : j;
        for(int t = 0; t <= kmax; ++t) {
            double l = (t == i) ? 1.0 : (double)output[i*size + t];
            double u = (double)output[t*size + j];
            sum += l*u;
        }

        double relErr = fabs(sum - (double)input[i*size + j])/sum;
        if(relErr > relativeTolerance) {
            printf("TEST FAILED at i = %d, j = %d, cpu = %0.3f, gpu = %0.3f\n\n", i, j, input[i*size  + j], sum);
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

