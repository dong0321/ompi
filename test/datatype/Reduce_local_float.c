#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

#include "mpi.h"

#define ARRAYSIZE  1024*1024

float in_float[ARRAYSIZE];
float inout_float[ARRAYSIZE];

uint8_t in_unit8[ARRAYSIZE];
uint8_t inout_uint8[ARRAYSIZE];

int main(int argc, char **argv) {

    char *num_elem = argv[1];
    int count = atoi(num_elem);
    char *sve_yes = argv[2];
    printf("#Sum %d elems, option %c \n",count, *sve_yes);

    for (int i=0; i<count; i++)
    {
        in_float[i] = 1.0+1;
        inout_float[i] = 1.0+2;

        in_uint8[i] = 2;
        inout_uint8[i] = 1;
    }
    int rank, size, len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double tstart, tend;

    if(*sve_yes=='s') {
        printf("#Local Reduce SUM: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_SUM);
        MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_UINT8_T, MPI_SUM);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='p') {
        printf("#Local Reduce PROD: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_PROD);
        MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_UINT8_T, MPI_PROD);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='a') {
        printf("#Local Reduce MAX: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_MAX);
        MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_UINT8_T, MPI_MAX);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='i') {
        printf("#Local Reduce MIN: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_MIN);
        MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_UINT8_T, MPI_MIN);
        tend = MPI_Wtime();
    }

    return 0;
}

