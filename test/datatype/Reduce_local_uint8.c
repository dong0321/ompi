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

uint8_t b[ARRAYSIZE];
uint8_t c[ARRAYSIZE];

int main(int argc, char **argv) {

    char *numbits = argv[1];
    int count = atoi(numbits);
    char *sve_yes = argv[2];
    printf("#Sum %d elems, option %c \n",count, *sve_yes);

    for (int i=0; i<count; i++)
    {
        b[i] = 2;
        c[i] = 1;
    }
    int rank, size, len;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double tstart, tend;

    if(*sve_yes=='s') {
        printf("#Local Reduce SUM: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_UINT8_T, MPI_SUM);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='p') {
        printf("#Local Reduce PROD: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_UINT8_T, MPI_PROD);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='a') {
        printf("#Local Reduce MAX: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_UINT8_T, MPI_MAX);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='i') {
        printf("#Local Reduce MIN: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_UINT8_T, MPI_MIN);
        tend = MPI_Wtime();
    }

    for (int i=0; i<count; i++)
    {
        printf("Psum%d %d \n",i, c[i] );
    }

    printf("#Local Reduce total_bytes %d time %.6f seconds\n", count, tend-tstart);
    return 0;
}

