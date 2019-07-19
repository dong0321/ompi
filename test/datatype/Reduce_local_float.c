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

float b[ARRAYSIZE];
float c[ARRAYSIZE];

int main(int argc, char **argv) {

    char *numbits = argv[1];
    int count = atoi(numbits);
    char *sve_yes = argv[2];
    printf("#Sum %d elems, option %c \n",count, *sve_yes);

    for (int i=0; i<count; i++)
    {
        b[i] = 1.0+1;
        c[i] = 1.0+2;
    }
    int rank, size, len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double tstart, tend;

    if(*sve_yes=='s') {
        printf("#Local Reduce SUM: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_FLOAT, MPI_SUM);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='p') {
        printf("#Local Reduce PROD: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_FLOAT, MPI_PROD);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='a') {
        printf("#Local Reduce MAX: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_FLOAT, MPI_MAX);
        tend = MPI_Wtime();
    }

    if(*sve_yes=='i') {
        printf("#Local Reduce MIN: %d \n", count);
        tstart = MPI_Wtime();
        MPI_Reduce_local(b,c,count, MPI_FLOAT, MPI_MIN);
        tend = MPI_Wtime();
    }
        for (int i=0; i<count; i++)
        {
            printf("Psum%d %f \n",i, c[i] );
        }

        return 0;
    }

