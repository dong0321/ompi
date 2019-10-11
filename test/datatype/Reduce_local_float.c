#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

#include "mpi.h"

#define ARRAYSIZE  1024*1024

float in_float[ARRAYSIZE];
float inout_float[ARRAYSIZE];

int8_t in_uint8[ARRAYSIZE];
int8_t inout_uint8[ARRAYSIZE];

uint16_t in_uint16[ARRAYSIZE];
uint16_t inout_uint16[ARRAYSIZE];

uint32_t in_uint32[ARRAYSIZE];
uint32_t inout_uint32[ARRAYSIZE];

uint64_t in_uint64[ARRAYSIZE];
uint64_t inout_uint64[ARRAYSIZE];

double in_double[ARRAYSIZE];
double inout_double[ARRAYSIZE];


int main(int argc, char **argv) {

    char *num_elem = argv[1];
    int count = atoi(num_elem);
    char *type = argv[2];
    char *elem_size = argv[3];
    int elem_size1  = atoi(elem_size);
    char *op = argv[4];

    printf("Sum %d elems, option %c \n",count, *type);
    int i;

    for (i=0; i<count; i++)
    {
        in_float[i] = 1000.0+1;
        inout_float[i] = 100.0+2;

        in_double[i] = 10.0+1;
        inout_double[i] = 1.0+2;

        in_uint8[i] = 2;
        inout_uint8[i] = 3;

        in_uint16[i] = 2;
        inout_uint16[i] = 3;

        in_uint32[i] = 2;
        inout_uint32[i] = 3;

        in_uint64[i] = 2;
        inout_uint64[i] = 3;
    }
    int rank, size, len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double tstart, tend;

    if(*type=='i') {
        if(strcmp(op, "sum") == 0){
            printf("#Local Reduce SUM: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_SUM);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_SUM);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_SUM);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_SUM);
             tend = MPI_Wtime();
        }

        if(strcmp(op, "max") == 0){
            printf("#Local Reduce MAX: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_MAX);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_MAX);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_MAX);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_MAX);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "min") == 0) {
            printf("#Local Reduce MIN: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_MIN);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_MIN);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_MIN);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_MIN);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "band") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_BAND);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_BAND);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BAND);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_BAND);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "bor") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_BOR);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_BOR);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BOR);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_BOR);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "bxor") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_BXOR);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_BXOR);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BXOR);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_BXOR);
            tend = MPI_Wtime();
        }


        if(strcmp(op, "mul") == 0) {
            printf("#Local Reduce Logical: %d \n", count);
            tstart = MPI_Wtime();
            if (elem_size1 == 8)
                MPI_Reduce_local(in_uint8,inout_uint8,count, MPI_INT8_T, MPI_PROD);
            if (elem_size1 == 16)
                MPI_Reduce_local(in_uint16,inout_uint16,count, MPI_INT16_T, MPI_PROD);
            if (elem_size1 == 32)
                MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_PROD);
            if (elem_size1 == 64)
                MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_PROD);
            tend = MPI_Wtime();
        }
    }


    if(*type=='f') {
        if(strcmp(op, "sum") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_SUM);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "max") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_MAX);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "min") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_MIN);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "mul") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_float,inout_float,count, MPI_FLOAT, MPI_PROD);
            tend = MPI_Wtime();
        }
    }

    if(*type=='d') {
        if(strcmp(op, "sum") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_SUM);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "max") == 0){
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_MAX);
            tend = MPI_Wtime();
        }

        if(strcmp(op, "min") == 0) {
            tstart = MPI_Wtime();
            MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_MIN);
            tend = MPI_Wtime();
        }

         if(strcmp(op, "mul") == 0) {
             tstart = MPI_Wtime();
             MPI_Reduce_local(in_double,inout_double,count, MPI_DOUBLE, MPI_PROD);
             tend = MPI_Wtime();
         }
    }
/*
    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" float: %f", inout_float[i]);
        if((i+1)%16==0)
            printf("\n");
    }

    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" out8: %d", inout_uint8[i]);
        if((i+1)%64==0)
            printf("\n");
    }
    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" out16: %d",inout_uint16[i]);
        if((i+1)%32==0)
            printf("\n");
    }
    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" out32: %d",inout_uint32[i]);
        if((i+1)%16==0)
            printf("\n");
    }

    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" out64: %d",inout_uint64[i]);
        if((i+1)%8==0)
            printf("\n");
    }

    printf("\n ================\n");
    for (i=0; i<count; i++)
    {
        printf(" float: %f", inout_double[i]);
        if((i+1)%8==0)
            printf("\n");
    }
*/
    printf("#Local Reduce time %.6f seconds\n", tend-tstart);
    MPI_Finalize();
    return 0;
}

