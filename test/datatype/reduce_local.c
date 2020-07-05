/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2019-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

#include "mpi.h"
#include "ompi/communicator/communicator.h"
#include "ompi/runtime/mpiruntime.h"
#include "ompi/datatype/ompi_datatype.h"

typedef struct op_name_s {
    char* name;
    MPI_Op op;
} op_name_t;
static op_name_t array_of_ops [] = {
    { "max", MPI_MAX },
    { "min", MPI_MIN },
    { "sum", MPI_SUM },
    { "prod", MPI_PROD },
    { "land", MPI_LAND },
    { "band", MPI_BAND },
    { "lor", MPI_LOR },
    { "bor", MPI_BOR },
    { "lxor", MPI_LXOR },
    { "bxor", MPI_BXOR },
    { "replace", MPI_REPLACE },
    { NULL, MPI_OP_NULL }
};

static int verbose = 0;

static void print_status(char* op, char* type, int type_size,
                         int count, double duration,
                         int correct )
{
    if(correct) {
        printf("%s %s %d%s ", op, type, type_size, (verbose ? " [\033[1;32msuccess\033[0m]" : ""));
    } else {
        printf("%s %s [\033[1;31mfail\033[0m]", op, type);
    }
    printf(" count  %d  time %.6f seconds\n", count, duration);
}

int main(int argc, char **argv)
{
    static void *in_buf = NULL, *inout_buf = NULL, *inout_check_buf = NULL;
    int count, type_size = 8, rank, size, provided, correctness = 1;
    int repeats = 1, i, c;
    double tstart, tend;
    bool check = true;
    char type = 'u', *op = "sum", *mpi_op, *mpi_type;
    int lower = 1, upper = 1000000;

    while( -1 != (c = getopt(argc, argv, "l:u:t:o:s:n:vfh")) ) {
        switch(c) {
        case 'l':
            lower = atoi(optarg);
            if( lower <= 0 ) {
                fprintf(stderr, "The number of elements must be positive\n");
                exit(-1);
            }
            break;
        case 'u':
            upper = atoi(optarg);
            break;
        case 'f':
            check = false;
            break;
        case 'v':
            verbose++;
            break;
        case 'r':
            repeats = atoi(optarg);
            if( repeats <= 0 ) {
                fprintf(stderr, "The number of repetitions (%d) must be positive\n", repeats);
                exit(-1);
            }
            break;
        case 't':
            if( 0 == strncmp(optarg, "i", 1) ) {
                type = 'i';
            } else if( 0 == strncmp(optarg, "u", 1) ) {
                type = 'u';
            } else if( 0 == strncmp(optarg, "f", 1) ) {
                type = 'f';
            } else if( 0 == strncmp(optarg, "d", 1) ) {
                type = 'd';
            } else {
                fprintf(stderr, "type must be i (signed int), u (unsigned int), f (float) or d (double)\n");
                exit(-1);
            }
            break;
        case 'o':
            for( i = 0; NULL != array_of_ops[i].name; i++ ) {
                if( 0 == strcmp(array_of_ops[i].name, optarg) ) {
                    op = optarg;
                    break;
                }
            }
            if( NULL == array_of_ops[i].name ) {
                fprintf(stderr, "Unknown op %s\n", optarg);
                exit(-1);
            }
            break;
        case 's':
            type_size = atoi(optarg);
            if( ! ((8 == type_size) || (16 == type_size) || (32 == type_size) || (64 == type_size)) ) {
                fprintf(stderr, "type_size must be 8, 16, 32 or 64. %d is an invalid value\n",
                        type_size);
                exit(-1);
            }
            break;
        case 'h':
            fprintf(stdout, "%s options are:\n"
                    " -l <number> : lower number of elements\n"
                    " -u <number> : upper number of elements\n"
                    " -s <type_size> : 8, 16, 32 or 64 bits elements\n"
                    " -t [i,u,f,d] : type of the elements to apply the operations on\n"
                    " -o <op> : comma separated list of operations to execute among\n"
                    "           sum, min, max, prod, bor, bxor, band\n"
                    " -h: this help message\n", argv[0]);
            exit(0);
        }
    }

    in_buf          = malloc(upper * sizeof(double));
    inout_buf       = malloc(upper * sizeof(double));
    inout_check_buf = malloc(upper * sizeof(double));

    ompi_mpi_init(argc, argv, MPI_THREAD_SERIALIZED, &provided, false);

    rank = ompi_comm_rank(MPI_COMM_WORLD); (void)rank;
    size = ompi_comm_size(MPI_COMM_WORLD); (void)size;

    for( count = lower; count <= upper; count += count ) {
        mpi_op = NULL; mpi_type = NULL;
        correctness = 1;
        if(type=='i') {
            if( 8 == type_size ) {
                int8_t *in_int8 = (int8_t*)in_buf,
                    *inout_int8 = (int8_t*)inout_buf,
                    *inout_int8_for_check = (int8_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_int8[i] = 5;
                    inout_int8[i] = inout_int8_for_check[i] = -3;
                }
                mpi_type = "MPI_INT8_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int8, inout_int8, count, MPI_INT8_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == (int8_t)(in_int8[i] + inout_int8_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                i, in_int8[i], op, inout_int8_for_check[i], inout_int8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int8, inout_int8, count, MPI_INT8_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == in_int8[i])
                                continue;
                            printf("First error at position %d (%d != %s(%d))\n",
                                   i, inout_int8[i], op, in_int8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {  //intentionly reversed in and out
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_int8, in_int8, count, MPI_INT8_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == in_int8[i])
                                continue;
                            printf("First error at position %d (%d != %s(%d))\n",
                                   i, inout_int8[i], op, in_int8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int8, inout_int8, count, MPI_INT8_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == (in_int8[i] | inout_int8_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int8[i], op, inout_int8_for_check[i], inout_int8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int8, inout_int8, count, MPI_INT8_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == (in_int8[i] ^ inout_int8_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int8[i], op, inout_int8_for_check[i], inout_int8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int8, inout_int8, count, MPI_INT8_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == (int8_t)(in_int8[i] * inout_int8_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int8[i], op, inout_int8_for_check[i], inout_int8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int8, inout_int8, count, MPI_INT8_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int8[i] == (in_int8[i] & inout_int8_for_check[i]) )
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int8[i], op, inout_int8_for_check[i], inout_int8[i]);
                                printf("First error at position %d\n", i);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
            if( 16 == type_size ) {
                int16_t *in_int16 = (int16_t*)in_buf,
                    *inout_int16 = (int16_t*)inout_buf,
                    *inout_int16_for_check = (int16_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_int16[i] = 5;
                    inout_int16[i] = inout_int16_for_check[i] = -3;
                }
                mpi_type = "MPI_INT16_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int16, inout_int16, count, MPI_INT16_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int16[i] == (int16_t)(in_int16[i] + inout_int16_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int16[i], op, inout_int16_for_check[i], inout_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int16, inout_int16, count, MPI_INT16_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ )  {
                            if(inout_int16[i] == in_int16[i])
                                continue;
                            printf("First error at position %d (%d != %s(%d))\n",
                                   i, inout_int16[i], op, in_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_int16, in_int16, count, MPI_INT16_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int16[i] == in_int16[i])
                                continue;
                            printf("First error at position %d (%d != %s(%d))\n",
                                   i, inout_int16[i], op, in_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int16, inout_int16, count, MPI_INT16_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int16[i] == (in_int16[i] | inout_int16_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int16[i], op, inout_int16_for_check[i], inout_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int16, inout_int16, count, MPI_INT16_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int16[i] == (in_int16[i] ^ inout_int16_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int16[i], op, inout_int16_for_check[i], inout_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int16, inout_int16, count, MPI_INT16_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int16[i] == (int16_t)(in_int16[i] * inout_int16_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int16[i], op, inout_int16_for_check[i], inout_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int16, inout_int16, count, MPI_INT16_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int16[i] == (in_int16[i] & inout_int16_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int16[i], op, inout_int16_for_check[i], inout_int16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
            if( 32 == type_size ) {
                int32_t *in_int32 = (int32_t*)in_buf,
                    *inout_int32 = (int32_t*)inout_buf,
                    *inout_int32_for_check = (int32_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_int32[i] = 5;
                    inout_int32[i] = inout_int32_for_check[i] = 3;
                }
                mpi_type = "MPI_INT32_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int32, inout_int32, count, MPI_INT32_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == (int32_t)(in_int32[i] + inout_int32_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int32[i], op, inout_int32_for_check[i], inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int32, inout_int32, count, MPI_INT32_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == in_int32[i])
                                continue;
                            printf("First error at position %d (%d != %s(%d))\n",
                                   i, in_int32[i], op, inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_int32, in_int32, count, MPI_INT32_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == in_int32[i])
                                continue;
                            printf("First error at position %d (%d != %s(%d))\n",
                                   i, in_int32[i], op, inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int32, inout_int32, count, MPI_INT32_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == (in_int32[i] | inout_int32_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int32[i], op, inout_int32_for_check[i], inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int32, inout_int32, count, MPI_INT32_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == (int32_t)(in_int32[i] * inout_int32_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int32[i], op, inout_int32_for_check[i], inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int32, inout_int32, count, MPI_INT32_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == (in_int32[i] & inout_int32_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int32[i], op, inout_int32_for_check[i], inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int32, inout_int32, count, MPI_INT32_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int32[i] == (in_int32[i] ^ inout_int32_for_check[i]))
                                continue;
                            printf("First error at position %d (%d %s %d != %d)\n",
                                   i, in_int32[i], op, inout_int32_for_check[i], inout_int32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
            if( 64 == type_size ) {
                int64_t *in_int64 = (int64_t*)in_buf,
                    *inout_int64 = (int64_t*)inout_buf,
                    *inout_int64_for_check = (int64_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_int64[i] = 5;
                    inout_int64[i] = inout_int64_for_check[i] = 3;
                }
                mpi_type = "MPI_INT64_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int64, inout_int64, count, MPI_INT64_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == (int64_t)(in_int64[i] + inout_int64_for_check[i]))
                                continue;
                            printf("First error at position %d (%lld %s %lld != %lld)\n",
                                   i, in_int64[i], op, inout_int64_for_check[i], inout_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int64, inout_int64, count, MPI_INT64_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == in_int64[i])
                                continue;
                            printf("First error at position %d (%lld != %s(%lld))\n",
                                   i, inout_int64[i], op, in_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_int64, in_int64, count, MPI_INT64_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == in_int64[i])
                                continue;
                            printf("First error at position %d (%lld != %s(%lld))\n",
                                   i, inout_int64[i], op, in_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_int64, in_int64, count, MPI_INT64_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == in_int64[i])
                                continue;
                            printf("First error at position %d (%lld %s %lld != %lld)\n",
                                   i, in_int64[i], op, inout_int64_for_check[i], inout_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int64, inout_int64, count, MPI_INT64_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == (in_int64[i] | inout_int64_for_check[i]))
                                continue;
                            printf("First error at position %d (%lld %s %lld != %lld)\n",
                                   i, in_int64[i], op, inout_int64_for_check[i], inout_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int64, inout_int64, count, MPI_INT64_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == (in_int64[i] ^ inout_int64_for_check[i]))
                                continue;
                            printf("First error at position %d (%lld %s %lld != %lld)\n",
                                   i, in_int64[i], op, inout_int64_for_check[i], inout_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int64,inout_int64,count, MPI_INT64_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == (int64_t)(in_int64[i] * inout_int64_for_check[i]))
                                continue;
                            printf("First error at position %d (%lld %s %lld != %lld)\n",
                                   i, in_int64[i], op, inout_int64_for_check[i], inout_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_int64, inout_int64, count, MPI_INT64_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_int64[i] == (in_int64[i] & inout_int64_for_check[i]) )
                                continue;
                            printf("First error at position %d (%lld %s %lld != %lld)\n",
                                   i, in_int64[i], op, inout_int64_for_check[i], inout_int64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
        }

        if( type == 'u' ) {
            if( 8 == type_size ) {
                uint8_t *in_uint8 = (uint8_t*)in_buf,
                    *inout_uint8 = (uint8_t*)inout_buf,
                    *inout_uint8_for_check = (uint8_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_uint8[i] = 5;
                    inout_uint8[i] = inout_uint8_for_check[i] = 3;
                }
                mpi_type = "MPI_UINT8_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == (uint8_t)(in_uint8[i] + inout_uint8_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint8[i], op, inout_uint8_for_check[i], inout_uint8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == inout_uint8_for_check[i])
                                continue;
                            printf("First error at position %d (%u != %s(%u))\n",
                                   i, inout_uint8[i], op, inout_uint8_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {  //intentionly reversed in and out
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == in_uint8[i])
                                continue;
                            printf("First error at position %d (%u != %s(%u))\n",
                                   i, inout_uint8[i], op, inout_uint8_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == (in_uint8[i] | inout_uint8_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint8[i], op, inout_uint8_for_check[i], inout_uint8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == (in_uint8[i] ^ inout_uint8_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint8[i], op, inout_uint8_for_check[i], inout_uint8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == (uint8_t)(in_uint8[i] * inout_uint8_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint8[i], op, inout_uint8_for_check[i], inout_uint8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint8, inout_uint8, count, MPI_UINT8_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint8[i] == (in_uint8[i] & inout_uint8_for_check[i]) )
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint8[i], op, inout_uint8_for_check[i], inout_uint8[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
            if( 16 == type_size ) {
                uint16_t *in_uint16 = (uint16_t*)in_buf,
                    *inout_uint16 = (uint16_t*)inout_buf,
                    *inout_uint16_for_check = (uint16_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_uint16[i] = 5;
                    inout_uint16[i] = inout_uint16_for_check[i] = 3;
                }
                mpi_type = "MPI_UINT16_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint16[i] == (uint16_t)(in_uint16[i] + inout_uint16_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint16[i], op, inout_uint16_for_check[i], inout_uint16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ )  {
                            if(inout_uint16[i] == inout_uint16_for_check[i])
                                continue;
                            printf("First error at position %d (%u != %s(%u))\n",
                                   i, inout_uint16[i], op, inout_uint16_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint16[i] == in_uint16[i])
                                continue;
                            printf("First error at position %d (%u != %s(%u))\n",
                                   i, inout_uint16[i], op, inout_uint16_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint16[i] == (in_uint16[i] | inout_uint16_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint16[i], op, inout_uint16_for_check[i], inout_uint16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint16[i] == (in_uint16[i] ^ inout_uint16_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint16[i], op, inout_uint16_for_check[i], inout_uint16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint16[i] == (uint16_t)(in_uint16[i] * inout_uint16_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint16[i], op, inout_uint16_for_check[i], inout_uint16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint16, inout_uint16, count, MPI_UINT16_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint16[i] == (in_uint16[i] & inout_uint16_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint16[i], op, inout_uint16_for_check[i], inout_uint16[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
            if( 32 == type_size ) {
                uint32_t *in_uint32 = (uint32_t*)in_buf,
                    *inout_uint32 = (uint32_t*)inout_buf,
                    *inout_uint32_for_check = (uint32_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_uint32[i] = 5;
                    inout_uint32[i] = inout_uint32_for_check[i] = 3;
                }
                mpi_type = "MPI_UINT32_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint32, inout_uint32, count, MPI_UINT32_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == (uint32_t)(in_uint32[i] + inout_uint32_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint32[i], op, inout_uint32_for_check[i], inout_uint32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint32, inout_uint32, count, MPI_UINT32_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == in_uint32[i])
                                continue;
                            printf("First error at position %d (%u != %s(%u))\n",
                                   i, inout_uint32[i], op, inout_uint32_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {  // we reverse the send and recv buffers
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_uint32, in_uint32, count, MPI_UINT32_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == in_uint32[i])
                                continue;
                            printf("First error at position %d (%u != %s(%u))\n",
                                   i, inout_uint32[i], op, inout_uint32_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint32,inout_uint32,count, MPI_UINT32_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == (in_uint32[i] | inout_uint32_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint32[i], op, inout_uint32_for_check[i], inout_uint32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint32, inout_uint32, count, MPI_UINT32_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == (uint32_t)(in_uint32[i] * inout_uint32_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint32[i], op, inout_uint32_for_check[i], inout_uint32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint32, inout_uint32, count, MPI_UINT32_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == (in_uint32[i] & inout_uint32_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint32[i], op, inout_uint32_for_check[i], inout_uint32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint32, inout_uint32, count, MPI_UINT32_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint32[i] == (in_uint32[i] ^ inout_uint32_for_check[i]))
                                continue;
                            printf("First error at position %d (%u %s %u != %u)\n",
                                   i, in_uint32[i], op, inout_uint32_for_check[i], inout_uint32[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
            if( 64 == type_size ) {
                int64_t *in_uint64 = (int64_t*)in_buf,
                    *inout_uint64 = (int64_t*)inout_buf,
                    *inout_uint64_for_check = (int64_t*)inout_check_buf;
                for( i = 0; i < count; i++ ) {
                    in_uint64[i] = 5;
                    inout_uint64[i] = inout_uint64_for_check[i] = 3;
                }
                mpi_type = "MPI_UINT64_T";

                if( 0 == strcmp(op, "sum") ) {
                    mpi_op = "MPI_SUM";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint64, inout_uint64, count, MPI_UINT64_T, MPI_SUM);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( correctness = 1, i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == (int64_t)(in_uint64[i] + inout_uint64_for_check[i]))
                                continue;
                            printf("First error at position %d (%llu %s %llu != %llu)\n",
                                   i, in_uint64[i], op, inout_uint64_for_check[i], inout_uint64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "max") ) {
                    mpi_op = "MPI_MAX";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint64, inout_uint64, count, MPI_UINT64_T, MPI_MAX);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == in_uint64[i])
                                continue;
                            printf("First error at position %d (%llu != %s(%llu))\n",
                                   i, inout_uint64[i], op, inout_uint64_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "min") ) {
                    mpi_op = "MPI_MIN";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(inout_uint64, in_uint64, count, MPI_UINT64_T, MPI_MIN);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == in_uint64[i])
                                continue;
                            printf("First error at position %d (%llu != %s(%llu))\n",
                                   i, inout_uint64[i], op, inout_uint64_for_check[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bor") ) {
                    mpi_op = "MPI_BOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint64, inout_uint64, count, MPI_UINT64_T, MPI_BOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == (in_uint64[i] | inout_uint64_for_check[i]))
                                continue;
                            printf("First error at position %d (%llu %s %llu != %llu)\n",
                                   i, in_uint64[i], op, inout_uint64_for_check[i], inout_uint64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "bxor") ) {
                    mpi_op = "MPI_BXOR";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint64, inout_uint64, count, MPI_UINT64_T, MPI_BXOR);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == (in_uint64[i] ^ inout_uint64_for_check[i]))
                                continue;
                            printf("First error at position %d (%llu %s %llu != %llu)\n",
                                   i, in_uint64[i], op, inout_uint64_for_check[i], inout_uint64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "mul") ) {
                    mpi_op = "MPI_PROD";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint64,inout_uint64,count, MPI_UINT64_T, MPI_PROD);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == (int64_t)(in_uint64[i] * inout_uint64_for_check[i]))
                                continue;
                            printf("First error at position %d (%llu %s %llu != %llu)\n",
                                   i, in_uint64[i], op, inout_uint64_for_check[i], inout_uint64[i]);
                            correctness = 0;
                            break;
                        }
                    }
                }
                if( 0 == strcmp(op, "band") ) {
                    mpi_op = "MPI_BAND";
                    tstart = MPI_Wtime();
                    MPI_Reduce_local(in_uint64, inout_uint64, count, MPI_UINT64_T, MPI_BAND);
                    tend = MPI_Wtime();
                    if( check ) {
                        for( i = 0; i < count; i++ ) {
                            if(inout_uint64[i] == (in_uint64[i] & inout_uint64_for_check[i]) )
                                continue;
                            printf("First error at position %d (%llu %s %llu != %llu)\n",
                                   i, in_uint64[i], op, inout_uint64_for_check[i], inout_uint64[i]);
                                printf("First error at position %d\n", i);
                            correctness = 0;
                            break;
                        }
                    }
                }
            }
        }

        if( type == 'f' ) {
            float *in_float = (float*)in_buf,
                *inout_float = (float*)inout_buf,
                *inout_float_for_check = (float*)inout_check_buf;
            for( i = 0; i < count; i++ ) {
                in_float[i] = 1000.0+1;
                inout_float[i] = inout_float_for_check[i] = 100.0+2;
            }
            mpi_type = "MPI_FLOAT";

            if( 0 == strcmp(op, "sum") ) {
                mpi_op = "MPI_SUM";
                tstart = MPI_Wtime();
                MPI_Reduce_local(in_float, inout_float, count, MPI_FLOAT, MPI_SUM);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_float[i] == inout_float_for_check[i]+in_float[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
            if( 0 == strcmp(op, "max") ) {
                mpi_op = "MPI_MAX";
                tstart = MPI_Wtime();
                MPI_Reduce_local(in_float, inout_float, count, MPI_FLOAT, MPI_MAX);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_float[i] == in_float[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
            if( 0 == strcmp(op, "min") ) {
                mpi_op = "MPI_MIN";
                tstart = MPI_Wtime();
                MPI_Reduce_local(inout_float,in_float,count, MPI_FLOAT, MPI_MIN);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_float[i] == in_float[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
            if( 0 == strcmp(op, "mul") ) {
                mpi_op = "MPI_PROD";
                tstart = MPI_Wtime();
                MPI_Reduce_local(in_float, inout_float, count, MPI_FLOAT, MPI_PROD);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_float[i] == in_float[i] * inout_float_for_check[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
        }

        if( type == 'd' ) {
            double *in_double = (double*)in_buf,
                *inout_double = (double*)inout_buf,
                *inout_double_for_check = (double*)inout_check_buf;
            for( i = 0; i < count; i++ ) {
                in_double[i] = 10.0+1;
                inout_double[i] = inout_double_for_check[i] = 1.0+2;
            }
            mpi_type = "MPI_DOUBLE";

            if( 0 == strcmp(op, "sum") ) {
                mpi_op = "MPI_SUM";
                tstart = MPI_Wtime();
                MPI_Reduce_local(in_double, inout_double, count, MPI_DOUBLE, MPI_SUM);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_double[i] == inout_double_for_check[i]+in_double[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
            if( 0 == strcmp(op, "max") ) {
                mpi_op = "MPI_MAX";
                tstart = MPI_Wtime();
                MPI_Reduce_local(in_double, inout_double, count, MPI_DOUBLE, MPI_MAX);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_double[i] == in_double[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
            if( 0 == strcmp(op, "min") ) {
                mpi_op = "MPI_MIN";
                tstart = MPI_Wtime();
                MPI_Reduce_local(inout_double, in_double, count, MPI_DOUBLE, MPI_MIN);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_double[i] == in_double[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
            if( 0 == strcmp(op, "mul") ) {
                mpi_op = "MPI_PROD";
                tstart = MPI_Wtime();
                MPI_Reduce_local(in_double, inout_double, count, MPI_DOUBLE, MPI_PROD);
                tend = MPI_Wtime();
                if( check ) {
                    for( i = 0; i < count; i++ ) {
                        if(inout_double[i] == inout_double_for_check[i]*in_double[i])
                            continue;
                        printf("First error at position %d\n", i);
                        correctness = 0;
                        break;
                    }
                }
            }
        }
        print_status(mpi_op, mpi_type, type_size, count, tend-tstart, correctness);
    }
    //tstart = MPI_Wtime(); 
    //memcpy(in_uint8,inout_uint8, count);
    //memcpy(in_float, inout_float, count);
    //memcpy(in_double, inout_double, count);
    ompi_mpi_finalize();

    free(in_buf);
    free(inout_buf);
    free(inout_check_buf);

    return correctness ? 0 : -1;
}

