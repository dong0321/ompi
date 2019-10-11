/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include "opal/util/output.h"

#include "ompi/op/op.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/mca/op/arm_sve_op/op_sve.h"
#include "ompi/mca/op/arm_sve_op/op_sve_functions.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

/*
 * Since all the functions in this file are essentially identical, we
 * use a macro to substitute in names and types.  The core operation
 * in all functions that use this macro is the same.
 *
 * This macro is for (out op in).
 *
 * Support ops: max, min, for signed/unsigned 8,16,32,64
 *              sum, for integer 8,16,32,64
 *
 */
#define OP_SVE_FUNC(name, type_sign, type_size, type, op) \
    static void ompi_op_sve_2buff_##name##_##type(void *in, void *out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size;                                        \
    printf("op: %s  %s \n ", #op, #type_size);\
    int size = *count/step; \
    int i; \
    int round = size*64; \
    svbool_t Pg = svptrue_b##type_size(); \
}

/*
 *  This macro is for bit-wise operations (out op in).
 *
 *  Support ops: or, xor, and of 512 bits (representing integer data)
 *
 */
#define OP_SVE_BIT_FUNC(name, type_size, type, op) \
    static void ompi_op_sve_2buff_##name##_##type(void *in, void *out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size; \
    int size = *count/step; \
    int i; \
    int round = size*64; \
}

#define OP_SVE_FLOAT_FUNC(op) \
    static void ompi_op_sve_2buff_##op##_float(void *in, void *out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 16;                                                          \
    int size = *count/step; \
    int i; \
    int round = size*64; \
}

#define OP_SVE_DOUBLE_FUNC(op) \
    static void ompi_op_sve_2buff_##op##_double(void *in, void *out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 8;                                                          \
    int size = *count/step; \
    int i; \
}


/*************************************************************************
 * Max
 *************************************************************************/
    OP_SVE_FUNC(max, i, 8,    int8_t, max)
    OP_SVE_FUNC(max, u, 8,   uint8_t, max)
    OP_SVE_FUNC(max, i, 16,  int16_t, max)
    OP_SVE_FUNC(max, u, 16, uint16_t, max)
    OP_SVE_FUNC(max, i, 32,  int32_t, max)
    OP_SVE_FUNC(max, u, 32, uint32_t, max)
    OP_SVE_FUNC(max, i, 64,  int64_t, max)
    OP_SVE_FUNC(max, u, 64, uint64_t, max)

    /* Floating point */
    OP_SVE_FLOAT_FUNC(max)
    OP_SVE_DOUBLE_FUNC(max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_SVE_FUNC(min, i, 8,    int8_t, min)
    OP_SVE_FUNC(min, u, 8,   uint8_t, min)
    OP_SVE_FUNC(min, i, 16,  int16_t, min)
    OP_SVE_FUNC(min, u, 16, uint16_t, min)
    OP_SVE_FUNC(min, i, 32,  int32_t, min)
    OP_SVE_FUNC(min, u, 32, uint32_t, min)
    OP_SVE_FUNC(min, i, 64,  int64_t, min)
    OP_SVE_FUNC(min, u, 64, uint64_t, min)

    /* Floating point */
    OP_SVE_FLOAT_FUNC(min)
    OP_SVE_DOUBLE_FUNC(min)

/*************************************************************************
 * Sum
 ************************************************************************/
    OP_SVE_FUNC(sum, i, 8,    int8_t, add)
    OP_SVE_FUNC(sum, i, 8,   uint8_t, add)
    OP_SVE_FUNC(sum, i, 16,  int16_t, add)
    OP_SVE_FUNC(sum, i, 16, uint16_t, add)
    OP_SVE_FUNC(sum, i, 32,  int32_t, add)
    OP_SVE_FUNC(sum, i, 32, uint32_t, add)
    OP_SVE_FUNC(sum, i, 64,  int64_t, add)
    OP_SVE_FUNC(sum, i, 64, uint64_t, add)

    /* Floating point */
    OP_SVE_FLOAT_FUNC(add)
    OP_SVE_DOUBLE_FUNC(add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_SVE_FUNC(prod, i, 16,  int16_t, mullo)
    OP_SVE_FUNC(prod, i, 16, uint16_t, mullo)
    OP_SVE_FUNC(prod, i, 32,  int32_t, mullo)
    OP_SVE_FUNC(prod, i ,32, uint32_t, mullo)
    OP_SVE_FUNC(prod, i, 64,  int64_t, mullo)
    OP_SVE_FUNC(prod, i, 64, uint64_t, mullo)

    /* Floating point */
    OP_SVE_FLOAT_FUNC(mul)
    OP_SVE_DOUBLE_FUNC(mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_SVE_BIT_FUNC(band, 8,    int8_t, and)
    OP_SVE_BIT_FUNC(band, 8,   uint8_t, and)
    OP_SVE_BIT_FUNC(band, 16,  int16_t, and)
    OP_SVE_BIT_FUNC(band, 16, uint16_t, and)
    OP_SVE_BIT_FUNC(band, 32,  int32_t, and)
    OP_SVE_BIT_FUNC(band, 32, uint32_t, and)
    OP_SVE_BIT_FUNC(band, 64,  int64_t, and)
    OP_SVE_BIT_FUNC(band, 64, uint64_t, and)

    OP_SVE_FLOAT_FUNC(and)
    OP_SVE_DOUBLE_FUNC(and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_SVE_BIT_FUNC(bor, 8,    int8_t, or)
    OP_SVE_BIT_FUNC(bor, 8,   uint8_t, or)
    OP_SVE_BIT_FUNC(bor, 16,  int16_t, or)
    OP_SVE_BIT_FUNC(bor, 16, uint16_t, or)
    OP_SVE_BIT_FUNC(bor, 32,  int32_t, or)
    OP_SVE_BIT_FUNC(bor, 32, uint32_t, or)
    OP_SVE_BIT_FUNC(bor, 64,  int64_t, or)
    OP_SVE_BIT_FUNC(bor, 64, uint64_t, or)

    OP_SVE_FLOAT_FUNC(or)
    OP_SVE_DOUBLE_FUNC(or)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_SVE_BIT_FUNC(bxor, 8,    int8_t, xor)
    OP_SVE_BIT_FUNC(bxor, 8,   uint8_t, xor)
    OP_SVE_BIT_FUNC(bxor, 16,  int16_t, xor)
    OP_SVE_BIT_FUNC(bxor, 16, uint16_t, xor)
    OP_SVE_BIT_FUNC(bxor, 32,  int32_t, xor)
    OP_SVE_BIT_FUNC(bxor, 32, uint32_t, xor)
    OP_SVE_BIT_FUNC(bxor, 64,  int64_t, xor)
    OP_SVE_BIT_FUNC(bxor, 64, uint64_t, xor)

    OP_SVE_FLOAT_FUNC(xor)
    OP_SVE_DOUBLE_FUNC(xor)

/*
 *  This is a three buffer (2 input and 1 output) version of the reduction
 *  routines, needed for some optimizations.
 */
#define OP_SVE_FUNC_3BUFF(name, type_sign, type_size, type, op)\
        static void ompi_op_sve_3buff_##name##_##type(void * restrict in1,   \
                void * restrict in2, void * restrict out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size; \
    int size = *count/step; \
    int i; \
    int round = size*64; \
}

#define OP_SVE_BIT_FUNC_3BUFF(name, type_size, type, op) \
        static void ompi_op_sve_3buff_##op##_##type(void *in1, void *in2, void *out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size; \
    int size = *count/step; \
    int i; \
}

#define OP_SVE_FLOAT_FUNC_3BUFF(op) \
        static void ompi_op_sve_3buff_##op##_float(void *in1, void *in2, void *out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 16;                                                          \
    int size = *count/step; \
    int i; \
    int round = size*64; \
}

#define OP_SVE_DOUBLE_FUNC_3BUFF(op) \
        static void ompi_op_sve_3buff_##op##_double(void *in1, void *in2, void *out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 8;                                                          \
    int size = *count/step; \
    int i; \
}

/*************************************************************************
 * Max
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(max, i, 8,    int8_t, max)
    OP_SVE_FUNC_3BUFF(max, u, 8,   uint8_t, max)
    OP_SVE_FUNC_3BUFF(max, i, 16,  int16_t, max)
    OP_SVE_FUNC_3BUFF(max, u, 16, uint16_t, max)
    OP_SVE_FUNC_3BUFF(max, i, 32,  int32_t, max)
    OP_SVE_FUNC_3BUFF(max, u, 32, uint32_t, max)
    OP_SVE_FUNC_3BUFF(max, i, 64,  int64_t, max)
    OP_SVE_FUNC_3BUFF(max, u, 64, uint64_t, max)

    /* Floating point */
    OP_SVE_FLOAT_FUNC_3BUFF(max)
    OP_SVE_DOUBLE_FUNC_3BUFF(max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(min, i, 8,    int8_t, min)
    OP_SVE_FUNC_3BUFF(min, u, 8,   uint8_t, min)
    OP_SVE_FUNC_3BUFF(min, i, 16,  int16_t, min)
    OP_SVE_FUNC_3BUFF(min, u, 16, uint16_t, min)
    OP_SVE_FUNC_3BUFF(min, i, 32,  int32_t, min)
    OP_SVE_FUNC_3BUFF(min, u, 32, uint32_t, min)
    OP_SVE_FUNC_3BUFF(min, i, 64,  int64_t, min)
    OP_SVE_FUNC_3BUFF(min, u, 64, uint64_t, min)

    /* Floating point */
    OP_SVE_FLOAT_FUNC_3BUFF(min)
    OP_SVE_DOUBLE_FUNC_3BUFF(min)

/*************************************************************************
 * Sum
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(sum, i, 8,    int8_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 8,   uint8_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 16,  int16_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 16, uint16_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 32,  int32_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 32, uint32_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 64,  int64_t, add)
    OP_SVE_FUNC_3BUFF(sum, i, 64, uint64_t, add)

    /* Floating point */
    OP_SVE_FLOAT_FUNC_3BUFF(add)
    OP_SVE_DOUBLE_FUNC_3BUFF(add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(prod, i, 16,  int16_t, mullo)
    OP_SVE_FUNC_3BUFF(prod, i, 16, uint16_t, mullo)
    OP_SVE_FUNC_3BUFF(prod, i, 32,  int32_t, mullo)
    OP_SVE_FUNC_3BUFF(prod, i ,32, uint32_t, mullo)
    OP_SVE_FUNC_3BUFF(prod, i, 64,  int64_t, mullo)
    OP_SVE_FUNC_3BUFF(prod, i, 64, uint64_t, mullo)

    /* Floating point */
    OP_SVE_FLOAT_FUNC_3BUFF(mul)
    OP_SVE_DOUBLE_FUNC_3BUFF(mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_SVE_BIT_FUNC_3BUFF(band, 8,    int8_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 8,   uint8_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 16,  int16_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 16, uint16_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 32,  int32_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 32, uint32_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 64,  int64_t, and)
    OP_SVE_BIT_FUNC_3BUFF(band, 64, uint64_t, and)

    OP_SVE_FLOAT_FUNC_3BUFF(and)
    OP_SVE_DOUBLE_FUNC_3BUFF(and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_SVE_BIT_FUNC_3BUFF(bor, 8,    int8_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 8,   uint8_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 16,  int16_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 16, uint16_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 32,  int32_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 32, uint32_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 64,  int64_t, or)
    OP_SVE_BIT_FUNC_3BUFF(bor, 64, uint64_t, or)

    OP_SVE_FLOAT_FUNC_3BUFF(or)
    OP_SVE_DOUBLE_FUNC_3BUFF(or)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_SVE_BIT_FUNC_3BUFF(bxor, 8,    int8_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 8,   uint8_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 16,  int16_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 16, uint16_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 32,  int32_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 32, uint32_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 64,  int64_t, xor)
    OP_SVE_BIT_FUNC_3BUFF(bxor, 64, uint64_t, xor)

    OP_SVE_FLOAT_FUNC_3BUFF(xor)
    OP_SVE_DOUBLE_FUNC_3BUFF(xor)


/** C integer ***********************************************************/
#define C_INTEGER(name, ftype)                                              \
    [OMPI_OP_BASE_TYPE_INT8_T] = ompi_op_sve_##ftype##_##name##_int8_t,     \
    [OMPI_OP_BASE_TYPE_UINT8_T] = ompi_op_sve_##ftype##_##name##_uint8_t,   \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_sve_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_sve_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_sve_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_sve_##ftype##_##name##_uint32_t, \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_sve_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_sve_##ftype##_##name##_uint64_t


/** Floating point, including all the Fortran reals *********************/
#define FLOAT(name, ftype) ompi_op_sve_##ftype##_##name##_float
#define DOUBLE(name, ftype) ompi_op_sve_##ftype##_##name##_double

#define FLOATING_POINT(name, ftype)                                                            \
    [OMPI_OP_BASE_TYPE_SHORT_FLOAT] = NULL, \
    [OMPI_OP_BASE_TYPE_FLOAT] = FLOAT(name, ftype),                                              \
    [OMPI_OP_BASE_TYPE_DOUBLE] = DOUBLE(name, ftype)

#define C_INTEGER_PROD(name, ftype)                                           \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_sve_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_sve_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_sve_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_sve_##ftype##_##name##_uint32_t, \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_sve_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_sve_##ftype##_##name##_uint64_t


/*
 * MPI_OP_NULL
 * All types
 */
#define FLAGS_NO_FLOAT \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | OMPI_OP_FLAGS_COMMUTE)
#define FLAGS \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | \
         OMPI_OP_FLAGS_FLOAT_ASSOC | OMPI_OP_FLAGS_COMMUTE)

ompi_op_base_handler_fn_t ompi_op_sve_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 2buff),
        FLOATING_POINT(max, 2buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 2buff),
        FLOATING_POINT(min, 2buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 2buff),
        FLOATING_POINT(add, 2buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER_PROD(prod, 2buff),
        FLOATING_POINT(mul, 2buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] = {
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(band, 2buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(bor, 2buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(bxor, 2buff),
    },
    /* Corresponds to MPI_REPLACE */
    [OMPI_OP_BASE_FORTRAN_REPLACE] = {
        /* (MPI_ACCUMULATE is handled differently than the other
           reductions, so just zero out its function
           impementations here to ensure that users don't invoke
           MPI_REPLACE with any reduction operations other than
           ACCUMULATE) */
        NULL,
    },

};

ompi_op_base_3buff_handler_fn_t ompi_op_sve_3buff_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 3buff),
        FLOATING_POINT(max, 3buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 3buff),
        FLOATING_POINT(min, 3buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 3buff),
        FLOATING_POINT(add, 3buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER_PROD(prod, 3buff),
        FLOATING_POINT(mul, 3buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] ={
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(and, 3buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(or, 3buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(xor, 3buff),
    },
    /* Corresponds to MPI_REPLACE */
    [OMPI_OP_BASE_FORTRAN_REPLACE] = {
        /* MPI_ACCUMULATE is handled differently than the other
           reductions, so just zero out its function
           impementations here to ensure that users don't invoke
           MPI_REPLACE with any reduction operations other than
           ACCUMULATE */
        NULL,
    },
};
