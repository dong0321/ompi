/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Copyright (c) 2019      ARM Ltd.  All rights reserved.
 *
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
 */
#define OP_SVE_FUNC(name, type_name, type_size, type, op) \
    static void ompi_op_sve_2buff_##name##_##type_name(void *in, void *out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    uint64_t i; \
    uint64_t step = 0; \
    svbool_t Pg = svptrue_b##type_size(); \
    switch(type_size) {                                                \
        case 8:                                                        \
               step = svcntb();                                        \
               break;                                                  \
        case 16:                                                       \
               step = svcnth();                                        \
               break;                                                  \
        case 32:                                                       \
               step = svcntw();                                        \
               break;                                                  \
        case 64:                                                       \
               step = svcntd();                                        \
    }    \
    uint64_t round = *count;                                           \
    uint64_t remain = *count % step;                                   \
    for(i=0; i< round; i=i+step)                                       \
    {                                                                  \
        sv##type  vsrc = svld1(Pg, (type *)in+i);                    \
        sv##type  vdst = svld1(Pg, (type *)out+i);                   \
        vdst=sv##op##_z(Pg,vdst,vsrc);                               \
        svst1(Pg, (type *)out+i,vdst);                                 \
    }                                                                  \
    \
    if (remain !=0){                                                   \
        Pg = svwhilelt_b##type_size##_u64(0, remain);                  \
        sv##type  vsrc = svld1(Pg, (type *)in+i);                    \
        sv##type  vdst = svld1(Pg, (type *)out+i);                   \
        vdst=sv##op##_z(Pg,vdst,vsrc);                               \
        svst1(Pg, (type *)out+i,vdst);                                 \
    } \
}

/*************************************************************************
 * Max
 *************************************************************************/
    OP_SVE_FUNC(max, int8_t  ,  8,   int8_t, max)
    OP_SVE_FUNC(max, uint8_t,   8,  uint8_t, max)
    OP_SVE_FUNC(max, int16_t,  16,  int16_t, max)
    OP_SVE_FUNC(max, uint16_t, 16, uint16_t, max)
    OP_SVE_FUNC(max, int32_t,  32,  int32_t, max)
    OP_SVE_FUNC(max, uint32_t, 32, uint32_t, max)
    OP_SVE_FUNC(max, int64_t,  64,  int64_t, max)
    OP_SVE_FUNC(max, uint64_t, 64, uint64_t, max)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(max, short_float, 16, float16_t, max)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(max, short_float, 16, float16_t, max)
#endif
    OP_SVE_FUNC(max, float, 32, float32_t, max)
    OP_SVE_FUNC(max, double, 64, float64_t, max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_SVE_FUNC(min, int8_t  ,  8,   int8_t, min)
    OP_SVE_FUNC(min, uint8_t,   8,  uint8_t, min)
    OP_SVE_FUNC(min, int16_t,  16,  int16_t, min)
    OP_SVE_FUNC(min, uint16_t, 16, uint16_t, min)
    OP_SVE_FUNC(min, int32_t,  32,  int32_t, min)
    OP_SVE_FUNC(min, uint32_t, 32, uint32_t, min)
    OP_SVE_FUNC(min, int64_t,  64,  int64_t, min)
    OP_SVE_FUNC(min, uint64_t, 64, uint64_t, min)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(min, short_float, 16, float16_t, min)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(min, short_float, 16, float16_t, min)
#endif
    OP_SVE_FUNC(min, float, 32, float32_t, min)
    OP_SVE_FUNC(min, double, 64, float64_t, min)

/*************************************************************************
 * Sum
 ************************************************************************/
    OP_SVE_FUNC(sum, int8_t  ,  8,   int8_t, add)
    OP_SVE_FUNC(sum, uint8_t,   8,  uint8_t, add)
    OP_SVE_FUNC(sum, int16_t,  16,  int16_t, add)
    OP_SVE_FUNC(sum, uint16_t, 16, uint16_t, add)
    OP_SVE_FUNC(sum, int32_t,  32,  int32_t, add)
    OP_SVE_FUNC(sum, uint32_t, 32, uint32_t, add)
    OP_SVE_FUNC(sum, int64_t,  64,  int64_t, add)
    OP_SVE_FUNC(sum, uint64_t, 64, uint64_t, add)
    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(sum, short_float, 16, float16_t, add)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(sum, short_float, 16, float16_t, add)
#endif
    OP_SVE_FUNC(sum, float, 32, float32_t, add)
    OP_SVE_FUNC(sum, double, 64, float64_t, add)


/*************************************************************************
 * Product
 *************************************************************************/
    OP_SVE_FUNC(prod,   int8_t, 8,    int8_t, mul)
    OP_SVE_FUNC(prod,  uint8_t, 8,   uint8_t, mul)
    OP_SVE_FUNC(prod,  int16_t, 16,  int16_t, mul)
    OP_SVE_FUNC(prod, uint16_t, 16, uint16_t, mul)
    OP_SVE_FUNC(prod,  int32_t, 32,  int32_t, mul)
    OP_SVE_FUNC(prod, uint32_t, 32, uint32_t, mul)
    OP_SVE_FUNC(prod,  int64_t, 64,  int64_t, mul)
    OP_SVE_FUNC(prod, uint64_t, 64, uint64_t, mul)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(prod, short_float, 16, float16_t, mul)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(prod, short_float, 16, float16_t, mul)
#endif
    OP_SVE_FUNC(prod, float, 32, float32_t, mul)
    OP_SVE_FUNC(prod, double, 64, float64_t, mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_SVE_FUNC(band,   int8_t, 8,    int8_t, and)
    OP_SVE_FUNC(band,  uint8_t, 8,   uint8_t, and)
    OP_SVE_FUNC(band,  int16_t, 16,  int16_t, and)
    OP_SVE_FUNC(band, uint16_t, 16, uint16_t, and)
    OP_SVE_FUNC(band,  int32_t, 32,  int32_t, and)
    OP_SVE_FUNC(band, uint32_t, 32, uint32_t, and)
    OP_SVE_FUNC(band,  int64_t, 64,  int64_t, and)
    OP_SVE_FUNC(band, uint64_t, 64, uint64_t, and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_SVE_FUNC(bor,   int8_t, 8,    int8_t, orr)
    OP_SVE_FUNC(bor,  uint8_t, 8,   uint8_t, orr)
    OP_SVE_FUNC(bor,  int16_t, 16,  int16_t, orr)
    OP_SVE_FUNC(bor, uint16_t, 16, uint16_t, orr)
    OP_SVE_FUNC(bor,  int32_t, 32,  int32_t, orr)
    OP_SVE_FUNC(bor, uint32_t, 32, uint32_t, orr)
    OP_SVE_FUNC(bor,  int64_t, 64,  int64_t, orr)
    OP_SVE_FUNC(bor, uint64_t, 64, uint64_t, orr)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_SVE_FUNC(bxor,   int8_t, 8,    int8_t, eor)
    OP_SVE_FUNC(bxor,  uint8_t, 8,   uint8_t, eor)
    OP_SVE_FUNC(bxor,  int16_t, 16,  int16_t, eor)
    OP_SVE_FUNC(bxor, uint16_t, 16, uint16_t, eor)
    OP_SVE_FUNC(bxor,  int32_t, 32,  int32_t, eor)
    OP_SVE_FUNC(bxor, uint32_t, 32, uint32_t, eor)
    OP_SVE_FUNC(bxor,  int64_t, 64,  int64_t, eor)
    OP_SVE_FUNC(bxor, uint64_t, 64, uint64_t, eor)


/*
 *  This is a three buffer (2 input and 1 output) version of the reduction
 *  routines, needed for some optimizations.
 */
#define OP_SVE_FUNC_3BUFF(name, type_name, type_size, type, op) \
        static void ompi_op_sve_3buff_##name##_##type_name(void * restrict in1,   \
                void * restrict in2, void * restrict out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    uint64_t i; \
    uint64_t step = 0; \
    svbool_t Pg = svptrue_b##type_size(); \
    switch(type_size) {                                                \
        case 8:                                                        \
               step = svcntb();                                        \
               break;                                                  \
        case 16:                                                       \
               step = svcnth();                                        \
               break;                                                  \
        case 32:                                                       \
               step = svcntw();                                        \
               break;                                                  \
        case 64:                                                       \
               step = svcntd();                                        \
    }    \
    uint64_t round = *count;                                           \
    uint64_t remain = *count % step;                                   \
    for(i=0; i< round; i=i+step)                                       \
    {                                                                  \
        sv##type  vsrc = svld1(Pg, (type *)in1+i);                     \
        sv##type  vdst = svld1(Pg, (type *)in2+i);                     \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, (type *)out+i,vdst);                                 \
    }                                                                  \
    if (remain !=0){                                                   \
        Pg = svwhilelt_b##type_size##_u64(0, remain);                  \
        sv##type  vsrc = svld1(Pg, (type *)in1+i);                     \
        sv##type  vdst = svld1(Pg, (type *)in2+i);                     \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, (type *)out+i,vdst);                                 \
    } \
}


/*************************************************************************
 * Max
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(max, int8_t  ,  8,   int8_t, max)
    OP_SVE_FUNC_3BUFF(max, uint8_t,   8,  uint8_t, max)
    OP_SVE_FUNC_3BUFF(max, int16_t,  16,  int16_t, max)
    OP_SVE_FUNC_3BUFF(max, uint16_t, 16, uint16_t, max)
    OP_SVE_FUNC_3BUFF(max, int32_t,  32,  int32_t, max)
    OP_SVE_FUNC_3BUFF(max, uint32_t, 32, uint32_t, max)
    OP_SVE_FUNC_3BUFF(max, int64_t,  64,  int64_t, max)
    OP_SVE_FUNC_3BUFF(max, uint64_t, 64, uint64_t, max)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(max, short_float, 16, float16_t, max)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(max, short_float, 16, float16_t, max)
#endif
    OP_SVE_FUNC_3BUFF(max, float, 32, float32_t, max)
    OP_SVE_FUNC_3BUFF(max, double, 64, float64_t, max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(min, int8_t  ,  8,   int8_t, min)
    OP_SVE_FUNC_3BUFF(min, uint8_t,   8,  uint8_t, min)
    OP_SVE_FUNC_3BUFF(min, int16_t,  16,  int16_t, min)
    OP_SVE_FUNC_3BUFF(min, uint16_t, 16, uint16_t, min)
    OP_SVE_FUNC_3BUFF(min, int32_t,  32,  int32_t, min)
    OP_SVE_FUNC_3BUFF(min, uint32_t, 32, uint32_t, min)
    OP_SVE_FUNC_3BUFF(min, int64_t,  64,  int64_t, min)
    OP_SVE_FUNC_3BUFF(min, uint64_t, 64, uint64_t, min)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(min, short_float, 16, float16_t, min)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(min, short_float, 16, float16_t, min)
#endif
    OP_SVE_FUNC_3BUFF(min, float, 32, float32_t, min)
    OP_SVE_FUNC_3BUFF(min, double, 64, float64_t, min)

/*************************************************************************
 * Sum
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(sum, int8_t  ,  8,   int8_t, add)
    OP_SVE_FUNC_3BUFF(sum, uint8_t,   8,  uint8_t, add)
    OP_SVE_FUNC_3BUFF(sum, int16_t,  16,  int16_t, add)
    OP_SVE_FUNC_3BUFF(sum, uint16_t, 16, uint16_t, add)
    OP_SVE_FUNC_3BUFF(sum, int32_t,  32,  int32_t, add)
    OP_SVE_FUNC_3BUFF(sum, uint32_t, 32, uint32_t, add)
    OP_SVE_FUNC_3BUFF(sum, int64_t,  64,  int64_t, add)
    OP_SVE_FUNC_3BUFF(sum, uint64_t, 64, uint64_t, add)
    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(sum, short_float, 16, float16_t, add)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(sum, short_float, 16, float16_t, add)
#endif
    OP_SVE_FUNC_3BUFF(sum, float, 32, float32_t, add)
    OP_SVE_FUNC_3BUFF(sum, double, 64, float64_t, add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(prod,   int8_t, 8,    int8_t, mul)
    OP_SVE_FUNC_3BUFF(prod,  uint8_t, 8,   uint8_t, mul)
    OP_SVE_FUNC_3BUFF(prod,  int16_t, 16,  int16_t, mul)
    OP_SVE_FUNC_3BUFF(prod, uint16_t, 16, uint16_t, mul)
    OP_SVE_FUNC_3BUFF(prod,  int32_t, 32,  int32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, uint32_t, 32, uint32_t, mul)
    OP_SVE_FUNC_3BUFF(prod,  int64_t, 64,  int64_t, mul)
    OP_SVE_FUNC_3BUFF(prod, uint64_t, 64, uint64_t, mul)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(prod, short_float, 16, float16_t, mul)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(prod, short_float, 16, float16_t, mul)
#endif
    OP_SVE_FUNC_3BUFF(prod, float, 32, float32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, double, 64, float64_t, mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(band,   int8_t, 8,    int8_t, and)
    OP_SVE_FUNC_3BUFF(band,  uint8_t, 8,   uint8_t, and)
    OP_SVE_FUNC_3BUFF(band,  int16_t, 16,  int16_t, and)
    OP_SVE_FUNC_3BUFF(band, uint16_t, 16, uint16_t, and)
    OP_SVE_FUNC_3BUFF(band,  int32_t, 32,  int32_t, and)
    OP_SVE_FUNC_3BUFF(band, uint32_t, 32, uint32_t, and)
    OP_SVE_FUNC_3BUFF(band,  int64_t, 64,  int64_t, and)
    OP_SVE_FUNC_3BUFF(band, uint64_t, 64, uint64_t, and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(bor,   int8_t, 8,    int8_t, orr)
    OP_SVE_FUNC_3BUFF(bor,  uint8_t, 8,   uint8_t, orr)
    OP_SVE_FUNC_3BUFF(bor,  int16_t, 16,  int16_t, orr)
    OP_SVE_FUNC_3BUFF(bor, uint16_t, 16, uint16_t, orr)
    OP_SVE_FUNC_3BUFF(bor,  int32_t, 32,  int32_t, orr)
    OP_SVE_FUNC_3BUFF(bor, uint32_t, 32, uint32_t, orr)
    OP_SVE_FUNC_3BUFF(bor,  int64_t, 64,  int64_t, orr)
    OP_SVE_FUNC_3BUFF(bor, uint64_t, 64, uint64_t, orr)

 /*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(bxor,   int8_t, 8,    int8_t, eor)
    OP_SVE_FUNC_3BUFF(bxor,  uint8_t, 8,   uint8_t, eor)
    OP_SVE_FUNC_3BUFF(bxor,  int16_t, 16,  int16_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, uint16_t, 16, uint16_t, eor)
    OP_SVE_FUNC_3BUFF(bxor,  int32_t, 32,  int32_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, uint32_t, 32, uint32_t, eor)
    OP_SVE_FUNC_3BUFF(bxor,  int64_t, 64,  int64_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, uint64_t, 64, uint64_t, eor)

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
#if defined(HAVE_SHORT_FLOAT) || defined(HAVE_OPAL_SHORT_FLOAT_T)
#define SHORT_FLOAT(name, ftype) ompi_op_sve_##ftype##_##name##_short_float
#else
#define SHORT_FLOAT(name, ftype) NULL
#endif
#define FLOAT(name, ftype) ompi_op_sve_##ftype##_##name##_float
#define DOUBLE(name, ftype) ompi_op_sve_##ftype##_##name##_double

#define FLOATING_POINT(name, ftype)                                        \
    [OMPI_OP_BASE_TYPE_SHORT_FLOAT] = SHORT_FLOAT(name, ftype),            \
    [OMPI_OP_BASE_TYPE_FLOAT] = FLOAT(name, ftype),                        \
    [OMPI_OP_BASE_TYPE_DOUBLE] = DOUBLE(name, ftype)

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
        FLOATING_POINT(sum, 2buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 2buff),
        FLOATING_POINT(prod, 2buff),
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
        FLOATING_POINT(sum, 3buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 3buff),
        FLOATING_POINT(prod, 3buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] ={
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(band, 3buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(bor, 3buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(bxor, 3buff),
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
