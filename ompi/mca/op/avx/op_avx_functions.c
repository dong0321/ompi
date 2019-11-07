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
#include "ompi/mca/op/avx/op_avx.h"
#include "ompi/mca/op/avx/op_avx_functions.h"

#include <immintrin.h>
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
#define OP_AVX_FUNC(name, type_sign, type_size, type, op) \
    static void ompi_op_avx_2buff_##name##_##type(void *_in, void *_out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size;                                        \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    type* in = (type*)_in; \
    type* out = (type*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512i vecA =  _mm512_loadu_si512((in+i));\
        __m512i vecB =  _mm512_loadu_si512((out+i));\
        __m512i res = _mm512_##op##_ep##type_sign##type_size(vecA, vecB); \
        _mm512_storeu_si512((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512i vecA = _mm512_maskz_loadu_epi##type_size(left_over, (in+round)); \
        __m512i vecB = _mm512_maskz_loadu_epi##type_size(left_over, (out+round)); \
        __m512i res = _mm512_##op##_ep##type_sign##type_size(vecA, vecB); \
        _mm512_mask_storeu_epi##type_size((out+round), left_over, res); \
    }\
}

/*
 *  This macro is for bit-wise operations (out op in).
 *
 *  Support ops: or, xor, and of 512 bits (representing integer data)
 *
 */
#define OP_AVX_BIT_FUNC(name, type_size, type, op) \
    static void ompi_op_avx_2buff_##name##_##type(void *_in, void *_out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size; \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    type* in = (type*)_in; \
    type* out = (type*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512i vecA =  _mm512_loadu_si512((in+i));\
        __m512i vecB =  _mm512_loadu_si512((out+i));\
        __m512i res = _mm512_##op##_si512(vecA, vecB); \
        _mm512_storeu_si512((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512i vecA = _mm512_maskz_loadu_epi##type_size(left_over, (in+round)); \
        __m512i vecB = _mm512_maskz_loadu_epi##type_size(left_over, (out+round)); \
        __m512i res =  _mm512_##op##_si512(vecA, vecB); \
        _mm512_mask_storeu_epi##type_size((out+round), left_over, res); \
    }\
}

#if 0
#define OP_AVX_FLOAT_FUNC(op)						\
  static void ompi_op_avx_2buff_##op##_float(void *in, void *out, int *count, \
					     struct ompi_datatype_t **dtype, \
					     struct ompi_op_base_module_1_0_0_t *module) \
  {									\
    int step = 16;							\
    int size = *count/step;						\
    int i;								\
    int round = size*64;						\
    for (i = 0; i < round; i+=64) {					\
      __m512 vecA =  _mm512_load_ps((in+i));				\
      __m512 vecB =  _mm512_load_ps((out+i));				\
      __m512 res = _mm512_##op##_ps(vecA, vecB);			\
	_mm512_store_ps((out+i), res);					\
    }									\
    uint64_t left_over = *count - (size*step);				\
    if(left_over!=0) {							\
      uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF;			\
      left_over = left_over64 >>(64-left_over);				\
      __m512 vecA = _mm512_maskz_load_ps(left_over, (in+round));	\
      __m512 vecB = _mm512_maskz_load_ps(left_over, (out+round));	\
      __m512 res = _mm512_##op##_ps(vecA, vecB);			\
	_mm512_mask_store_ps((out+round), left_over, res);		\
    }									\
  }

#else
#define OP_AVX_FLOAT_FUNC(op)				\
static void ompi_op_avx_2buff_##op##_float(void *_in, void *_out, int *count, \
					   struct ompi_datatype_t **dtype, \
					   struct ompi_op_base_module_1_0_0_t *module) \
{									\
  int types_per_step = 512 / (8 * sizeof(float));			\
  int left_over = *count;						\
  float* in = (float*)_in;						\
  float* out = (float*)_out;						\
  for (; left_over >= types_per_step; left_over -= types_per_step) {	\
    __m512 vecA =  _mm512_load_ps(in);					\
    __m512 vecB =  _mm512_load_ps(out);					\
    in += types_per_step;						\
    __m512 res = _mm512_##op##_ps(vecA, vecB);				\
      _mm512_store_ps(out, res);					\
      out += types_per_step;						\
  }									\
  if( 0 != left_over ) {						\
    types_per_step >>= 1;  /* 256 / (8 * sizeof(float));	*/	\
    if( left_over >= types_per_step ) {					\
      __m256 vecA =  _mm256_load_ps(in);				\
      __m256 vecB =  _mm256_load_ps(out);				\
      __m256 res = _mm256_##op##_ps(vecA, vecB);			\
	_mm256_store_ps(out, res);					\
	in += types_per_step;						\
	out += types_per_step;						\
	left_over -= types_per_step;					\
    }									\
  }									\
  if( 0 != left_over ) {						\
    switch(left_over) {							\
    case 7: out[6] += in[6];						\
    case 6: out[5] += in[5];						\
    case 5: out[4] += in[4];						\
    case 4: out[3] += in[3];						\
    case 3: out[2] += in[2];						\
    case 2: out[1] += in[1];						\
    case 1: out[0] += in[0];						\
    }									\
  }									\
}
#endif

#define OP_AVX_DOUBLE_FUNC(op) \
    static void ompi_op_avx_2buff_##op##_double(void *_in, void *_out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 8;                                                          \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    double* in = (double*)_in; \
    double* out = (double*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512d vecA =  _mm512_load_pd((in+i));\
        __m512d vecB =  _mm512_load_pd((out+i));\
        __m512d res = _mm512_##op##_pd(vecA, vecB); \
        _mm512_store_pd((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512d vecA = _mm512_maskz_load_pd(left_over, (in+round)); \
        __m512d vecB = _mm512_maskz_load_pd(left_over, (out+round)); \
        __m512d res = _mm512_##op##_pd(vecA, vecB); \
        _mm512_mask_store_pd((out+round), left_over, res); \
    }\
}


/*************************************************************************
 * Max
 *************************************************************************/
    OP_AVX_FUNC(max, i, 8,    int8_t, max)
    OP_AVX_FUNC(max, u, 8,   uint8_t, max)
    OP_AVX_FUNC(max, i, 16,  int16_t, max)
    OP_AVX_FUNC(max, u, 16, uint16_t, max)
    OP_AVX_FUNC(max, i, 32,  int32_t, max)
    OP_AVX_FUNC(max, u, 32, uint32_t, max)
    OP_AVX_FUNC(max, i, 64,  int64_t, max)
    OP_AVX_FUNC(max, u, 64, uint64_t, max)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(max)
    OP_AVX_DOUBLE_FUNC(max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_AVX_FUNC(min, i, 8,    int8_t, min)
    OP_AVX_FUNC(min, u, 8,   uint8_t, min)
    OP_AVX_FUNC(min, i, 16,  int16_t, min)
    OP_AVX_FUNC(min, u, 16, uint16_t, min)
    OP_AVX_FUNC(min, i, 32,  int32_t, min)
    OP_AVX_FUNC(min, u, 32, uint32_t, min)
    OP_AVX_FUNC(min, i, 64,  int64_t, min)
    OP_AVX_FUNC(min, u, 64, uint64_t, min)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(min)
    OP_AVX_DOUBLE_FUNC(min)

/*************************************************************************
 * Sum
 ************************************************************************/
    OP_AVX_FUNC(sum, i, 8,    int8_t, add)
    OP_AVX_FUNC(sum, i, 8,   uint8_t, add)
    OP_AVX_FUNC(sum, i, 16,  int16_t, add)
    OP_AVX_FUNC(sum, i, 16, uint16_t, add)
    OP_AVX_FUNC(sum, i, 32,  int32_t, add)
    OP_AVX_FUNC(sum, i, 32, uint32_t, add)
    OP_AVX_FUNC(sum, i, 64,  int64_t, add)
    OP_AVX_FUNC(sum, i, 64, uint64_t, add)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(add)
#if 0
static void ompi_op_avx_2buff_add_float(void *_in, void *_out, int *count,
					struct ompi_datatype_t **dtype,
					struct ompi_op_base_module_1_0_0_t *module) 
{
  int types_per_step = 512 / (8 * sizeof(float));
  int left_over = *count;
  float* in = (float*)_in;
  float* out = (float*)_out;
  for (; left_over >= types_per_step; left_over -= types_per_step) {
    __m512 vecA =  _mm512_load_ps(in);
    __m512 vecB =  _mm512_load_ps(out);
    in += types_per_step;
    __m512 res = _mm512_add_ps(vecA, vecB);
      _mm512_store_ps(out, res);
      out += types_per_step;
  }
  if( 0 != left_over ) {
    types_per_step >>= 1;  /* 256 / (8 * sizeof(float)); */
    if( left_over >= types_per_step ) {
      __m256 vecA =  _mm256_load_ps(in);
      __m256 vecB =  _mm256_load_ps(out);
      __m256 res = _mm256_add_ps(vecA, vecB);
	_mm256_store_ps(out, res);
	in += types_per_step;
	out += types_per_step;
	left_over -= types_per_step;
    }
    if( 0 != left_over ) {
      switch(left_over) {
      case 7: out[6] += in[6];
      case 6: out[5] += in[5];
      case 5: out[4] += in[4];
      case 4: out[3] += in[3];
      case 3: out[2] += in[2];
      case 2: out[1] += in[1];
      case 1: out[0] += in[0];
      }
    }
  }
}
#endif
    OP_AVX_DOUBLE_FUNC(add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_AVX_FUNC(prod, i, 16,  int16_t, mullo)
    OP_AVX_FUNC(prod, i, 16, uint16_t, mullo)
    OP_AVX_FUNC(prod, i, 32,  int32_t, mullo)
    OP_AVX_FUNC(prod, i ,32, uint32_t, mullo)
    OP_AVX_FUNC(prod, i, 64,  int64_t, mullo)
    OP_AVX_FUNC(prod, i, 64, uint64_t, mullo)

    /* Floating point */
    OP_AVX_FLOAT_FUNC(mul)
    OP_AVX_DOUBLE_FUNC(mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_AVX_BIT_FUNC(band, 8,    int8_t, and)
    OP_AVX_BIT_FUNC(band, 8,   uint8_t, and)
    OP_AVX_BIT_FUNC(band, 16,  int16_t, and)
    OP_AVX_BIT_FUNC(band, 16, uint16_t, and)
    OP_AVX_BIT_FUNC(band, 32,  int32_t, and)
    OP_AVX_BIT_FUNC(band, 32, uint32_t, and)
    OP_AVX_BIT_FUNC(band, 64,  int64_t, and)
    OP_AVX_BIT_FUNC(band, 64, uint64_t, and)

    // not defined - OP_AVX_FLOAT_FUNC(and)
    // not defined - OP_AVX_DOUBLE_FUNC(and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_AVX_BIT_FUNC(bor, 8,    int8_t, or)
    OP_AVX_BIT_FUNC(bor, 8,   uint8_t, or)
    OP_AVX_BIT_FUNC(bor, 16,  int16_t, or)
    OP_AVX_BIT_FUNC(bor, 16, uint16_t, or)
    OP_AVX_BIT_FUNC(bor, 32,  int32_t, or)
    OP_AVX_BIT_FUNC(bor, 32, uint32_t, or)
    OP_AVX_BIT_FUNC(bor, 64,  int64_t, or)
    OP_AVX_BIT_FUNC(bor, 64, uint64_t, or)

    // not defined - OP_AVX_FLOAT_FUNC(or)
    // not defined - OP_AVX_DOUBLE_FUNC(or)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_AVX_BIT_FUNC(bxor, 8,    int8_t, xor)
    OP_AVX_BIT_FUNC(bxor, 8,   uint8_t, xor)
    OP_AVX_BIT_FUNC(bxor, 16,  int16_t, xor)
    OP_AVX_BIT_FUNC(bxor, 16, uint16_t, xor)
    OP_AVX_BIT_FUNC(bxor, 32,  int32_t, xor)
    OP_AVX_BIT_FUNC(bxor, 32, uint32_t, xor)
    OP_AVX_BIT_FUNC(bxor, 64,  int64_t, xor)
    OP_AVX_BIT_FUNC(bxor, 64, uint64_t, xor)

    // not defined - OP_AVX_FLOAT_FUNC(xor)
    // not defined - OP_AVX_DOUBLE_FUNC(xor)

/*
 *  This is a three buffer (2 input and 1 output) version of the reduction
 *  routines, needed for some optimizations.
 */
#define OP_AVX_FUNC_3BUFF(name, type_sign, type_size, type, op)\
        static void ompi_op_avx_3buff_##name##_##type(void * restrict _in1,   \
                void * restrict _in2, void * restrict _out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size; \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    type* in1 = (type*)_in1; \
    type* in2 = (type*)_in2; \
    type* out = (type*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512i vecA =  _mm512_loadu_si512((in1+i));\
        __m512i vecB =  _mm512_loadu_si512((in2+i));\
        __m512i res = _mm512_##op##_ep##type_sign##type_size(vecA, vecB); \
        _mm512_storeu_si512((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512i vecA = _mm512_maskz_loadu_epi##type_size(left_over, (in1+round)); \
        __m512i vecB = _mm512_maskz_loadu_epi##type_size(left_over, (in2+round)); \
        __m512i res = _mm512_##op##_ep##type_sign##type_size(vecA, vecB); \
        _mm512_mask_storeu_epi##type_size((out+round), left_over, res); \
    }\
}

#define OP_AVX_BIT_FUNC_3BUFF(name, type_size, type, op) \
        static void ompi_op_avx_3buff_##op##_##type(void *_in1, void *_in2, void *_out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 512 / type_size; \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    type* in1 = (type*)_in1; \
    type* in2 = (type*)_in2; \
    type* out = (type*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512i vecA =  _mm512_loadu_si512((in1+i));\
        __m512i vecB =  _mm512_loadu_si512((in2+i));\
        __m512i res = _mm512_##op##_si512(vecA, vecB); \
        _mm512_storeu_si512((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512i vecA = _mm512_maskz_loadu_epi##type_size(left_over, (in1+round)); \
        __m512i vecB = _mm512_maskz_loadu_epi##type_size(left_over, (in2+round)); \
        __m512i res =  _mm512_##op##_si512(vecA, vecB); \
        _mm512_mask_storeu_epi##type_size((out+round), left_over, res); \
    }\
}

#define OP_AVX_FLOAT_FUNC_3BUFF(op) \
        static void ompi_op_avx_3buff_##op##_float(void *_in1, void *_in2, void *_out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 16;                                                          \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    float* in1 = (float*)_in1; \
    float* in2 = (float*)_in2; \
    float* out = (float*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512 vecA =  _mm512_load_ps((in1+i));\
        __m512 vecB =  _mm512_load_ps((in2+i));\
        __m512 res = _mm512_##op##_ps(vecA, vecB); \
        _mm512_store_ps((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512 vecA = _mm512_maskz_load_ps(left_over, (in1+round)); \
        __m512 vecB = _mm512_maskz_load_ps(left_over, (in2+round)); \
        __m512 res = _mm512_##op##_ps(vecA, vecB); \
        _mm512_mask_store_ps((out+round), left_over, res); \
    }\
}

#define OP_AVX_DOUBLE_FUNC_3BUFF(op) \
        static void ompi_op_avx_3buff_##op##_double(void *_in1, void *_in2, void *_out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int step = 8;                                                          \
    int size = *count/step; \
    int i; \
    int round = size*64; \
    double* in1 = (double*)_in1; \
    double* in2 = (double*)_in2; \
    double* out = (double*)_out; \
    for (i = 0; i < round; i+=64) { \
        __m512d vecA =  _mm512_load_pd((in1+i));\
        __m512d vecB =  _mm512_load_pd((in2+i));\
        __m512d res = _mm512_##op##_pd(vecA, vecB); \
        _mm512_store_pd((out+i), res); \
    }\
    uint64_t left_over = *count - (size*step); \
    if(left_over!=0){ \
        uint64_t left_over64 = 0xFFFFFFFFFFFFFFFF; \
        left_over = left_over64 >>(64-left_over); \
        __m512d vecA = _mm512_maskz_load_pd(left_over, (in1+round)); \
        __m512d vecB = _mm512_maskz_load_pd(left_over, (in2+round)); \
        __m512d res = _mm512_##op##_pd(vecA, vecB); \
        _mm512_mask_store_pd((out+round), left_over, res); \
    }\
}

/*************************************************************************
 * Max
 *************************************************************************/
    OP_AVX_FUNC_3BUFF(max, i, 8,    int8_t, max)
    OP_AVX_FUNC_3BUFF(max, u, 8,   uint8_t, max)
    OP_AVX_FUNC_3BUFF(max, i, 16,  int16_t, max)
    OP_AVX_FUNC_3BUFF(max, u, 16, uint16_t, max)
    OP_AVX_FUNC_3BUFF(max, i, 32,  int32_t, max)
    OP_AVX_FUNC_3BUFF(max, u, 32, uint32_t, max)
    OP_AVX_FUNC_3BUFF(max, i, 64,  int64_t, max)
    OP_AVX_FUNC_3BUFF(max, u, 64, uint64_t, max)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3BUFF(max)
    OP_AVX_DOUBLE_FUNC_3BUFF(max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_AVX_FUNC_3BUFF(min, i, 8,    int8_t, min)
    OP_AVX_FUNC_3BUFF(min, u, 8,   uint8_t, min)
    OP_AVX_FUNC_3BUFF(min, i, 16,  int16_t, min)
    OP_AVX_FUNC_3BUFF(min, u, 16, uint16_t, min)
    OP_AVX_FUNC_3BUFF(min, i, 32,  int32_t, min)
    OP_AVX_FUNC_3BUFF(min, u, 32, uint32_t, min)
    OP_AVX_FUNC_3BUFF(min, i, 64,  int64_t, min)
    OP_AVX_FUNC_3BUFF(min, u, 64, uint64_t, min)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3BUFF(min)
    OP_AVX_DOUBLE_FUNC_3BUFF(min)

/*************************************************************************
 * Sum
 *************************************************************************/
    OP_AVX_FUNC_3BUFF(sum, i, 8,    int8_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 8,   uint8_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 16,  int16_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 16, uint16_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 32,  int32_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 32, uint32_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 64,  int64_t, add)
    OP_AVX_FUNC_3BUFF(sum, i, 64, uint64_t, add)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3BUFF(add)
    OP_AVX_DOUBLE_FUNC_3BUFF(add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_AVX_FUNC_3BUFF(prod, i, 16,  int16_t, mullo)
    OP_AVX_FUNC_3BUFF(prod, i, 16, uint16_t, mullo)
    OP_AVX_FUNC_3BUFF(prod, i, 32,  int32_t, mullo)
    OP_AVX_FUNC_3BUFF(prod, i ,32, uint32_t, mullo)
    OP_AVX_FUNC_3BUFF(prod, i, 64,  int64_t, mullo)
    OP_AVX_FUNC_3BUFF(prod, i, 64, uint64_t, mullo)

    /* Floating point */
    OP_AVX_FLOAT_FUNC_3BUFF(mul)
    OP_AVX_DOUBLE_FUNC_3BUFF(mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_AVX_BIT_FUNC_3BUFF(band, 8,    int8_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 8,   uint8_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 16,  int16_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 16, uint16_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 32,  int32_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 32, uint32_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 64,  int64_t, and)
    OP_AVX_BIT_FUNC_3BUFF(band, 64, uint64_t, and)

    // not defined - OP_AVX_FLOAT_FUNC_3BUFF(and)
    // not defined - OP_AVX_DOUBLE_FUNC_3BUFF(and)

/*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_AVX_BIT_FUNC_3BUFF(bor, 8,    int8_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 8,   uint8_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 16,  int16_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 16, uint16_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 32,  int32_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 32, uint32_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 64,  int64_t, or)
    OP_AVX_BIT_FUNC_3BUFF(bor, 64, uint64_t, or)

    // not defined - OP_AVX_FLOAT_FUNC_3BUFF(or)
    // not defined - OP_AVX_DOUBLE_FUNC_3BUFF(or)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_AVX_BIT_FUNC_3BUFF(bxor, 8,    int8_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 8,   uint8_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 16,  int16_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 16, uint16_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 32,  int32_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 32, uint32_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 64,  int64_t, xor)
    OP_AVX_BIT_FUNC_3BUFF(bxor, 64, uint64_t, xor)

    // not defined - OP_AVX_FLOAT_FUNC_3BUFF(xor)
    // not defined - OP_AVX_DOUBLE_FUNC_3BUFF(xor)


/** C integer ***********************************************************/
#define C_INTEGER(name, ftype)                                              \
    [OMPI_OP_BASE_TYPE_INT8_T] = ompi_op_avx_##ftype##_##name##_int8_t,     \
    [OMPI_OP_BASE_TYPE_UINT8_T] = ompi_op_avx_##ftype##_##name##_uint8_t,   \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_avx_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_avx_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_avx_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_avx_##ftype##_##name##_uint32_t, \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_avx_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_avx_##ftype##_##name##_uint64_t


/** Floating point, including all the Fortran reals *********************/
#define FLOAT(name, ftype) ompi_op_avx_##ftype##_##name##_float
#define DOUBLE(name, ftype) ompi_op_avx_##ftype##_##name##_double

#define FLOATING_POINT(name, ftype)                                                            \
    [OMPI_OP_BASE_TYPE_SHORT_FLOAT] = NULL, \
    [OMPI_OP_BASE_TYPE_FLOAT] = FLOAT(name, ftype),                                              \
    [OMPI_OP_BASE_TYPE_DOUBLE] = DOUBLE(name, ftype)

#define C_INTEGER_PROD(name, ftype)                                           \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_avx_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_avx_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_avx_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_avx_##ftype##_##name##_uint32_t, \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_avx_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_avx_##ftype##_##name##_uint64_t


/*
 * MPI_OP_NULL
 * All types
 */
#define FLAGS_NO_FLOAT \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | OMPI_OP_FLAGS_COMMUTE)
#define FLAGS \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | \
         OMPI_OP_FLAGS_FLOAT_ASSOC | OMPI_OP_FLAGS_COMMUTE)

ompi_op_base_handler_fn_t ompi_op_avx_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
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

ompi_op_base_3buff_handler_fn_t ompi_op_avx_3buff_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
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
