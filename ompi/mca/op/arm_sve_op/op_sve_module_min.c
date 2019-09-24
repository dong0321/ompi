/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (C) 2019      Arm Ltd.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/** @file
 *
 * This is the min module source code.  It contains
 * functions that will create a module for the MPI_MIN MPI_Op.
 */

#include "ompi_config.h"

#include "opal/class/opal_object.h"
#include "opal/util/output.h"

#include "ompi/constants.h"
#include "ompi/op/op.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/mca/op/arm_sve_op/op_sve.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

/**
 * Derive a struct from the base op module struct, allowing us to
 * cache some module-specific information for MAX.  Note that
 * information that should be shared across all modules should be put
 * on the sve component.
 */
typedef struct {
    ompi_op_base_module_1_0_0_t super;

    ompi_op_base_handler_fn_t fallback_float;
    ompi_op_base_module_t *fallback_float_module;
    ompi_op_base_handler_fn_t fallback_uint8;
    ompi_op_base_module_t *fallback_uint8_module;

    ompi_op_base_handler_fn_t fallback_double;
    ompi_op_base_module_t *fallback_double_module;
    ompi_op_base_handler_fn_t fallback_double_precision;
    ompi_op_base_module_t *fallback_double_precision_module;
} module_min_t;

/**
 * "Constructor" for the min module class
 */
static void module_min_constructor(module_min_t *m)
{
    /* Use this function to initialize any data in the class that is
       specific to this class (i.e. do *not* initialize the parent
       data members!). */
    m->fallback_float = NULL;
    m->fallback_float_module = NULL;
    m->fallback_uint8 = NULL;
    m->fallback_uint8_module = NULL;

    m->fallback_double = NULL;
    m->fallback_double_module = NULL;
    m->fallback_double_precision = NULL;
    m->fallback_double_precision_module = NULL;
}

/**
 * "Destructor" for the min module class
 */
static void module_min_destructor(module_min_t *m)
{
    /* Use this function to clean up any data members that may be
       necessary.  This may include freeing resources and/or setting
       members to sentinel values to know that the object has been
       destructed. */
    m->fallback_float = (ompi_op_base_handler_fn_t) 0xdeadbeef;
    m->fallback_float_module = (ompi_op_base_module_t*) 0xdeadbeef;
    m->fallback_uint8 = (ompi_op_base_handler_fn_t) 0xdeadbeef;
    m->fallback_uint8_module = (ompi_op_base_module_t*) 0xdeadbeef;

    m->fallback_double = (ompi_op_base_handler_fn_t) 0xdeadbeef;
    m->fallback_double_module = (ompi_op_base_module_t*) 0xdeadbeef;
    m->fallback_double_precision = (ompi_op_base_handler_fn_t) 0xdeadbeef;
    m->fallback_double_precision_module = (ompi_op_base_module_t*) 0xdeadbeef;
}

/**
 * Setup the class for the min module, listing:
 * - the name of the class
 * - the "parent" of the class
 * - function pointer for the constructor (or NULL)
 * - function pointer for the destructor (or NULL)
 */
static OBJ_CLASS_INSTANCE(module_min_t,
                          ompi_op_base_module_t,
                          module_min_constructor,
                          module_min_destructor);

/**
 * Min function for C float
 */
static void min_float(void *in, void *out, int *count,
                      ompi_datatype_t **type, ompi_op_base_module_t *module)
{
    opal_output(0, "In sve min float function");
    uint64_t i;
    svbool_t Pg = svptrue_b32();

    /* Count the number of 32-bit elements in a pattern */
    uint64_t step = svcntw();
    uint64_t round = *count;
    uint64_t remain = *count % step;
    printf("Round: %lu Remain %lu Step %lu \n", round, remain, step);

    for(i=0; i< round; i=i+step)
    {
        svfloat32_t  vsrc = svld1(Pg, (float*)in+i);
        svfloat32_t  vdst = svld1(Pg, (float*)out+i);
        vdst=svmin_z(Pg,vdst,vsrc);
        svst1(Pg, (float*)out+i,vdst);
    }

    if (remain !=0){
        Pg = svwhilelt_b32_u64(0, remain);
        svfloat32_t  vsrc = svld1(Pg, (float*)in+i);
        svfloat32_t  vdst = svld1(Pg, (float*)out+i);
        vdst=svmin_z(Pg,vdst,vsrc);
        svst1(Pg, (float*)out+i,vdst);
    }
}

/**
 * Min function for C double
 */
static void min_double(void *in, void *out, int *count,
                       ompi_datatype_t **type, ompi_op_base_module_t *module)
{
    module_min_t *m = (module_min_t*) module;
    opal_output(0, "In sve min double function");

    m->fallback_double(in, out, count, type, m->fallback_double_module);
}

/**
 * Min function for Fortran UINT8_T
 */
static void min_uint8(void *in, void *out, int *count,
                     ompi_datatype_t **type, ompi_op_base_module_t *module)
{
    opal_output(0, "In sve min uint8 function");
    uint64_t i;
    svbool_t Pg = svptrue_b8();

    /* Count the number of 8-bit elements in a pattern */
    uint64_t step = svcntb();
    uint64_t round = *count;
    uint64_t remain = *count % step;
    printf("Round: %lu Remain %lu Step %lu \n", round, remain, step);

    for(i=0; i< round; i=i+step)
    {
        svuint8_t  vsrc = svld1(Pg, (uint8_t*)in+i);
        svuint8_t  vdst = svld1(Pg, (uint8_t*)out+i);
        vdst=svmin_z(Pg,vdst,vsrc);
        svst1(Pg, (uint8_t*)out+i,vdst);
    }

    if (remain !=0){
        Pg = svwhilelt_b8_u64(0, remain);
        svuint8_t  vsrc = svld1(Pg, (uint8_t*)in+i);
        svuint8_t  vdst = svld1(Pg, (uint8_t*)out+i);
        vdst=svmin_z(Pg,vdst,vsrc);
        svst1(Pg, (uint8_t*)out+i,vdst);
    }
}

/**
 * Min function for Fortran DOUBLE PRECISION
 */
static void min_double_precision(void *in, void *out, int *count,
                                 ompi_datatype_t **type,
                                 ompi_op_base_module_t *module)
{
    module_min_t *m = (module_min_t*) module;
    opal_output(0, "In sve min double precision function");

    m->fallback_double_precision(in, out, count, type,
                                 m->fallback_double_precision_module);
}

ompi_op_base_module_t *ompi_op_sve_min(ompi_op_t *op)
{
    module_min_t *module = OBJ_NEW(module_min_t);

    /* Remember that we created an *sve* module (vs. a *base*
       module), so we can cache extra information on there that is
       specific for the MIN operation.  Let's cache the original
       fallback function pointers, that were passed to us in this call
       (i.e., they're already assigned on the op). */

    /* C float */
    module->super.opm_fns[OMPI_OP_BASE_TYPE_FLOAT] = min_float;
    module->fallback_float = op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_FLOAT];
    module->fallback_float_module =
        op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_FLOAT];
    /* If you cache a fallback function, you *must* RETAIN (i.e.,
       increase the refcount) its module so that the module knows that
       it is being used and won't be freed/destructed. */
    OBJ_RETAIN(module->fallback_float_module);

    /* Fortran UINT8_T */
    module->super.opm_fns[OMPI_OP_BASE_TYPE_UINT8_T] = min_uint8;
    module->fallback_uint8 =
        op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_UINT8_T];
    module->fallback_uint8_module =
        op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_UINT8_T];
    OBJ_RETAIN(module->fallback_uint8_module);

    /* Does our hardware support double precision? */

    if (mca_op_sve_component.double_supported) {
        /* C double */
        module->super.opm_fns[OMPI_OP_BASE_TYPE_DOUBLE] = min_double;
        module->fallback_double =
            op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_DOUBLE];
        module->fallback_double_module =
            op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_DOUBLE];
        OBJ_RETAIN(module->fallback_double_module);

        /* Fortran DOUBLE PRECISION */
        module->super.opm_fns[OMPI_OP_BASE_TYPE_DOUBLE_PRECISION] =
            min_double_precision;
        module->fallback_double_precision =
            op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_DOUBLE_PRECISION];
        module->fallback_double_precision_module =
            op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_DOUBLE_PRECISION];
        OBJ_RETAIN(module->fallback_double_precision_module);
    }

    /* ...not listing the rest of the floating point-typed functions
       in this sve... */

    return (ompi_op_base_module_t*) module;
}
