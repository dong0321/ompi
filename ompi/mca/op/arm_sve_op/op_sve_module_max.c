/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2007 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2009 Cisco Systems, Inc.  All rights reserved.
 * Copyright (C) 2019      Arm Ltd.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/** @file
 *
 * This is the max module source code.  It contains the "setup"
 * functions that will create a module for the MPI_MAX MPI_Op.
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

    /* Just like the ompi_op_sve_component_t, this struct is meant to
       cache information on a per-module basis.  What follows are
       sves; replace them with whatever is relevant for your
       component/module.  Keep in mind that there will be one distinct
       module for each MPI_Op; you may want to have different data
       cached on the module, depending on the MPI_Op that it is
       supporting.

       In this sve, we'll keep the fallback function pointers for
       several integer types. */
    ompi_op_base_handler_fn_t fallback_float;
    ompi_op_base_module_t *fallback_float_module;
    ompi_op_base_handler_fn_t fallback_uint8;
    ompi_op_base_module_t *fallback_uint8_module;

    ompi_op_base_handler_fn_t fallback_double;
    ompi_op_base_module_t *fallback_double_module;
    ompi_op_base_handler_fn_t fallback_double_precision;
    ompi_op_base_module_t *fallback_double_precision_module;
} module_max_t;

/**
 * "Constructor" for the max module class
 */
static void module_max_constructor(module_max_t *m)
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
 * "Destructor" for the max module class
 */
static void module_max_destructor(module_max_t *m)
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
 * Setup the class for the max module, listing:
 * - the name of the class
 * - the "parent" of the class
 * - function pointer for the constructor (or NULL)
 * - function pointer for the destructor (or NULL)
 */
static OBJ_CLASS_INSTANCE(module_max_t,
                          ompi_op_base_module_t,
                          module_max_constructor,
                          module_max_destructor);

/**
 * Max function for C float
 */
static void max_float(void *in, void *out, int *count,
                      ompi_datatype_t **type, ompi_op_base_module_t *module)
{
    /* Be chatty to the output, just so that we can see that this
       function was called */
    opal_output(0, "In sve max float function");
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
        vdst=svmax_z(Pg,vdst,vsrc);
        svst1(Pg, (float*)out+i,vdst);
    }

    if (remain !=0){
        Pg = svwhilelt_b32_u64(0, remain);
        svfloat32_t  vsrc = svld1(Pg, (float*)in+i);
        svfloat32_t  vdst = svld1(Pg, (float*)out+i);
        vdst=svmax_z(Pg,vdst,vsrc);
        svst1(Pg, (float*)out+i,vdst);
    }

}

/**
 * Max function for C double
 */
static void max_double(void *in, void *out, int *count,
                       ompi_datatype_t **type, ompi_op_base_module_t *module)
{
    module_max_t *m = (module_max_t*) module;
    opal_output(0, "In sve max double function");

    /* Just another sve function -- similar to max_int() */

    m->fallback_double(in, out, count, type, m->fallback_double_module);
}

/**
 * Max function for Fortran UINT8_T
 */
static void max_uint8(void *in, void *out, int *count,
                     ompi_datatype_t **type, ompi_op_base_module_t *module)
{
    opal_output(0, "In sve max uint8 function");
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
        vdst=svmax_z(Pg,vdst,vsrc);
        svst1(Pg, (uint8_t*)out+i,vdst);
    }

    if (remain !=0){
        Pg = svwhilelt_b8_u64(0, remain);
        svuint8_t  vsrc = svld1(Pg, (uint8_t*)in+i);
        svuint8_t  vdst = svld1(Pg, (uint8_t*)out+i);
        vdst=svmax_z(Pg,vdst,vsrc);
        svst1(Pg, (uint8_t*)out+i,vdst);
    }


}

/**
 * Max function for Fortran DOUBLE PRECISION
 */
static void max_double_precision(void *in, void *out, int *count,
                                 ompi_datatype_t **type,
                                 ompi_op_base_module_t *module)
{
    module_max_t *m = (module_max_t*) module;
    opal_output(0, "In sve max double precision function");

    /* Just another sve function -- similar to max_int() */

    m->fallback_double_precision(in, out, count, type,
                                 m->fallback_double_precision_module);
}

/**
 * Setup function for MPI_MAX.  If we get here, we can assume that a)
 * the hardware is present, b) the MPI thread scenario is what we
 * want, and c) the MAX operation is supported.  So this function's
 * job is to create a module and fill in function pointers for the
 * functions that this hardware supports.
 */
ompi_op_base_module_t *ompi_op_sve_setup_max(ompi_op_t *op)
{
    module_max_t *module = OBJ_NEW(module_max_t);

    /* We defintely support the single precision floating point types */

    /* Remember that we created an *sve* module (vs. a *base*
       module), so we can cache extra information on there that is
       specific for the MAX operation.  Let's cache the original
       fallback function pointers, that were passed to us in this call
       (i.e., they're already assigned on the op). */

    /* C float */
    module->super.opm_fns[OMPI_OP_BASE_TYPE_FLOAT] = max_float;
    module->fallback_float = op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_FLOAT];
    module->fallback_float_module =
        op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_FLOAT];
    /* If you cache a fallback function, you *must* RETAIN (i.e.,
       increase the refcount) its module so that the module knows that
       it is being used and won't be freed/destructed. */
    OBJ_RETAIN(module->fallback_float_module);

    /* Fortran UINT8_T */
    module->super.opm_fns[OMPI_OP_BASE_TYPE_UINT8_T] = max_uint8;
    module->fallback_uint8 =
        op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_UINT8_T];
    module->fallback_uint8_module =
        op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_UINT8_T];
    OBJ_RETAIN(module->fallback_uint8_module);

    /* Does our hardware support double precision? */

    if (mca_op_sve_component.double_supported) {
        /* C double */
        module->super.opm_fns[OMPI_OP_BASE_TYPE_DOUBLE] = max_double;
        module->fallback_double =
            op->o_func.intrinsic.fns[OMPI_OP_BASE_TYPE_DOUBLE];
        module->fallback_double_module =
            op->o_func.intrinsic.modules[OMPI_OP_BASE_TYPE_DOUBLE];
        OBJ_RETAIN(module->fallback_double_module);

        /* Fortran DOUBLE PRECISION */
        module->super.opm_fns[OMPI_OP_BASE_TYPE_DOUBLE_PRECISION] =
            max_double_precision;
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
