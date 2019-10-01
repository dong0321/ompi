/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (C) 2019      Arm Ltd.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_OP_SVE_EXPORT_H
#define MCA_OP_SVE_EXPORT_H

#include "ompi_config.h"

#include "ompi/mca/mca.h"
#include "opal/class/opal_object.h"

#include "ompi/mca/op/op.h"

BEGIN_C_DECLS

/**
 * Derive a struct from the base op component struct, allowing us to
 * cache some component-specific information on our well-known
 * component struct.
 */
typedef struct {
    /** The base op component struct */
    ompi_op_base_component_1_0_0_t super;

    /* What follows is sve-component-specific cached information.  We
       tend to use this scheme (caching information on the sve
       component itself) instead of lots of individual global
       variables for the component.  The following data fields are
       sves; replace them with whatever is relevant for your
       component. */

    /** A simple boolean indicating that the hardware is available. */
    bool hardware_available;

    /** A simple boolean indicating whether double precision is
        supported. */
    bool double_supported;
} ompi_op_sve_component_t;

/**
 * Derive a struct from the base op module struct, allowing us to
 * cache some module-specific information for BXOR.  Note that
 * information that should be shared across all modules should be put
 * on the sve component.
 */
typedef struct {
    ompi_op_base_module_1_0_0_t super;

    double some_bxor_data;
} ompi_op_sve_module_bxor_t;

/**
 * To use OMPI's OBJ system, you have to declare each "class".
 */
OBJ_CLASS_DECLARATION(ompi_op_sve_module_bxor_t);

/**
 * Globally exported variable.  Note that it is a *sve* component
 * (defined above), which has the ompi_op_base_component_t as its
 * first member.  Hence, the MCA/op framework will find the data that
 * it expects in the first memory locations, but then the component
 * itself can cache additional information after that that can be used
 * by both the component and modules.
 */
OMPI_DECLSPEC extern ompi_op_sve_component_t
    mca_op_sve_component;

/**
 * Setup for MPI_MAX and return a module.
 */
OMPI_DECLSPEC ompi_op_base_module_t *
    ompi_op_sve_max(ompi_op_t *op);

/**
 * Setup for MPI_BXOR and return a module.
 */
OMPI_DECLSPEC ompi_op_base_module_t *
    ompi_op_sve_bxor(ompi_op_t *op);

OMPI_DECLSPEC ompi_op_base_module_t *
    ompi_op_sve_min(ompi_op_t *op);

OMPI_DECLSPEC ompi_op_base_module_t *
    ompi_op_sve_sum(ompi_op_t *op);

OMPI_DECLSPEC ompi_op_base_module_t *
    ompi_op_sve_prod2buf(ompi_op_t *op);
/*
OMPI_DECLSPEC extern ompi_op_base_handler_fn_t
ompi_op_sve_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX];
OMPI_DECLSPEC extern ompi_op_base_3buff_handler_fn_t
ompi_op_sve_3buff_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX];
*/
END_C_DECLS

#endif /* MCA_OP_SVE_EXPORT_H */
