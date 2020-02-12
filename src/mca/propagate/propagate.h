/*
 * Copyright (c) 2017-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/** @file:
 *
 * The PRRTE propagator
 *
 * The PRRTE Propagator framework provides error propagating services
 * through daemons.
 */

#ifndef MCA_PROPAGATE_H
#define MCA_PROPAGATE_H

/*
 * includes
 */

#include "prrte_config.h"
#include "constants.h"
#include "types.h"

#include "src/mca/mca.h"
#include "src/class/prrte_object.h"
#include "src/class/prrte_bitmap.h"
#include "src/dss/dss_types.h"
#include "src/mca/state/state.h"
#include "src/mca/rml/rml_types.h"

BEGIN_C_DECLS

/* define a callback function to be invoked upon
 * collective completion */
typedef void (*prrte_propagate_cbfunc_t)(int status, prrte_buffer_t *buf, void *cbdata);

typedef int (*prrte_propagate_rbcast_cb_t)(prrte_buffer_t* buffer);

/*
 * Component functions - all MUST be provided!
 */

/* initialize the selected module */
typedef int (*prrte_propagate_base_module_init_fn_t)(void);

/* finalize the selected module */
typedef int (*prrte_propagate_base_module_finalize_fn_t)(void);

typedef int (*prrte_propagate_base_module_prp_fn_t)(prrte_jobid_t *job,
        prrte_process_name_t *source,
        prrte_process_name_t *sickproc,
        prrte_proc_state_t state);

typedef int (*prrte_propagate_base_module_registercb_fn_t)(void);

/*
 * Module Structure
 */
struct prrte_propagate_base_module_2_3_0_t {
    prrte_propagate_base_module_init_fn_t          init;
    prrte_propagate_base_module_finalize_fn_t      finalize;
    prrte_propagate_base_module_prp_fn_t           prp;
    prrte_propagate_base_module_registercb_fn_t    register_cb;
};

typedef struct prrte_propagate_base_module_2_3_0_t prrte_propagate_base_module_2_3_0_t;
typedef prrte_propagate_base_module_2_3_0_t prrte_propagate_base_module_t;

PRRTE_DECLSPEC extern prrte_propagate_base_module_t prrte_propagate;

/*
 *Propagate Component
 */
struct prrte_propagate_base_component_3_0_0_t {
    /** MCA base component */
    prrte_mca_base_component_t base_version;
    /** MCA base data */
    prrte_mca_base_component_data_t base_data;

    /** Verbosity Level */
    int verbose;
    /** Output Handle for prrte_output */
    int output_handle;
    /** Default Priority */
    int priority;
};

typedef struct prrte_propagate_base_component_3_0_0_t prrte_propagate_base_component_3_0_0_t;
typedef prrte_propagate_base_component_3_0_0_t prrte_propagate_base_component_t;

/*
 * Macro for use in components that are of type propagate v3.0.0
 */
#define PRRTE_PROPAGATE_BASE_VERSION_3_0_0 \
    /* propagate v3.0 is chained to MCA v2.0 */ \
    PRRTE_MCA_BASE_VERSION_2_1_0("propagate", 3, 0, 0)

END_C_DECLS

#endif
