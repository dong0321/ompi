/*
 * Copyright (c) 2010-2012 Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "orte_config.h"

#include <sys/types.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif  /* HAVE_UNISTD_H */
#include <string.h>
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#include "opal/util/output.h"
#include "opal/dss/dss.h"
#include "opal/mca/pmix/pmix.h"

#include "orte/mca/rml/rml.h"
#include "orte/mca/odls/odls.h"
#include "orte/mca/odls/base/base.h"
#include "orte/mca/odls/base/odls_private.h"
#include "orte/mca/plm/base/plm_private.h"
#include "orte/mca/plm/plm.h"
#include "orte/mca/rmaps/rmaps_types.h"
#include "orte/mca/routed/routed.h"
#include "orte/mca/grpcomm/grpcomm.h"
#include "orte/mca/grpcomm/bmg/grpcomm_bmg.h"
#include "orte/mca/ess/ess.h"
#include "orte/mca/state/state.h"

#include "orte/util/error_strings.h"
#include "orte/util/name_fns.h"
#include "orte/util/proc_info.h"
#include "orte/util/show_help.h"
#include "orte/util/nidmap.h"

#include "orte/runtime/orte_globals.h"
#include "orte/runtime/orte_locks.h"
#include "orte/runtime/orte_quit.h"
#include "orte/runtime/data_type_support/orte_dt_support.h"

#include "orte/mca/errmgr/errmgr.h"
#include "orte/mca/errmgr/base/base.h"
#include "orte/mca/errmgr/base/errmgr_private.h"

#include "errmgr_detector.h"

static int orte_errmgr_failure_propagator_cb_type = -1;

int orte_errmgr_failure_propagate(orte_jobid_t *job, orte_process_name_t *daemon, orte_proc_state_t state);
int orte_errmgr_failure_propagate_recv(opal_buffer_t* buffer);

/*
 *Local functions
 */

int orte_errmgr_init_failure_propagate(void)
{
    int ret;
    printf("dong orte errmgr init faliure propagator %p\n", orte_errmgr_failure_propagate_recv);

    //ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_errmgr_failure_propagate_recv);
    // ret = orte_grpcomm_rbcast_register_cb_type((orte_grpcomm_rbcast_cb_t)orte_errmgr_failure_propagate_recv);
    
    if ( 0 <= ret ){
        orte_errmgr_failure_propagator_cb_type = ret;
        return ORTE_SUCCESS;
    }
    printf("dong orte errmgr init faliure propagator 1\n");
    return ret;
}

int orte_errmgr_finalize_failure_propagate(void)
{
    int ret;
    if ( -1 == orte_errmgr_failure_propagator_cb_type){
        return ORTE_SUCCESS;
    }
    ret = orte_grpcomm.unregister_cb(orte_errmgr_failure_propagator_cb_type);
    //ret = orte_grpcomm_rbcast_unregister_cb_type(orte_errmgr_failure_propagator_cb_type);
    orte_errmgr_failure_propagator_cb_type = -1;
    return ret;
}

/*
 * uplevel call from the error handler to initiate a failure_propagator
 */
int orte_errmgr_failure_propagate(orte_jobid_t *job, orte_process_name_t *daemon, orte_proc_state_t state) {
    int rc = ORTE_SUCCESS;

    orte_grpcomm_signature_t *sig;
    opal_buffer_t propagete_buff;

    /*packing variables*/    
    opal_list_t *procs = NULL, *edaemons = NULL, *info = NULL;
    int n,status, nprocs, nedaemons;
    opal_namelist_t *nm;
    status = state;
    nprocs = 0;
    nedaemons = 1;

    /* change the error daemon state*/ 
    ORTE_ACTIVATE_PROC_STATE(daemon, ORTE_PROC_STATE_HEARTBEAT_FAILED);

    OBJ_CONSTRUCT(&propagete_buff, opal_buffer_t);        

    /* pack the callback type */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &orte_errmgr_failure_propagator_cb_type, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }

    /* pack the status */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &status, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }

    /* pack the nprocs,default val is 0 */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &nprocs, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }
    /* if targets is not 0 , add them to the list */
    if (0 < nprocs) {
        procs = OBJ_NEW(opal_list_t);
        for (n=0; n < nprocs; n++) {
            nm = OBJ_NEW(opal_namelist_t);
            opal_list_append(procs, &nm->super);
            if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &nm->name, 1, OPAL_NAME))) {
                ORTE_ERROR_LOG(rc);
                OPAL_LIST_RELEASE(procs);
                OBJ_DESTRUCT(&propagete_buff);
                return rc;
            }
        }
    }
     
   /* pack the num of error daemons */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &nedaemons, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        if (NULL != procs) {
            OPAL_LIST_RELEASE(procs);
        }
        OBJ_DESTRUCT(&propagete_buff);
        return rc;
    }
    /* if error daemons is not 0, add them to the list */
    if (0 < nedaemons) {
        edaemons = OBJ_NEW(opal_list_t);
        for (n=0; n < nedaemons; n++) {
            nm = OBJ_NEW(opal_namelist_t);
            opal_list_append(edaemons, &nm->super);
            if (ORTE_SUCCESS != (rc = opal_dss.pack(&propagete_buff, &nm->name, 1, OPAL_NAME))) {
                ORTE_ERROR_LOG(rc);
                if (NULL != procs) {
                    OPAL_LIST_RELEASE(procs);
                }
                OPAL_LIST_RELEASE(edaemons);
                OBJ_DESTRUCT(&propagete_buff);
                return rc;
            }
        }
    }

    /* notify this error locally */
   // if (OPAL_SUCCESS != (rc = opal_pmix.server_notify_event(status, daemon, info, NULL, NULL))) {
   {
        ORTE_ERROR_LOG(rc);
        if (NULL != procs) {
            OPAL_LIST_RELEASE(procs);
        }
        if (NULL != edaemons) {
            OPAL_LIST_RELEASE(edaemons);
        }
        if (NULL != info) {
            OPAL_LIST_RELEASE(info);
        }
    }

    /* goes to all daemons */
    sig = OBJ_NEW(orte_grpcomm_signature_t);
    sig->signature = (orte_process_name_t*)malloc(sizeof(orte_process_name_t));
    sig->signature[0].jobid = daemon->jobid;
    /* all daemons hosting this jobid are participating */
    sig->signature[0].vpid = ORTE_VPID_WILDCARD;
    if (ORTE_SUCCESS != (rc = orte_grpcomm.rbcast(sig, ORTE_RML_TAG_PROPAGATE, &propagete_buff))) {
        ORTE_ERROR_LOG(rc);
    }

    OBJ_DESTRUCT(&propagete_buff);
    OBJ_RELEASE(sig);

    /* we're done! */
    return ORTE_SUCCESS;
}

int orte_errmgr_failure_propagate_recv(opal_buffer_t* buffer)
{
    int ret, cnt, state;
    orte_process_name_t temp_proc_name;
    orte_proc_t *temp_orte_proc;
 
    orte_grpcomm_signature_t *sig;
    orte_rml_tag_t tag;
    int cbtype, nprocs; 
    /* get the signature for propagating */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &sig, &cnt, ORTE_SIGNATURE))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return false;
    }
    OBJ_RELEASE(sig);

    /* get the target tag */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &tag, &cnt, ORTE_RML_TAG))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return false;
    }

    /* get the cbtype */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &cbtype, &cnt,OPAL_INT ))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return false;
    }

    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &state, &cnt, OPAL_INT))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return false;
    }

    /* unpack the target procs that are to be notified,default value is 0 */
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &nprocs, &cnt, OPAL_INT))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }

    /* unpack the procs that were impacted by the error */
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &nprocs, &cnt, OPAL_INT))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return false;
    }

    /* for propagate, only one daemon is affected per call */
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &temp_proc_name, &cnt, OPAL_NAME))) {
        ORTE_ERROR_LOG(ret);
        ORTE_FORCED_TERMINATE(ret);
        return false;
    }

    temp_orte_proc = orte_get_proc_object(&temp_proc_name);
    if( ORTE_PROC_STATE_HEARTBEAT_FAILED != temp_orte_proc->state ) {
        orte_errmgr_failure_propagate(&orte_process_info.my_daemon.jobid, &temp_proc_name, state);  
    } 
    return false;    
}

