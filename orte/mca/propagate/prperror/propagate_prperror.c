/*
 * Copyright (c) 2017      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
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

#include "orte/mca/propagate/propagate.h"
#include "orte/mca/propagate/base/base.h"

#include "orte/mca/errmgr/errmgr.h"
#include "orte/mca/errmgr/base/base.h"
#include "orte/mca/errmgr/base/errmgr_private.h"
#include "orte/mca/errmgr/detector/errmgr_detector.h"

#include "propagate_prperror.h"

static int orte_propagate_error_cb_type = -1;

static int init(void);
static int finalize(void);
static int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *sickproc, orte_proc_state_t state);

int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *daemon, orte_proc_state_t state);
int orte_propagate_prperror_recv(opal_buffer_t* buffer);

/* flag use to register callback for grpcomm rbcast forward */
int cbflag = 1;

orte_propagate_base_module_t orte_propagate_prperror_module ={
    init,
    finalize,
    orte_propagate_prperror
};

/*
 *Local functions
 */

static int init(void)
{
        return ORTE_SUCCESS;
}

static int finalize(void)
{
    int ret;
    if ( -1 == orte_propagate_error_cb_type){
        return ORTE_SUCCESS;
    }
    ret = orte_grpcomm.unregister_cb(orte_propagate_error_cb_type);
    orte_propagate_error_cb_type = -1;
    return ret;
}

/*
 * uplevel call from the error handler to initiate a failure_propagator
 */
static int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *sickproc, orte_proc_state_t state) {
    int rc = ORTE_SUCCESS;
    assert(*job == sickproc->jobid);

    orte_grpcomm_signature_t *sig;
    int cnt=0;
    /* -------------------------------------------------------
     * | cb_type | status | sickproc | nprocs afftected | procs | |||
     * ------------------------------------------------------*/
    opal_buffer_t prperror_buffer;
    opal_list_t affected_list;
    opal_list_t *info;
    opal_value_t *val;
    //opal_value_t *kv, *kvptr;

    /*set the status for pmix to use*/
    int status;
    //status =OPAL_ERR_PROC_ABORTED; // this should come from the caller
    status = state;
    /*use a tempproc to get the orte_proc obj*/
    orte_process_name_t temp_daemon;

    // register callback for rbcast for forwarding
    int ret;
    if(cbflag)
    {
        if(orte_grpcomm.register_cb!=NULL)
        ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_propagate_prperror_recv);
        if ( 0 <= ret ){
            orte_propagate_error_cb_type = ret;
        }
        cbflag = 0;
    }

    temp_daemon.jobid = sickproc->jobid;
    temp_daemon.vpid  = sickproc->vpid;

    /* change the error daemon state*/
    orte_proc_t *temp_orte_proc;
    temp_orte_proc = (orte_proc_t*)orte_get_proc_object(sickproc);
    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                        "propagate:daemon %d prperror daemon %d with state",
                        ORTE_PROC_MY_NAME->vpid,
                        sickproc->vpid,
                        errmgr_get_daemon_status(temp_daemon)));

    if(!errmgr_get_daemon_status(temp_daemon))
        return rc;
    OBJ_CONSTRUCT(&prperror_buffer, opal_buffer_t);
    info = OBJ_NEW(opal_list_t);
    /* pack the callback type */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &orte_propagate_error_cb_type, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&prperror_buffer);
        return rc;
    }

    /* pack the status */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &status, 1, OPAL_INT))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&prperror_buffer);
        return rc;
    }
    /* pack dead proc first */
    if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, sickproc, 1, ORTE_NAME))) {
        ORTE_ERROR_LOG(rc);
        OBJ_DESTRUCT(&prperror_buffer);
        return rc;
    }
    val = OBJ_NEW(opal_value_t);
    val->key = strdup(OPAL_PMIX_EVENT_AFFECTED_PROC);
    val->super = temp_orte_proc->super;
    opal_list_append(info, &val->super);
    /* check this is a daemon or not*/
    if (sickproc->vpid == orte_get_proc_daemon_vpid(sickproc)){
        /* Given a node name, return an array of processes within the specified jobid
         * on that node. If the specified node does not currently host any processes,
         * then the returned list will be empty.
         */
        OBJ_CONSTRUCT(&affected_list, opal_list_t);
        opal_pmix.resolve_peers(orte_get_proc_hostname(sickproc), sickproc->jobid, &affected_list);

        /* add those procs in the buffer*/
        if (!opal_list_is_empty(&affected_list)){
            cnt =affected_list.opal_list_length;
            if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &cnt, 1, OPAL_INT))) {
                ORTE_ERROR_LOG(rc);
                OBJ_DESTRUCT(&prperror_buffer);
                return rc;
            }
            if (0 < cnt) {
                OPAL_LIST_FOREACH(val, &affected_list, opal_value_t) {
                    if (OPAL_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &val, 1, OPAL_VALUE))) {
                        ORTE_ERROR_LOG(rc);
                        OBJ_DESTRUCT(&prperror_buffer);
                        return rc;
                     }
                    opal_list_append(info, &val->super);
                }
            }
        }
    }

    //kv->key = strdup(OPAL_PMIX_EVENT_AFFECTED_PROC);
    //opal_list_append(info, &kv->super);

    /* notify this error locally */
   if (OPAL_SUCCESS != (rc = opal_pmix.server_notify_event(status, ORTE_PROC_MY_NAME, info, NULL, NULL))) {
        ORTE_ERROR_LOG(rc);
        if (NULL != info) {
            OPAL_LIST_RELEASE(info);
        }
    }

    /* goes to all daemons */
    sig = OBJ_NEW(orte_grpcomm_signature_t);
    sig->signature = (orte_process_name_t*)malloc(sizeof(orte_process_name_t));
    sig->signature[0].jobid = sickproc->jobid;
    /* all daemons hosting this jobid are participating */
    sig->signature[0].vpid = ORTE_VPID_WILDCARD;
    if (ORTE_SUCCESS != (rc = orte_grpcomm.rbcast(sig, ORTE_RML_TAG_PROPAGATE, &prperror_buffer))) {
        ORTE_ERROR_LOG(rc);
    }

    OBJ_DESTRUCT(&prperror_buffer);
    OBJ_RELEASE(sig);
    /* we're done! */
    return ORTE_SUCCESS;
}

int orte_propagate_prperror_recv(opal_buffer_t* buffer)
{
    int ret, cnt, state;
    orte_process_name_t temp_proc_name;
    orte_proc_t *temp_orte_proc;
    int cbtype;

    /* get the cbtype */
    cnt=1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &cbtype, &cnt,OPAL_INT ))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &state, &cnt, OPAL_INT))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }
    /* for propagate, only one major sickproc is affected per call */
    cnt = 1;
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &temp_proc_name, &cnt, OPAL_NAME))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }
    temp_orte_proc = orte_get_proc_object(&temp_proc_name);
    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                "propagate:proerror daemon %d received %d",
                orte_process_info.my_name.vpid,
                temp_proc_name.vpid));

    if(errmgr_get_daemon_status(temp_proc_name)){ 
    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                "propagete:prperror daemon %d begin forwarding state is %d",
                orte_process_info.my_name.vpid,
                 errmgr_get_daemon_status(temp_proc_name)));

        orte_propagate_prperror(&orte_process_info.my_name.jobid, &temp_proc_name, state);
        errmgr_set_daemon_status(temp_proc_name , false) ;
    }
    return false;
}
