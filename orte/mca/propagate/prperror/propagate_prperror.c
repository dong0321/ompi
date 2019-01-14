/*
 * Copyright (c) 2017-2018 The University of Tennessee and The University
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
#include <pmix.h>
#include <pmix_server.h>

#include "opal/util/output.h"
#include "opal/dss/dss.h"

#include "opal/pmix/pmix-internal.h"
#include "orte/orted/pmix/pmix_server_internal.h"
#include "orte/orted/pmix/pmix_server.h"
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

opal_list_t orte_error_procs = {{0}};

static int orte_propagate_error_cb_type = -1;

static int init(void);
static int finalize(void);
static int register_prp_callback(void);
static int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *source,
        orte_process_name_t *sickproc, orte_proc_state_t state);

int orte_propagate_prperror_recv(opal_buffer_t* buffer);

/* flag use to register callback for grpcomm rbcast forward */
int enable_callback_register_flag = 1;

orte_propagate_base_module_t orte_propagate_prperror_module ={
    init,
    finalize,
    orte_propagate_prperror,
    register_prp_callback
};

/*
 *Local functions
 */
static int init(void)
{
    OBJ_CONSTRUCT(&orte_error_procs, opal_list_t);
    return ORTE_SUCCESS;
}

static int register_prp_callback(void)
{
    int ret;
    if(enable_callback_register_flag)
    {
        if(orte_grpcomm.register_cb!=NULL)
            ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_propagate_prperror_recv);
        if ( 0 <= ret ){
            orte_propagate_error_cb_type = ret;
        }
        enable_callback_register_flag = 0;
    }
    return ORTE_SUCCESS;
}

static int finalize(void)
{
    int ret=0;
    if ( -1 == orte_propagate_error_cb_type){
        return ORTE_SUCCESS;
    }
    /* do we need unregister ? cause all module is unloading, those memory maybe collecred may have illegal access */
    /* ret = orte_grpcomm.unregister_cb(orte_propagate_error_cb_type); */
    orte_propagate_error_cb_type = -1;
    OBJ_DESTRUCT(&orte_error_procs);
    return ret;
}

/*
 * uplevel call from the error handler to initiate a failure_propagator
 */
static int orte_propagate_prperror(orte_jobid_t *job, orte_process_name_t *source,
        orte_process_name_t *sickproc, orte_proc_state_t state) {

    int rc = ORTE_SUCCESS;
    /* don't need to check jobid because this can be different: daemon and process has different jobids */

    bool daemon_error_flag = false;

    /* namelist for tracking error procs */
    orte_namelist_t *nmcheck, *nm;
    nmcheck = OBJ_NEW(orte_namelist_t);

    OPAL_LIST_FOREACH(nmcheck, &orte_error_procs, orte_namelist_t){
        if ((nmcheck->name.jobid == sickproc->jobid) && (nmcheck->name.vpid == sickproc->vpid))
        {
            OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                        "propagate: prperror: already propagated this msg: error proc is %s", ORTE_NAME_PRINT(sickproc) ));
            return rc;
        }
    }
    nm = OBJ_NEW(orte_namelist_t);
    nm->name.jobid = sickproc->jobid;
    nm->name.vpid = sickproc->vpid;
    opal_list_append(&orte_error_procs, &(nm->super));

    orte_grpcomm_signature_t *sig;
    int cnt=0;
    /* ---------------------------------------------------------
     * | cb_type | status | sickproc | nprocs afftected | procs|
     * --------------------------------------------------------*/
    opal_buffer_t prperror_buffer;

    /* set the status for pmix to use */
    orte_proc_state_t status;
    status = state;

    /* register callback for rbcast for forwarding */
    int ret;
    if(enable_callback_register_flag)
    {
        if(orte_grpcomm.register_cb!=NULL)
            ret= orte_grpcomm.register_cb((orte_grpcomm_rbcast_cb_t)orte_propagate_prperror_recv);
        if ( 0 <= ret ){
            orte_propagate_error_cb_type = ret;
        }
        enable_callback_register_flag = 0;
    }

    /* change the error daemon state*/
    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                "propagate: prperror: daemon %s rbcast state %d of proc %s",
                ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),state, ORTE_NAME_PRINT(sickproc)));

    OBJ_CONSTRUCT(&prperror_buffer, opal_buffer_t);
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

    orte_node_t *node;
    /* check this is a daemon or not, if vpid is same cannot ensure this is daemon also need check jobid*/
    if (sickproc->vpid == orte_get_proc_daemon_vpid(sickproc) && (sickproc->jobid == ORTE_PROC_MY_NAME->jobid) ){
        /* Given a node name, return an array of processes within the specified jobid
         * on that node. If the specified node does not currently host any processes,
         * then the returned list will be empty.
         */
        OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                    "%s propagate:daemon prperror this is a daemon error on %s \n",
                    ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                    orte_get_proc_hostname(sickproc)));

        node = (orte_node_t*)opal_pointer_array_get_item(orte_node_pool, sickproc->vpid);

        cnt=node->num_procs;
        if (ORTE_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &cnt, 1, OPAL_INT))) {
            ORTE_ERROR_LOG(rc);
            OBJ_DESTRUCT(&prperror_buffer);
            return rc;
        }
        daemon_error_flag = true;
    }
    pmix_proc_t pname;
    pmix_info_t *pinfo;
    PMIX_INFO_CREATE(pinfo, 1+cnt);

    OPAL_PMIX_CONVERT_NAME(&pname, sickproc);
    PMIX_INFO_LOAD(&pinfo[0], PMIX_EVENT_AFFECTED_PROC, &pname, PMIX_PROC );

    if(daemon_error_flag) {
        orte_proc_t *pptr;
        opal_buffer_t *alert;
        orte_plm_cmd_flag_t cmd;

        int i;
        for (i=0; i < node->procs->size; i++) {
            if (NULL != (pptr = (orte_proc_t*)opal_pointer_array_get_item(node->procs, i))) {
                OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                            " %d children are afftected  %s\n",cnt, ORTE_NAME_PRINT(&pptr->name)));

                if (OPAL_SUCCESS != (rc = opal_dss.pack(&prperror_buffer, &pptr->name, 1, OPAL_NAME))) {
                    ORTE_ERROR_LOG(rc);
                    OBJ_DESTRUCT(&prperror_buffer);
                    return rc;
                }
                OPAL_PMIX_CONVERT_NAME(&pname, &pptr->name);
                PMIX_INFO_LOAD(&pinfo[i+1], PMIX_EVENT_AFFECTED_PROC, &pname, PMIX_PROC );

                alert = OBJ_NEW(opal_buffer_t);
                /* pack update state command */
                cmd = ORTE_PLM_UPDATE_PROC_STATE;
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &cmd, 1, ORTE_PLM_CMD))) {
                    ORTE_ERROR_LOG(rc);
                    return rc;
                }

                /* pack jobid */
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &(pptr->name.jobid), 1, ORTE_JOBID))) {
                    ORTE_ERROR_LOG(rc);
                    return rc;
                }

                /* proc state now is ORTE_PROC_STATE_ABORTED_BY_SIG, cause odls set state to this; code is 128+9 */
                pptr->state = ORTE_PROC_STATE_ABORTED_BY_SIG;
                /* pack the child's vpid */
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &(pptr->name.vpid), 1, ORTE_VPID))) {
                    ORTE_ERROR_LOG(rc);
                    return rc;
                }
                /* pack the pid */
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &pptr->pid, 1, OPAL_PID))) {
                    ORTE_ERROR_LOG(rc);
                    return rc;
                }
                /* pack its state */
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &pptr->state, 1, ORTE_PROC_STATE))) {
                    ORTE_ERROR_LOG(rc);
                    return rc;
                }
                /* pack its exit code */
                if (ORTE_SUCCESS != (rc = opal_dss.pack(alert, &pptr->exit_code, 1, ORTE_EXIT_CODE))) {
                    ORTE_ERROR_LOG(rc);
                    return rc;
                }

                /* send this process's info to hnp */
                if (0 > (rc = orte_rml.send_buffer_nb(orte_mgmt_conduit,
                                ORTE_PROC_MY_HNP, alert,
                                ORTE_RML_TAG_PLM,
                                orte_rml_send_callback, NULL))) {
                    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                                "%s errmgr:detector: send to hnp failed",
                                ORTE_NAME_PRINT(ORTE_PROC_MY_NAME)));
                    ORTE_ERROR_LOG(rc);
                    OBJ_RELEASE(alert);
                }
                if (ORTE_FLAG_TEST(pptr, ORTE_PROC_FLAG_IOF_COMPLETE) &&
                        ORTE_FLAG_TEST(pptr, ORTE_PROC_FLAG_WAITPID) &&
                        !ORTE_FLAG_TEST(pptr, ORTE_PROC_FLAG_RECORDED)) {
                    ORTE_ACTIVATE_PROC_STATE(&(pptr->name), ORTE_PROC_STATE_TERMINATED);
                }
            }
        }
    }

    /* goes to all daemons */
    sig = OBJ_NEW(orte_grpcomm_signature_t);
    sig->signature = (orte_process_name_t*)malloc(sizeof(orte_process_name_t));
    sig->signature[0].jobid = ORTE_PROC_MY_NAME->jobid;
    /* all daemons hosting this jobid are participating */
    sig->signature[0].vpid = ORTE_VPID_WILDCARD;
    if (ORTE_SUCCESS != (rc = orte_grpcomm.rbcast(sig, ORTE_RML_TAG_PROPAGATE, &prperror_buffer))) {
        ORTE_ERROR_LOG(rc);
    }
    /* notify this error locally, only from rbcast dont have a source id */
    if( source==NULL ) {
        if (OPAL_SUCCESS != PMIx_Notify_event(opal_pmix_convert_rc(state), NULL,
                    PMIX_RANGE_LOCAL, pinfo, cnt+1,
                    NULL,NULL )) {
            OBJ_RELEASE(pinfo);
        }
    }
    OBJ_DESTRUCT(&prperror_buffer);
    OBJ_RELEASE(sig);
    /* we're done! */
    return ORTE_SUCCESS;
}

int orte_propagate_prperror_recv(opal_buffer_t* buffer)
{
    int ret, cnt, state;
    orte_process_name_t sickproc;
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
    if (ORTE_SUCCESS != (ret = opal_dss.unpack(buffer, &sickproc, &cnt, OPAL_NAME))) {
        ORTE_ERROR_LOG(ret);
        return false;
    }

    OPAL_OUTPUT_VERBOSE((5, orte_propagate_base_framework.framework_output,
                "%s propagete: prperror: daemon received %s gone forwarding with status %d",
                ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), ORTE_NAME_PRINT(&sickproc), state));

    orte_propagate_prperror(&orte_process_info.my_name.jobid, NULL, &sickproc, state);
    return false;
}
