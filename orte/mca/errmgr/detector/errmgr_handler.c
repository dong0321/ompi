/*
 * Copyright (c) 2016      The University of Tennessee and The University
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

#include "opal/mca/base/mca_base_var.h"
#include "opal/mca/timer/base/base.h"
#include "opal/threads/threads.h"
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

#include "orte/mca/propagate/propagate.h"
#include "errmgr_detector.h"


static void register_cbfunc(int status, size_t errhndler, void *cbdata)
{
    opal_list_t *codes = (opal_list_t*)cbdata;
    myerrhandle = errhndler;
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "errmgr:detector:event register cbfunc with status %d", status));
    //OPAL_LIST_RELEASE(codes);
}

static void error_notify_cbfunc(int status,
        const opal_process_name_t *source,
        opal_list_t *info, opal_list_t *results,
        opal_pmix_notification_complete_fn_t cbfunc, void *cbdata)
{
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,"errmgr:detector:notify call back called"));
    orte_process_name_t proc;
    opal_value_t *kv;
    OPAL_LIST_FOREACH(kv, info, opal_value_t) {
        if (0 == strcmp(kv->key, OPAL_PMIX_EVENT_AFFECTED_PROC)) {
            proc.jobid = kv->data.name.jobid;
            proc.vpid = kv->data.name.vpid;
            break;
        }
    }
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "register evhandler error callback.."));
    if (NULL != cbfunc) {
        cbfunc(ORTE_SUCCESS, NULL, NULL, NULL, cbdata);
    }
    //orte_propagate.prp(&proc.jobid, &proc, ORTE_PROC_FLAG_ABORT);
}

void reg_errhandler()
{
    opal_list_t *codes;
    opal_value_t *ekv;
    orte_pmix_server_op_caddy_t *cd;
    ekv = OBJ_NEW(opal_value_t);
    ekv->key = strdup("errorcode");
    ekv->type = OPAL_INT;
    ekv->data.integer = ORTE_ERR_PROC_ABORTED;
    opal_list_append(codes, &ekv->super);
    OPAL_OUTPUT_VERBOSE((5, orte_errmgr_base_framework.framework_output,
                "errmgr:detector: register evhandler in errmgr.."));
    opal_pmix.register_evhandler(codes, NULL, error_notify_cbfunc, register_cbfunc,codes);
}


