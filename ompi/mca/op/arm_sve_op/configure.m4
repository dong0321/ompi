#
# Copyright (c) 2019      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (C) 2019      Arm Ltd.  ALL RIGHTS RESERVED.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# SVE op component configure.m4 file.  This file is slurped in by
# Open MPI's autogen.pl to be part of the top-level configure script.
# This script must define (via AC_DEFUN) an m4 macro named
# MCA_<framework>_<component>_CONFIG that executes either $1 if the
# component wants to build itself, or $2 if the component does not
# want to build itself.

# Do *NOT* invoke AC_MSG_ERROR, or any other macro that will abort
# configure, except upon catastrophic error.  For sve, it *is* a
# catastropic error if the user specifically requested your component
# but it cannot be built.  If it *not* a catastropic error if your
# component cannot be built (but was not specifically requested).

# See https://github.com/open-mpi/ompi/wiki/devel-CreateComponent
# for more details on how to make Open MPI components.

# MCA_op_sve_CONFIG([action-if-found], [action-if-not-found])
# -----------------------------------------------------------
AC_DEFUN([MCA_ompi_op_arm_sve_op_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/op/arm_sve_op/Makefile])

    # Add checks here for any necessary header files and/or libraries
    # that must be present to compile your component.

    # This sve performs a fairly simple test (checking for the
    # "struct sockaddr_in" C type), just for the sake of showing you
    # one test and executing either $1 or $2, depending on the output
    # of the test.

    # check for sockaddr_in (a good sign we have TCP)
    AC_CHECK_TYPES([struct sockaddr_in],
                   [$1],
                   [$2],
                   [AC_INCLUDES_DEFAULT
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif])

    # Let's pretend that we found version A.B.C of the "libfoo"
    # support library that is necessary to compile/link this
    # component.  We'll AC_DEFINE the A, B, and C values so that they
    # can be printed as information MCA parameters via ompi_info.  See
     op_sve_component.c to see how these values are used.

    AC_DEFINE_UNQUOTED(OP_SVE_LIBFOO_VERSION_MAJOR, ["17"],
                       [Major version number of the "libfoo" library])
    AC_DEFINE_UNQUOTED(OP_SVE_LIBFOO_VERSION_MINOR, ["38"],
                       [Minor version number of the "libfoo" library])
    AC_DEFINE_UNQUOTED(OP_SVE_LIBFOO_VERSION_RELEASE, ["4"],
                       [Release version number of the "libfoo" library])
])dnl
