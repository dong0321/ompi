dnl
dnl Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
dnl                         University Research and Technology
dnl                         Corporation.  All rights reserved.
dnl Copyright (c) 2004-2018 The University of Tennessee and The University
dnl                         of Tennessee Research Foundation.  All rights
dnl                         reserved.
dnl Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
dnl                         University of Stuttgart.  All rights reserved.
dnl Copyright (c) 2004-2005 The Regents of the University of California.
dnl                         All rights reserved.
dnl Copyright (c) 2008-2018 Cisco Systems, Inc.  All rights reserved.
dnl Copyright (c) 2010      Oracle and/or its affiliates.  All rights reserved.
dnl Copyright (c) 2015-2017 Research Organization for Information Science
dnl                         and Technology (RIST). All rights reserved.
dnl Copyright (c) 2014-2018 Los Alamos National Security, LLC. All rights
dnl                         reserved.
dnl Copyright (c) 2017      Amazon.com, Inc. or its affiliates.  All Rights
dnl                         reserved.
dnl Copyright (c) 2018-2019 Intel, Inc.  All rights reserved.
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

dnl This is a C test to see if 128-bit __atomic_compare_exchange_n()
dnl actually works (e.g., it compiles and links successfully on
dnl ARM64+clang, but returns incorrect answers as of August 2018).
AC_DEFUN([PRRTE_ATOMIC_COMPARE_EXCHANGE_N_TEST_SOURCE],[[
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef union {
    uint64_t fake@<:@2@:>@;
    __int128 real;
} prrte128;

static void test1(void)
{
    // As of Aug 2018, we could not figure out a way to assign 128-bit
    // constants -- the compilers would not accept it.  So use a fake
    // union to assign 2 uin64_t's to make a single __int128.
    prrte128 ptr      = { .fake = { 0xFFEEDDCCBBAA0099, 0x8877665544332211 }};
    prrte128 expected = { .fake = { 0x11EEDDCCBBAA0099, 0x88776655443322FF }};
    prrte128 desired  = { .fake = { 0x1122DDCCBBAA0099, 0x887766554433EEFF }};
    bool r = __atomic_compare_exchange_n(&ptr.real, &expected.real,
                                         desired.real, true,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    if ( !(r == false && ptr.real == expected.real)) {
        exit(1);
    }
}

static void test2(void)
{
    prrte128 ptr =      { .fake = { 0xFFEEDDCCBBAA0099, 0x8877665544332211 }};
    prrte128 expected = ptr;
    prrte128 desired =  { .fake = { 0x1122DDCCBBAA0099, 0x887766554433EEFF }};
    bool r = __atomic_compare_exchange_n(&ptr.real, &expected.real,
                                         desired.real, true,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    if (!(r == true && ptr.real == desired.real)) {
        exit(2);
    }
}

int main(int argc, char** argv)
{
    test1();
    test2();
    return 0;
}
]])

dnl ------------------------------------------------------------------

dnl This is a C test to see if 128-bit __sync_bool_compare_and_swap()
dnl actually works (e.g., it compiles and links successfully on
dnl ARM64+clang, but returns incorrect answers as of August 2018).
AC_DEFUN([PRRTE_SYNC_BOOL_COMPARE_AND_SWAP_TEST_SOURCE],[[
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef union {
    uint64_t fake@<:@2@:>@;
    __int128 real;
} prrte128;

static void test1(void)
{
    // As of Aug 2018, we could not figure out a way to assign 128-bit
    // constants -- the compilers would not accept it.  So use a fake
    // union to assign 2 uin64_t's to make a single __int128.
    prrte128 ptr    = { .fake = { 0xFFEEDDCCBBAA0099, 0x8877665544332211 }};
    prrte128 oldval = { .fake = { 0x11EEDDCCBBAA0099, 0x88776655443322FF }};
    prrte128 newval = { .fake = { 0x1122DDCCBBAA0099, 0x887766554433EEFF }};
    bool r = __sync_bool_compare_and_swap(&ptr.real, oldval.real, newval.real);
    if (!(r == false && ptr.real != newval.real)) {
        exit(1);
    }
}

static void test2(void)
{
    prrte128 ptr    = { .fake = { 0xFFEEDDCCBBAA0099, 0x8877665544332211 }};
    prrte128 oldval = ptr;
    prrte128 newval = { .fake = { 0x1122DDCCBBAA0099, 0x887766554433EEFF }};
    bool r = __sync_bool_compare_and_swap(&ptr.real, oldval.real, newval.real);
    if (!(r == true && ptr.real == newval.real)) {
        exit(2);
    }
}

int main(int argc, char** argv)
{
    test1();
    test2();
    return 0;
}
]])

dnl This is a C test to see if 128-bit __atomic_compare_exchange_n()
dnl actually works (e.g., it compiles and links successfully on
dnl ARM64+clang, but returns incorrect answers as of August 2018).
AC_DEFUN([PRRTE_ATOMIC_COMPARE_EXCHANGE_STRONG_TEST_SOURCE],[[
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdatomic.h>

typedef union {
    uint64_t fake@<:@2@:>@;
    _Atomic __int128 real;
} prrte128;

static void test1(void)
{
    // As of Aug 2018, we could not figure out a way to assign 128-bit
    // constants -- the compilers would not accept it.  So use a fake
    // union to assign 2 uin64_t's to make a single __int128.
    prrte128 ptr      = { .fake = { 0xFFEEDDCCBBAA0099, 0x8877665544332211 }};
    prrte128 expected = { .fake = { 0x11EEDDCCBBAA0099, 0x88776655443322FF }};
    prrte128 desired  = { .fake = { 0x1122DDCCBBAA0099, 0x887766554433EEFF }};
    bool r = atomic_compare_exchange_strong (&ptr.real, &expected.real,
                                             desired.real, true,
					     atomic_relaxed, atomic_relaxed);
    if ( !(r == false && ptr.real == expected.real)) {
        exit(1);
    }
}

static void test2(void)
{
    prrte128 ptr =      { .fake = { 0xFFEEDDCCBBAA0099, 0x8877665544332211 }};
    prrte128 expected = ptr;
    prrte128 desired =  { .fake = { 0x1122DDCCBBAA0099, 0x887766554433EEFF }};
    bool r = atomic_compare_exchange_strong (&ptr.real, &expected.real,
                                             desired.real, true,
					     atomic_relaxed, atomic_relaxed);
    if (!(r == true && ptr.real == desired.real)) {
        exit(2);
    }
}

int main(int argc, char** argv)
{
    test1();
    test2();
    return 0;
}
]])

dnl ------------------------------------------------------------------

dnl
dnl Check to see if a specific function is linkable.
dnl
dnl Check with:
dnl 1. No compiler/linker flags.
dnl 2. CFLAGS += -mcx16
dnl 3. LIBS += -latomic
dnl 4. Finally, if it links ok with any of #1, #2, or #3, actually try
dnl to run the test code (if we're not cross-compiling) and verify
dnl that it actually gives us the correct result.
dnl
dnl Note that we unfortunately can't use AC SEARCH_LIBS because its
dnl check incorrectly fails (because these functions are special compiler
dnl intrinsics -- SEARCH_LIBS tries with "check FUNC()", which the
dnl compiler complains doesn't match the internal prototype).  So we have
dnl to use our own LINK_IFELSE tests.  Indeed, since these functions are
dnl so special, we actually need a valid source code that calls the
dnl functions with correct arguments, etc.  It's not enough, for example,
dnl to do the usual "try to set a function pointer to the symbol" trick to
dnl determine if these functions are available, because the compiler may
dnl not implement these as actual symbols.  So just try to link a real
dnl test code.
dnl
dnl $1: function name to print
dnl $2: program to test
dnl $3: action if any of 1, 2, or 3 succeeds
dnl #4: action if all of 1, 2, and 3 fail
dnl
AC_DEFUN([PRRTE_ASM_CHECK_ATOMIC_FUNC],[
    PRRTE_VAR_SCOPE_PUSH([prrte_asm_check_func_happy prrte_asm_check_func_CFLAGS_save prrte_asm_check_func_LIBS_save])

    prrte_asm_check_func_CFLAGS_save=$CFLAGS
    prrte_asm_check_func_LIBS_save=$LIBS

    dnl Check with no compiler/linker flags
    AC_MSG_CHECKING([for $1])
    AC_LINK_IFELSE([$2],
        [prrte_asm_check_func_happy=1
         AC_MSG_RESULT([yes])],
        [prrte_asm_check_func_happy=0
         AC_MSG_RESULT([no])])

    dnl If that didn't work, try again with CFLAGS+=mcx16
    AS_IF([test $prrte_asm_check_func_happy -eq 0],
        [AC_MSG_CHECKING([for $1 with -mcx16])
         CFLAGS="$CFLAGS -mcx16"
         AC_LINK_IFELSE([$2],
             [prrte_asm_check_func_happy=1
              AC_MSG_RESULT([yes])],
             [prrte_asm_check_func_happy=0
              CFLAGS=$prrte_asm_check_func_CFLAGS_save
              AC_MSG_RESULT([no])])
         ])

    dnl If that didn't work, try again with LIBS+=-latomic
    AS_IF([test $prrte_asm_check_func_happy -eq 0],
        [AC_MSG_CHECKING([for $1 with -latomic])
         LIBS="$LIBS -latomic"
         AC_LINK_IFELSE([$2],
             [prrte_asm_check_func_happy=1
              AC_MSG_RESULT([yes])],
             [prrte_asm_check_func_happy=0
              LIBS=$prrte_asm_check_func_LIBS_save
              AC_MSG_RESULT([no])])
         ])

    dnl If we have it, try it and make sure it gives a correct result.
    dnl As of Aug 2018, we know that it links but does *not* work on clang
    dnl 6 on ARM64.
    AS_IF([test $prrte_asm_check_func_happy -eq 1],
        [AC_MSG_CHECKING([if $1() gives correct results])
         AC_RUN_IFELSE([$2],
              [AC_MSG_RESULT([yes])],
              [prrte_asm_check_func_happy=0
               AC_MSG_RESULT([no])],
              [AC_MSG_RESULT([cannot test -- assume yes (cross compiling)])])
         ])

    dnl If we were unsuccessful, restore CFLAGS/LIBS
    AS_IF([test $prrte_asm_check_func_happy -eq 0],
        [CFLAGS=$prrte_asm_check_func_CFLAGS_save
         LIBS=$prrte_asm_check_func_LIBS_save])

    dnl Run the user actions
    AS_IF([test $prrte_asm_check_func_happy -eq 1], [$3], [$4])

    PRRTE_VAR_SCOPE_POP
])

dnl ------------------------------------------------------------------

AC_DEFUN([PRRTE_CHECK_SYNC_BUILTIN_CSWAP_INT128], [
  PRRTE_VAR_SCOPE_PUSH([sync_bool_compare_and_swap_128_result])

  # Do we have __sync_bool_compare_and_swap?
  # Use a special macro because we need to check with a few different
  # CFLAGS/LIBS.
  PRRTE_ASM_CHECK_ATOMIC_FUNC([__sync_bool_compare_and_swap],
      [AC_LANG_SOURCE(PRRTE_SYNC_BOOL_COMPARE_AND_SWAP_TEST_SOURCE)],
      [sync_bool_compare_and_swap_128_result=1],
      [sync_bool_compare_and_swap_128_result=0])

  AC_DEFINE_UNQUOTED([PRRTE_HAVE_SYNC_BUILTIN_CSWAP_INT128],
        [$sync_bool_compare_and_swap_128_result],
        [Whether the __sync builtin atomic compare and swap supports 128-bit values])

  PRRTE_VAR_SCOPE_POP
])

AC_DEFUN([PRRTE_CHECK_SYNC_BUILTINS], [
  AC_MSG_CHECKING([for __sync builtin atomics])

  AC_TRY_LINK([long tmp;], [__sync_synchronize();
__sync_bool_compare_and_swap(&tmp, 0, 1);
__sync_add_and_fetch(&tmp, 1);],
    [AC_MSG_RESULT([yes])
     $1],
    [AC_MSG_RESULT([no])
     $2])

  AC_MSG_CHECKING([for 64-bit __sync builtin atomics])

  AC_TRY_LINK([
#include <stdint.h>
uint64_t tmp;], [
__sync_bool_compare_and_swap(&tmp, 0, 1);
__sync_add_and_fetch(&tmp, 1);],
    [AC_MSG_RESULT([yes])
     prrte_asm_sync_have_64bit=1],
    [AC_MSG_RESULT([no])
     prrte_asm_sync_have_64bit=0])

  AC_DEFINE_UNQUOTED([PRRTE_ASM_SYNC_HAVE_64BIT],[$prrte_asm_sync_have_64bit],
                     [Whether 64-bit is supported by the __sync builtin atomics])

  # Check for 128-bit support
  PRRTE_CHECK_SYNC_BUILTIN_CSWAP_INT128
])


AC_DEFUN([PRRTE_CHECK_GCC_BUILTIN_CSWAP_INT128], [
  PRRTE_VAR_SCOPE_PUSH([atomic_compare_exchange_n_128_result atomic_compare_exchange_n_128_CFLAGS_save atomic_compare_exchange_n_128_LIBS_save])

  atomic_compare_exchange_n_128_CFLAGS_save=$CFLAGS
  atomic_compare_exchange_n_128_LIBS_save=$LIBS

  # Do we have __sync_bool_compare_and_swap?
  # Use a special macro because we need to check with a few different
  # CFLAGS/LIBS.
  PRRTE_ASM_CHECK_ATOMIC_FUNC([__atomic_compare_exchange_n],
      [AC_LANG_SOURCE(PRRTE_ATOMIC_COMPARE_EXCHANGE_N_TEST_SOURCE)],
      [atomic_compare_exchange_n_128_result=1],
      [atomic_compare_exchange_n_128_result=0])

  # If we have it and it works, check to make sure it is always lock
  # free.
  AS_IF([test $atomic_compare_exchange_n_128_result -eq 1],
        [AC_MSG_CHECKING([if __int128 atomic compare-and-swap is always lock-free])
         AC_RUN_IFELSE([AC_LANG_PROGRAM([], [if (!__atomic_always_lock_free(16, 0)) { return 1; }])],
              [AC_MSG_RESULT([yes])],
              [atomic_compare_exchange_n_128_result=0
               # If this test fails, need to reset CFLAGS/LIBS (the
               # above tests atomically set CFLAGS/LIBS or not; this
               # test is running after the fact, so we have to undo
               # the side-effects of setting CFLAGS/LIBS if the above
               # tests passed).
               CFLAGS=$atomic_compare_exchange_n_128_CFLAGS_save
               LIBS=$atomic_compare_exchange_n_128_LIBS_save
               AC_MSG_RESULT([no])],
              [AC_MSG_RESULT([cannot test -- assume yes (cross compiling)])])
        ])

  AC_DEFINE_UNQUOTED([PRRTE_HAVE_GCC_BUILTIN_CSWAP_INT128],
        [$atomic_compare_exchange_n_128_result],
        [Whether the __atomic builtin atomic compare swap is both supported and lock-free on 128-bit values])

  dnl If we could not find decent support for 128-bits __atomic let's
  dnl try the GCC _sync
  AS_IF([test $atomic_compare_exchange_n_128_result -eq 0],
      [PRRTE_CHECK_SYNC_BUILTIN_CSWAP_INT128])

  PRRTE_VAR_SCOPE_POP
])

AC_DEFUN([PRRTE_CHECK_GCC_ATOMIC_BUILTINS], [
  AC_MSG_CHECKING([for __atomic builtin atomics])

  AC_TRY_LINK([
#include <stdint.h>
uint32_t tmp, old = 0;
uint64_t tmp64, old64 = 0;], [
__atomic_thread_fence(__ATOMIC_SEQ_CST);
__atomic_compare_exchange_n(&tmp, &old, 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
__atomic_add_fetch(&tmp, 1, __ATOMIC_RELAXED);
__atomic_compare_exchange_n(&tmp64, &old64, 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
__atomic_add_fetch(&tmp64, 1, __ATOMIC_RELAXED);],
    [AC_MSG_RESULT([yes])
     $1],
    [AC_MSG_RESULT([no])
     $2])

  # Check for 128-bit support
  PRRTE_CHECK_GCC_BUILTIN_CSWAP_INT128
])

AC_DEFUN([PRRTE_CHECK_C11_CSWAP_INT128], [
  PRRTE_VAR_SCOPE_PUSH([atomic_compare_exchange_result atomic_compare_exchange_CFLAGS_save atomic_compare_exchange_LIBS_save])

  atomic_compare_exchange_CFLAGS_save=$CFLAGS
  atomic_compare_exchange_LIBS_save=$LIBS

  # Do we have C11 atomics on 128-bit integers?
  # Use a special macro because we need to check with a few different
  # CFLAGS/LIBS.
  PRRTE_ASM_CHECK_ATOMIC_FUNC([atomic_compare_exchange_strong_16],
      [AC_LANG_SOURCE(PRRTE_ATOMIC_COMPARE_EXCHANGE_STRONG_TEST_SOURCE)],
      [atomic_compare_exchange_result=1],
      [atomic_compare_exchange_result=0])

  # If we have it and it works, check to make sure it is always lock
  # free.
  AS_IF([test $atomic_compare_exchange_result -eq 1],
        [AC_MSG_CHECKING([if C11 __int128 atomic compare-and-swap is always lock-free])
         AC_RUN_IFELSE([AC_LANG_PROGRAM([#include <stdatomic.h>], [_Atomic __int128_t x; if (!atomic_is_lock_free(&x)) { return 1; }])],
              [AC_MSG_RESULT([yes])],
              [atomic_compare_exchange_result=0
               # If this test fails, need to reset CFLAGS/LIBS (the
               # above tests atomically set CFLAGS/LIBS or not; this
               # test is running after the fact, so we have to undo
               # the side-effects of setting CFLAGS/LIBS if the above
               # tests passed).
               CFLAGS=$atomic_compare_exchange_CFLAGS_save
               LIBS=$atomic_compare_exchange_LIBS_save
               AC_MSG_RESULT([no])],
              [AC_MSG_RESULT([cannot test -- assume yes (cross compiling)])])
        ])

  AC_DEFINE_UNQUOTED([PRRTE_HAVE_C11_CSWAP_INT128],
        [$atomic_compare_exchange_result],
        [Whether C11 atomic compare swap is both supported and lock-free on 128-bit values])

  dnl If we could not find decent support for 128-bits atomic let's
  dnl try the GCC _sync
  AS_IF([test $atomic_compare_exchange_result -eq 0],
      [PRRTE_CHECK_SYNC_BUILTIN_CSWAP_INT128])

  PRRTE_VAR_SCOPE_POP
])

AC_DEFUN([PRRTE_CHECK_GCC_ATOMIC_BUILTINS], [
  AC_MSG_CHECKING([for __atomic builtin atomics])

  AC_TRY_LINK([
#include <stdint.h>
uint32_t tmp, old = 0;
uint64_t tmp64, old64 = 0;], [
__atomic_thread_fence(__ATOMIC_SEQ_CST);
__atomic_compare_exchange_n(&tmp, &old, 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
__atomic_add_fetch(&tmp, 1, __ATOMIC_RELAXED);
__atomic_compare_exchange_n(&tmp64, &old64, 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
__atomic_add_fetch(&tmp64, 1, __ATOMIC_RELAXED);],
    [AC_MSG_RESULT([yes])
     $1],
    [AC_MSG_RESULT([no])
     $2])

  # Check for 128-bit support
  PRRTE_CHECK_GCC_BUILTIN_CSWAP_INT128
])


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_TEXT
dnl
dnl Determine how to set current mode as text.
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_TEXT],[
    AC_MSG_CHECKING([directive for setting text section])
    prrte_cv_asm_text=""
    if test "$prrte_cv_c_compiler_vendor" = "microsoft" ; then
        # text section will be brought in with the rest of
        # header for MS - leave blank for now
        prrte_cv_asm_text=""
    else
        case $host in
            *-aix*)
                prrte_cv_asm_text=[".csect .text[PR]"]
            ;;
            *)
                prrte_cv_asm_text=".text"
            ;;
        esac
    fi
    AC_MSG_RESULT([$prrte_cv_asm_text])
    AC_DEFINE_UNQUOTED([PRRTE_ASM_TEXT], ["$prrte_cv_asm_text"],
                       [Assembly directive for setting text section])
    PRRTE_ASM_TEXT="$prrte_cv_asm_text"
    AC_SUBST(PRRTE_ASM_TEXT)
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_GLOBAL
dnl
dnl Sets PRRTE_ASM_GLOBAL to the value to prefix global values
dnl
dnl I'm sure if I don't have a test for this, there will be some
dnl dumb platform that uses something else
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_GLOBAL],[
    AC_MSG_CHECKING([directive for exporting symbols])
    prrte_cv_asm_global=""
    if test "$prrte_cv_c_compiler_vendor" = "microsoft" ; then
        prrte_cv_asm_global="PUBLIC"
    else
        case $host in
            *)
                prrte_cv_asm_global=".globl"
            ;;
        esac
    fi
    AC_MSG_RESULT([$prrte_cv_asm_global])
    AC_DEFINE_UNQUOTED([PRRTE_ASM_GLOBAL], ["$prrte_cv_asm_global"],
                       [Assembly directive for exporting symbols])
    PRRTE_ASM_GLOBAL="$prrte_cv_asm_global"
    AC_SUBST(PRRTE_AS_GLOBAL)
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_LSYM
dnl
dnl Sets PRRTE_ASM_LSYM to the prefix value on a symbol to make it
dnl an internal label (jump target and whatnot)
dnl
dnl We look for L .L $ L$ (in that order) for something that both
dnl assembles and does not leave a label in the output of nm.  Fall
dnl back to L if nothing else seems to work :/
dnl
dnl #################################################################

# _PRRTE_CHECK_ASM_LSYM([variable-to-set])
# ---------------------------------------
AC_DEFUN([_PRRTE_CHECK_ASM_LSYM],[
    AC_REQUIRE([AC_PROG_GREP])

    $1="L"

    for sym in L .L $ L$ ; do
        asm_result=0
        echo "configure: trying $sym" >&AC_FD_CC
        PRRTE_TRY_ASSEMBLE([foobar$prrte_cv_asm_label_suffix
${sym}mytestlabel$prrte_cv_asm_label_suffix],
            [# ok, we succeeded at assembling.  see if we can nm,
             # throwing the results in a file
            if $NM conftest.$OBJEXT > conftest.out 2>&AC_FD_CC ; then
                if test "`$GREP mytestlabel conftest.out`" = "" ; then
                    # there was no symbol...  looks promising to me
                    $1="$sym"
                    asm_result=1
                elif test ["`$GREP ' [Nt] .*mytestlabel' conftest.out`"] = "" ; then
                    # see if we have a non-global-ish symbol
                    # but we should see if we can do better.
                    $1="$sym"
                fi
            else
                # not so much on the NM goodness :/
                echo "$NM failed.  Output from NM was:" >&AC_FD_CC
                cat conftest.out >&AC_FD_CC
                AC_MSG_WARN([$NM could not read object file])
            fi
            ])
        if test "$asm_result" = "1" ; then
            break
        fi
    done
    rm -f conftest.out
    unset asm_result sym
])

# PRRTE_CHECK_ASM_LSYM()
# ---------------------
AC_DEFUN([PRRTE_CHECK_ASM_LSYM],[
    AC_REQUIRE([AC_PROG_NM])

    AC_CACHE_CHECK([prefix for lsym labels],
                   [prrte_cv_asm_lsym],
                   [_PRRTE_CHECK_ASM_LSYM([prrte_cv_asm_lsym])])
    AC_DEFINE_UNQUOTED([PRRTE_ASM_LSYM], ["$prrte_cv_asm_lsym"],
                       [Assembly prefix for lsym labels])
    PRRTE_ASM_LSYM="$prrte_cv_asm_lsym"
    AC_SUBST(PRRTE_ASM_LSYM)
])dnl

dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_PROC
dnl
dnl Sets a cv-flag, if the compiler needs a proc/endp-definition to
dnl link with C.
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_PROC],[
    AC_CACHE_CHECK([if .proc/endp is needed],
                   [prrte_cv_asm_need_proc],
                   [prrte_cv_asm_need_proc="no"
                    PRRTE_TRY_ASSEMBLE([
     .proc mysym
mysym:
     .endp mysym],
                          [prrte_cv_asm_need_proc="yes"])
                    rm -f conftest.out])

    if test "$prrte_cv_asm_need_proc" = "yes" ; then
       prrte_cv_asm_proc=".proc"
       prrte_cv_asm_endproc=".endp"
    else
       prrte_cv_asm_proc="#"
       prrte_cv_asm_endproc="#"
    fi
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_GSYM
dnl
dnl Sets PRRTE_ASM_GSYM to the prefix value on a symbol to make it
dnl a global linkable from C.  Basically, an _ or not.
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_GSYM],[
    AC_CACHE_CHECK([prefix for global symbol labels],
                   [prrte_cv_asm_gsym],
                   [_PRRTE_CHECK_ASM_GSYM])

    if test "$prrte_cv_asm_gsym" = "none" ; then
       AC_MSG_ERROR([Could not determine global symbol label prefix])
    fi

    AC_DEFINE_UNQUOTED([PRRTE_ASM_GSYM], ["$prrte_cv_asm_gsym"],
                       [Assembly prefix for gsym labels])
    PRRTE_ASM_GSYM="$prrte_cv_asm_gsym"
    AC_SUBST(PRRTE_ASM_GSYM)

])

AC_DEFUN([_PRRTE_CHECK_ASM_GSYM],[
    prrte_cv_asm_gsym="none"

    for sym in "_" "" "." ; do
        asm_result=0
        echo "configure: trying $sym" >&AC_FD_CC
cat > conftest_c.c <<EOF
#ifdef __cplusplus
extern "C" {
#endif
void gsym_test_func(void);
#ifdef __cplusplus
}
#endif
int
main()
{
    gsym_test_func();
    return 0;
}
EOF
        PRRTE_TRY_ASSEMBLE([
$prrte_cv_asm_text
$prrte_cv_asm_proc ${sym}gsym_test_func
$prrte_cv_asm_global ${sym}gsym_test_func
${sym}gsym_test_func${prrte_cv_asm_label_suffix}
$prrte_cv_asm_endproc ${sym}gsym_test_func
            ],
            [prrte_compile="$CC $CFLAGS -I. conftest_c.c -c > conftest.cmpl 2>&1"
             if AC_TRY_EVAL(prrte_compile) ; then
                # save the warnings
                 cat conftest.cmpl >&AC_FD_CC
                 prrte_link="$CC $CFLAGS conftest_c.$OBJEXT conftest.$OBJEXT -o conftest  $LDFLAGS $LIBS > conftest.link 2>&1"
                 if AC_TRY_EVAL(prrte_link) ; then
                     # save the warnings
                     cat conftest.link >&AC_FD_CC
                     asm_result=1
                 else
                     cat conftest.link >&AC_FD_CC
                     echo "configure: failed C program was: " >&AC_FD_CC
                     cat conftest_c.c >&AC_FD_CC
                     echo "configure: failed ASM program was: " >&AC_FD_CC
                     cat conftest.s >&AC_FD_CC
                     asm_result=0
                 fi
             else
                # save output and failed program
                 cat conftest.cmpl >&AC_FD_CC
                 echo "configure: failed C program was: " >&AC_FD_CC
                 cat conftest.c >&AC_FD_CC
                 asm_result=0
             fi],
            [asm_result=0])
        if test "$asm_result" = "1" ; then
            prrte_cv_asm_gsym="$sym"
            break
        fi
    done
    rm -rf conftest.*
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_LABEL_SUFFIX
dnl
dnl Sets PRRTE_ASM_LABEL_SUFFIX to the value to suffix for labels
dnl
dnl I'm sure if I don't have a test for this, there will be some
dnl dumb platform that uses something else
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_LABEL_SUFFIX],[
    AC_MSG_CHECKING([suffix for labels])
    prrte_cv_asm_label_suffix=""
    case $host in
        *)
                prrte_cv_asm_label_suffix=":"
        ;;
    esac
    AC_MSG_RESULT([$prrte_cv_asm_label_suffix])
    AC_DEFINE_UNQUOTED([PRRTE_ASM_LABEL_SUFFIX], ["$prrte_cv_asm_label_suffix"],
                       [Assembly suffix for labels])
    PRRTE_ASM_LABEL_SUFFIX="$prrte_cv_asm_label_suffix"
    AC_SUBST(PRRTE_AS_LABEL_SUFFIX)
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_ALIGN_LOG
dnl
dnl Sets PRRTE_ASM_ALIGN_LOG to 1 if align is specified
dnl logarithmically, 0 otherwise
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_ALIGN_LOG],[
    AC_REQUIRE([AC_PROG_NM])
    AC_REQUIRE([AC_PROG_GREP])

    AC_CACHE_CHECK([if .align directive takes logarithmic value],
                   [prrte_cv_asm_align_log],
                   [ PRRTE_TRY_ASSEMBLE([        $prrte_cv_asm_text
        .align 4
        $prrte_cv_asm_global foo
        .byte 1
        .align 4
foo$prrte_cv_asm_label_suffix
        .byte 2],
        [prrte_asm_addr=[`$NM conftest.$OBJEXT | $GREP foo | sed -e 's/.*\([0-9a-fA-F][0-9a-fA-F]\).*foo.*/\1/'`]],
        [prrte_asm_addr=""])
    # test for both 16 and 10 (decimal and hex notations)
    echo "configure: .align test address offset is $prrte_asm_addr" >&AC_FD_CC
    if test "$prrte_asm_addr" = "16" || test "$prrte_asm_addr" = "10" ; then
       prrte_cv_asm_align_log="yes"
    else
        prrte_cv_asm_align_log="no"
    fi])

    if test "$prrte_cv_asm_align_log" = "yes" || test "$prrte_cv_asm_align_log" = "1" ; then
        prrte_asm_align_log_result=1
    else
        prrte_asm_align_log_result=0
    fi

    AC_DEFINE_UNQUOTED([PRRTE_ASM_ALIGN_LOG],
                       [$asm_align_log_result],
                       [Assembly align directive expects logarithmic value])

    unset omp_asm_addr asm_result
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_TYPE
dnl
dnl Sets PRRTE_ASM_TYPE to the prefix for the function type to
dnl set a symbol's type as function (needed on ELF for shared
dnl libraries).  If no .type directive is needed, sets PRRTE_ASM_TYPE
dnl to an empty string
dnl
dnl We look for @ \# %
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_TYPE],[
        AC_CACHE_CHECK([prefix for function in .type],
                       [prrte_cv_asm_type],
                       [_PRRTE_CHECK_ASM_TYPE])

    AC_DEFINE_UNQUOTED([PRRTE_ASM_TYPE], ["$prrte_cv_asm_type"],
                       [How to set function type in .type directive])
    PRRTE_ASM_TYPE="$prrte_cv_asm_type"
    AC_SUBST(PRRTE_ASM_TYPE)
])

AC_DEFUN([_PRRTE_CHECK_ASM_TYPE],[
    prrte_cv_asm_type=""

    case "${host}" in
    *-sun-solaris*)
        # GCC on solaris seems to accept just about anything, not
        # that what it defines actually works...  So just hardwire
        # to the right answer
        prrte_cv_asm_type="#"
    ;;
    *)
        for type  in @ \# % ; do
            asm_result=0
            echo "configure: trying $type" >&AC_FD_CC
            PRRTE_TRY_ASSEMBLE([     .type mysym, ${type}function
mysym:],
                 [prrte_cv_asm_type="${type}"
                    asm_result=1])
            if test "$asm_result" = "1" ; then
                break
            fi
        done
    ;;
    esac
    rm -f conftest.out

    unset asm_result type
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_ASM_SIZE
dnl
dnl Sets PRRTE_ASM_SIZE to 1 if we should set .size directives for
dnl each function, 0 otherwise.
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_ASM_SIZE],[
    AC_CACHE_CHECK([if .size is needed],
                   [prrte_cv_asm_need_size],
                   [prrte_cv_asm_need_size="no"
                    PRRTE_TRY_ASSEMBLE([     .size mysym, 1],
                          [prrte_cv_asm_need_size="yes"])
                    rm -f conftest.out])

    if test "$prrte_cv_asm_need_size" = "yes" ; then
       prrte_asm_size=1
    else
       prrte_asm_size=0
    fi

    AC_DEFINE_UNQUOTED([PRRTE_ASM_SIZE], ["$prrte_asm_size"],
                       [Do we need to give a .size directive])
    PRRTE_ASM_SIZE="$prrte_asm_size"
    AC_SUBST(PRRTE_ASM_TYPE)
    unset asm_result
])dnl


# PRRTE_CHECK_ASM_GNU_STACKEXEC(var)
# ----------------------------------
# sets shell variable var to the things necessary to
# disable execable stacks with GAS
AC_DEFUN([PRRTE_CHECK_ASM_GNU_STACKEXEC], [
    AC_REQUIRE([AC_PROG_GREP])

    AC_CHECK_PROG([OBJDUMP], [objdump], [objdump])
    AC_CACHE_CHECK([if .note.GNU-stack is needed],
        [prrte_cv_asm_gnu_stack_result],
        [AS_IF([test "$OBJDUMP" != ""],
            [ # first, see if a simple C program has it set
             cat >conftest.c <<EOF
int testfunc() {return 0; }
EOF
             PRRTE_LOG_COMMAND([$CC $CFLAGS -c conftest.c -o conftest.$OBJEXT],
                 [$OBJDUMP -x conftest.$OBJEXT | $GREP '\.note\.GNU-stack' > /dev/null && prrte_cv_asm_gnu_stack_result=yes],
                 [PRRTE_LOG_MSG([the failed program was:], 1)
                  PRRTE_LOG_FILE([conftest.c])
                  prrte_cv_asm_gnu_stack_result=no])
             if test "$prrte_cv_asm_gnu_stack_result" != "yes" ; then
                 prrte_cv_asm_gnu_stack_result="no"
             fi
             rm -rf conftest.*],
            [prrte_cv_asm_gnu_stack_result="no"])])
    if test "$prrte_cv_asm_gnu_stack_result" = "yes" ; then
        prrte_cv_asm_gnu_stack=1
    else
        prrte_cv_asm_gnu_stack=0
    fi
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_POWERPC_REG
dnl
dnl See if the notation for specifying registers is X (most everyone)
dnl or rX (OS X)
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_POWERPC_REG],[
    AC_MSG_CHECKING([if PowerPC registers have r prefix])
    PRRTE_TRY_ASSEMBLE([$prrte_cv_asm_text
        addi 1,1,0],
        [prrte_cv_asm_powerpc_r_reg=0],
        [PRRTE_TRY_ASSEMBLE([$prrte_cv_asm_text
        addi r1,r1,0],
            [prrte_cv_asm_powerpc_r_reg=1],
            [AC_MSG_ERROR([Can not determine how to use PPC registers])])])
    if test "$prrte_cv_asm_powerpc_r_reg" = "1" ; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
    fi

    AC_DEFINE_UNQUOTED([PRRTE_POWERPC_R_REGISTERS],
                       [$prrte_cv_asm_powerpc_r_reg],
                       [Whether r notation is used for ppc registers])
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_POWERPC_64BIT
dnl
dnl On some powerpc chips (the PPC970 or G5), the OS usually runs in
dnl 32 bit mode, even though the hardware can do 64bit things.  If
dnl the compiler will let us, emit code for 64bit test and set type
dnl operations (on a long long).
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_POWERPC_64BIT],[
    if test "$ac_cv_sizeof_long" != "4" ; then
        # this function should only be called in the 32 bit case
        AC_MSG_ERROR([CHECK_POWERPC_64BIT called on 64 bit platform.  Internal error.])
    fi
    AC_MSG_CHECKING([for 64-bit PowerPC assembly support])
        case $host in
            *-darwin*)
                ppc64_result=0
                if test "$prrte_cv_asm_powerpc_r_reg" = "1" ; then
                   ldarx_asm="        ldarx r1,r1,r1";
                else
                   ldarx_asm="        ldarx 1,1,1";
                fi
                PRRTE_TRY_ASSEMBLE([$prrte_cv_asm_text
        $ldarx_asm],
                    [ppc64_result=1],
                    [ppc64_result=0])
            ;;
            *)
                ppc64_result=0
            ;;
        esac

    if test "$ppc64_result" = "1" ; then
        AC_MSG_RESULT([yes])
        ifelse([$1],,:,[$1])
    else
        AC_MSG_RESULT([no])
        ifelse([$2],,:,[$2])
    fi

    unset ppc64_result ldarx_asm
])dnl


dnl #################################################################
dnl
dnl PRRTE_CHECK_SPARCV8PLUS
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_SPARCV8PLUS],[
    AC_MSG_CHECKING([if have Sparc v8+/v9 support])
    sparc_result=0
    PRRTE_TRY_ASSEMBLE([$prrte_cv_asm_text
        casa [%o0] 0x80, %o1, %o2],
                [sparc_result=1],
                [sparc_result=0])
    if test "$sparc_result" = "1" ; then
        AC_MSG_RESULT([yes])
        ifelse([$1],,:,[$1])
    else
        AC_MSG_RESULT([no])
        ifelse([$2],,:,[$2])
    fi

    unset sparc_result
])dnl

dnl #################################################################
dnl
dnl PRRTE_CHECK_CMPXCHG16B
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CMPXCHG16B_TEST_SOURCE],[[
#include <stdint.h>
#include <assert.h>

union prrte_counted_pointer_t {
    struct {
        uint64_t counter;
        uint64_t item;
    } data;
#if defined(HAVE___INT128) && HAVE___INT128
    __int128 value;
#elif defined(HAVE_INT128_T) && HAVE_INT128_T
    int128_t value;
#endif
};
typedef union prrte_counted_pointer_t prrte_counted_pointer_t;

int main(int argc, char* argv) {
    volatile prrte_counted_pointer_t a;
    prrte_counted_pointer_t b;

    a.data.counter = 0;
    a.data.item = 0x1234567890ABCDEF;

    b.data.counter = a.data.counter;
    b.data.item = a.data.item;

    /* bozo checks */
    assert(16 == sizeof(prrte_counted_pointer_t));
    assert(a.data.counter == b.data.counter);
    assert(a.data.item == b.data.item);
    /*
     * the following test fails on buggy compilers
     * so far, with icc -o conftest conftest.c
     *  - intel icc 14.0.0.080 (aka 2013sp1)
     *  - intel icc 14.0.1.106 (aka 2013sp1u1)
     * older and more recents compilers work fine
     * buggy compilers work also fine but only with -O0
     */
#if (defined(HAVE___INT128) && HAVE___INT128) || (defined(HAVE_INT128_T) && HAVE_INT128_T)
    return (a.value != b.value);
#else
    return 0;
#endif
}
]])

AC_DEFUN([PRRTE_CHECK_CMPXCHG16B],[
    PRRTE_VAR_SCOPE_PUSH([cmpxchg16b_result])

    PRRTE_ASM_CHECK_ATOMIC_FUNC([cmpxchg16b],
                               [AC_LANG_PROGRAM([[unsigned char tmp[16];]],
                                                [[__asm__ __volatile__ ("lock cmpxchg16b (%%rsi)" : : "S" (tmp) : "memory", "cc");]])],
                               [cmpxchg16b_result=1],
                               [cmpxchg16b_result=0])
    # If we have it, make sure it works.
    AS_IF([test $cmpxchg16b_result -eq 1],
          [AC_MSG_CHECKING([if cmpxchg16b_result works])
           AC_RUN_IFELSE([AC_LANG_SOURCE(PRRTE_CMPXCHG16B_TEST_SOURCE)],
                         [AC_MSG_RESULT([yes])],
                         [cmpxchg16b_result=0
                          AC_MSG_RESULT([no])],
                         [AC_MSG_RESULT([cannot test -- assume yes (cross compiling)])])
          ])

    AC_DEFINE_UNQUOTED([PRRTE_HAVE_CMPXCHG16B], [$cmpxchg16b_result],
        [Whether the processor supports the cmpxchg16b instruction])
    PRRTE_VAR_SCOPE_POP
])dnl

dnl #################################################################
dnl
dnl PRRTE_CHECK_INLINE_GCC
dnl
dnl Check if the compiler is capable of doing GCC-style inline
dnl assembly.  Some compilers emit a warning and ignore the inline
dnl assembly (xlc on OS X) and compile without error.  Therefore,
dnl the test attempts to run the emitted code to check that the
dnl assembly is actually run.  To run this test, one argument to
dnl the macro must be an assembly instruction in gcc format to move
dnl the value 0 into the register containing the variable ret.
dnl For PowerPC, this would be:
dnl
dnl   "li %0,0" : "=&r"(ret)
dnl
dnl For testing ia32 assembly, the assembly instruction xaddl is
dnl tested.  The xaddl instruction is used by some of the atomic
dnl implementations so it makes sense to test for it.  In addition,
dnl some compilers (i.e. earlier versions of Sun Studio 12) do not
dnl necessarily handle xaddl properly, so that needs to be detected
dnl during configure time.
dnl
dnl DEFINE PRRTE_GCC_INLINE_ASSEMBLY to 0 or 1 depending on GCC
dnl                support
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CHECK_INLINE_C_GCC],[
    assembly="$1"
    asm_result="unknown"

    AC_MSG_CHECKING([if $CC supports GCC inline assembly])

    if test ! "$assembly" = "" ; then
        AC_RUN_IFELSE([AC_LANG_PROGRAM([AC_INCLUDES_DEFAULT],[[
int ret = 1;
int negone = -1;
__asm__ __volatile__ ($assembly);
return ret;
                                                             ]])],
                      [asm_result="yes"], [asm_result="no"],
                      [asm_result="unknown"])
    else
        assembly="test skipped - assuming no"
    fi

    # if we're cross compiling, just try to compile and figure good enough
    if test "$asm_result" = "unknown" ; then
        AC_LINK_IFELSE([AC_LANG_PROGRAM([AC_INCLUDES_DEFAULT],[[
int ret = 1;
int negone = -1;
__asm__ __volatile__ ($assembly);
return ret;
                                                              ]])],
                       [asm_result="yes"], [asm_result="no"])
    fi

    AC_MSG_RESULT([$asm_result])

    if test "$asm_result" = "yes" ; then
        PRRTE_C_GCC_INLINE_ASSEMBLY=1
        prrte_cv_asm_inline_supported="yes"
    else
        PRRTE_C_GCC_INLINE_ASSEMBLY=0
    fi

    AC_DEFINE_UNQUOTED([PRRTE_C_GCC_INLINE_ASSEMBLY],
                       [$PRRTE_C_GCC_INLINE_ASSEMBLY],
                       [Whether C compiler supports GCC style inline assembly])

    unset PRRTE_C_GCC_INLINE_ASSEMBLY assembly asm_result
])dnl

dnl #################################################################
dnl
dnl PRRTE_CONFIG_ASM
dnl
dnl DEFINE PRRTE_ASSEMBLY_ARCH to something in sys/architecture.h
dnl DEFINE PRRTE_ASSEMBLY_FORMAT to string containing correct
dnl                             format for assembly (not user friendly)
dnl SUBST PRRTE_ASSEMBLY_FORMAT to string containing correct
dnl                             format for assembly (not user friendly)
dnl
dnl #################################################################
AC_DEFUN([PRRTE_CONFIG_ASM],[
    AC_REQUIRE([PRRTE_SETUP_CC])
    AC_REQUIRE([AM_PROG_AS])

    AC_ARG_ENABLE([c11-atomics],[AC_HELP_STRING([--enable-c11-atomics],
                  [Enable use of C11 atomics if available (default: enabled)])])

    AC_ARG_ENABLE([builtin-atomics],
      [AC_HELP_STRING([--enable-builtin-atomics],
         [Enable use of __sync builtin atomics (default: disabled)])])

    PRRTE_CHECK_C11_CSWAP_INT128

    if test "x$enable_c11_atomics" != "xno" && test "$prrte_cv_c11_supported" = "yes" ; then
        prrte_cv_asm_builtin="BUILTIN_C11"
        PRRTE_CHECK_C11_CSWAP_INT128
    elif test "x$enable_c11_atomics" = "xyes"; then
        AC_MSG_WARN([C11 atomics were requested but are not supported])
        AC_MSG_ERROR([Cannot continue])
    else
        prrte_cv_asm_builtin="BUILTIN_NO"
        AS_IF([test "$prrte_cv_asm_builtin" = "BUILTIN_NO" && test "$enable_builtin_atomics" = "yes"],
              [PRRTE_CHECK_GCC_ATOMIC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_GCC"], [])])
        AS_IF([test "$prrte_cv_asm_builtin" = "BUILTIN_NO" && test "$enable_builtin_atomics" = "yes"],
              [PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"], [])])
        AS_IF([test "$prrte_cv_asm_builtin" = "BUILTIN_NO" && test "$enable_builtin_atomics" = "yes"],
              [AC_MSG_ERROR([__sync builtin atomics requested but not found.])])
    fi

        PRRTE_CHECK_ASM_PROC
        PRRTE_CHECK_ASM_TEXT
        PRRTE_CHECK_ASM_GLOBAL
        PRRTE_CHECK_ASM_GNU_STACKEXEC
        PRRTE_CHECK_ASM_LABEL_SUFFIX
        PRRTE_CHECK_ASM_GSYM
        PRRTE_CHECK_ASM_LSYM
        PRRTE_CHECK_ASM_TYPE
        PRRTE_CHECK_ASM_SIZE
        PRRTE_CHECK_ASM_ALIGN_LOG

        # find our architecture for purposes of assembly stuff
        prrte_cv_asm_arch="UNSUPPORTED"
        PRRTE_GCC_INLINE_ASSIGN=""
        PRRTE_ASM_SUPPORT_64BIT=0
        case "${host}" in
        x86_64-*x32)
            prrte_cv_asm_arch="X86_64"
            PRRTE_ASM_SUPPORT_64BIT=1
            PRRTE_GCC_INLINE_ASSIGN='"xaddl %1,%0" : "=m"(ret), "+r"(negone) : "m"(ret)'
            ;;
        i?86-*|x86_64*|amd64*)
            if test "$ac_cv_sizeof_long" = "4" ; then
                prrte_cv_asm_arch="IA32"
            else
                prrte_cv_asm_arch="X86_64"
            fi
            PRRTE_ASM_SUPPORT_64BIT=1
            PRRTE_GCC_INLINE_ASSIGN='"xaddl %1,%0" : "=m"(ret), "+r"(negone) : "m"(ret)'
            PRRTE_CHECK_CMPXCHG16B
            ;;

        ia64-*)
            prrte_cv_asm_arch="IA64"
            PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"],
              [AC_MSG_ERROR([No atomic primitives available for $host])])
            ;;
        aarch64*)
            prrte_cv_asm_arch="ARM64"
            PRRTE_ASM_SUPPORT_64BIT=1
            PRRTE_ASM_ARM_VERSION=8
            AC_DEFINE_UNQUOTED([PRRTE_ASM_ARM_VERSION], [$PRRTE_ASM_ARM_VERSION],
                               [What ARM assembly version to use])
            PRRTE_GCC_INLINE_ASSIGN='"mov %0, #0" : "=&r"(ret)'
            ;;

        armv7*|arm-*-linux-gnueabihf)
            prrte_cv_asm_arch="ARM"
            PRRTE_ASM_SUPPORT_64BIT=1
            PRRTE_ASM_ARM_VERSION=7
            AC_DEFINE_UNQUOTED([PRRTE_ASM_ARM_VERSION], [$PRRTE_ASM_ARM_VERSION],
                               [What ARM assembly version to use])
            PRRTE_GCC_INLINE_ASSIGN='"mov %0, #0" : "=&r"(ret)'
            ;;

        armv6*)
            prrte_cv_asm_arch="ARM"
            PRRTE_ASM_SUPPORT_64BIT=0
            PRRTE_ASM_ARM_VERSION=6
            CCASFLAGS="$CCASFLAGS -march=armv7-a"
            AC_DEFINE_UNQUOTED([PRRTE_ASM_ARM_VERSION], [$PRRTE_ASM_ARM_VERSION],
                               [What ARM assembly version to use])
            PRRTE_GCC_INLINE_ASSIGN='"mov %0, #0" : "=&r"(ret)'
            ;;

        armv5*linux*|armv4*linux*|arm-*-linux-gnueabi)
            # uses Linux kernel helpers for some atomic operations
            prrte_cv_asm_arch="ARM"
            PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"],
              [AC_MSG_ERROR([No atomic primitives available for $host])])
            ;;

        mips-*|mips64*)
            # Should really find some way to make sure that we are on
            # a MIPS III machine (r4000 and later)
            prrte_cv_asm_arch="MIPS"
            PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"],
              [AC_MSG_ERROR([No atomic primitives available for $host])])
            ;;

        powerpc-*|powerpc64-*|powerpcle-*|powerpc64le-*|rs6000-*|ppc-*)
            PRRTE_CHECK_POWERPC_REG
            if test "$ac_cv_sizeof_long" = "4" ; then
                prrte_cv_asm_arch="POWERPC32"

                # Note that on some platforms (Apple G5), even if we are
                # compiling in 32 bit mode (and therefore should assume
                # sizeof(long) == 4), we can use the 64 bit test and set
                # operations.
                PRRTE_CHECK_POWERPC_64BIT(PRRTE_ASM_SUPPORT_64BIT=1)
            elif test "$ac_cv_sizeof_long" = "8" ; then
                PRRTE_ASM_SUPPORT_64BIT=1
                prrte_cv_asm_arch="POWERPC64"
            else
                AC_MSG_ERROR([Could not determine PowerPC word size: $ac_cv_sizeof_long])
            fi
            PRRTE_GCC_INLINE_ASSIGN='"1: li %0,0" : "=&r"(ret)'
            ;;
        # There is no current difference between s390 and s390x
        # But use two different defines in case some come later
        # as s390 is 31bits while s390x is 64bits
        s390-*)
            prrte_cv_asm_arch="S390"
            PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"],
              [AC_MSG_ERROR([No atomic primitives available for $host])])
            ;;
        s390x-*)
            prrte_cv_asm_arch="S390X"
            PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"],
              [AC_MSG_ERROR([No atomic primitives available for $host])])
            ;;
        sparc*-*)
            # SPARC v9 (and above) are the only ones with 64bit support
            # if compiling 32 bit, see if we are v9 (aka v8plus) or
            # earlier (casa is v8+/v9).
            if test "$ac_cv_sizeof_long" = "4" ; then
                have_v8plus=0
                PRRTE_CHECK_SPARCV8PLUS([have_v8plus=1])
                if test "$have_v8plus" = "0" ; then
                    PRRTE_ASM_SUPPORT_64BIT=0
                    prrte_cv_asm_arch="SPARC"
AC_MSG_WARN([Sparc v8 target is not supported in this release of Open MPI.])
AC_MSG_WARN([You must specify the target architecture v8plus to compile])
AC_MSG_WARN([Open MPI in 32 bit mode on Sparc processors (see the README).])
AC_MSG_ERROR([Can not continue.])
                else
                    PRRTE_ASM_SUPPORT_64BIT=1
                    prrte_cv_asm_arch="SPARCV9_32"
                fi

            elif test "$ac_cv_sizeof_long" = "8" ; then
                PRRTE_ASM_SUPPORT_64BIT=1
                prrte_cv_asm_arch="SPARCV9_64"
            else
                AC_MSG_ERROR([Could not determine Sparc word size: $ac_cv_sizeof_long])
            fi
            PRRTE_GCC_INLINE_ASSIGN='"mov 0,%0" : "=&r"(ret)'
            ;;

        *)
            PRRTE_CHECK_SYNC_BUILTINS([prrte_cv_asm_builtin="BUILTIN_SYNC"],
              [AC_MSG_ERROR([No atomic primitives available for $host])])
            ;;
        esac

        if test "x$PRRTE_ASM_SUPPORT_64BIT" = "x1" && test "$prrte_cv_asm_builtin" = "BUILTIN_SYNC" &&
                test "$prrte_asm_sync_have_64bit" = "0" ; then
            # __sync builtins exist but do not implement 64-bit support. Fall back on inline asm.
            prrte_cv_asm_builtin="BUILTIN_NO"
        fi

      if test "$prrte_cv_asm_builtin" = "BUILTIN_SYNC" || test "$prrte_cv_asm_builtin" = "BUILTIN_GCC" ; then
        AC_DEFINE([PRRTE_C_GCC_INLINE_ASSEMBLY], [1],
          [Whether C compiler supports GCC style inline assembly])
      else
        AC_DEFINE_UNQUOTED([PRRTE_ASM_SUPPORT_64BIT],
            [$PRRTE_ASM_SUPPORT_64BIT],
            [Whether we can do 64bit assembly operations or not.  Should not be used outside of the assembly header files])
        AC_SUBST([PRRTE_ASM_SUPPORT_64BIT])

        #
        # figure out if we need any special function start / stop code
        #
        case $host_os in
        aix*)
            prrte_asm_arch_config="aix"
            ;;
        *)
            prrte_asm_arch_config="default"
            ;;
         esac

         prrte_cv_asm_inline_supported="no"
         # now that we know our architecture, try to inline assemble
         PRRTE_CHECK_INLINE_C_GCC([$PRRTE_GCC_INLINE_ASSIGN])

         # format:
         #   config_file-text-global-label_suffix-gsym-lsym-type-size-align_log-ppc_r_reg-64_bit-gnu_stack
         asm_format="${prrte_asm_arch_config}"
         asm_format="${asm_format}-${prrte_cv_asm_text}-${prrte_cv_asm_global}"
         asm_format="${asm_format}-${prrte_cv_asm_label_suffix}-${prrte_cv_asm_gsym}"
         asm_format="${asm_format}-${prrte_cv_asm_lsym}"
         asm_format="${asm_format}-${prrte_cv_asm_type}-${prrte_asm_size}"
         asm_format="${asm_format}-${prrte_asm_align_log_result}"
         if test "$prrte_cv_asm_arch" = "POWERPC32" || test "$prrte_cv_asm_arch" = "POWERPC64" ; then
             asm_format="${asm_format}-${prrte_cv_asm_powerpc_r_reg}"
         else
             asm_format="${asm_format}-1"
         fi
         asm_format="${asm_format}-${PRRTE_ASM_SUPPORT_64BIT}"
         prrte_cv_asm_format="${asm_format}-${prrte_cv_asm_gnu_stack}"
         # For the Makefile, need to escape the $ as $$.  Don't display
         # this version, but make sure the Makefile gives the right thing
         # when regenerating the files because the base has been touched.
         PRRTE_ASSEMBLY_FORMAT=`echo "$prrte_cv_asm_format" | sed -e 's/\\\$/\\\$\\\$/'`

        AC_MSG_CHECKING([for assembly format])
        AC_MSG_RESULT([$prrte_cv_asm_format])
        AC_DEFINE_UNQUOTED([PRRTE_ASSEMBLY_FORMAT], ["$PRRTE_ASSEMBLY_FORMAT"],
                           [Format of assembly file])
        AC_SUBST([PRRTE_ASSEMBLY_FORMAT])
      fi # if prrte_cv_asm_builtin = BUILTIN_SYNC

    result="PRRTE_$prrte_cv_asm_arch"
    PRRTE_ASSEMBLY_ARCH="$prrte_cv_asm_arch"
    AC_MSG_CHECKING([for assembly architecture])
    AC_MSG_RESULT([$prrte_cv_asm_arch])
    AC_DEFINE_UNQUOTED([PRRTE_ASSEMBLY_ARCH], [$result],
        [Architecture type of assembly to use for atomic operations and CMA])
    AC_SUBST([PRRTE_ASSEMBLY_ARCH])

    # Check for RDTSCP support
    result=0
    AS_IF([test "$prrte_cv_asm_arch" = "PRRTE_X86_64" || test "$prrte_cv_asm_arch" = "PRRTE_IA32"],
          [AC_MSG_CHECKING([for RDTSCP assembly support])
           AC_LANG_PUSH([C])
           AC_TRY_RUN([[
int main(int argc, char* argv[])
{
  unsigned int rax, rdx;
  __asm__ __volatile__ ("rdtscp\n": "=a" (rax), "=d" (rdx):: "%rax", "%rdx");
  return 0;
}
           ]],
           [result=1
            AC_MSG_RESULT([yes])],
           [AC_MSG_RESULT([no])],
           [#cross compile not supported
            AC_MSG_RESULT(["no (cross compiling)"])])
           AC_LANG_POP([C])])
    AC_DEFINE_UNQUOTED([PRRTE_ASSEMBLY_SUPPORTS_RDTSCP], [$result],
                       [Whether we have support for RDTSCP instruction])

    result="PRRTE_$prrte_cv_asm_builtin"
    PRRTE_ASSEMBLY_BUILTIN="$prrte_cv_asm_builtin"
    AC_MSG_CHECKING([for builtin atomics])
    AC_MSG_RESULT([$prrte_cv_asm_builtin])
    AC_DEFINE_UNQUOTED([PRRTE_ASSEMBLY_BUILTIN], [$result],
        [Whether to use builtin atomics])
    AC_SUBST([PRRTE_ASSEMBLY_BUILTIN])

    PRRTE_ASM_FIND_FILE

    unset result asm_format
])dnl


dnl #################################################################
dnl
dnl PRRTE_ASM_FIND_FILE
dnl
dnl
dnl do all the evil mojo to provide a working assembly file
dnl
dnl #################################################################
AC_DEFUN([PRRTE_ASM_FIND_FILE], [
    AC_REQUIRE([AC_PROG_GREP])
    AC_REQUIRE([AC_PROG_FGREP])

if test "$prrte_cv_asm_arch" != "WINDOWS" && test "$prrte_cv_asm_builtin" != "BUILTIN_SYNC" && test "$prrte_cv_asm_builtin" != "BUILTIN_GCC" && test "$prrte_cv_asm_builtin" != "BUILTIN_OSX"  && test "$prrte_cv_asm_inline_arch" = "no" ; then
    AC_MSG_ERROR([no atomic support available. exiting])
else
    # On windows with VC++, atomics are done with compiler primitives
    prrte_cv_asm_file=""
fi
])dnl
