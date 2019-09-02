#  Copyright (c) 2019 ETH Zurich, Simon Frasch
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


#.rst:
# FindMKLSequential
# -----------
#
# This module searches for the sequential 32-bit integer MKL library.
# Only looks for static libraries by default.
#
#
# The following variables are set
#
# ::
#
#   MKLSequential_FOUND                - True if double precision fftw library is found
#   MKLSequential_LIBRARIES            - The required libraries
#   MKLSequential_INCLUDE_DIRS         - The required include directory
#   MKLSequential_FFTW_INCLUDE_DIRS    - The required fftw interface include directory
#
# The following import target is created
#
# ::
#
#   MKL::Sequential

# set paths to look for MKL
set(_MKLSequential_PATHS ${MKLSequential_ROOT} $ENV{MKLROOT})
set(_MKLSequential_INCLUDE_PATHS)

set(_MKLSequential_DEFAULT_PATH_SWITCH)

if(_MKLSequential_PATHS)
    # do not look at any default paths if a custom path was set
    set(_MKLSequential_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    # try to detect location with pkgconfig
    if(NOT MKLSequential_ROOT)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
           # look for dynmic module, such that a -L flag can be parsed
          pkg_check_modules(PKG_MKL QUIET "mkl-dynamic-lp64-seq")
            set(_MKLSequential_PATHS ${PKG_MKL_LIBRARY_DIRS})
            set(_MKLSequential_INCLUDE_PATHS ${PKG_MKL_INCLUDE_DIRS})
        endif()
    endif()
endif()


# find all MKL libraries / include directories
find_library(
    _MKLSequential_INT_LIB
    NAMES "mkl_intel_lp64"
    HINTS ${_MKLSequential_PATHS}
    PATH_SUFFIXES "intel64_lin" "intel64" "lib/intel64_lin" "lib/intel64"
    ${_MKLSequential_DEFAULT_PATH_SWITCH}
)
find_library(
    _MKLSequential_SEQ_LIB
    NAMES "mkl_sequential"
    HINTS ${_MKLSequential_PATHS}
    PATH_SUFFIXES "intel64_lin" "intel64" "lib/intel64_lin" "lib/intel64"
    ${_MKLSequential_DEFAULT_PATH_SWITCH}
)
find_library(
    _MKLSequential_CORE_LIB
    NAMES "mkl_core"
    HINTS ${_MKLSequential_PATHS}
    PATH_SUFFIXES "intel64_lin" "intel64" "lib/intel64_lin" "lib/intel64"
    ${_MKLSequential_DEFAULT_PATH_SWITCH}
)
find_path(MKLSequential_INCLUDE_DIRS
    NAMES "mkl.h"
    HINTS ${_MKLSequential_PATHS} ${_MKLSequential_INCLUDE_PATHS}
    PATH_SUFFIXES "include"
    ${_MKLSequential_DEFAULT_PATH_SWITCH}
)
find_path(MKLSequential_FFTW_INCLUDE_DIRS
    NAMES "fftw3.h"
    HINTS ${_MKLSequential_PATHS} ${_MKLSequential_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "include/fftw" "fftw"
    ${_MKLSequential_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLSequential REQUIRED_VARS _MKLSequential_INT_LIB 
    _MKLSequential_SEQ_LIB _MKLSequential_CORE_LIB MKLSequential_INCLUDE_DIRS MKLSequential_FFTW_INCLUDE_DIRS)

# add target to link against
if(MKLSequential_FOUND)
    # libries have inter-dependencies, therefore use link group on Linux
    if(UNIX AND NOT APPLE)
        set(MKLSequential_LIBRARIES "-Wl,--start-group" ${_MKLSequential_INT_LIB} ${_MKLSequential_SEQ_LIB} ${_MKLSequential_CORE_LIB} "-Wl,--end-group")
    else()
        set(MKLSequential_LIBRARIES ${_MKLSequential_INT_LIB} ${_MKLSequential_SEQ_LIB} ${_MKLSequential_CORE_LIB})
    endif()
    # external libries required on unix
    if(UNIX)
        list(APPEND MKLSequential_LIBRARIES -lpthread -lm -ldl)
    endif()

    # create interface target
    if(NOT TARGET MKL::Sequential)
        add_library(MKL::Sequential INTERFACE IMPORTED)
    endif()
    set_property(TARGET MKL::Sequential PROPERTY INTERFACE_LINK_LIBRARIES ${MKLSequential_LIBRARIES})
    set_property(TARGET MKL::Sequential PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MKLSequential_INCLUDE_DIRS} ${MKLSequential_FFTW_INCLUDE_DIRS})
endif()

# prevent clutter in gui
MARK_AS_ADVANCED(MKLSequential_FOUND MKLSequential_LIBRARIES MKLSequential_INCLUDE_DIRS
    _MKLSequential_INT_LIB _MKLSequential_SEQ_LIB _MKLSequential_CORE_LIB MKLSequential_FFTW_INCLUDE_DIRS
    _MKLSequential_DEFAULT_PATH_SWITCH _MKLSequential_PATHS)

MARK_AS_ADVANCED(pkgcfg_lib_PKG_MKL_dl pkgcfg_lib_PKG_MKL_m pkgcfg_lib_PKG_MKL_mkl_core
    pkgcfg_lib_PKG_MKL_mkl_sequential pkgcfg_lib_PKG_MKL_mkl_intel_lp64 pkgcfg_lib_PKG_MKL_pthread)
