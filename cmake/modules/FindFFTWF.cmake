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
# FindFFTWF
# -----------
#
# This module looks for the fftw3f library.
#
# The following variables are set
#
# ::
#
#   FFTWF_FOUND           - True if single precision fftw library is found
#   FFTWF_LIBRARIES       - The required libraries
#   FFTWF_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   FFTWF::FFTWF



# set paths to look for library
set(_FFTWF_PATHS ${FFTW_ROOT} $ENV{FFTW_ROOT} ${FFTWF_ROOT} $ENV{FFTWF_ROOT})
set(_FFTWF_INCLUDE_PATHS)

set(_FFTWF_DEFAULT_PATH_SWITCH)

if(_FFTWF_PATHS)
    # disable default paths if ROOT is set
    set(_FFTWF_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    # try to detect location with pkgconfig
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(PKG_FFTWF QUIET "fftw3")
    endif()
    set(_FFTWF_PATHS ${PKG_FFTWF_LIBRARY_DIRS})
    set(_FFTWF_INCLUDE_PATHS ${PKG_FFTWF_INCLUDE_DIRS})
endif()


find_library(
    FFTWF_LIBRARIES
    NAMES "fftw3f"
    HINTS ${_FFTWF_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_FFTWF_DEFAULT_PATH_SWITCH}
)
find_path(FFTWF_INCLUDE_DIRS
    NAMES "fftw3.h"
    HINTS ${_FFTWF_PATHS} ${_FFTWF_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "include/fftw"
    ${_FFTWF_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTWF REQUIRED_VARS FFTWF_INCLUDE_DIRS FFTWF_LIBRARIES )


# add target to link against
if(FFTWF_FOUND)
    if(NOT TARGET FFTWF::FFTWF)
        add_library(FFTWF::FFTWF INTERFACE IMPORTED)
    endif()
    set_property(TARGET FFTWF::FFTWF PROPERTY INTERFACE_LINK_LIBRARIES ${FFTWF_LIBRARIES})
    set_property(TARGET FFTWF::FFTWF PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTWF_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(FFTWF_FOUND FFTWF_LIBRARIES FFTWF_INCLUDE_DIRS pkgcfg_lib_PKG_FFTWF_fftw3)
