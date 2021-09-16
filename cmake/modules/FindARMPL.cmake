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
# FindARMPL
# -----------
#
# This module searches for the sequential 32-bit integer ARM library.
#
#
# The following variables are set
#
# ::
#
#   ARMPL_FOUND                - True if double precision fftw library is found
#   ARMPL_LIBRARIES            - The required libraries
#   ARMPL_INCLUDE_DIRS         - The required include directory
#
# The following import target is created
#
# ::
#
#   ARM::pl

# set paths to look for ARM
set(_ARMPL_PATHS ${ARMPL_ROOT} $ENV{MKLROOT})

set(_ARMPL_DEFAULT_PATH_SWITCH)

if(_ARMPL_PATHS)
    # do not look at any default paths if a custom path was set
    set(_ARMPL_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    set(_ARMPL_PATHS /opt/arm)
endif()


# find all ARM libraries / include directories
find_library(
    ARMPL_LIBRARIES
    NAMES "armpl_lp64"
    HINTS ${_ARMPL_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_ARMPL_DEFAULT_PATH_SWITCH}
)
find_path(ARMPL_INCLUDE_DIRS
    NAMES "fftw3.h"
    HINTS ${_ARMPL_PATHS}
    PATH_SUFFIXES "include" "include/fftw" "fftw"
    ${_ARMPL_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARMPL REQUIRED_VARS ARMPL_LIBRARIES ARMPL_INCLUDE_DIRS)

# add target to link against
if(ARMPL_FOUND)
    # create interface target
    if(NOT TARGET ARM::pl)
        add_library(ARM::pl INTERFACE IMPORTED)
    endif()
    set_property(TARGET ARM::pl PROPERTY INTERFACE_LINK_LIBRARIES ${ARMPL_LIBRARIES})
    set_property(TARGET ARM::pl PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARMPL_INCLUDE_DIRS})
endif()

# prevent clutter in gui
MARK_AS_ADVANCED(REQUIRED_VARS ARMPL_LIBRARIES ARMPL_INCLUDE_DIRS)

