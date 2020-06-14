#  Copyright (c) 2019 ETH Zurich
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
# FindROCFFT
# -----------
#
# This module searches for the fftw3 library.
#
# The following variables are set
#
# ::
#
#   ROCFFT_FOUND           - True if rocfft is found
#   ROCFFT_LIBRARIES       - The required libraries
#   ROCFFT_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   ROCFFT::rocfft

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_ROCFFT_PATHS ${ROCFFT_ROOT} $ENV{ROCFFT_ROOT})
endif()

if(NOT _ROCFFT_PATHS)
    set(_ROCFFT_PATHS /opt/rocm $ENV{ROCM_HOME})
endif()

find_library(
    ROCFFT_LIBRARIES
    NAMES "rocfft"
    HINTS ${_ROCFFT_PATHS}
    PATH_SUFFIXES "rocfft/lib" "rocfft" 
)
find_path(
    ROCFFT_INCLUDE_DIRS
    NAMES "rocfft.h"
    HINTS ${_ROCFFT_PATHS}
    PATH_SUFFIXES "rocfft/include" "include"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCFFT REQUIRED_VARS ROCFFT_INCLUDE_DIRS ROCFFT_LIBRARIES )

# add target to link against
if(ROCFFT_FOUND)
    if(NOT TARGET ROCFFT::rocfft)
        add_library(ROCFFT::rocfft INTERFACE IMPORTED)
    endif()
    set_property(TARGET ROCFFT::rocfft PROPERTY INTERFACE_LINK_LIBRARIES ${ROCFFT_LIBRARIES})
    set_property(TARGET ROCFFT::rocfft PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ROCFFT_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(ROCFFT_FOUND ROCFFT_LIBRARIES ROCFFT_INCLUDE_DIRS)
