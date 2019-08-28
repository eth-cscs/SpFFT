/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SPFFT_ERRORS_H
#define SPFFT_ERRORS_H

#include "spfft/config.h"

enum SpfftError {
  /**
   * Success. No error.
   */
  SPFFT_SUCCESS,
  /**
   * Unknown error.
   */
  SPFFT_UNKNOWN_ERROR,
  /**
   * Invalid Grid or Transform handle.
   */
  SPFFT_INVALID_HANDLE_ERROR,
  /**
   * Integer overflow.
   */
  SPFFT_OVERFLOW_ERROR,
  /**
   * Failed to allocate memory on host.
   */
  SPFFT_ALLOCATION_ERROR,
  /**
   * Invalid parameter.
   */
  SPFFT_INVALID_PARAMETER_ERROR,
  /**
   * Duplicate indices given to transform. May indicate non-local z-coloumn between MPI ranks.
   */
  SPFFT_DUPLICATE_INDICES_ERROR,
  /**
   * Invalid indices given to transform.
   */
  SPFFT_INVALID_INDICES_ERROR,
  /**
   * Library not compiled with MPI support.
   */
  SPFFT_MPI_SUPPORT_ERROR,
  /**
   * MPI error. Only returned if error code of MPI API calls is non-zero.
   */
  SPFFT_MPI_ERROR,
  /**
   * Parameters differ between MPI ranks.
   */
  SPFFT_MPI_PARAMETER_MISMATCH_ERROR,
  /**
   * Failed execution on host.
   */
  SPFFT_HOST_EXECUTION_ERROR,
  /**
   * FFTW library error.
   */
  SPFFT_FFTW_ERROR,
  /**
   * Generic GPU error.
   */
  SPFFT_GPU_ERROR,
  /**
   * Detected error on GPU from previous GPU API / kernel calls.
   */
  SPFFT_GPU_PRECEDING_ERROR,
  /**
   * Library not compiled with GPU support.
   */
  SPFFT_GPU_SUPPORT_ERROR,
  /**
   * Failed allocation on GPU.
   */
  SPFFT_GPU_ALLOCATION_ERROR,
  /**
   * Failed to launch kernel on GPU.
   */
  SPFFT_GPU_LAUNCH_ERROR,
  /**
   * No GPU device detected.
   */
  SPFFT_GPU_NO_DEVICE_ERROR,
  /**
   * Invalid value passed to GPU API.
   */
  SPFFT_GPU_INVALID_VALUE_ERROR,
  /**
   * Invalid device pointer used.
   */
  SPFFT_GPU_INVALID_DEVICE_PTR_ERROR,
  /**
   * Failed to copy from / to GPU.
   */
  SPFFT_GPU_COPY_ERROR,
  /**
   * Failure in GPU FFT library call.
   */
  SPFFT_GPU_FFT_ERROR
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum SpfftError SpfftError;
/*! \endcond */
#endif // cpp

#endif
