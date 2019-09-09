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
#ifndef SPFFT_EXCEPTIONS_H
#define SPFFT_EXCEPTIONS_H

#include <stdexcept>
#include "spfft/config.h"
#include "spfft/errors.h"

namespace spfft {

/**
 * A generic error. Base type for all other exceptions.
 */
class GenericError : public std::exception {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Generic error"; }

  virtual auto error_code() const noexcept -> SpfftError { return SpfftError::SPFFT_UNKNOWN_ERROR; }
};

/**
 * Overflow of integer values.
 */
class OverflowError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Overflow error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_OVERFLOW_ERROR;
  }
};

/**
 * Failed allocation on host.
 */
class HostAllocationError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Host allocation error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_ALLOCATION_ERROR;
  }
};

/**
 * Invalid parameter.
 */
class InvalidParameterError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Invalid parameter error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_INVALID_PARAMETER_ERROR;
  }
};

/**
 * Duplicate indices given to transform. May indicate non-local z-coloumn between MPI ranks.
 */
class DuplicateIndicesError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Duplicate indices error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_DUPLICATE_INDICES_ERROR;
  }
};

/**
 * Invalid indices given to transform.
 */
class InvalidIndicesError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Invalid indices error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_INVALID_INDICES_ERROR;
  }
};

/**
 * Library not compiled with MPI support.
 */
class MPISupportError : public GenericError {
public:
  auto what() const noexcept -> const char* override {
    return "SpFFT: Not compiled with MPI support error";
  }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_MPI_SUPPORT_ERROR;
  }
};

/**
 * MPI error. Only thrown if error code of MPI API calls is non-zero.
 */
class MPIError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: MPI error"; }

  auto error_code() const noexcept -> SpfftError override { return SpfftError::SPFFT_MPI_ERROR; }
};

/**
 * Parameters differ between MPI ranks.
 */
class MPIParameterMismatchError : public GenericError {
public:
  auto what() const noexcept -> const char* override {
    return "SpFFT: Mismatched parameters between MPI ranks";
  }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_MPI_PARAMETER_MISMATCH_ERROR;
  }
};

/**
 * Failed execution on host.
 */
class HostExecutionError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Host execution error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_HOST_EXECUTION_ERROR;
  }
};

/**
 * FFTW library error.
 */
class FFTWError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: FFTW error"; }

  auto error_code() const noexcept -> SpfftError override { return SpfftError::SPFFT_FFTW_ERROR; }
};

/**
 * Unknown internal error.
 */
class InternalError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Internal error"; }

  auto error_code() const noexcept -> SpfftError override { return SpfftError::SPFFT_FFTW_ERROR; }
};

// ==================================
// GPU Errors
// ==================================
/**
 * Generic GPU error. Base type for all GPU related exceptions.
 */
class GPUError : public GenericError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: GPU error"; }

  auto error_code() const noexcept -> SpfftError override { return SpfftError::SPFFT_GPU_ERROR; }
};

/**
 * Library not compiled with GPU support.
 */
class GPUSupportError : public GPUError {
public:
  auto what() const noexcept -> const char* override {
    return "SpFFT: Not compiled with GPU support";
  }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_SUPPORT_ERROR;
  }
};

/**
 * Detected error on GPU from previous GPU API / kernel calls.
 */
class GPUPrecedingError : public GPUError {
public:
  auto what() const noexcept -> const char* override {
    return "SpFFT: Detected error from preceding gpu calls.";
  }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_PRECEDING_ERROR;
  }
};

/**
 * Failed allocation on GPU.
 */
class GPUAllocationError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: GPU allocation error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_ALLOCATION_ERROR;
  }
};

/**
 * Failed to launch kernel on GPU.
 */
class GPULaunchError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: GPU launch error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_LAUNCH_ERROR;
  }
};

/**
 * No GPU device detected.
 */
class GPUNoDeviceError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: no GPU available"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_NO_DEVICE_ERROR;
  }
};

/**
 * Invalid value passed to GPU API.
 */
class GPUInvalidValueError : public GPUError {
public:
  auto what() const noexcept -> const char* override {
    return "SpFFT: GPU call with invalid value";
  }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_INVALID_VALUE_ERROR;
  }
};

/**
 * Invalid device pointer used.
 */
class GPUInvalidDevicePointerError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: Invalid GPU pointer"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_INVALID_DEVICE_PTR_ERROR;
  }
};

/**
 * Failed to copy from / to GPU.
 */
class GPUCopyError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: GPU Memory copy error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_COPY_ERROR;
  }
};

/**
 * Failure in GPU FFT library call.
 */
class GPUFFTError : public GPUError {
public:
  auto what() const noexcept -> const char* override { return "SpFFT: GPU FFT error"; }

  auto error_code() const noexcept -> SpfftError override {
    return SpfftError::SPFFT_GPU_FFT_ERROR;
  }
};

}  // namespace spfft

#endif
