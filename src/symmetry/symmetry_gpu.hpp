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
#ifndef SPFFT_SYMMETRY_GPU_HPP
#define SPFFT_SYMMETRY_GPU_HPP

#include <complex>
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array_view.hpp"
#include "spfft/config.h"
#include "symmetry/gpu_kernels/symmetry_kernels.hpp"
#include "symmetry/symmetry.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {

// This class will apply the 1D hermitian symmetry along the inner dimension on the plane with mid
// index 0
template <typename T>
class PlaneSymmetryGPU : public Symmetry {
public:
  PlaneSymmetryGPU(GPUStreamHandle stream,
                   const GPUArrayView3D<typename gpu::fft::ComplexType<T>::type>& data)
      : stream_(std::move(stream)), data_(data) {}

  auto apply() -> void override {
    if (data_.dim_mid() > 2 && data_.size() > 0) {
      symmetrize_plane_gpu(stream_.get(), data_);
    }
  }

private:
  GPUStreamHandle stream_;
  GPUArrayView3D<typename gpu::fft::ComplexType<T>::type> data_;
};

// This class will apply the hermitian symmetry in 1d
template <typename T>
class StickSymmetryGPU : public Symmetry {
public:
  StickSymmetryGPU(GPUStreamHandle stream,
                   const GPUArrayView1D<typename gpu::fft::ComplexType<T>::type>& stick)
      : stream_(std::move(stream)), stick_(stick) {}

  auto apply() -> void override {
    if (stick_.size() > 2) {
      symmetrize_stick_gpu(stream_.get(), stick_);
    }
  }

private:
  GPUStreamHandle stream_;
  GPUArrayView1D<typename gpu::fft::ComplexType<T>::type> stick_;
};
}  // namespace spfft

#endif
