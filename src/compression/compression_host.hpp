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
#ifndef SPFFT_COMPRESSION_HOST_HPP
#define SPFFT_COMPRESSION_HOST_HPP

#include <complex>
#include <cstring>
#include <memory>
#include <vector>
#include "compression/indices.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {
// Handles packing and unpacking of sparse frequency values for single or double precision on Host
class CompressionHost {
public:
  explicit CompressionHost(const std::shared_ptr<Parameters>& param) : param_(param) {}

  // Pack values into output buffer
  template <typename T>
  auto compress(const HostArrayView2D<std::complex<T>> input2d, T* output, bool useScaling,
                const T scalingFactor = 1.0) const -> void {
    const auto& indices = param_->local_value_indices();
    auto input =
        HostArrayConstView1D<std::complex<T>>(input2d.data(), input2d.size(), input2d.pinned());

    if (useScaling) {
      SPFFT_OMP_PRAGMA("omp for schedule(static)")
      for (SizeType i = 0; i < indices.size(); ++i) {
        const auto value = scalingFactor * input(indices[i]);
        output[2 * i] = value.real();
        output[2 * i + 1] = value.imag();
      }
    } else {
      SPFFT_OMP_PRAGMA("omp for schedule(static)")
      for (SizeType i = 0; i < indices.size(); ++i) {
        const auto value = input(indices[i]);
        output[2 * i] = value.real();
        output[2 * i + 1] = value.imag();
      }
    }
  }

  // Unpack values into z-stick collection
  template <typename T>
  auto decompress(const T* input, HostArrayView2D<std::complex<T>> output2d) const -> void {
    const auto& indices = param_->local_value_indices();
    auto output =
        HostArrayView1D<std::complex<T>>(output2d.data(), output2d.size(), output2d.pinned());

    // ensure values are padded with zeros
    SPFFT_OMP_PRAGMA("omp for schedule(static)")  // implicit barrier
    for (SizeType stick = 0; stick < output2d.dim_outer(); ++stick) {
      std::memset(static_cast<void*>(&output2d(stick, 0)), 0,
                  sizeof(typename decltype(output2d)::ValueType) * output2d.dim_inner());
    }

    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType i = 0; i < indices.size(); ++i) {
      output(indices[i]) = std::complex<T>(input[2 * i], input[2 * i + 1]);
    }
  }

private:
  std::shared_ptr<Parameters> param_;
};
}  // namespace spfft

#endif
