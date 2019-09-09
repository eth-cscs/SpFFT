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
#ifndef SPFFT_SYMMETRY_HOST_HPP
#define SPFFT_SYMMETRY_HOST_HPP

#include <complex>
#include "memory/host_array_view.hpp"
#include "spfft/config.h"
#include "symmetry/symmetry.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {

// This class will apply the 1D hermitian symmetry along the inner dimension on the plane with mid
// index 0
template <typename T>
class PlaneSymmetryHost : public Symmetry {
public:
  explicit PlaneSymmetryHost(const HostArrayView3D<std::complex<T>>& data) : data_(data) {}

  auto apply() -> void override {
    constexpr std::complex<T> zeroElement;
    // Data may be conjugated twice, but this way symmetry is applied independent of positive or
    // negative frequencies provided
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType idxOuter = 0; idxOuter < data_.dim_outer(); ++idxOuter) {
      for (SizeType idxInner = 1; idxInner < data_.dim_inner(); ++idxInner) {
        const auto value = data_(idxOuter, 0, idxInner);
        if (value != zeroElement) {
          data_(idxOuter, 0, data_.dim_inner() - idxInner) = std::conj(value);
        }
      }
    }
  }

private:
  HostArrayView3D<std::complex<T>> data_;
};

// This class will apply the hermitian symmetry in 1d
template <typename T>
class StickSymmetryHost : public Symmetry {
public:
  explicit StickSymmetryHost(const HostArrayView1D<std::complex<T>>& stick) : stick_(stick) {}

  auto apply() -> void override {
    constexpr std::complex<T> zeroElement;
    // Data may be conjugated twice, but this way symmetry is applied independent of positive or
    // negative frequencies provided
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType idxInner = 1; idxInner < stick_.size() / 2 + 1; ++idxInner) {
      const auto value = stick_(idxInner);
      if (value != zeroElement) {
        stick_(stick_.size() - idxInner) = std::conj(value);
      }
    }
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType idxInner = stick_.size() / 2 + 1; idxInner < stick_.size(); ++idxInner) {
      const auto value = stick_(idxInner);
      if (value != zeroElement) {
        stick_(stick_.size() - idxInner) = std::conj(value);
      }
    }
  }

private:
  HostArrayView1D<std::complex<T>> stick_;
};
}  // namespace spfft

#endif
