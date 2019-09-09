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
#ifndef SPFFT_TRANSPOSE_HOST_HPP
#define SPFFT_TRANSPOSE_HOST_HPP

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

namespace spfft {
// Transpose Z sticks, such that data is represented by xy planes, where the y-dimension is
// continous and vice versa
template <typename T>
class TransposeHost : public Transpose {
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ValueType = T;
  using ComplexType = std::complex<T>;

public:
  TransposeHost(const std::shared_ptr<Parameters>& param,
                HostArrayView3D<ComplexType> spaceDomainData,
                HostArrayView2D<ComplexType> freqDomainData)
      : spaceDomainData_(spaceDomainData), freqDomainData_(freqDomainData), param_(param) {
    // single rank only checks
    assert(spaceDomainData.dim_outer() == freqDomainData.dim_inner());

    // check data dimensions and parameters
    assert(param->dim_x_freq() == spaceDomainData.dim_mid());
    assert(param->dim_y() == spaceDomainData.dim_inner());
    assert(param->dim_z() == spaceDomainData.dim_outer());
    assert(param->dim_z() == freqDomainData.dim_inner());
    assert(param->num_z_sticks(0) == freqDomainData.dim_outer());

    // data must be disjoint
    assert(disjoint(spaceDomainData, freqDomainData));
  }

  auto exchange_backward_start(const bool) -> void override {}

  auto unpack_backward() -> void override {
    SPFFT_OMP_PRAGMA("omp for schedule(static)")  // implicit barrier
    for (SizeType z = 0; z < spaceDomainData_.dim_outer(); ++z) {
      std::memset(static_cast<void*>(&spaceDomainData_(z, 0, 0)), 0,
                  sizeof(typename decltype(spaceDomainData_)::ValueType) *
                      spaceDomainData_.dim_inner() * spaceDomainData_.dim_mid());
    }

    const SizeType unrolledLoopEnd =
        freqDomainData_.dim_outer() < 4 ? 0 : freqDomainData_.dim_outer() - 3;

    auto stickIndicesView = param_->z_stick_xy_indices(0);

    auto spaceDomainDataFlat =
        create_2d_view(spaceDomainData_, 0, spaceDomainData_.dim_outer(),
                       spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner());

    // unrolled loop
    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = 0; zStickIndex < unrolledLoopEnd; zStickIndex += 4) {
      const SizeType xyIndex1 = stickIndicesView(zStickIndex);
      const SizeType xyIndex2 = stickIndicesView(zStickIndex + 1);
      const SizeType xyIndex3 = stickIndicesView(zStickIndex + 2);
      const SizeType xyIndex4 = stickIndicesView(zStickIndex + 3);
      for (SizeType zIndex = 0; zIndex < freqDomainData_.dim_inner(); ++zIndex) {
        spaceDomainDataFlat(zIndex, xyIndex1) = freqDomainData_(zStickIndex, zIndex);
        spaceDomainDataFlat(zIndex, xyIndex2) = freqDomainData_(zStickIndex + 1, zIndex);
        spaceDomainDataFlat(zIndex, xyIndex3) = freqDomainData_(zStickIndex + 2, zIndex);
        spaceDomainDataFlat(zIndex, xyIndex4) = freqDomainData_(zStickIndex + 3, zIndex);
      }
    }

    // transpose remaining elements
    SPFFT_OMP_PRAGMA("omp for schedule(static)")  // keep barrier
    for (SizeType zStickIndex = unrolledLoopEnd; zStickIndex < freqDomainData_.dim_outer();
         zStickIndex += 1) {
      const SizeType xyIndex = stickIndicesView(zStickIndex);
      for (SizeType zIndex = 0; zIndex < freqDomainData_.dim_inner(); ++zIndex) {
        spaceDomainDataFlat(zIndex, xyIndex) = freqDomainData_(zStickIndex, zIndex);
      }
    }
  }

  auto exchange_forward_start(const bool) -> void override {}

  auto unpack_forward() -> void override {
    const SizeType unrolledLoopEnd =
        freqDomainData_.dim_outer() < 4 ? 0 : freqDomainData_.dim_outer() - 3;

    auto stickIndicesView = param_->z_stick_xy_indices(0);

    auto spaceDomainDataFlat =
        create_2d_view(spaceDomainData_, 0, spaceDomainData_.dim_outer(),
                       spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner());

    // unrolled loop
    SPFFT_OMP_PRAGMA("omp for schedule(static) nowait")
    for (SizeType zStickIndex = 0; zStickIndex < unrolledLoopEnd; zStickIndex += 4) {
      const SizeType xyIndex1 = stickIndicesView(zStickIndex);
      const SizeType xyIndex2 = stickIndicesView(zStickIndex + 1);
      const SizeType xyIndex3 = stickIndicesView(zStickIndex + 2);
      const SizeType xyIndex4 = stickIndicesView(zStickIndex + 3);
      for (SizeType zIndex = 0; zIndex < freqDomainData_.dim_inner(); ++zIndex) {
        freqDomainData_(zStickIndex, zIndex) = spaceDomainDataFlat(zIndex, xyIndex1);
        freqDomainData_(zStickIndex + 1, zIndex) = spaceDomainDataFlat(zIndex, xyIndex2);
        freqDomainData_(zStickIndex + 2, zIndex) = spaceDomainDataFlat(zIndex, xyIndex3);
        freqDomainData_(zStickIndex + 3, zIndex) = spaceDomainDataFlat(zIndex, xyIndex4);
      }
    }

    // transpose remaining elements
    SPFFT_OMP_PRAGMA("omp for schedule(static)")  // keep barrier
    for (SizeType zStickIndex = unrolledLoopEnd; zStickIndex < freqDomainData_.dim_outer();
         zStickIndex += 1) {
      const SizeType xyIndex = stickIndicesView(zStickIndex);
      for (SizeType zIndex = 0; zIndex < freqDomainData_.dim_inner(); ++zIndex) {
        freqDomainData_(zStickIndex, zIndex) = spaceDomainDataFlat(zIndex, xyIndex);
      }
    }
  }

private:
  HostArrayView3D<ComplexType> spaceDomainData_;
  HostArrayView2D<ComplexType> freqDomainData_;
  std::shared_ptr<Parameters> param_;
};
}  // namespace spfft
#endif
