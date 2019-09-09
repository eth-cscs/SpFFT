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
#ifndef SPFFT_TEST_CHECK_VALUES_HPP
#define SPFFT_TEST_CHECK_VALUES_HPP

#include <cassert>
#include <complex>
#include <vector>
#include "gtest/gtest.h"
#include "memory/host_array_view.hpp"
#include "spfft/config.h"

namespace spfft {

inline void check_c2c_space_domain(const HostArrayView3D<std::complex<double>>& realView,
                                   const HostArrayView3D<std::complex<double>>& fftwView,
                                   const SizeType planeOffset, const SizeType numLocalXYPlanes) {
  for (SizeType z = 0; z < numLocalXYPlanes; ++z) {
    for (SizeType x = 0; x < fftwView.dim_outer(); ++x) {
      for (SizeType y = 0; y < fftwView.dim_mid(); ++y) {
        ASSERT_NEAR(realView(z, y, x).real(), fftwView(x, y, z + planeOffset).real(), 1e-6);
        ASSERT_NEAR(realView(z, y, x).imag(), fftwView(x, y, z + planeOffset).imag(), 1e-6);
      }
    }
  }
}

inline void check_r2c_space_domain(const HostArrayView3D<double>& realView,
                                   const HostArrayView3D<std::complex<double>>& fftwView,
                                   const SizeType planeOffset, const SizeType numLocalXYPlanes) {
  for (SizeType z = 0; z < numLocalXYPlanes; ++z) {
    for (SizeType x = 0; x < fftwView.dim_outer(); ++x) {
      for (SizeType y = 0; y < fftwView.dim_mid(); ++y) {
        ASSERT_NEAR(realView(z, y, x), fftwView(x, y, z + planeOffset).real(), 1e-6);
      }
    }
  }
}

inline void check_freq_domain(const std::vector<std::complex<double>>& freqValues,
                              const HostArrayView3D<std::complex<double>>& fftwView,
                              const std::vector<int>& indices) {
  assert(indices.size() == freqValues.size() * 3);

  for (SizeType i = 0; i < freqValues.size(); ++i) {
    int x = indices[i * 3];
    int y = indices[i * 3 + 1];
    int z = indices[i * 3 + 2];
    if (x < 0) x = fftwView.dim_outer() + x;
    if (y < 0) y = fftwView.dim_mid() + y;
    if (z < 0) z = fftwView.dim_inner() + z;
    ASSERT_NEAR(freqValues[i].real(), fftwView(x, y, z).real(), 1e-6);
    ASSERT_NEAR(freqValues[i].imag(), fftwView(x, y, z).imag(), 1e-6);
  }
}

}  // namespace spfft

#endif
