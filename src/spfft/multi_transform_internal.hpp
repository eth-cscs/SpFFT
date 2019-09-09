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
#ifndef SPFFT_MULTI_TRANSFORM_INTERNAL_HPP
#define SPFFT_MULTI_TRANSFORM_INTERNAL_HPP

#include "spfft/exceptions.hpp"
#include "spfft/transform.hpp"
#include "spfft/transform_internal.hpp"
#include "timing/timing.hpp"

#ifdef SPFFT_SINGLE_PRECISION
#include "spfft/transform_float.hpp"
#endif

namespace spfft {

template <typename TransformType>
class MultiTransformInternal {
public:
  using ValueType = typename TransformType::ValueType;

  inline static auto forward(const int numTransforms, TransformType* transforms,
                             SpfftProcessingUnitType* inputLocations, ValueType** outputPointers,
                             SpfftScalingType* scalingTypes) -> void {
    HOST_TIMING_SCOPED("forward")

    // transforms must not share grids
    for (int t1 = 0; t1 < numTransforms; ++t1) {
      for (int t2 = t1 + 1; t2 < numTransforms; ++t2) {
        if (transforms[t1].transform_->shared_grid(*(transforms[t2].transform_))) {
          throw InvalidParameterError();
        }
      }
    }

    // launch all gpu transforms first
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() == SPFFT_PU_GPU) {
        transforms[t].transform_->forward_xy(inputLocations[t]);
      }
    }

    // launch all cpu transforms including MPI exchange
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() != SPFFT_PU_GPU) {
        transforms[t].transform_->forward_xy(inputLocations[t]);
        transforms[t].transform_->forward_exchange();
      }
    }

    // launch all GPU MPI exhanges and transform
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() == SPFFT_PU_GPU) {
        transforms[t].transform_->forward_exchange();
        transforms[t].transform_->forward_z(outputPointers[t], scalingTypes[t]);
      }
    }

    // launch all remaining cpu transforms
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() != SPFFT_PU_GPU) {
        transforms[t].transform_->forward_z(outputPointers[t], scalingTypes[t]);
      }
    }

    // synchronize all transforms
    for (int t = 0; t < numTransforms; ++t) {
      transforms[t].transform_->synchronize();
    }
  }

  inline static auto backward(const int numTransforms, TransformType* transforms,
                              ValueType** inputPointers, SpfftProcessingUnitType* outputLocations)
      -> void {
    HOST_TIMING_SCOPED("backward")

    // transforms must not share grids
    for (int t1 = 0; t1 < numTransforms; ++t1) {
      for (int t2 = t1 + 1; t2 < numTransforms; ++t2) {
        if (transforms[t1].transform_->shared_grid(*(transforms[t2].transform_))) {
          throw InvalidParameterError();
        }
      }
    }

    // launch all gpu transforms first
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() == SPFFT_PU_GPU) {
        transforms[t].transform_->backward_z(inputPointers[t]);
      }
    }

    // launch all cpu transforms including MPI exchange
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() != SPFFT_PU_GPU) {
        transforms[t].transform_->backward_z(inputPointers[t]);
        transforms[t].transform_->backward_exchange();
      }
    }

    // launch all GPU MPI exhanges and transform
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() == SPFFT_PU_GPU) {
        transforms[t].transform_->backward_exchange();
        transforms[t].transform_->backward_xy(outputLocations[t]);
      }
    }

    // launch all remaining cpu transforms
    for (int t = 0; t < numTransforms; ++t) {
      if (transforms[t].transform_->processing_unit() != SPFFT_PU_GPU) {
        transforms[t].transform_->backward_xy(outputLocations[t]);
      }
    }

    // synchronize all transforms
    for (int t = 0; t < numTransforms; ++t) {
      transforms[t].transform_->synchronize();
    }
  }
};
}  // namespace spfft

#endif
