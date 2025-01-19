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
#include "spfft/multi_transform.h"

#include <algorithm>
#include <vector>

#include "spfft/config.h"
#include "spfft/multi_transform.hpp"
#include "spfft/multi_transform_internal.hpp"
#include "spfft/types.h"

namespace spfft {

void multi_transform_forward(int numTransforms, Transform* transforms,
                             const SpfftProcessingUnitType* inputLocations,
                             double* const* outputPointers, const SpfftScalingType* scalingTypes) {
  std::vector<Transform*> transformPtrs(numTransforms);
  std::transform(transforms, transforms + numTransforms, transformPtrs.begin(),
                 [](Transform& t) { return &t; });

  MultiTransformInternal<Transform>::forward(numTransforms, transformPtrs.data(), inputLocations,
                                             outputPointers, scalingTypes);
}

void multi_transform_backward(int numTransforms, Transform* transforms,
                              const double* const* inputPointers,
                              const SpfftProcessingUnitType* outputLocations) {
  std::vector<Transform*> transformPtrs(numTransforms);
  std::transform(transforms, transforms + numTransforms, transformPtrs.begin(),
                 [](Transform& t) { return &t; });

  MultiTransformInternal<Transform>::backward(numTransforms, transformPtrs.data(), inputPointers,
                                              outputLocations);
}

void multi_transform_forward(int numTransforms, Transform* transforms,
                             const double* const* inputPointers, double* const* outputPointers,
                             const SpfftScalingType* scalingTypes) {
  std::vector<Transform*> transformPtrs(numTransforms);
  std::transform(transforms, transforms + numTransforms, transformPtrs.begin(),
                 [](Transform& t) { return &t; });

  MultiTransformInternal<Transform>::forward(numTransforms, transformPtrs.data(), inputPointers,
                                             outputPointers, scalingTypes);
}

void multi_transform_backward(int numTransforms, Transform* transforms,
                              const double* const* inputPointers, double* const* outputPointers) {
  std::vector<Transform*> transformPtrs(numTransforms);
  std::transform(transforms, transforms + numTransforms, transformPtrs.begin(),
                 [](Transform& t) { return &t; });

  MultiTransformInternal<Transform>::backward(numTransforms, transformPtrs.data(), inputPointers,
                                              outputPointers);
}
}  // namespace spfft

extern "C" {

SpfftError spfft_multi_transform_forward(int numTransforms, SpfftTransform* transforms,
                                         const SpfftProcessingUnitType* inputLocations,
                                         double* const* outputPointers,
                                         const SpfftScalingType* scalingTypes) {
  try {
    spfft::MultiTransformInternal<spfft::Transform>::forward(
        numTransforms, reinterpret_cast<spfft::Transform**>(transforms), inputLocations,
        outputPointers, scalingTypes);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_multi_transform_forward_ptr(int numTransforms, SpfftTransform* transforms,
                                             const double* const* inputPointers,
                                             double* const* outputPointers,
                                             const SpfftScalingType* scalingTypes) {
  try {
    spfft::MultiTransformInternal<spfft::Transform>::forward(
        numTransforms, reinterpret_cast<spfft::Transform**>(transforms), inputPointers,
        outputPointers, scalingTypes);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_multi_transform_backward(int numTransforms, SpfftTransform* transforms,
                                          const double* const* inputPointers,
                                          const SpfftProcessingUnitType* outputLocations) {
  try {
    spfft::MultiTransformInternal<spfft::Transform>::backward(
        numTransforms, reinterpret_cast<spfft::Transform**>(transforms), inputPointers,
        outputLocations);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_multi_transform_backward_ptr(int numTransforms, SpfftTransform* transforms,
                                              const double* const* inputPointers,
                                              double* const* outputPointers) {
  try {
    spfft::MultiTransformInternal<spfft::Transform>::backward(
        numTransforms, reinterpret_cast<spfft::Transform**>(transforms), inputPointers,
        outputPointers);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}
}
