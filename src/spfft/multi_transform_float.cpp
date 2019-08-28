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
#include "spfft/config.h"
#include "spfft/types.h"
#include "spfft/multi_transform_float.h"
#include "spfft/multi_transform_float.hpp"
#include "spfft/multi_transform_internal.hpp"

namespace spfft {

#ifdef SPFFT_SINGLE_PRECISION

void multi_transform_forward(int numTransforms, TransformFloat* transforms,
                             SpfftProcessingUnitType* inputLocations, float** outputPointers,
                             SpfftScalingType* scalingTypes) {
  MultiTransformInternal<TransformFloat>::forward(numTransforms, transforms, inputLocations,
                                                  outputPointers, scalingTypes);
}

void multi_transform_backward(int numTransforms, TransformFloat* transforms, float** inputPointers,
                              SpfftProcessingUnitType* outputLocations) {
  MultiTransformInternal<TransformFloat>::backward(numTransforms, transforms, inputPointers,
                                                   outputLocations);
}
#endif

} // namespace spfft


extern "C" {

SpfftError spfft_float_multi_transform_forward(int numTransforms, SpfftFloatTransform* transforms,
                                               SpfftProcessingUnitType* inputLocations,
                                               float** outputPointers,
                                               SpfftScalingType* scalingTypes) {
  try {
    multi_transform_forward(numTransforms, reinterpret_cast<spfft::TransformFloat*>(transforms),
                            inputLocations, outputPointers, scalingTypes);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_multi_transform_backward(int numTransforms, SpfftFloatTransform* transforms,
                                                float** inputPointers,
                                                SpfftProcessingUnitType* outputLocations) {
  try {
    multi_transform_backward(numTransforms, reinterpret_cast<spfft::TransformFloat*>(transforms),
                             inputPointers, outputLocations);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}
}
