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
#ifndef SPFFT_MULTI_TRANSFORM_FLOAT_H
#define SPFFT_MULTI_TRANSFORM_FLOAT_H

#include "spfft/config.h"
#include "spfft/transform_float.h"
#include "spfft/types.h"

#ifdef SPFFT_SINGLE_PRECISION

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute multiple independent forward transforms at once by internal pipelining.
 *
 * @param[in] numTransforms Number of transforms to execute.
 * @param[in] transforms Transforms to execute.
 * @param[in] inputLocations Input locations for each transform.
 * @param[out] outputPointers Output pointers for each transform.
 * @param[in] scalingTypes Scaling types for each transform.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_multi_transform_forward(
    int numTransforms, SpfftFloatTransform* transforms,
    const SpfftProcessingUnitType* inputLocations, float* const* outputPointers,
    const SpfftScalingType* scalingTypes);

/**
 * Execute multiple independent forward transforms at once by internal pipelining.
 *
 * @param[in] numTransforms Number of transforms to execute.
 * @param[in] transforms Transforms to execute.
 * @param[in] inputPointers Input pointers for each transform.
 * @param[out] outputPointers Output pointers for each transform.
 * @param[in] scalingTypes Scaling types for each transform.
 * @throw GenericError SpFFT error. Can be a derived type.
 * @throw std::exception Error from standard library calls. Can be a derived type.
 */
SPFFT_EXPORT SpfftError spfft_float_multi_transform_forward_ptr(
    int numTransforms, SpfftFloatTransform* transforms, const float* const* inputPointers,
    float* const* outputPointers, const SpfftScalingType* scalingTypes);

/**
 * Execute multiple independent backward transforms at once by internal pipelining.
 *
 * @param[in] numTransforms Number of transforms to execute.
 * @param[in] transforms Transforms to execute.
 * @param[in] inputPointers Input pointers for each transform.
 * @param[in] outputLocations Output locations for each transform.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_multi_transform_backward(
    int numTransforms, SpfftFloatTransform* transforms, const float* const* inputPointers,
    const SpfftProcessingUnitType* outputLocations);

/**
 * Execute multiple independent backward transforms at once by internal pipelining.
 *
 * @param[in] numTransforms Number of transforms to execute.
 * @param[in] transforms Transforms to execute.
 * @param[in] inputPointers Input pointers for each transform.
 * @param[in] outputPointers Output pointers for each transform.
 * @throw GenericError SpFFT error. Can be a derived type.
 * @throw std::exception Error from standard library calls. Can be a derived type.
 */
SPFFT_EXPORT SpfftError spfft_float_multi_transform_backward_ptr(int numTransforms,
                                                                 SpfftFloatTransform* transforms,
                                                                 const float* const* inputPointers,
                                                                 float* const* outputPointers);

#ifdef __cplusplus
}
#endif

#endif

#endif
