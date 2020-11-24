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
#ifndef SPFFT_TRANSFORM_REAL_1D_HOST_HPP
#define SPFFT_TRANSFORM_REAL_1D_HOST_HPP

#include <cassert>
#include <complex>
#include <set>
#include <vector>

#include "fft/fftw_plan_1d.hpp"
#include "fft/transform_interface.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

namespace spfft {

// Computes the FFT in 1D along either the innermost dimension (not transposed) or the second
// innermost dimension (transposed)
// The transforms are computed in batches aligned to inner 2d planes
template <typename T>
class R2CTransform1DPlanesHost : public TransformHost<T> {
public:
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ValueType = T;
  using ComplexType = std::complex<T>;

  // r2c
  R2CTransform1DPlanesHost(HostArrayView3D<T> inputData, HostArrayView3D<ComplexType> outputData,
                           bool transposeInputData, bool transposeOutputData, int maxNumThreads) {
    assert(inputData.dim_outer() == outputData.dim_outer());

    assert(disjoint(inputData, outputData));

    // set fftw plan parameters
    const SizeType size = transposeInputData ? inputData.dim_mid() : inputData.dim_inner();
    const SizeType inputStride = transposeInputData ? inputData.dim_inner() : 1;
    const SizeType outputStride = transposeOutputData ? outputData.dim_inner() : 1;

    const SizeType inputDist = transposeInputData ? 1 : inputData.dim_inner();
    const SizeType outputDist = transposeOutputData ? 1 : outputData.dim_inner();

    // make sure maxNumThreads is at least 1
    SizeType numSplitsPerPlane = maxNumThreads < 1 ? 1 : maxNumThreads;
    // only use at most as many splits as required to create work for every thread
    if (numSplitsPerPlane > 1 && inputData.dim_outer() > numSplitsPerPlane) {
      numSplitsPerPlane = 2;
    }
    const SizeType numTransformsPerPlane =
        transposeInputData ? inputData.dim_inner() : inputData.dim_mid();
    // make sure there are at most as many splits as transforms per plane
    numSplitsPerPlane =
        numTransformsPerPlane < numSplitsPerPlane ? numTransformsPerPlane : numSplitsPerPlane;

    const SizeType numTransformsPerSplit = numTransformsPerPlane / numSplitsPerPlane;

    const SizeType inputSplitStrideMid = transposeInputData ? 0 : numTransformsPerSplit;
    const SizeType inputSplitStrideInner = transposeInputData ? numTransformsPerSplit : 0;
    const SizeType outputSplitStrideMid = transposeOutputData ? 0 : numTransformsPerSplit;
    const SizeType outputSplitStrideInner = transposeOutputData ? numTransformsPerSplit : 0;

    // check for non-transposed output
    assert((transposeOutputData) || (size / 2 + 1 == outputData.dim_inner()));

    // check for transposed output
    assert((!transposeOutputData) || (size / 2 + 1 == outputData.dim_mid()));

    // determine number of transforms per plane
    // create plans within each plane
    transforms_.reserve(inputData.dim_outer() * numSplitsPerPlane);
    for (SizeType idxOuter = 0; idxOuter < inputData.dim_outer(); ++idxOuter) {
      for (SizeType idxSplit = 0; idxSplit < numSplitsPerPlane; ++idxSplit) {
        const SizeType howmany =
            idxSplit == numSplitsPerPlane - 1
                ? numTransformsPerSplit + numTransformsPerPlane % numSplitsPerPlane
                : numTransformsPerSplit;
        transforms_.emplace_back(
            FlexibleFFTWPlan<ValueType>{&(inputData(idxOuter, idxSplit * inputSplitStrideMid,
                                                    idxSplit * inputSplitStrideInner)),
                                        &(outputData(idxOuter, idxSplit * outputSplitStrideMid,
                                                     idxSplit * outputSplitStrideInner)),
                                        size, inputStride, outputStride, inputDist, outputDist,
                                        howmany},
            inputData.index(idxOuter, idxSplit * inputSplitStrideMid,
                            idxSplit * inputSplitStrideInner),
            outputData.index(idxOuter, idxSplit * outputSplitStrideMid,
                             idxSplit * outputSplitStrideInner)

        );
      }
    }
  }

  auto execute(const T* input, T* output) -> void override {
    ComplexType* outputComplex = reinterpret_cast<ComplexType*>(output);
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType i = 0; i < transforms_.size(); ++i) {
      auto& triplet = transforms_[i];
      std::get<0>(triplet).execute(input + std::get<1>(triplet),
                                   outputComplex + std::get<2>(triplet));
    }
  }

  auto execute() -> void override {
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType i = 0; i < transforms_.size(); ++i) {
      auto& triplet = transforms_[i];
      std::get<0>(triplet).execute();
    }
  }

private:
  std::vector<std::tuple<FlexibleFFTWPlan<ValueType>, SizeType, SizeType>> transforms_;
};

// Computes the FFT in 1D along either the innermost dimension (not transposed) or the second
// innermost dimension (transposed)
// The transforms are computed in batches aligned to inner 2d planes
template <typename T>
class C2RTransform1DPlanesHost : public TransformHost<T> {
public:
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ValueType = T;
  using ComplexType = std::complex<T>;
  // c2r
  C2RTransform1DPlanesHost(HostArrayView3D<ComplexType> inputData, HostArrayView3D<T> outputData,
                           bool transposeInputData, bool transposeOutputData, int maxNumThreads) {
    assert(inputData.dim_outer() == outputData.dim_outer());

    assert(disjoint(inputData, outputData));

    // set fftw plan parameters
    const SizeType size = transposeOutputData ? outputData.dim_mid() : outputData.dim_inner();
    const SizeType inputStride = transposeInputData ? inputData.dim_inner() : 1;
    const SizeType outputStride = transposeOutputData ? outputData.dim_inner() : 1;

    const SizeType inputDist = transposeInputData ? 1 : inputData.dim_inner();
    const SizeType outputDist = transposeOutputData ? 1 : outputData.dim_inner();

    // make sure maxNumThreads is at least 1
    SizeType numSplitsPerPlane = maxNumThreads < 1 ? 1 : maxNumThreads;
    // only use at most as many splits as required to create work for every thread
    if (numSplitsPerPlane > 1 && inputData.dim_outer() > numSplitsPerPlane) {
      numSplitsPerPlane = 2;
    }
    const SizeType numTransformsPerPlane =
        transposeInputData ? inputData.dim_inner() : inputData.dim_mid();
    // make sure there are at most as many splits as transforms per plane
    numSplitsPerPlane =
        numTransformsPerPlane < numSplitsPerPlane ? numTransformsPerPlane : numSplitsPerPlane;

    const SizeType numTransformsPerSplit = numTransformsPerPlane / numSplitsPerPlane;

    const SizeType inputSplitStrideMid = transposeInputData ? 0 : numTransformsPerSplit;
    const SizeType inputSplitStrideInner = transposeInputData ? numTransformsPerSplit : 0;
    const SizeType outputSplitStrideMid = transposeOutputData ? 0 : numTransformsPerSplit;
    const SizeType outputSplitStrideInner = transposeOutputData ? numTransformsPerSplit : 0;

    // check for non-transposed output
    assert((transposeInputData) || (size / 2 + 1 == inputData.dim_inner()));

    // check for transposed output
    assert((!transposeInputData) || (size / 2 + 1 == inputData.dim_mid()));

    // determine number of transforms per plane
    // create plans within each plane
    transforms_.reserve(inputData.dim_outer() * numSplitsPerPlane);
    for (SizeType idxOuter = 0; idxOuter < inputData.dim_outer(); ++idxOuter) {
      for (SizeType idxSplit = 0; idxSplit < numSplitsPerPlane; ++idxSplit) {
        const SizeType howmany =
            idxSplit == numSplitsPerPlane - 1
                ? numTransformsPerSplit + numTransformsPerPlane % numSplitsPerPlane
                : numTransformsPerSplit;
        transforms_.emplace_back(
            FlexibleFFTWPlan<ValueType>{&(inputData(idxOuter, idxSplit * inputSplitStrideMid,
                                                    idxSplit * inputSplitStrideInner)),
                                        &(outputData(idxOuter, idxSplit * outputSplitStrideMid,
                                                     idxSplit * outputSplitStrideInner)),
                                        size, inputStride, outputStride, inputDist, outputDist,
                                        howmany},
            inputData.index(idxOuter, idxSplit * inputSplitStrideMid,
                            idxSplit * inputSplitStrideInner),
            outputData.index(idxOuter, idxSplit * outputSplitStrideMid,
                             idxSplit * outputSplitStrideInner)

        );
      }
    }
  }

  auto execute(const T* input, T* output) -> void override {
    const ComplexType* inputComplex = reinterpret_cast<const ComplexType*>(input);
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType i = 0; i < transforms_.size(); ++i) {
      auto& triplet = transforms_[i];
      std::get<0>(triplet).execute(inputComplex + std::get<1>(triplet),
                                   output + std::get<2>(triplet));
    }
  }

  auto execute() -> void override {
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType i = 0; i < transforms_.size(); ++i) {
      auto& triplet = transforms_[i];
      std::get<0>(triplet).execute();
    }
  }

private:
  std::vector<std::tuple<FlexibleFFTWPlan<ValueType>, SizeType, SizeType>> transforms_;
};

}  // namespace spfft

#endif
