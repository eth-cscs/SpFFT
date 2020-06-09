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
#ifndef SPFFT_FFTW_REAL_PLAN_HPP
#define SPFFT_FFTW_REAL_PLAN_HPP

#include <fftw3.h>
#include <cassert>
#include <complex>
#include <mutex>
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"
#include "util/type_check.hpp"
#include "fft/fftw_mutex.hpp"

namespace spfft {

template <typename T>
class FFTWRealPlan;

template <>
class FFTWRealPlan<double> {
public:
  using ComplexType = std::complex<double>;

  // Create strided 1d fftw real plan r2c
  FFTWRealPlan(double* input, ComplexType* output, const SizeType size, const SizeType istride,
               const SizeType ostride, const SizeType idist, const SizeType odist,
               const SizeType howmany) {
    assert(reinterpret_cast<void*>(input) !=
           reinterpret_cast<void*>(output));  // must not be in place
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;
    {
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      plan_ = fftw_plan_many_dft_r2c(rank, n, (int)howmany, input, inembed, (int)istride,
                                     (int)idist, reinterpret_cast<fftw_complex*>(output), onembed,
                                     (int)ostride, (int)odist, flags);
    }
    if (!plan_) throw FFTWError();
  }

  // Create strided 1d fftw real plan c2r
  FFTWRealPlan(ComplexType* input, double* output, const SizeType size, const SizeType istride,
               const SizeType ostride, const SizeType idist, const SizeType odist,
               const SizeType howmany) {
    assert(reinterpret_cast<void*>(input) !=
           reinterpret_cast<void*>(output));  // must not be in place
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;
    {
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      plan_ = fftw_plan_many_dft_c2r(rank, n, (int)howmany, reinterpret_cast<fftw_complex*>(input),
                                     inembed, (int)istride, (int)idist, output, onembed,
                                     (int)ostride, (int)odist, flags);
    }
    if (!plan_) throw FFTWError();
  }

  FFTWRealPlan(const FFTWRealPlan& other) = delete;

  FFTWRealPlan(FFTWRealPlan&& other) noexcept {
    if (plan_) {
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      fftw_destroy_plan(plan_);
    }
    plan_ = other.plan_;
    other.plan_ = nullptr;
  }

  auto operator=(const FFTWRealPlan& other) -> FFTWRealPlan& = delete;

  auto operator=(FFTWRealPlan&& other) noexcept -> FFTWRealPlan& {
    if (plan_){
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      fftw_destroy_plan(plan_);
    }
    plan_ = other.plan_;
    other.plan_ = nullptr;
    return *this;
  }

  // Get plan handle
  inline auto get() -> fftw_plan { return plan_; };

  // Release ownership of plan handle
  inline auto release() -> fftw_plan {
    fftw_plan planLocal = plan_;
    plan_ = nullptr;
    return planLocal;
  };

  inline auto empty() const noexcept -> bool { return !plan_; }

  inline auto size() const noexcept -> SizeType { return size_; }

  // Plan created with in-place transform
  inline auto in_place() const noexcept -> bool { return inPlace_; }

  // Execute on input / output provided to constructor.
  // Undefinded behaviour if empty().
  auto execute() -> void { fftw_execute(plan_); }

  ~FFTWRealPlan() {
    if (plan_){
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      fftw_destroy_plan(plan_);
    }
    plan_ = nullptr;
  }

private:
  fftw_plan plan_ = nullptr;
  SizeType size_ = 0;
  bool inPlace_ = false;
};

#ifdef SPFFT_SINGLE_PRECISION
template <>
class FFTWRealPlan<float> {
public:
  using ComplexType = std::complex<float>;

  // Create strided 1d fftwf real plan r2c
  FFTWRealPlan(float* input, ComplexType* output, const SizeType size, const SizeType istride,
               const SizeType ostride, const SizeType idist, const SizeType odist,
               const SizeType howmany) {
    assert(reinterpret_cast<void*>(input) !=
           reinterpret_cast<void*>(output));  // must not be in place
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

    {
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      plan_ = fftwf_plan_many_dft_r2c(rank, n, (int)howmany, input, inembed, (int)istride,
                                      (int)idist, reinterpret_cast<fftwf_complex*>(output), onembed,
                                      (int)ostride, (int)odist, flags);
    }
    if (!plan_) throw FFTWError();
  }

  // Create strided 1d fftwf real plan c2r
  FFTWRealPlan(ComplexType* input, float* output, const SizeType size, const SizeType istride,
               const SizeType ostride, const SizeType idist, const SizeType odist,
               const SizeType howmany) {
    assert(reinterpret_cast<void*>(input) !=
           reinterpret_cast<void*>(output));  // must not be in place
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;
    {
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      plan_ = fftwf_plan_many_dft_c2r(
          rank, n, (int)howmany, reinterpret_cast<fftwf_complex*>(input), inembed, (int)istride,
          (int)idist, output, onembed, (int)ostride, (int)odist, flags);
    }
    if (!plan_) throw FFTWError();
  }

  FFTWRealPlan(const FFTWRealPlan& other) = delete;

  FFTWRealPlan(FFTWRealPlan&& other) noexcept {
    if (plan_) fftwf_destroy_plan(plan_);
    plan_ = other.plan_;
    other.plan_ = nullptr;
  }

  auto operator=(const FFTWRealPlan& other) -> FFTWRealPlan& = delete;

  auto operator=(FFTWRealPlan&& other) noexcept -> FFTWRealPlan& {
    if (plan_) fftwf_destroy_plan(plan_);
    plan_ = other.plan_;
    other.plan_ = nullptr;
    return *this;
  }

  // Get plan handle
  inline auto get() -> fftwf_plan { return plan_; };

  // Release ownership of plan handle
  inline auto release() -> fftwf_plan {
    fftwf_plan planLocal = plan_;
    plan_ = nullptr;
    return planLocal;
  };

  inline auto empty() const noexcept -> bool { return !plan_; }

  inline auto size() const noexcept -> SizeType { return size_; }

  // Plan created with in-place transform
  inline auto in_place() const noexcept -> bool { return inPlace_; }

  // Execute on input / output provided to constructor.
  // Undefinded behaviour if empty().
  auto execute() -> void { fftwf_execute(plan_); }

  // Execute on given input / output.
  // The alignment of input and output must match the pointers given to the constructor.
  // If the plan was not setup for in-place transforms, input and output must not be equal
  // Undefinded behaviour if empty().
  auto execute(ComplexType* input, ComplexType* output) -> void {
    assert(inPlace_ == (input == output));
    assert(fftwf_alignment_of(reinterpret_cast<float*>(input)) == alignmentInput_);
    assert(fftwf_alignment_of(reinterpret_cast<float*>(output)) == alignmentOutput_);
    fftwf_execute_dft(plan_, reinterpret_cast<fftwf_complex*>(input),
                      reinterpret_cast<fftwf_complex*>(output));
  }

  ~FFTWRealPlan() {
    if (plan_){ 
      std::lock_guard<std::mutex> guard(global_fftw_mutex());
      fftwf_destroy_plan(plan_);
    }
    plan_ = nullptr;
  }

private:
  fftwf_plan plan_ = nullptr;
  SizeType size_ = 0;
  bool inPlace_ = false;
  int alignmentInput_ = 0;
  int alignmentOutput_ = 0;
};

#endif

}  // namespace spfft

#endif
