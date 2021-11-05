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
#ifndef SPFFT_FFTW_PLAN_HPP
#define SPFFT_FFTW_PLAN_HPP

#include <cassert>
#include <complex>
#include <unordered_map>
#include <tuple>
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"
#include "util/type_check.hpp"
#include "fft/fftw_interface.hpp"

namespace spfft {

// Hash for tuple of int alignment values. Assumption is that alignments are small numbers (less than
// half the maximum value of an int)
struct FFTWPropHash {
  std::size_t operator()(const std::tuple<bool, int, int>& tuple) const {
    assert(std::get<1>(tuple) >= 0);
    assert(std::get<2>(tuple) >= 0);
    assert(std::get<1>(tuple) < (1 << (sizeof(int) * 4 - 1)));
    assert(std::get<2>(tuple) < (1 << (sizeof(int) * 4 - 1)));
    const int sign = 2 * static_cast<int>(std::get<0>(tuple)) - 1;
    return std::hash<int>()(
        sign * ((std::get<1>(tuple) << (sizeof(int) * 4 - 1)) + std::get<2>(tuple) + 1));
  }
};

enum class FFTWPlanType {
  C2C,
  R2C,
  C2R
};

template <typename T>
class FFTWPlan {
public:
  using ComplexType = std::complex<T>;

  // Create strided 1d fftw plan.
  // If input and output pointers are equal, in-place transform is created.
  FFTWPlan(const ComplexType* input, ComplexType* output, const SizeType size,
           const SizeType istride, const SizeType ostride, const SizeType idist,
           const SizeType odist, const SizeType howmany, const int sign)
      : size_(size),
        sign_(sign),
        inPlace_(input == output),
        alignmentInput_(
            FFTW<T>::alignment_of(reinterpret_cast<T*>(const_cast<ComplexType*>(input)))),
        alignmentOutput_(FFTW<T>::alignment_of(reinterpret_cast<T*>(output))),
        type_(FFTWPlanType::C2C) {
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE;

    plan_ = FFTW<T>::plan_many_dft(
        rank, n, (int)howmany,
        reinterpret_cast<typename FFTW<T>::ComplexType*>(const_cast<ComplexType*>(input)), inembed,
        (int)istride, (int)idist, reinterpret_cast<typename FFTW<T>::ComplexType*>(output), onembed,
        (int)ostride, (int)odist, sign, flags);
    if (!plan_) throw FFTWError();
  }

  // C2R
  FFTWPlan(const ComplexType* input, T* output, const SizeType size, const SizeType istride,
           const SizeType ostride, const SizeType idist, const SizeType odist,
           const SizeType howmany)
      : size_(size),
        sign_(FFTW_BACKWARD),
        inPlace_(reinterpret_cast<const void*>(input) == reinterpret_cast<void*>(output)),
        alignmentInput_(
            FFTW<T>::alignment_of(reinterpret_cast<T*>(const_cast<ComplexType*>(input)))),
        alignmentOutput_(FFTW<T>::alignment_of(output)),
        type_(FFTWPlanType::C2R) {
    assert(reinterpret_cast<const void*>(input) !=
           reinterpret_cast<void*>(output));  // must not be in place
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE;
    plan_ = FFTW<T>::plan_many_dft_c2r(
        rank, n, (int)howmany,
        reinterpret_cast<typename FFTW<T>::ComplexType*>(const_cast<ComplexType*>(input)), inembed,
        (int)istride, (int)idist, output, onembed, (int)ostride, (int)odist, flags);
    if (!plan_) throw FFTWError();
  }

  // R2C
  FFTWPlan(const T* input, ComplexType* output, const SizeType size, const SizeType istride,
           const SizeType ostride, const SizeType idist, const SizeType odist,
           const SizeType howmany)
      : size_(size),
        sign_(FFTW_FORWARD),
        inPlace_(reinterpret_cast<const void*>(input) == reinterpret_cast<void*>(output)),
        alignmentInput_(FFTW<T>::alignment_of(const_cast<T*>(input))),
        alignmentOutput_(FFTW<T>::alignment_of(reinterpret_cast<T*>(output))),
        type_(FFTWPlanType::R2C) {
    assert(reinterpret_cast<const void*>(input) !=
           reinterpret_cast<void*>(output));  // must not be in place
    int rank = 1;
    int n[] = {(int)size};
    int inembed[] = {n[0]};
    int onembed[] = {n[0]};
    auto flags = FFTW_ESTIMATE;
    plan_ = FFTW<T>::plan_many_dft_r2c(rank, n, (int)howmany, const_cast<T*>(input), inembed,
                                       (int)istride, (int)idist,
                                       reinterpret_cast<typename FFTW<T>::ComplexType*>(output),
                                       onembed, (int)ostride, (int)odist, flags);
    if (!plan_) throw FFTWError();
  }

  FFTWPlan(const FFTWPlan& other) = delete;

  FFTWPlan(FFTWPlan&& other) noexcept {
    *this = std::move(other);
  }

  auto operator=(const FFTWPlan& other) -> FFTWPlan& = delete;

  auto operator=(FFTWPlan&& other) noexcept -> FFTWPlan& {
      FFTW<T>::destroy_plan(plan_);

    plan_ = other.plan_;
    size_ = other.size_;
    sign_ = other.sign_;
    inPlace_ = other.inPlace_;
    alignmentInput_ = other.alignmentInput_;
    alignmentOutput_ = other.alignmentOutput_;
    type_ = other.type_;

    other.plan_ = nullptr;
    other.size_ = 0;
    other.sign_ = 0;
    other.inPlace_ = false;
    other.alignmentInput_ = 0;
    other.alignmentOutput_ = 0;
    other.type_ = FFTWPlanType::C2C;

    return *this;
  }

  // Get plan handle
  inline auto get() -> fftw_plan { return plan_; };

  // Release ownership of plan handle
  inline auto release() -> fftw_plan {
    typename FFTW<T>::PlanType planLocal = plan_;
    plan_ = nullptr;
    return planLocal;
  };

  inline auto empty() const noexcept -> bool { return !plan_; }

  inline auto size() const noexcept -> SizeType { return size_; }

  inline auto sign() const noexcept -> int { return sign_; }

  inline auto type() const noexcept -> FFTWPlanType { return type_; }

  // Plan created with in-place transform
  inline auto in_place() const noexcept -> bool { return inPlace_; }

  // Execute on input / output provided to constructor.
  // Undefinded behaviour if empty().
  auto execute() -> void { FFTW<T>::execute(plan_); }

  // Execute on given input / output.
  // The alignment of input and output must match the pointers given to the constructor.
  // If the plan was not setup for in-place transforms, input and output must not be equal
  // Undefinded behaviour if empty().
  auto execute(const void* inputConst, void* output) -> void {
    void* input = const_cast<void*>(inputConst);
    assert(inPlace_ == (input == output));
    assert(FFTW<T>::alignment_of(reinterpret_cast<T*>(input)) == alignmentInput_);
    assert(FFTW<T>::alignment_of(reinterpret_cast<T*>(output)) == alignmentOutput_);
    if(type_ == FFTWPlanType::C2C)
      FFTW<T>::execute_dft(plan_, reinterpret_cast<typename FFTW<T>::ComplexType*>(input),
                           reinterpret_cast<typename FFTW<T>::ComplexType*>(output));
    else if (type_== FFTWPlanType::C2R)
      FFTW<T>::execute_dft_c2r(plan_, reinterpret_cast<typename FFTW<T>::ComplexType*>(input),
                               reinterpret_cast<T*>(output));
    else
      FFTW<T>::execute_dft_r2c(plan_, reinterpret_cast<T*>(input),
                               reinterpret_cast<typename FFTW<T>::ComplexType*>(output));
  }

  ~FFTWPlan() {
    if (plan_) {
      FFTW<T>::destroy_plan(plan_);
    }
    plan_ = nullptr;
  }

private:
  typename FFTW<T>::PlanType plan_ = nullptr;
  SizeType size_ = 0;
  int sign_;
  bool inPlace_ = false;
  int alignmentInput_ = 0;
  int alignmentOutput_ = 0;
  FFTWPlanType type_ = FFTWPlanType::C2C;
};


template <typename T>
class FlexibleFFTWPlan {
public:
  using ComplexType = typename FFTWPlan<T>::ComplexType;

  FlexibleFFTWPlan(const ComplexType* input, ComplexType* output, const SizeType size,
                   const SizeType istride, const SizeType ostride, const SizeType idist,
                   const SizeType odist, const SizeType howmany, const int sign)
      : originalKey_(input == output,
                     FFTW<T>::alignment_of(reinterpret_cast<T*>(const_cast<ComplexType*>(input))),
                     FFTW<T>::alignment_of(reinterpret_cast<T*>(output))),
        size_(size),
        istride_(istride),
        ostride_(ostride),
        idist_(idist),
        odist_(odist),
        howmany_(howmany),
        sign_(sign),
        type_(FFTWPlanType::C2C) {
    plans_.insert({originalKey_, FFTWPlan<T>(input, output, size, istride, ostride, idist, odist,
                                             howmany, sign)});
  }

  FlexibleFFTWPlan(const ComplexType* input, T* output, const SizeType size, const SizeType istride,
                   const SizeType ostride, const SizeType idist, const SizeType odist,
                   const SizeType howmany)
      : originalKey_(reinterpret_cast<const T*>(input) == output,
                     FFTW<T>::alignment_of(reinterpret_cast<T*>(const_cast<ComplexType*>(input))),
                     FFTW<T>::alignment_of(output)),
        size_(size),
        istride_(istride),
        ostride_(ostride),
        idist_(idist),
        odist_(odist),
        howmany_(howmany),
        sign_(FFTW_BACKWARD),
        type_(FFTWPlanType::C2R) {
    plans_.insert(
        {originalKey_, FFTWPlan<T>(input, output, size, istride, ostride, idist, odist, howmany)});
  }

  FlexibleFFTWPlan(const T* input, ComplexType* output, const SizeType size, const SizeType istride,
                   const SizeType ostride, const SizeType idist, const SizeType odist,
                   const SizeType howmany)
      : originalKey_(input == reinterpret_cast<T*>(output),
                     FFTW<T>::alignment_of(const_cast<T*>(input)),
                     FFTW<T>::alignment_of(reinterpret_cast<T*>(output))),
        size_(size),
        istride_(istride),
        ostride_(ostride),
        idist_(idist),
        odist_(odist),
        howmany_(howmany),
        sign_(FFTW_FORWARD),
        type_(FFTWPlanType::R2C) {
    plans_.insert(
        {originalKey_, FFTWPlan<T>(input, output, size, istride, ostride, idist, odist, howmany)});
  }

  inline auto sign() const noexcept -> int { return sign_; }

  auto execute(const void* input, void* output) -> void {
    std::tuple<bool, int, int> key{
        input == output, FFTW<T>::alignment_of(reinterpret_cast<T*>(const_cast<void*>(input))),
        FFTW<T>::alignment_of(reinterpret_cast<T*>(output))};
    auto it = plans_.find(key);
    // Create plan if no matching one is found
    if (it == plans_.end()) {
      if (type_ == FFTWPlanType::C2C)
        it = plans_
                 .insert({key, FFTWPlan<T>(reinterpret_cast<const ComplexType*>(input),
                                           reinterpret_cast<ComplexType*>(output), size_, istride_,
                                           ostride_, idist_, odist_, howmany_, sign_)})
                 .first;
      else if (type_ == FFTWPlanType::C2R)
        it = plans_
                 .insert({key, FFTWPlan<T>(reinterpret_cast<const ComplexType*>(input),
                                           reinterpret_cast<T*>(output), size_, istride_, ostride_,
                                           idist_, odist_, howmany_)})
                 .first;
      else
        it = plans_
                 .insert({key, FFTWPlan<T>(reinterpret_cast<const T*>(input),
                                           reinterpret_cast<ComplexType*>(output), size_, istride_,
                                           ostride_, idist_, odist_, howmany_)})
                 .first;
    }
    it->second.execute(input, output);
  }

  auto execute() -> void {
    auto it = plans_.find(originalKey_);
    assert(it != plans_.end());
    it->second.execute();
  }

private:
  std::unordered_map<std::tuple<bool, int, int>, FFTWPlan<T>, FFTWPropHash> plans_;

  const std::tuple<bool, int, int> originalKey_;
  const SizeType size_;
  const SizeType istride_;
  const SizeType ostride_;
  const SizeType idist_;
  const SizeType odist_;
  const SizeType howmany_;
  const int sign_;
  const FFTWPlanType type_;
};
}  // namespace spfft

#endif
