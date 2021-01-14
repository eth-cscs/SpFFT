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
#ifndef SPFFT_FFTW_INTERFACE_HPP
#define SPFFT_FFTW_INTERFACE_HPP

#include <mutex>
#include <fftw3.h>
#include "fft/fftw_mutex.hpp"
#include "spfft/config.h"

namespace spfft {

template<typename T>
struct FFTW;

template <>
struct FFTW<double> {
  using ValueType = double;
  using ComplexType = fftw_complex;
  using PlanType = fftw_plan;

  template <typename... ARGS>
  static auto alignment_of(ARGS&&... args) -> int {
    return fftw_alignment_of(args...);
  }

  template <typename... ARGS>
  static auto plan_dft_1d(ARGS&&... args) -> fftw_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftw_plan_dft_1d(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft(ARGS&&... args) -> fftw_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftw_plan_many_dft(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft_c2r(ARGS&&... args) -> fftw_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftw_plan_many_dft_c2r(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft_r2c(ARGS&&... args) -> fftw_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftw_plan_many_dft_r2c(args...);
  }

  template <typename... ARGS>
  static auto destroy_plan(ARGS&&... args) -> void {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    fftw_destroy_plan(args...);
  }

  template <typename... ARGS>
  static auto execute(ARGS&&... args) -> void {
    fftw_execute(args...);
  }

  template <typename... ARGS>
  static auto execute_dft(ARGS&&... args) -> void {
    fftw_execute_dft(args...);
  }

  template <typename... ARGS>
  static auto execute_dft_r2c(ARGS&&... args) -> void {
    fftw_execute_dft_r2c(args...);
  }

  template <typename... ARGS>
  static auto execute_dft_c2r(ARGS&&... args) -> void {
    fftw_execute_dft_c2r(args...);
  }
};

#ifdef SPFFT_SINGLE_PRECISION
template <>
struct FFTW<float> {
  using ValueType = float;
  using ComplexType = fftwf_complex;
  using PlanType = fftwf_plan;

  template <typename... ARGS>
  static auto alignment_of(ARGS&&... args) -> int {
    return fftwf_alignment_of(args...);
  }

  template <typename... ARGS>
  static auto plan_dft_1d(ARGS&&... args) -> fftwf_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftwf_plan_dft_1d(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft(ARGS&&... args) -> fftwf_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftwf_plan_many_dft(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft_c2r(ARGS&&... args) -> fftwf_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftwf_plan_many_dft_c2r(args...);
  }

  template <typename... ARGS>
  static auto plan_many_dft_r2c(ARGS&&... args) -> fftwf_plan {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    return fftwf_plan_many_dft_r2c(args...);
  }

  template <typename... ARGS>
  static auto destroy_plan(ARGS&&... args) -> void {
    std::lock_guard<std::mutex> guard(global_fftw_mutex());
    fftwf_destroy_plan(args...);
  }

  template <typename... ARGS>
  static auto execute(ARGS&&... args) -> void {
    fftwf_execute(args...);
  }

  template <typename... ARGS>
  static auto execute_dft(ARGS&&... args) -> void {
    fftwf_execute_dft(args...);
  }

  template <typename... ARGS>
  static auto execute_dft_r2c(ARGS&&... args) -> void {
    fftwf_execute_dft_r2c(args...);
  }

  template <typename... ARGS>
  static auto execute_dft_c2r(ARGS&&... args) -> void {
    fftwf_execute_dft_c2r(args...);
  }
};
#endif


}  // namespace spfft

#endif
