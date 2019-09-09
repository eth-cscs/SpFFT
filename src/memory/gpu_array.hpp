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

#ifndef SPFFT_GPU_ARRAY_HPP
#define SPFFT_GPU_ARRAY_HPP

#include <cassert>
#include "gpu_util/gpu_runtime_api.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"

namespace spfft {

template <typename T>
class GPUArray {
public:
  using ValueType = T;
  static constexpr SizeType ORDER = 1;

  GPUArray() = default;

  GPUArray(const SizeType size);

  GPUArray(const GPUArray& array) = delete;

  GPUArray(GPUArray&& array) noexcept;

  ~GPUArray();

  auto operator=(const GPUArray& array) -> GPUArray& = delete;

  auto operator=(GPUArray&& array) noexcept -> GPUArray&;

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return size_ == 0; }

  inline auto size() const noexcept -> SizeType { return size_; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

private:
  SizeType size_ = 0;
  ValueType* data_ = nullptr;
  int deviceId_ = 0;
};

// ======================
// Implementation
// ======================
template <typename T>
GPUArray<T>::GPUArray(const SizeType size) : size_(size), data_(nullptr), deviceId_(0) {
  assert(size >= 0);
  gpu::check_status(gpu::get_device(&deviceId_));
  if (size > 0) {
    gpu::check_status(gpu::malloc(reinterpret_cast<void**>(&data_), size * sizeof(ValueType)));
  }
}

template <typename T>
GPUArray<T>::~GPUArray() {
  if (data_) {
    // don't check error to avoid throwing exception in destructor
    gpu::free(data_);
    data_ = nullptr;
    size_ = 0;
  }
}

template <typename T>
GPUArray<T>::GPUArray(GPUArray&& array) noexcept
    : size_(array.size_), data_(array.data_), deviceId_(array.deviceId_) {
  array.data_ = nullptr;
  array.size_ = 0;
}

template <typename T>
auto GPUArray<T>::operator=(GPUArray&& array) noexcept -> GPUArray& {
  if (data_) {
    gpu::free(data_);
  }
  data_ = array.data_;
  size_ = array.size_;
  deviceId_ = array.deviceId_;

  array.data_ = nullptr;
  array.size_ = 0;

  return *this;
}

}  // namespace spfft

#endif
