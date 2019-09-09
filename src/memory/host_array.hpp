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

#ifndef SPFFT_HOST_ARRAY_HPP
#define SPFFT_HOST_ARRAY_HPP

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/aligned_allocation.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"

namespace spfft {

// Fixed sized array with data aligned to page boundaries
// and requirements for pinned memory with ROCm.
// The data can be pinned in memory, if GPU support is enabled.
// The destructor of type T must not throw.
template <typename T>
class HostArray {
public:
  static_assert(std::is_nothrow_destructible<T>::value,
                "Destructor of ValueType for HostArray must be noexcept.");

  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;
  static constexpr SizeType ORDER = 1;

  // Construct empty array
  HostArray() noexcept;

  // Create array with given size. Additional parameters are passed to the
  // constructor of each element of type T.
  // Throws exception upon allocation or element construction failure
  template <typename... ARGS>
  HostArray(SizeType size, ARGS... args);

  HostArray(const HostArray& array) = delete;

  HostArray(HostArray&& array) noexcept;

  ~HostArray() noexcept(std::is_nothrow_destructible<T>::value);

  auto operator=(const HostArray& array) -> HostArray& = delete;

  auto operator=(HostArray&& array) noexcept -> HostArray&;

  inline auto operator[](const SizeType idx) -> ValueType& {
    assert(idx < size_);
    return data_[idx];
  }

  inline auto operator[](const SizeType idx) const -> const ValueType& {
    assert(idx < size_);
    return data_[idx];
  }

  inline auto operator()(const SizeType idx) -> ValueType& {
    assert(idx < size_);
    return data_[idx];
  }

  inline auto operator()(const SizeType idx) const -> const ValueType& {
    assert(idx < size_);
    return data_[idx];
  }

  inline auto size() const noexcept -> SizeType { return size_; }

  inline auto pinned() const noexcept -> bool { return pinned_; }

  // Attempt to pin memory. Return true on success and false otherwise
  auto pin_memory() noexcept -> bool;

  // Unpin memory if pinned. Does nothing otherwise
  auto unpin_memory() noexcept -> void;

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size_; }

  inline auto end() const noexcept -> ConstIterator { return data_ + size_; }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size_; }

  // undefined behaviour for empty array
  inline auto front() -> ValueType& { return data_[0]; }

  // undefined behaviour for empty array
  inline auto front() const -> const ValueType& { return data_[0]; }

  // undefined behaviour for empty array
  inline auto back() -> ValueType& { return data_[size_ - 1]; }

  // undefined behaviour for empty array
  inline auto back() const -> const ValueType& { return data_[size_ - 1]; }

  inline auto empty() const noexcept -> bool { return size_ == 0; }

private:
  T* data_ = nullptr;
  SizeType size_ = 0;
  bool pinned_ = false;
};

// ======================
// Implementation
// ======================

template <typename T>
HostArray<T>::HostArray() noexcept : data_(nullptr), size_(0), pinned_(false) {}

template <typename T>
template <typename... ARGS>
HostArray<T>::HostArray(SizeType size, ARGS... args)
    : data_(static_cast<T*>(memory::allocate_aligned(size * sizeof(T)))),
      size_(size),
      pinned_(false) {
  try {
    memory::construct_elements_in_place(data_, size, std::forward<ARGS>(args)...);
  } catch (...) {
    size_ = 0;
    memory::free_aligned(data_);
    data_ = nullptr;
    throw;
  }
}

template <typename T>
HostArray<T>::HostArray(HostArray&& array) noexcept : data_(nullptr), size_(0), pinned_(false) {
  data_ = array.data_;
  array.data_ = nullptr;

  size_ = array.size_;
  array.size_ = 0;

  pinned_ = array.pinned_;
  array.pinned_ = false;
}

template <typename T>
HostArray<T>::~HostArray() noexcept(std::is_nothrow_destructible<T>::value) {
  if (data_) {
    this->unpin_memory();
    memory::deconstruct_elements(data_, size_);
    memory::free_aligned(data_);
    data_ = nullptr;
    size_ = 0;
  }
  assert(data_ == nullptr);
  assert(size_ == 0);
  assert(!pinned_);
}

template <typename T>
auto HostArray<T>::operator=(HostArray&& array) noexcept -> HostArray& {
  if (data_) {
    this->unpin_memory();
    memory::deconstruct_elements(data_, size_);
    memory::free_aligned(data_);
  }

  data_ = array.data_;
  array.data_ = nullptr;

  size_ = array.size_;
  array.size_ = 0;

  pinned_ = array.pinned_;
  array.pinned_ = false;

  return *this;
}

template <typename T>
auto HostArray<T>::pin_memory() noexcept -> bool {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  if (!pinned_ && data_) {
    if (gpu::host_register(static_cast<void*>(data_), size_ * sizeof(ValueType),
                           gpu::flag::HostRegisterDefault) == gpu::status::Success) {
      pinned_ = true;
    }
  }
#endif
  return pinned_;
}

template <typename T>
auto HostArray<T>::unpin_memory() noexcept -> void {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  if (pinned_) {
    gpu::host_unregister((void*)data_);
    pinned_ = false;
  }
#endif
}

}  // namespace spfft

#endif
