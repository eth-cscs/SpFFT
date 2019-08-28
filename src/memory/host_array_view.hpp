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

#ifndef SPFFT_HOST_ARRAY_VIEW_HPP
#define SPFFT_HOST_ARRAY_VIEW_HPP

#include <array>
#include <cassert>
#include "spfft/config.h"
#include "util/common_types.hpp"

namespace spfft {

template <typename T>
class HostArrayView1D {
public:
  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;

  static constexpr SizeType ORDER = 1;

  HostArrayView1D() = default;

  HostArrayView1D(ValueType* data, const SizeType size, const bool pinned);

  inline auto operator()(const SizeType idx) -> ValueType& {
    assert(idx < size_);
    return data_[idx];
  }

  inline auto operator()(const SizeType idx) const -> const ValueType& {
    assert(idx < size_);
    return data_[idx];
  }

  inline auto pinned() const noexcept -> bool { return pinned_; }

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return size_ == 0; }

  inline auto size() const noexcept -> SizeType { return size_; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size_; }

  inline auto end() const noexcept -> ConstIterator { return data_ + size_; }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size_; }

private:
  SizeType size_ = 0;
  bool pinned_ = false;
  ValueType* data_ = nullptr;
};

template <typename T>
class HostArrayView2D {
public:
  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;

  static constexpr SizeType ORDER = 2;

  HostArrayView2D() = default;

  HostArrayView2D(ValueType* data, const SizeType dimOuter, const SizeType dimInner,
                  const bool pinned);

  HostArrayView2D(ValueType* data, const std::array<SizeType, 2>& dims, const bool pinned);

  inline auto operator()(const SizeType idxOuter, const SizeType idxInner) -> ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxInner < dims_[1]);
    return data_[(idxOuter * dims_[1]) + idxInner];
  }

  inline auto operator()(const SizeType idxOuter, const SizeType idxInner) const
      -> const ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxInner < dims_[1]);
    return data_[(idxOuter * dims_[1]) + idxInner];
  }

  inline auto index(const SizeType idxOuter, const SizeType idxInner) const noexcept -> SizeType {
    return (idxOuter * dims_[1]) + idxInner;
  }

  inline auto pinned() const noexcept -> bool { return pinned_; }

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> SizeType { return dims_[0] * dims_[1]; }

  inline auto dim_inner() const noexcept -> SizeType { return dims_[1]; }

  inline auto dim_outer() const noexcept -> SizeType { return dims_[0]; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size(); }

  inline auto end() const noexcept -> ConstIterator { return data_ + size(); }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size(); }

private:
  std::array<SizeType, 2> dims_ = {0, 0};
  bool pinned_ = false;
  ValueType* data_ = nullptr;
};

template <typename T>
class HostArrayView3D {
public:
  using ValueType = T;
  using Iterator = T*;
  using ConstIterator = const T*;

  static constexpr SizeType ORDER = 3;

  HostArrayView3D() = default;

  HostArrayView3D(ValueType* data, const SizeType dimOuter, const SizeType dimMid,
                  const SizeType dimInner, const bool pinned);

  HostArrayView3D(ValueType* data, const std::array<SizeType, 3>& dims, const bool pinned);

  inline auto operator()(const SizeType idxOuter, const SizeType idxMid,
                         const SizeType idxInner) noexcept -> ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxMid < dims_[1]);
    assert(idxInner < dims_[2]);
    return data_[(idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner];
  }

  inline auto operator()(const SizeType idxOuter, const SizeType idxMid,
                         const SizeType idxInner) const noexcept -> const ValueType& {
    assert(idxOuter < dims_[0]);
    assert(idxMid < dims_[1]);
    assert(idxInner < dims_[2]);
    return data_[(idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner];
  }

  inline auto index(const SizeType idxOuter, const SizeType idxMid, const SizeType idxInner) const
      noexcept -> SizeType {
    return (idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner;
  }

  inline auto pinned() const noexcept -> bool { return pinned_; }

  inline auto data() noexcept -> ValueType* { return data_; }

  inline auto data() const noexcept -> const ValueType* { return data_; }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> SizeType { return dims_[0] * dims_[1] * dims_[2]; }

  inline auto dim_inner() const noexcept -> SizeType { return dims_[2]; }

  inline auto dim_mid() const noexcept -> SizeType { return dims_[1]; }

  inline auto dim_outer() const noexcept -> SizeType { return dims_[0]; }

  inline auto begin() noexcept -> Iterator { return data_; }

  inline auto begin() const noexcept -> ConstIterator { return data_; }

  inline auto cbegin() const noexcept -> ConstIterator { return data_; }

  inline auto end() noexcept -> Iterator { return data_ + size(); }

  inline auto end() const noexcept -> ConstIterator { return data_ + size(); }

  inline auto cend() const noexcept -> ConstIterator { return data_ + size(); }

private:
  std::array<SizeType, 3> dims_ = {0, 0, 0};
  bool pinned_ = false;
  ValueType* data_ = nullptr;
};

// ======================
// Implementation
// ======================

template <typename T>
HostArrayView1D<T>::HostArrayView1D(ValueType* data, const SizeType size, const bool pinned)
    : size_(size), pinned_(pinned), data_(data) {
  assert(!(size != 0 && data == nullptr));
}

template <typename T>
HostArrayView2D<T>::HostArrayView2D(ValueType* data, const SizeType dimOuter,
                                    const SizeType dimInner, const bool pinned)
    : dims_({dimOuter, dimInner}), pinned_(pinned), data_(data) {}

template <typename T>
HostArrayView2D<T>::HostArrayView2D(ValueType* data, const std::array<SizeType, 2>& dims,
                                    const bool pinned)
    : dims_(dims), pinned_(pinned), data_(data) {}

template <typename T>
HostArrayView3D<T>::HostArrayView3D(ValueType* data, const SizeType dimOuter, const SizeType dimMid,
                                    const SizeType dimInner, const bool pinned)
    : dims_({dimOuter, dimMid, dimInner}), pinned_(pinned), data_(data) {}

template <typename T>
HostArrayView3D<T>::HostArrayView3D(ValueType* data, const std::array<SizeType, 3>& dims,
                                    const bool pinned)
    : dims_(dims), pinned_(pinned), data_(data) {}
} // namespace spfft
#endif

