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

#ifndef SPFFT_GPU_ARRAY_CONST_VIEW_HPP
#define SPFFT_GPU_ARRAY_CONST_VIEW_HPP

#include <cassert>
#include <limits>
#include "memory/gpu_array_view.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "gpu_util/gpu_runtime.hpp"
#endif

namespace spfft {

// T must be build-in type
template <typename T>
class GPUArrayConstView1D {
public:
  using ValueType = T;
  static constexpr SizeType ORDER = 1;

  GPUArrayConstView1D() = default;

  GPUArrayConstView1D(const ValueType* data, const int size, const int deviceId);

  GPUArrayConstView1D(const GPUArrayView1D<T>&);  // conversion allowed

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ inline auto operator()(const int idx) const -> ValueType {
    assert(idx < size_);
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    return __ldg(data_ + idx);
#else
    return data_[idx];
#endif
  }

  __host__ __device__ inline auto empty() const noexcept -> bool { return size_ == 0; }

  __host__ __device__ inline auto size() const noexcept -> int { return size_; }

  __host__ __device__ inline auto device_id() const noexcept -> int { return deviceId_; }

#else

  inline auto empty() const noexcept -> bool { return size_ == 0; }

  inline auto size() const noexcept -> int { return size_; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

#endif

private:
  int size_ = 0;
  const ValueType* data_ = nullptr;
  int deviceId_ = 0;
};

// T must be build-in type
template <typename T>
class GPUArrayConstView2D {
public:
  using ValueType = T;
  static constexpr SizeType ORDER = 2;

  GPUArrayConstView2D() = default;

  GPUArrayConstView2D(const ValueType* data, const int dimOuter, const int dimInner,
                      const int deviceId);

  GPUArrayConstView2D(const GPUArrayView2D<T>&);  // conversion allowed

#if defined(__CUDACC__) || defined(__HIPCC__)

  __device__ inline auto operator()(const int idxOuter, const int idxInner) const -> ValueType {
    assert(idxOuter < dims_[0]);
    assert(idxInner < dims_[1]);
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    return __ldg(data_ + (idxOuter * dims_[1]) + idxInner);
#else
    return data_[(idxOuter * dims_[1]) + idxInner];
#endif
  }

  __host__ __device__ inline auto index(const int idxOuter, const int idxInner) const noexcept
      -> int {
    return (idxOuter * dims_[1]) + idxInner;
  }

  __host__ __device__ inline auto empty() const noexcept -> bool { return this->size() == 0; }

  __host__ __device__ inline auto size() const noexcept -> int { return dims_[0] * dims_[1]; }

  __host__ __device__ inline auto dim_inner() const noexcept -> int { return dims_[1]; }

  __host__ __device__ inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  __host__ __device__ inline auto device_id() const noexcept -> int { return deviceId_; }

#else

  inline auto index(const int idxOuter, const int idxInner) const noexcept -> int {
    return (idxOuter * dims_[1]) + idxInner;
  }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> int { return dims_[0] * dims_[1]; }

  inline auto dim_inner() const noexcept -> int { return dims_[1]; }

  inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

#endif
private:
  int dims_[2];
  const ValueType* data_ = nullptr;
  int deviceId_ = 0;
};
// T must be build-in type
template <typename T>
class GPUArrayConstView3D {
public:
  using ValueType = T;
  static constexpr SizeType ORDER = 3;

  GPUArrayConstView3D() = default;

  GPUArrayConstView3D(const ValueType* data, const int dimOuter, const int dimMid,
                      const int dimInner, const int deviceId);

  GPUArrayConstView3D(const GPUArrayView3D<T>&);  // conversion allowed

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ inline auto operator()(const int idxOuter, const int idxMid, const int idxInner) const
      -> ValueType {
    assert(idxOuter < dims_[0]);
    assert(idxMid < dims_[1]);
    assert(idxInner < dims_[2]);
#if __CUDA_ARCH__ >= 350 || defined(__HIPCC__)
    return __ldg(data_ + (idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner);
#else
    return data_[(idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner];
#endif
  }

  __host__ __device__ inline auto index(const int idxOuter, const int idxMid,
                                        const int idxInner) const noexcept -> int {
    return (idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner;
  }

  __host__ __device__ inline auto empty() const noexcept -> bool { return this->size() == 0; }

  __host__ __device__ inline auto size() const noexcept -> int {
    return dims_[0] * dims_[1] * dims_[2];
  }

  __host__ __device__ inline auto dim_inner() const noexcept -> int { return dims_[2]; }

  __host__ __device__ inline auto dim_mid() const noexcept -> int { return dims_[1]; }

  __host__ __device__ inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  __host__ __device__ inline auto device_id() const noexcept -> int { return deviceId_; }

#else

  inline auto index(const int idxOuter, const int idxMid, const int idxInner) const noexcept
      -> int {
    return (idxOuter * dims_[1] + idxMid) * dims_[2] + idxInner;
  }

  inline auto empty() const noexcept -> bool { return this->size() == 0; }

  inline auto size() const noexcept -> int { return dims_[0] * dims_[1] * dims_[2]; }

  inline auto dim_inner() const noexcept -> int { return dims_[2]; }

  inline auto dim_mid() const noexcept -> int { return dims_[1]; }

  inline auto dim_outer() const noexcept -> int { return dims_[0]; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

#endif

private:
  int dims_[3];
  const ValueType* data_ = nullptr;
  int deviceId_ = 0;
};

// ======================
// Implementation
// ======================

template <typename T>
GPUArrayConstView1D<T>::GPUArrayConstView1D(const ValueType* data, const int size,
                                            const int deviceId)
    : size_(size), data_(data), deviceId_(deviceId) {
  assert(!(size != 0 && data == nullptr));
}

template <typename T>
GPUArrayConstView1D<T>::GPUArrayConstView1D(const GPUArrayView1D<T>& view)
    : size_(view.size()), data_(view.data()), deviceId_(view.device_id()) {}

template <typename T>
GPUArrayConstView2D<T>::GPUArrayConstView2D(const ValueType* data, const int dimOuter,
                                            const int dimInner, const int deviceId)
    : dims_{dimOuter, dimInner}, data_(data), deviceId_(deviceId) {
  assert(!(dimOuter != 0 && dimInner != 0 && data == nullptr));
  assert(dimOuter >= 0);
  assert(dimInner >= 0);
}

template <typename T>
GPUArrayConstView2D<T>::GPUArrayConstView2D(const GPUArrayView2D<T>& view)
    : dims_{view.dim_outer(), view.dim_inner()}, data_(view.data()), deviceId_(view.device_id()) {}

template <typename T>
GPUArrayConstView3D<T>::GPUArrayConstView3D(const ValueType* data, const int dimOuter,
                                            const int dimMid, const int dimInner,
                                            const int deviceId)
    : dims_{dimOuter, dimMid, dimInner}, data_(data), deviceId_(deviceId) {
  assert(!(dimOuter != 0 && dimMid != 0 && dimInner != 0 && data == nullptr));
  assert(dimOuter >= 0);
  assert(dimMid >= 0);
  assert(dimInner >= 0);
}

template <typename T>
GPUArrayConstView3D<T>::GPUArrayConstView3D(const GPUArrayView3D<T>& view)
    : dims_{view.dim_outer(), view.dim_mid(), view.dim_inner()},
      data_(view.data()),
      deviceId_(view.device_id()) {}
}  // namespace spfft

#endif
