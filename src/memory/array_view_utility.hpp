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

#ifndef SPFFT_ARRAY_VIEW_UTILITY_HPP
#define SPFFT_ARRAY_VIEW_UTILITY_HPP

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <utility>
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/memory_type_trait.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"

namespace spfft {

template <typename T, typename U>
auto disjoint(const T& array1, const U& array2) -> bool {
  const void* start1 = static_cast<const void*>(array1.data());
  const void* end1 = static_cast<const void*>(array1.data() + array1.size());
  const void* start2 = static_cast<const void*>(array2.data());
  const void* end2 = static_cast<const void*>(array2.data() + array2.size());
  return !(start1 >= start2 && start1 < end2) && !(start2 >= start1 && start2 < end1);
}

namespace gpu_array_utility_internal {
inline auto checked_cast_to_int(const SizeType value) -> int {
  static_assert(std::is_unsigned<SizeType>::value, "Expected unsigend SizeType");
  if (value > static_cast<SizeType>(std::numeric_limits<int>::max())) {
    throw OverflowError();
  }
  return static_cast<int>(value);
}
} // namespace gpu_array_utility_internal

// ----------------------
// Create array view
// ----------------------

template <typename T, typename std::enable_if<!IsDeviceMemory<T>::value, int>::type = 0>
auto create_1d_view(T& array, const SizeType startIdx, const SizeType size)
    -> HostArrayView1D<typename T::ValueType> {
  assert(array.size() >= startIdx + size);
  return HostArrayView1D<typename T::ValueType>(array.data() + startIdx, size, array.pinned());
}

template <typename T, typename std::enable_if<IsDeviceMemory<T>::value, int>::type = 0>
auto create_1d_view(T& array, const SizeType startIdx, const SizeType size)
    -> GPUArrayView1D<typename T::ValueType> {
  assert(array.size() >= startIdx + size);
  return GPUArrayView1D<typename T::ValueType>(
      array.data() + startIdx, gpu_array_utility_internal::checked_cast_to_int(size),
      array.device_id());
}

template <typename T, typename std::enable_if<!IsDeviceMemory<T>::value, int>::type = 0>
auto create_2d_view(T& array, const SizeType startIdx, const SizeType dimOuter,
                    const SizeType dimInner) -> HostArrayView2D<typename T::ValueType> {
  assert(array.size() >= startIdx + dimInner * dimOuter);
  return HostArrayView2D<typename T::ValueType>(array.data() + startIdx, dimOuter, dimInner,
                                                array.pinned());
}

template <typename T, typename std::enable_if<IsDeviceMemory<T>::value, int>::type = 0>
auto create_2d_view(T& array, const SizeType startIdx, const SizeType dimOuter,
                    const SizeType dimInner) -> GPUArrayView2D<typename T::ValueType> {
  assert(array.size() >= startIdx + dimInner * dimOuter);
  // check that entire memory can be adressed with int
  gpu_array_utility_internal::checked_cast_to_int(dimOuter * dimInner);
  return GPUArrayView2D<typename T::ValueType>(
      array.data() + startIdx, gpu_array_utility_internal::checked_cast_to_int(dimOuter),
      gpu_array_utility_internal::checked_cast_to_int(dimInner), array.device_id());
}

template <typename T, typename std::enable_if<!IsDeviceMemory<T>::value, int>::type = 0>
auto create_3d_view(T& array, const SizeType startIdx, const SizeType dimOuter,
                    const SizeType dimMid, const SizeType dimInner)
    -> HostArrayView3D<typename T::ValueType> {
  assert(array.size() >= startIdx + dimOuter * dimMid * dimInner);
  return HostArrayView3D<typename T::ValueType>(array.data() + startIdx, dimOuter, dimMid, dimInner,
                                                array.pinned());
}

template <typename T, typename std::enable_if<IsDeviceMemory<T>::value, int>::type = 0>
auto create_3d_view(T& array, const SizeType startIdx, const SizeType dimOuter,
                    const SizeType dimMid, const SizeType dimInner)
    -> GPUArrayView3D<typename T::ValueType> {
  assert(array.size() >= startIdx + dimOuter * dimMid * dimInner);
  // check that entire memory can be adressed with int
  gpu_array_utility_internal::checked_cast_to_int(dimOuter * dimMid * dimInner);
  return GPUArrayView3D<typename T::ValueType>(
      array.data() + startIdx, gpu_array_utility_internal::checked_cast_to_int(dimOuter),
      gpu_array_utility_internal::checked_cast_to_int(dimMid),
      gpu_array_utility_internal::checked_cast_to_int(dimInner), array.device_id());
}

// -------------------------------
// Create array view with new type
// ------------------------------
template <typename U, typename T, typename std::enable_if<!IsDeviceMemory<T>::value, int>::type = 0>
auto create_new_type_1d_view(T& array, const SizeType size) -> HostArrayView1D<U> {
  assert(array.size() * sizeof(typename T::ValueType) >= size * sizeof(U));
  static_assert(alignof(typename T::ValueType) % alignof(U) == 0,
                "Alignment of old type must be multiple of new type alignment");
  return HostArrayView1D<U>(reinterpret_cast<U*>(array.data()), size, array.pinned());
}

template <typename U, typename T, typename std::enable_if<IsDeviceMemory<T>::value, int>::type = 0>
auto create_new_type_1d_view(T& array, const SizeType size) -> GPUArrayView1D<U> {
  assert(array.size() * sizeof(typename T::ValueType) >= size * sizeof(U));
  static_assert(alignof(typename T::ValueType) % alignof(U) == 0,
                "Alignment of old type must be multiple of new type alignment");
  return GPUArrayView1D<U>(reinterpret_cast<U*>(array.data()),
                           gpu_array_utility_internal::checked_cast_to_int(size),
                           array.device_id());
}

template <typename U, typename T, typename std::enable_if<!IsDeviceMemory<T>::value, int>::type = 0>
auto create_new_type_2d_view(T& array, const SizeType dimOuter, const SizeType dimInner)
    -> HostArrayView2D<U> {
  assert(array.size() * sizeof(typename T::ValueType) >= dimOuter * dimInner * sizeof(U));
  static_assert(alignof(typename T::ValueType) % alignof(U) == 0,
                "Alignment of old type must be multiple of new type alignment");
  return HostArrayView2D<U>(reinterpret_cast<U*>(array.data()), dimOuter, dimInner, array.pinned());
}

template <typename U, typename T, typename std::enable_if<IsDeviceMemory<T>::value, int>::type = 0>
auto create_new_type_2d_view(T& array, const SizeType dimOuter, const SizeType dimInner)
    -> GPUArrayView2D<U> {
  assert(array.size() * sizeof(typename T::ValueType) >= dimOuter * dimInner * sizeof(U));
  static_assert(alignof(typename T::ValueType) % alignof(U) == 0,
                "Alignment of old type must be multiple of new type alignment");
  // check that entire memory can be adressed with int
  gpu_array_utility_internal::checked_cast_to_int(dimOuter * dimInner);
  return GPUArrayView2D<U>(
      reinterpret_cast<U*>(array.data()), gpu_array_utility_internal::checked_cast_to_int(dimOuter),
      gpu_array_utility_internal::checked_cast_to_int(dimInner), array.device_id());
}

template <typename U, typename T, typename std::enable_if<!IsDeviceMemory<T>::value, int>::type = 0>
auto create_new_type_3d_view(T& array, const SizeType dimOuter, const SizeType dimMid,
                             const SizeType dimInner) -> HostArrayView3D<U> {
  assert(array.size() * sizeof(typename T::ValueType) >= dimOuter * dimMid * dimInner * sizeof(U));
  static_assert(alignof(typename T::ValueType) % alignof(U) == 0,
                "Alignment of old type must be multiple of new type alignment");
  return HostArrayView3D<U>(reinterpret_cast<U*>(array.data()), dimOuter, dimMid, dimInner,
                            array.pinned());
}

template <typename U, typename T, typename std::enable_if<IsDeviceMemory<T>::value, int>::type = 0>
auto create_new_type_3d_view(T& array, const SizeType dimOuter, const SizeType dimMid,
                             const SizeType dimInner) -> GPUArrayView3D<U> {
  assert(array.size() * sizeof(typename T::ValueType) >= dimOuter * dimMid * dimInner * sizeof(U));
  static_assert(alignof(typename T::ValueType) % alignof(U) == 0,
                "Alignment of old type must be multiple of new type alignment");
  // check that entire memory can be adressed with int
  gpu_array_utility_internal::checked_cast_to_int(dimOuter * dimMid * dimInner);
  return GPUArrayView3D<U>(
      reinterpret_cast<U*>(array.data()), gpu_array_utility_internal::checked_cast_to_int(dimOuter),
      gpu_array_utility_internal::checked_cast_to_int(dimMid),
      gpu_array_utility_internal::checked_cast_to_int(dimInner), array.device_id());
}

// --------------------------------
// convert scalar and complex views
// --------------------------------
template <typename T>
auto convert_to_complex_view(HostArrayView1D<T> view) -> HostArrayView1D<std::complex<T>> {
  assert(view.size() % 2 == 0);
  return HostArrayView1D<std::complex<T>>(reinterpret_cast<std::complex<T>*>(view.data()),
                                          view.size() / 2, view.pinned());
}

template <typename T>
auto convert_to_complex_view(HostArrayView2D<T> view) -> HostArrayView2D<std::complex<T>> {
  assert(view.dim_inner() % 2 == 0);
  return HostArrayView2D<std::complex<T>>(reinterpret_cast<std::complex<T>*>(view.data()),
                                          view.dim_outer(), view.dim_inner() / 2, view.pinned());
}

template <typename T>
auto convert_to_complex_view(HostArrayView3D<T> view) -> HostArrayView3D<std::complex<T>> {
  assert(view.dim_inner() % 2 == 0);
  return HostArrayView3D<std::complex<T>>(reinterpret_cast<std::complex<T>*>(view.data()),
                                          view.dim_outer(), view.dim_mid(), view.dim_inner() / 2,
                                          view.pinned());
}

template <typename T>
auto convert_from_complex_view(HostArrayView2D<std::complex<T>> view) -> HostArrayView1D<T> {
  return HostArrayView1D<T>(reinterpret_cast<T*>(view.data()), view.size() * 2, view.pinned());
}

template <typename T>
auto convert_from_complex_view(HostArrayView2D<std::complex<T>> view) -> HostArrayView3D<T> {
  return HostArrayView2D<T>(reinterpret_cast<T*>(view.data()), view.dim_outer(),
                            view.dim_inner() * 2, view.pinned());
}

template <typename T>
auto convert_from_complex_view(HostArrayView3D<std::complex<T>> view) -> HostArrayView3D<T> {
  return HostArrayView3D<T>(reinterpret_cast<T*>(view.data()), view.dim_outer(), view.dim_mid(),
                            view.dim_inner() * 2, view.pinned());
}

} // namespace spfft

#endif
