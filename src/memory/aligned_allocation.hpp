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

#ifndef SPFFT_ALIGNED_ALLOCATOR_HPP
#define SPFFT_ALIGNED_ALLOCATOR_HPP

#include <type_traits>
#include <utility>
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"

namespace spfft {

namespace memory {

// Allocate given number of bytes at adress with given alignment.
// The alignment must be a multiple of sizeof(void*) and a power of 2
// Throws upon failure.
auto allocate_aligned(SizeType numBytes, SizeType alignment) -> void*;

// Allocate memory aligned at page boundaries
auto allocate_aligned(SizeType numBytes) -> void*;

// Free memory allocated with allocate_aligned() function
auto free_aligned(void* ptr) noexcept -> void;

// construct numElements elements of type T with arguments args at location pointed to by ptr
template <typename T, typename... ARGS>
auto construct_elements_in_place(T* ptr, SizeType numElements, ARGS&&... args) -> void;

// deconstruct elements of trivially destructable type in array
template <typename T,
          typename std::enable_if<std::is_trivially_destructible<T>::value, int>::type = 0>
auto deconstruct_elements(T* ptr, SizeType numElements) noexcept -> void;

// deconstruct elements of non-trivially destructable type in array
template <typename T,
          typename std::enable_if<!std::is_trivially_destructible<T>::value, int>::type = 0>
auto deconstruct_elements(T* ptr,
                          SizeType numElements) noexcept(std::is_nothrow_destructible<T>::value)
    -> void;

// ======================
// Implementation
// ======================
template <typename T, typename... ARGS>
auto construct_elements_in_place(T* ptr, SizeType numElements, ARGS&&... args) -> void {
  SizeType constructIdx = 0;
  try {
    // construct all elements
    for (; constructIdx < numElements; ++constructIdx) {
      new (ptr + constructIdx) T(std::forward<ARGS>(args)...);
    }
  } catch (...) {
    // destruct all elements which did not throw in case of error
    deconstruct_elements(ptr, constructIdx);
    throw;
  }
}

template <typename T, typename std::enable_if<std::is_trivially_destructible<T>::value, int>::type>
auto deconstruct_elements(T*, SizeType) noexcept -> void {}

template <typename T, typename std::enable_if<!std::is_trivially_destructible<T>::value, int>::type>
auto deconstruct_elements(T* ptr,
                          SizeType numElements) noexcept(std::is_nothrow_destructible<T>::value)
    -> void {
  for (SizeType destructIdx = 0; destructIdx < numElements; ++destructIdx) {
    ptr[destructIdx].~T();
  }
}

}  // namespace memory
}  // namespace spfft

#endif
