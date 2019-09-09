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

#include "memory/aligned_allocation.hpp"
#include <stdlib.h>
#include <unistd.h>

namespace spfft {

namespace memory {

auto allocate_aligned(SizeType numBytes, SizeType alignment) -> void* {
  // check if sizeof(void*) is power of 2
  static_assert((sizeof(void*) & (sizeof(void*) - 1)) == 0,
                "size of void* must by power of 2 for alignment!");
  // check if alignment is power of 2 and multiple of sizeof(void*)
  if (alignment % sizeof(void*) != 0 || ((alignment & (alignment - 1)) != 0))
    throw HostAllocationError();
  void* ptr;
  if (posix_memalign(&ptr, alignment, numBytes) != 0) throw HostAllocationError();
  return ptr;
}

auto allocate_aligned(SizeType numBytes) -> void* {
  static auto pageSize = sysconf(_SC_PAGESIZE);
  return allocate_aligned(numBytes, static_cast<SizeType>(pageSize));
}

auto free_aligned(void* ptr) noexcept -> void { free(ptr); }

}  // namespace memory

}  // namespace spfft
