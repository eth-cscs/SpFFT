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

#ifndef SPFFT_TIMING_HPP
#define SPFFT_TIMING_HPP

#include "spfft/config.h"

#ifdef SPFFT_TIMING
#include <chrono>
#include <string>
#include "timing/rt_graph.hpp"

namespace spfft {
namespace timing {
extern ::rt_graph::Timer GlobalTimer;
}  // namespace timing
}  // namespace spfft

#define HOST_TIMING_CONCAT_IMPL(x, y) x##y
#define HOST_TIMING_MACRO_CONCAT(x, y) HOST_TIMING_CONCAT_IMPL(x, y)

#define HOST_TIMING_SCOPED(identifier)                        \
  ::rt_graph::ScopedTiming HOST_TIMING_MACRO_CONCAT( \
      scopedHostTimerMacroGenerated, __COUNTER__)(identifier, ::spfft::timing::GlobalTimer);

#define HOST_TIMING_START(identifier) ::spfft::timing::GlobalTimer.start(identifier);

#define HOST_TIMING_STOP(identifier) ::spfft::timing::GlobalTimer.stop(identifier);

#else

#define HOST_TIMING_START(identifier)
#define HOST_TIMING_STOP(identifier)
#define HOST_TIMING_SCOPED(identifier)

#endif

#endif
