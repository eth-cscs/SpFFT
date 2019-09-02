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
#ifndef SPFFT_GPU_EVENT_HANDLE_HPP
#define SPFFT_GPU_EVENT_HANDLE_HPP

#include "spfft/config.h"
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include <memory>
#include "gpu_util/gpu_runtime_api.hpp"
#include "spfft/exceptions.hpp"

namespace spfft {
class GPUEventHandle {
public:

  explicit GPUEventHandle(const bool enableTiming) : deviceId_(0) {
    gpu::check_status(gpu::get_device(&deviceId_));
    gpu::EventType event;

    const auto flag = enableTiming ? gpu::flag::EventDefault : gpu::flag::EventDisableTiming;
    gpu::check_status(gpu::event_create_with_flags(&event, flag));

    event_ =
        std::shared_ptr<gpu::EventType>(new gpu::EventType(event), [](gpu::EventType* ptr) {
          gpu::event_destroy(*ptr);
          delete ptr;
        });
  };

  inline auto get() const -> gpu::EventType { return *event_; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

  inline auto record(const gpu::StreamType& stream) const -> void {
    gpu::check_status(gpu::event_record(*event_, stream));
  }

  inline auto stream_wait(const gpu::StreamType& stream) const -> void {
    gpu::check_status(gpu::stream_wait_event(stream, *event_, 0));
  }

private:
  std::shared_ptr<gpu::EventType> event_;
  int deviceId_ = 0;
};
} // namespace spfft

#endif
#endif
