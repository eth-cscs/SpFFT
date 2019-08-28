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

#ifndef SPFFT_HOST_TIMING_HPP
#define SPFFT_HOST_TIMING_HPP

#include <chrono>
#include <cstddef>
#include <list>
#include <string>
#include <vector>
#include "spfft/config.h"

namespace spfft {
namespace timing {

using HostClockType = std::chrono::high_resolution_clock;

enum class TimeStampType { Start, Stop, Empty };

struct HostTimeStamp {
  HostTimeStamp() : type(TimeStampType::Empty) {}

  // Identifier pointer must point to compile time string literal
  HostTimeStamp(const char* identifier, const TimeStampType& stampType)
      : time(HostClockType::now()), identifierPtr(identifier), type(stampType) {}

  HostClockType::time_point time;
  const char* identifierPtr;
  TimeStampType type;
};

struct TimingResult {
  std::string identifier;
  std::vector<double> timings;
  std::list<TimingResult> subNodes;
};

class HostTimingScoped;

class HostTiming {
public:
  // reserve 1000'000 time stamps by default
  HostTiming() { timeStamps_.reserve(1000 * 1000); }

  // explicit reserve size
  explicit HostTiming(std::size_t reserveCount) { timeStamps_.reserve(reserveCount); }

  // start with string literal
  template <std::size_t N>
  inline auto start(const char (&identifierPtr)[N]) -> void {
    asm volatile("" ::: "memory"); // prevent compiler reordering
    timeStamps_.emplace_back(identifierPtr, TimeStampType::Start);
    asm volatile("" ::: "memory");
  }

  // start with string; more overhead than with string literals
  inline auto start(std::string identifier) -> void {
    asm volatile("" ::: "memory");
    identifierStrings_.emplace_back(std::move(identifier));
    timeStamps_.emplace_back(identifierStrings_.back().c_str(), TimeStampType::Start);
    asm volatile("" ::: "memory");
  }

  // stop with string literal
  template <std::size_t N>
  inline auto stop(const char (&identifierPtr)[N]) -> void {
    asm volatile("" ::: "memory");
    timeStamps_.emplace_back(identifierPtr, TimeStampType::Stop);
    asm volatile("" ::: "memory");
  }

  // stop with string; more overhead than with string literals
  inline auto stop(std::string identifier) -> void {
    asm volatile("" ::: "memory");
    identifierStrings_.emplace_back(std::move(identifier));
    timeStamps_.emplace_back(identifierStrings_.back().c_str(), TimeStampType::Stop);
    asm volatile("" ::: "memory");
  }

  // reset timer
  inline auto reset() -> void {
    timeStamps_.clear();
    timeStamps_.reserve(1000 * 1000);
    identifierStrings_.clear();
  }

  // pretty print to cout
  auto print_timings() -> void;

  // process timings as tree structure
  auto process_timings() -> std::list<TimingResult>;

  // simple json export
  auto export_json() -> std::string;

private:
  inline auto stop_unchecked(const char* identifierPtr) -> void {
    asm volatile("" ::: "memory");
    timeStamps_.emplace_back(identifierPtr, TimeStampType::Stop);
    asm volatile("" ::: "memory");
  }

  friend HostTimingScoped;

  std::vector<HostTimeStamp> timeStamps_;
  std::list<std::string> identifierStrings_;
};

// Helper class, which calls start() upon creation and stop() on deconstruction
class HostTimingScoped {
public:
  // timer reference must be valid for the entire lifetime
  template <std::size_t N>
  HostTimingScoped(const char (&identifierPtr)[N], HostTiming& timer)
      : identifierPtr_(identifierPtr), timer_(timer) {
    timer_.start(identifierPtr);
  }

  HostTimingScoped(std::string identifier, HostTiming& timer)
      : identifierPtr_(nullptr), identifier_(std::move(identifier)), timer_(timer) {
    timer_.start(identifier_);
  }

  HostTimingScoped(const HostTimingScoped&) = delete;
  HostTimingScoped(HostTimingScoped&&) = delete;
  auto operator=(const HostTimingScoped&) -> HostTimingScoped& = delete;
  auto operator=(HostTimingScoped &&) -> HostTimingScoped& = delete;

  ~HostTimingScoped() {
    if (identifierPtr_) {
      timer_.stop_unchecked(identifierPtr_);
    } else {
      timer_.stop(std::move(identifier_));
    }
  }

private:
  const char* identifierPtr_;
  std::string identifier_;
  HostTiming& timer_;
};

} // namespace timing
} // namespace spfft

#endif
