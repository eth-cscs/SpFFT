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
#ifndef SPFFT_PARAMETERS_HPP
#define SPFFT_PARAMETERS_HPP

#include <cassert>
#include <limits>
#include <utility>
#include <vector>
#include "compression/indices.hpp"
#include "memory/host_array_const_view.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "spfft/types.h"
#include "util/common_types.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_communicator_handle.hpp"
#endif

namespace spfft {

class Parameters {
public:
#ifdef SPFFT_MPI
  Parameters(const MPICommunicatorHandle& comm, const SpfftTransformType transformType,
             const SizeType dimX, const SizeType dimY, const SizeType dimZ,
             const SizeType numLocalXYPlanes, const SizeType numLocalElements,
             SpfftIndexFormatType indexFormat, const int* indices);
#endif

  Parameters(const SpfftTransformType transformType, const SizeType dimX, const SizeType dimY,
             const SizeType dimZ, const SizeType numLocalElements, SpfftIndexFormatType indexFormat,
             const int* indices);

  inline auto dim_x() const noexcept -> SizeType { return dimX_; }

  inline auto dim_x_freq() const noexcept -> SizeType { return dimXFreq_; }

  inline auto dim_y() const noexcept -> SizeType { return dimY_; }

  inline auto dim_z() const noexcept -> SizeType { return dimZ_; }

  inline auto max_num_z_sticks() const noexcept -> SizeType { return maxNumZSticks_; }

  inline auto max_num_xy_planes() const noexcept -> SizeType { return maxNumXYPlanes_; }

  inline auto total_num_z_sticks() const noexcept -> SizeType { return totalNumZSticks_; }

  inline auto total_num_xy_planes() const noexcept -> SizeType { return totalNumXYPlanes_; }

  inline auto transform_type() const noexcept -> SpfftTransformType { return transformType_; }

  inline auto zero_zero_stick_index() const noexcept -> SizeType { return zeroZeroStickIndex_; }

  inline auto num_xy_planes(const SizeType rank) const -> SizeType {
    assert(rank < numXYPlanesPerRank_.size());
    return numXYPlanesPerRank_[rank];
  }

  inline auto local_num_xy_planes() const -> SizeType {
    assert(comm_rank_ < numXYPlanesPerRank_.size());
    return numXYPlanesPerRank_[comm_rank_];
  }

  inline auto xy_plane_offset(const SizeType rank) const -> SizeType {
    assert(rank < numXYPlanesPerRank_.size());
    return xyPlaneOffsets_[rank];
  }

  inline auto local_xy_plane_offset() const -> SizeType {
    assert(comm_rank_ < numXYPlanesPerRank_.size());
    return xyPlaneOffsets_[comm_rank_];
  }

  inline auto num_z_sticks(const SizeType rank) const -> SizeType {
    assert(rank < numZSticksPerRank_.size());
    return numZSticksPerRank_[rank];
  }

  inline auto local_num_z_sticks() const -> SizeType {
    assert(comm_rank_ < numZSticksPerRank_.size());
    return numZSticksPerRank_[comm_rank_];
  }

  inline auto z_stick_xy_indices(const SizeType rank) const -> HostArrayConstView1D<int> {
    assert(rank < stickIndicesPerRank_.size());
    assert(num_z_sticks(rank) == stickIndicesPerRank_[rank].size());
    return HostArrayConstView1D<int>(stickIndicesPerRank_[rank].data(),
                                     stickIndicesPerRank_[rank].size(), false);
  }

  inline auto local_z_stick_xy_indices() const -> HostArrayConstView1D<int> {
    assert(comm_rank_ < stickIndicesPerRank_.size());
    assert(num_z_sticks(comm_rank_) == stickIndicesPerRank_[comm_rank_].size());
    return HostArrayConstView1D<int>(stickIndicesPerRank_[comm_rank_].data(),
                                     stickIndicesPerRank_[comm_rank_].size(), false);
  }

  inline auto local_value_indices() const -> const std::vector<int>& { return freqValueIndices_; }

  inline auto local_num_elements() const -> SizeType { return freqValueIndices_.size(); }

  inline auto global_num_elements() const -> SizeType { return totalNumFrequencyDomainElements_; }

  inline auto global_size() const -> SizeType { return dimX_ * dimY_ * dimZ_; }

  inline auto comm_rank() const -> SizeType { return comm_rank_; }

  inline auto comm_size() const -> SizeType { return comm_size_; }

private:
  SpfftTransformType transformType_;
  SizeType zeroZeroStickIndex_ = std::numeric_limits<SizeType>::max();
  SizeType dimX_ = 0;
  SizeType dimXFreq_ = 0;
  SizeType dimY_ = 0;
  SizeType dimZ_ = 0;
  SizeType maxNumZSticks_ = 0;
  SizeType maxNumXYPlanes_ = 0;
  SizeType totalNumZSticks_ = 0;
  SizeType totalNumXYPlanes_ = 0;
  SizeType totalNumFrequencyDomainElements_ = 0;
  SizeType comm_rank_ = 0;
  SizeType comm_size_ = 1;
  std::vector<SizeType> numZSticksPerRank_;
  std::vector<SizeType> numXYPlanesPerRank_;
  std::vector<SizeType> xyPlaneOffsets_;
  std::vector<std::vector<int>> stickIndicesPerRank_;
  std::vector<int> freqValueIndices_;
};

}  // namespace spfft

#endif
