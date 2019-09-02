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

#include "parameters/parameters.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>

#ifdef SPFFT_MPI
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#endif

namespace spfft {

#ifdef SPFFT_MPI
Parameters::Parameters(const MPICommunicatorHandle& comm, const SpfftTransformType transformType,
                       const SizeType dimX, const SizeType dimY, const SizeType dimZ,
                       const SizeType numLocalXYPlanes, const SizeType numLocalElements,
                       SpfftIndexFormatType indexFormat, const int* indices)
    : transformType_(transformType),
      dimX_(dimX),
      dimXFreq_(transformType == SPFFT_TRANS_R2C ? dimX / 2 + 1 : dimX),
      dimY_(dimY),
      dimZ_(dimZ),
      totalNumXYPlanes_(dimZ),
      comm_rank_(comm.rank()),
      comm_size_(comm.size()) {
  // helper struct to exchange information
  struct TransposeParameter {
    SizeType dimX;
    SizeType dimY;
    SizeType dimZ;
    SizeType numLocalXYPlanes;
    SizeType numLocalZSticks;
    SizeType numLocalElements;
  };


  // Only index triplets supported (for now)
  if(indexFormat != SPFFT_INDEX_TRIPLETS) {
    throw InternalError();
  }

  // convert indices to internal format
  std::vector<int> localStickIndices;
  std::tie(freqValueIndices_, localStickIndices) =
      convert_index_triplets(transformType == SPFFT_TRANS_R2C, dimX, dimY, dimZ, numLocalElements,
                             indices, indices + 1, indices + 2, 3);

  stickIndicesPerRank_ = create_distributed_transform_indices(comm, std::move(localStickIndices));
  check_stick_duplicates(stickIndicesPerRank_);

  const SizeType numLocalZSticks = stickIndicesPerRank_[comm.rank()].size();

  TransposeParameter paramLocal =
      TransposeParameter{dimX, dimY, dimZ, numLocalXYPlanes, numLocalZSticks, numLocalElements};

  // exchange local parameters
  MPIDatatypeHandle parameterType = MPIDatatypeHandle::create_contiguous(
      sizeof(TransposeParameter) / sizeof(SizeType), MPIMatchElementaryType<SizeType>::get());

  std::vector<TransposeParameter> paramPerRank(comm.size());
  mpi_check_status(MPI_Allgather(&paramLocal, 1, parameterType.get(), paramPerRank.data(), 1,
                                 parameterType.get(), comm.get()));

  // Check parameters
  SizeType numZSticksTotal = 0;
  SizeType numXYPlanesTotal = 0;
  for (const auto& p : paramPerRank) {
    // dimensions must match for all ranks
    if (p.dimX != paramLocal.dimX || p.dimY != paramLocal.dimY || p.dimZ != paramLocal.dimZ) {
      throw MPIParameterMismatchError();
    }
    numZSticksTotal += p.numLocalZSticks;
    numXYPlanesTotal += p.numLocalXYPlanes;
  }
  if (numZSticksTotal > dimX * dimY) {
    // More z sticks than possible
    throw MPIParameterMismatchError();
  }
  if (numXYPlanesTotal != dimZ) {
    throw MPIParameterMismatchError();
  }

  // store all parameters in members
  numZSticksPerRank_.reserve(comm.size());
  numXYPlanesPerRank_.reserve(comm.size());
  xyPlaneOffsets_.reserve(comm.size());
  SizeType startIndex = 0;
  SizeType xyPlaneOffset = 0;
  for (const auto& p : paramPerRank) {
    numZSticksPerRank_.emplace_back(p.numLocalZSticks);
    numXYPlanesPerRank_.emplace_back(p.numLocalXYPlanes);
    xyPlaneOffsets_.emplace_back(xyPlaneOffset);
    startIndex += p.numLocalZSticks;
    xyPlaneOffset += p.numLocalXYPlanes;
    totalNumFrequencyDomainElements_ += p.numLocalElements;
  }

  maxNumZSticks_ = *std::max_element(numZSticksPerRank_.begin(), numZSticksPerRank_.end());
  maxNumXYPlanes_ = *std::max_element(numXYPlanesPerRank_.begin(), numXYPlanesPerRank_.end());
  totalNumZSticks_ =
      std::accumulate(numZSticksPerRank_.begin(), numZSticksPerRank_.end(), SizeType(0));

  // check if this rank holds the x=0, y=0 z-stick, which is treated specially for the real to
  // complex case
  zeroZeroStickIndex_ = 0;
  for (const auto& index : stickIndicesPerRank_[comm.rank()]) {
    if (index == 0) {
      break;
    }
    ++zeroZeroStickIndex_;
  }
}
#endif

Parameters::Parameters(const SpfftTransformType transformType, const SizeType dimX,
                       const SizeType dimY, const SizeType dimZ, const SizeType numLocalElements,
                       SpfftIndexFormatType indexFormat, const int* indices)
    : transformType_(transformType),
      dimX_(dimX),
      dimXFreq_(transformType == SPFFT_TRANS_R2C ? dimX / 2 + 1 : dimX),
      dimY_(dimY),
      dimZ_(dimZ),
      maxNumXYPlanes_(dimZ),
      totalNumXYPlanes_(dimZ),
      totalNumFrequencyDomainElements_(numLocalElements),
      comm_rank_(0),
      comm_size_(1),
      numXYPlanesPerRank_(1, dimZ),
      xyPlaneOffsets_(1, 0) {
  // Only index triplets supported (for now)
  if (indexFormat != SPFFT_INDEX_TRIPLETS) {
    throw InternalError();
  }

  std::vector<int> localStickIndices;
  std::tie(freqValueIndices_, localStickIndices) =
      convert_index_triplets(transformType == SPFFT_TRANS_R2C, dimX, dimY, dimZ, numLocalElements,
                             indices, indices + 1, indices + 2, 3);
  stickIndicesPerRank_.emplace_back(std::move(localStickIndices));
  check_stick_duplicates(stickIndicesPerRank_);

  maxNumZSticks_ = stickIndicesPerRank_[0].size();
  totalNumZSticks_ = stickIndicesPerRank_[0].size();
  numZSticksPerRank_.assign(1, stickIndicesPerRank_[0].size());
  zeroZeroStickIndex_ = 0;
  for (const auto& index : stickIndicesPerRank_[0]) {
    if (index == 0) {
      break;
    }
    ++zeroZeroStickIndex_;
  }
}

} // namespace spfft
