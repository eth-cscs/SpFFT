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
#ifndef SPFFT_INDICES_HPP
#define SPFFT_INDICES_HPP

#include <complex>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#endif

namespace spfft {

// convert [-N, N) frequency index to [0, N) for FFT input
inline auto to_storage_index(const int dim, const int index) -> int {
  if (index < 0) {
    return dim + index;
  } else {
    return index;
  }
}

#ifdef SPFFT_MPI
inline auto create_distributed_transform_indices(const MPICommunicatorHandle& comm,
                                                 std::vector<int> localSticks)
    -> std::vector<std::vector<int>> {
  std::vector<MPI_Request> sendRequests(comm.size());

  constexpr int tag = 442;  // random tag (must be less than 32768)

  // send local stick indices
  for (int r = 0; r < static_cast<int>(comm.size()); ++r) {
    if (r != static_cast<int>(comm.rank())) {
      mpi_check_status(MPI_Isend(localSticks.data(), localSticks.size(), MPI_INT, r, tag,
                                 comm.get(), &(sendRequests[r])));
    }
  }

  std::vector<std::vector<int>> globalXYIndices(comm.size());

  // recv all other stick indices
  for (int r = 0; r < static_cast<int>(comm.size()); ++r) {
    if (r != static_cast<int>(comm.rank())) {
      // get recv count
      MPI_Status status;
      MPI_Probe(r, tag, comm.get(), &status);
      int recvCount = 0;
      MPI_Get_count(&status, MPI_INT, &recvCount);

      // recv data
      globalXYIndices[r].resize(recvCount);
      MPI_Recv(globalXYIndices[r].data(), recvCount, MPI_INT, r, tag, comm.get(),
               MPI_STATUS_IGNORE);
    }
  }

  // wait for all sends to finish
  for (int r = 0; r < static_cast<int>(comm.size()); ++r) {
    if (r != static_cast<int>(comm.rank())) {
      MPI_Wait(&(sendRequests[r]), MPI_STATUS_IGNORE);
    }
  }

  // move local sticks into transform indices object AFTER sends are finished
  globalXYIndices[comm.rank()] = std::move(localSticks);

  return globalXYIndices;
}
#endif

inline auto check_stick_duplicates(const std::vector<std::vector<int>>& indicesPerRank) -> void {
  // check for z-sticks indices
  std::set<int> globalXYIndices;
  for (const auto& rankIndices : indicesPerRank) {
    for (const auto& index : rankIndices) {
      if (globalXYIndices.count(index)) {
        throw DuplicateIndicesError();
      }

      globalXYIndices.insert(index);
    }
  }
}

// convert index triplets for every value into stick/z indices and z-stick index pairs.
inline auto convert_index_triplets(const bool hermitianSymmetry, const int dimX, const int dimY,
                                   const int dimZ, const int numValues, const int* xIndices,
                                   const int* yIndices, const int* zIndices, const int stride)
    -> std::pair<std::vector<int>, std::vector<int>> {
  if (static_cast<SizeType>(numValues) >
      static_cast<SizeType>(dimX) * static_cast<SizeType>(dimY) * static_cast<SizeType>(dimZ)) {
    throw InvalidParameterError();
  }
  // check if indices are non-negative or centered
  bool centeredIndices = false;
  for (int i = 0; i < numValues; ++i) {
    if (xIndices[i * stride] < 0 || yIndices[i * stride] < 0 || zIndices[i * stride] < 0) {
      centeredIndices = true;
      break;
    }
  }

  const int maxX = (hermitianSymmetry || centeredIndices ? dimX / 2 + 1 : dimX) - 1;
  const int maxY = (centeredIndices ? dimY / 2 + 1 : dimY) - 1;
  const int maxZ = (centeredIndices ? dimZ / 2 + 1 : dimZ) - 1;
  const int minX = hermitianSymmetry ? 0 : maxX - dimX + 1;
  const int minY = maxY - dimY + 1;
  const int minZ = maxZ - dimZ + 1;

  // check if indices are inside bounds
  for (int i = 0; i < numValues; ++i) {
    if (xIndices[i * stride] < minX || xIndices[i * stride] > maxX) throw InvalidIndicesError();
    if (yIndices[i * stride] < minY || yIndices[i * stride] > maxY) throw InvalidIndicesError();
    if (zIndices[i * stride] < minZ || zIndices[i * stride] > maxZ) throw InvalidIndicesError();
  }

  // store all unique xy index pairs in an ordered container
  std::map<int, int> sortedXYIndices;  // key = index in xy-plane, value = stick index
  for (int i = 0; i < numValues; ++i) {
    const auto x = to_storage_index(dimX, xIndices[i * stride]);
    const auto y = to_storage_index(dimY, yIndices[i * stride]);

    sortedXYIndices[x * dimY + y] = 0;
  }

  // assign z-stick indices
  int count = 0;
  for (auto& pair : sortedXYIndices) {
    pair.second = count;
    ++count;
  }

  // store index for each element. Each z-stick is continous
  std::vector<int> valueIndices;
  valueIndices.reserve(numValues);
  for (int i = 0; i < numValues; ++i) {
    const auto x = to_storage_index(dimX, xIndices[i * stride]);
    const auto y = to_storage_index(dimY, yIndices[i * stride]);
    const auto z = to_storage_index(dimZ, zIndices[i * stride]);

    valueIndices.emplace_back(sortedXYIndices[x * dimY + y] * dimZ + z);
  }

  // store ordered unique xy-index pairs
  std::vector<int> stickIndices;
  stickIndices.reserve(sortedXYIndices.size());
  for (auto& pair : sortedXYIndices) {
    stickIndices.emplace_back(pair.first);
  }

  return {std::move(valueIndices), std::move(stickIndices)};
}

}  // namespace spfft

#endif
