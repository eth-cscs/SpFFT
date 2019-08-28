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
#ifndef SPFFT_GENERATE_INDICES_HPP
#define SPFFT_GENERATE_INDICES_HPP

#include <algorithm>
#include <random>
#include <vector>
#include "spfft/config.h"

namespace spfft {
// creates randomly distributed indices for all ranks according to the input distributions
template <typename T>
auto create_value_indices(T& sharedRandGen, const std::vector<double>& zStickDistribution,
                                 const double totalZStickFraction, const double zStickFillFraction,
                                 const int dimX, const int dimY, const int dimZ,
                                 const bool hermitianSymmetry) -> std::vector<std::vector<int>> {
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);
  std::discrete_distribution<int> rankSelectDis(zStickDistribution.begin(),
                                                zStickDistribution.end());

  const double zStickFractionSum =
      std::accumulate(zStickDistribution.begin(), zStickDistribution.end(), 0.0);

  std::vector<std::vector<std::pair<int, int>>> xyIndicesPerRank(zStickDistribution.size());

  const int dimXFreq = hermitianSymmetry ? dimX / 2 + 1 : dimX;
  const int dimYFreq = hermitianSymmetry ? dimY / 2 + 1 : dimY;
  for (int x = 0; x < dimXFreq; ++x) {
    for (int y = 0; y < dimY; ++y) {
      if (!(x == 0 && y >= dimYFreq) && uniformRandDis(sharedRandGen) < totalZStickFraction) {
        const auto selectedRank = rankSelectDis(sharedRandGen);
        xyIndicesPerRank[selectedRank].emplace_back(std::make_pair(x, y));
      }
    }
  }

  const int dimZFreq = hermitianSymmetry ? dimZ / 2 + 1 : dimZ;
  std::vector<std::vector<int>> valueIndices(zStickDistribution.size());
  auto valueIndicesIt = valueIndices.begin();
  for (const auto& xyIndices : xyIndicesPerRank) {
    for (const auto& xyIndex : xyIndices) {
      for (int z = 0; z < dimZ; ++z) {
        // only add half x=0, y=0 stick if hermitian symmetry is used
        if (!(hermitianSymmetry && xyIndex.first == 0 && xyIndex.second == 0 && z >= dimZFreq) &&
            uniformRandDis(sharedRandGen) < zStickFillFraction) {
          valueIndicesIt->emplace_back(xyIndex.first);
          valueIndicesIt->emplace_back(xyIndex.second);
          valueIndicesIt->emplace_back(z);
        }
      }
    }
    ++valueIndicesIt;
  }

  return valueIndices;
}

inline auto center_indices(const int dimX, const int dimY, const int dimZ,
                           std::vector<std::vector<int>>& indicesPerRank) -> void {
  const int positiveSizeX = dimX / 2 + 1;
  const int positiveSizeY = dimY / 2 + 1;
  const int positiveSizeZ = dimZ / 2 + 1;
  for (auto& rankIndices : indicesPerRank) {
	for (std::size_t i = 0; i < rankIndices.size() ; i += 3) {
	  if (rankIndices[i] >= positiveSizeX) rankIndices[i] -= dimX;
	  if (rankIndices[i + 1] >= positiveSizeY) rankIndices[i + 1] -= dimY;
	  if (rankIndices[i + 2] >= positiveSizeZ) rankIndices[i + 2] -= dimZ;
	}
  }
}

// assigns a number of xy planes to the local rank according to the xy plane distribution
inline auto calculate_num_local_xy_planes(const int rank, const int dimZ,
                                          const std::vector<double>& planeRankDistribution) -> int {
  const double planeDistriSum =
      std::accumulate(planeRankDistribution.begin(), planeRankDistribution.end(), 0.0);
  std::vector<int> numXYPlanesPerRank(planeRankDistribution.size());
  for (std::size_t i = 0; i < planeRankDistribution.size(); ++i) {
    numXYPlanesPerRank[i] = planeRankDistribution[i] / planeDistriSum * dimZ;
  }

  int numMissingPlanes =
      dimZ - std::accumulate(numXYPlanesPerRank.begin(), numXYPlanesPerRank.end(), 0);
  for (auto& val : numXYPlanesPerRank) {
    // add missing planes to rank with non-zero number
    if (val > 0 && numMissingPlanes > 0) {
      val += numMissingPlanes;
      numMissingPlanes = 0;
      break;
    }
    // substract extra planes
    if (numMissingPlanes < 0) {
      val -= std::min(val, -numMissingPlanes);
      numMissingPlanes += val;
      if (numMissingPlanes >= 0) {
        numMissingPlanes = 0;
        break;
      }
    }
  }

  // if all ranks have 0 planes, some planes have to be assigned somewhere
  if (numMissingPlanes > 0) {
    numXYPlanesPerRank[0] = numMissingPlanes;
  }
  return numXYPlanesPerRank[rank];
}

} // namespace spfft

#endif

