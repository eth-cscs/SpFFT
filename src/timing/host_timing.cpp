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

#include "timing/host_timing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace spfft {
namespace timing {

// ======================
// Local helper
// ======================
namespace {
// Helper struct for creating a tree of timings
struct HostTimeStampPair {
  std::string identifier;
  double time = 0.0;
  std::size_t startIdx = 0;
  std::size_t stopIdx = 0;
  TimingResult* nodePtr = nullptr;
};
auto calc_median(const std::vector<double>::iterator& begin,
                 const std::vector<double>::iterator& end) -> double {
  const auto n = end - begin;
  if (n == 0) return *begin;
  if (n % 2 == 0) {
    return (*(begin + n / 2) + *(begin + n / 2 - 1)) / 2.0;
  } else {
    return *(begin + n / 2);
  }
}

auto calculate_statistic(std::vector<double> values)
    -> std::tuple<double, double, double, double, double, double, double> {
  if (values.empty()) return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  std::sort(values.begin(), values.end());

  const double min = values.front();
  const double max = values.back();

  const double median = calc_median(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / values.size();

  const double lowerQuartile = calc_median(values.begin(), values.begin() + values.size() / 2);
  const double upperQuartile = calc_median(
      values.begin() + values.size() / 2 + (values.size() % 2) * (values.size() > 1), values.end());

  return std::make_tuple(sum, mean, median, min, max, lowerQuartile, upperQuartile);
}

// format time input in seconds into string with appropriate unit
auto format_time(const double time_seconds) -> std::string {
  if (time_seconds <= 0.0) return std::string("0 s");

  // time is always greater than 0 here
  const double exponent = std::log10(std::abs(time_seconds));
  const int siExponent = static_cast<int>(std::floor(exponent / 3.0) * 3);

  std::stringstream result;
  result << std::fixed << std::setprecision(2);
  result << time_seconds * std::pow(10.0, static_cast<double>(-siExponent));
  result << " ";
  switch (siExponent) {
    case 24:
      result << "Y";
      break;
    case 21:
      result << "Z";
      break;
    case 18:
      result << "E";
      break;
    case 15:
      result << "P";
      break;
    case 12:
      result << "T";
      break;
    case 9:
      result << "G";
      break;
    case 6:
      result << "M";
      break;
    case 3:
      result << "k";
      break;
    case 0:
      break;
    case -3:
      result << "m";
      break;
    case -6:
      result << "u";
      break;
    case -9:
      result << "n";
      break;
    case -12:
      result << "p";
      break;
    case -15:
      result << "f";
      break;
    case -18:
      result << "a";
      break;
    case -21:
      result << "z";
      break;
    case -24:
      result << "y";
      break;
    default:
      result << "?";
  }
  result << "s";
  return result.str();
}

// print timing nodes in tree recursively
auto print_node(const std::size_t identifierSpace, const std::string& nodePrefix,
                const TimingResult& node, const bool isSubNode, const double parentTotTime)
    -> void {
  double sum, mean, median, min, max, lowerQuartile, upperQuartile;
  std::tie(sum, mean, median, min, max, lowerQuartile, upperQuartile) =
      calculate_statistic(node.timings);

  const double percentage =
      (parentTotTime < sum || parentTotTime == 0) ? 100.0 : sum / parentTotTime * 100.0;

  std::stringstream percentageStream;
  percentageStream << std::fixed << std::setprecision(2) << percentage;

  std::cout << std::left << std::setw(identifierSpace);
  if (isSubNode)
    std::cout << nodePrefix + "- " + node.identifier;
  else
    std::cout << nodePrefix + node.identifier;
  std::cout << std::right << std::setw(8) << node.timings.size();
  std::cout << std::right << std::setw(15) << format_time(sum);
  std::cout << std::right << std::setw(15) << percentageStream.str();
  std::cout << std::right << std::setw(15) << format_time(mean);
  std::cout << std::right << std::setw(15) << format_time(median);
  std::cout << std::right << std::setw(15) << format_time(min);
  std::cout << std::right << std::setw(15) << format_time(max);
  std::cout << std::endl;

  for (const auto& subNode : node.subNodes) {
    print_node(identifierSpace, nodePrefix + std::string(" |"), subNode, true, sum);
  }
}

// determine length of padding required for printing entire tree identifiers recursively
auto max_node_identifier_length(const TimingResult& node, const std::size_t recursionDepth,
                                const std::size_t addPerLevel, const std::size_t parentMax)
    -> std::size_t {
  std::size_t currentLength = node.identifier.length() + recursionDepth * addPerLevel;
  std::size_t max = currentLength > parentMax ? currentLength : parentMax;
  for (const auto& subNode : node.subNodes) {
    const std::size_t subMax =
        max_node_identifier_length(subNode, recursionDepth + 1, addPerLevel, max);
    if (subMax > max) max = subMax;
  }

  return max;
}

auto export_node_json(const std::string& padding, const std::list<TimingResult>& nodeList,
                      std::stringstream& stream) -> void {
  stream << "{" << std::endl;
  const std::string nodePadding = padding + "  ";
  const std::string subNodePadding = nodePadding + "  ";
  for (const auto& node : nodeList) {
    stream << nodePadding << "\"" << node.identifier << "\" : {" << std::endl;
    stream << subNodePadding << "\"timings\" : [";
    for (const auto& value : node.timings) {
      stream << value;
      if (&value != &(node.timings.back())) stream << ", ";
    }
    stream << "]," << std::endl;
    stream << subNodePadding << "\"sub-timings\" : ";
    export_node_json(subNodePadding, node.subNodes, stream);
    stream << nodePadding << "}";
    if (&node != &(nodeList.back())) stream << ",";
    stream << std::endl;
  }
  stream << padding << "}" << std::endl;
}
}  // namespace

// ======================
// HostTiming
// ======================
auto HostTiming::process_timings() -> std::list<TimingResult> {
  std::list<TimingResult> results;

  std::vector<HostTimeStampPair> timePairs;
  timePairs.reserve(timeStamps_.size() / 2);

  // create pairs of start / stop timings
  for (std::size_t i = 0; i < timeStamps_.size(); ++i) {
    if (timeStamps_[i].type == TimeStampType::Start) {
      HostTimeStampPair pair;
      pair.startIdx = i;
      pair.identifier = std::string(timeStamps_[i].identifierPtr);
      std::size_t numInnerMatchingIdentifiers = 0;
      // search for matching stop after start
      for (std::size_t j = i + 1; j < timeStamps_.size(); ++j) {
        // only consider matching identifiers
        if (std::string(timeStamps_[j].identifierPtr) ==
            std::string(timeStamps_[i].identifierPtr)) {
          if (timeStamps_[j].type == TimeStampType::Stop && numInnerMatchingIdentifiers == 0) {
            // Matching stop found
            std::chrono::duration<double> duration = timeStamps_[j].time - timeStamps_[i].time;
            pair.time = duration.count();
            pair.stopIdx = j;
            timePairs.push_back(pair);
            if (pair.time < 0) {
              std::cerr
                  << "WARNING: Host Timing -> Measured time is negative. Non-steady system-clock?!"
                  << std::endl;
            }
            break;
          } else if (timeStamps_[j].type == TimeStampType::Stop &&
                     numInnerMatchingIdentifiers > 0) {
            // inner stop with matching identifier
            --numInnerMatchingIdentifiers;
          } else if (timeStamps_[j].type == TimeStampType::Start) {
            // inner start with matching identifier
            ++numInnerMatchingIdentifiers;
          }
        }
      }
      if (pair.stopIdx == 0) {
        std::cerr << "WARNING: Host Timing -> Start / stop time stamps do not match for \""
                  << timeStamps_[i].identifierPtr << "\"!" << std::endl;
      }
    }
  }

  // create tree of timings where sub-nodes represent timings fully enclosed by another start / stop
  // pair Use the fact that timePairs is sorted by startIdx
  for (std::size_t i = 0; i < timePairs.size(); ++i) {
    auto& pair = timePairs[i];

    // find potential parent by going backwards through pairs, starting with the current pair
    // position
    for (auto timePairIt = timePairs.rbegin() + (timePairs.size() - i);
         timePairIt != timePairs.rend(); ++timePairIt) {
      if (timePairIt->stopIdx > pair.stopIdx && timePairIt->nodePtr != nullptr) {
        auto& parentNode = *(timePairIt->nodePtr);
        // check if sub-node with identifier exists
        bool nodeFound = false;
        for (auto& subNode : parentNode.subNodes) {
          if (subNode.identifier == pair.identifier) {
            nodeFound = true;
            subNode.timings.push_back(pair.time);
            // mark node position in pair for finding sub-nodes
            pair.nodePtr = &(subNode);
            break;
          }
        }
        if (!nodeFound) {
          // create new sub-node
          TimingResult newNode;
          newNode.identifier = pair.identifier;
          newNode.timings.push_back(pair.time);
          parentNode.subNodes.push_back(std::move(newNode));
          // mark node position in pair for finding sub-nodes
          pair.nodePtr = &(parentNode.subNodes.back());
        }
        break;
      }
    }

    // No parent found, must be top level node
    if (pair.nodePtr == nullptr) {
      // Check if top level node with same name exists
      for (auto& topNode : results) {
        if (topNode.identifier == pair.identifier) {
          topNode.timings.push_back(pair.time);
          pair.nodePtr = &(topNode);
          break;
        }
      }
    }

    // New top level node
    if (pair.nodePtr == nullptr) {
      TimingResult newNode;
      newNode.identifier = pair.identifier;
      newNode.timings.push_back(pair.time);
      // newNode.parent = nullptr;
      results.push_back(std::move(newNode));

      // mark node position in pair for finding sub-nodes
      pair.nodePtr = &(results.back());
    }
  }

  return results;
}

auto HostTiming::print_timings() -> void {
  auto timings = process_timings();
  // calculate space for printing identifiers
  std::size_t identifierSpace = 0;
  for (const auto& node : timings) {
    const auto nodeMax = max_node_identifier_length(node, 0, 2, identifierSpace);
    if (nodeMax > identifierSpace) identifierSpace = nodeMax;
  }
  identifierSpace += 3;

  const auto totalLength = identifierSpace + 8 + 6 * 15;
  std::cout << std::string(totalLength, '=') << std::endl;

  // header
  std::cout << std::right << std::setw(identifierSpace + 8) << "#";
  std::cout << std::right << std::setw(15) << "Total";
  std::cout << std::right << std::setw(15) << "%";
  std::cout << std::right << std::setw(15) << "Mean";
  std::cout << std::right << std::setw(15) << "Median";
  std::cout << std::right << std::setw(15) << "Min";
  std::cout << std::right << std::setw(15) << "Max";
  std::cout << std::endl;

  std::cout << std::string(totalLength, '-') << std::endl;

  // print all timings
  for (auto& node : timings) {
    print_node(identifierSpace, std::string(), node, false, 0.0);
    std::cout << std::endl;
  }
  std::cout << std::string(totalLength, '=') << std::endl;
}

auto HostTiming::export_json() -> std::string {
  auto nodeList = process_timings();
  std::stringstream jsonStream;
  jsonStream << std::scientific;
  export_node_json("", nodeList, jsonStream);
  return jsonStream.str();
}

}  // namespace timing
}  // namespace spfft
