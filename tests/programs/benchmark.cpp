#include <algorithm>
#include <chrono>
#include <complex>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "CLI/CLI.hpp"
#include "fft/transform_1d_host.hpp"
#include "memory/host_array.hpp"
#include "spfft/config.h"

#include "nlohmann/json.hpp"
#include "timing/timing.hpp"

#include <mpi.h>
#include "memory/array_view_utility.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#include "util/omp_definitions.hpp"

#include "spfft/grid.hpp"
#include "spfft/multi_transform.hpp"
#include "spfft/transform.hpp"

#include <unistd.h>  // for MPI debugging

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/gpu_array.hpp"
#endif

namespace std {
void to_json(nlohmann::json& j, const std::list<::spfft::timing::TimingResult>& resList) {
  for (const auto& res : resList) {
    if (res.subNodes.empty())
      j[res.identifier] = {{"values", res.timings}};
    else
      j[res.identifier] = {{"values", res.timings}, {"sub-timings", res.subNodes}};
  }
}
}  // namespace std

using namespace spfft;

void run_benchmark(const int dimX, const int dimY, const int dimZ, const int numLocalZSticks,
                   const int numLocalXYPlanes, const SpfftProcessingUnitType executionUnit,
                   const SpfftProcessingUnitType targetUnit, const int numThreads,
                   const SpfftExchangeType exchangeType, const std::vector<int>& indices,
                   const int numRepeats, const int numTransforms, double** freqValuesPTR) {
  std::vector<Transform> transforms;
  for (int t = 0; t < numTransforms; ++t) {
    Grid grid(dimX, dimY, dimZ, numLocalZSticks, numLocalXYPlanes, executionUnit, numThreads,
              MPI_COMM_WORLD, exchangeType);

    auto transform = grid.create_transform(
        executionUnit, SpfftTransformType::SPFFT_TRANS_C2C, dimX, dimY, dimZ, numLocalXYPlanes,
        indices.size() / 3, SpfftIndexFormatType::SPFFT_INDEX_TRIPLETS, indices.data());
    transforms.emplace_back(std::move(transform));
  }
  std::vector<SpfftProcessingUnitType> targetUnits(numTransforms, targetUnit);
  std::vector<SpfftScalingType> scalingTypes(numTransforms, SPFFT_NO_SCALING);

  // run once for warm cache
  {
    HOST_TIMING_SCOPED("Warming")
    multi_transform_backward(transforms.size(), transforms.data(), freqValuesPTR,
                             targetUnits.data());
    multi_transform_forward(transforms.size(), transforms.data(), targetUnits.data(), freqValuesPTR,
                            scalingTypes.data());
  }

  std::string exchName("Compact buffered");
  if (exchangeType == SpfftExchangeType::SPFFT_EXCH_BUFFERED) {
    exchName = "Buffered";
  } else if (exchangeType == SpfftExchangeType::SPFFT_EXCH_UNBUFFERED) {
    exchName = "Unbuffered";
  } else if (exchangeType == SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED_FLOAT) {
    exchName = "Compact buffered float";
  } else if (exchangeType == SpfftExchangeType::SPFFT_EXCH_BUFFERED_FLOAT) {
    exchName = "Buffered float";
  }

  HOST_TIMING_SCOPED(exchName)
  if (numTransforms == 1) {
    for (int repeat = 0; repeat < numRepeats; ++repeat) {
      transforms.front().backward(*freqValuesPTR, targetUnits.front());
      transforms.front().forward(targetUnits.front(), *freqValuesPTR, scalingTypes.front());
    }
  } else {
    for (int repeat = 0; repeat < numRepeats; ++repeat) {
      multi_transform_backward(transforms.size(), transforms.data(), freqValuesPTR,
                               targetUnits.data());
      multi_transform_forward(transforms.size(), transforms.data(), targetUnits.data(),
                              freqValuesPTR, scalingTypes.data());
    }
  }
}

int main(int argc, char** argv) {
  MPIInitHandle initHandle(argc, argv, true);
  MPICommunicatorHandle comm(MPI_COMM_WORLD);

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  // set device for multi-gpu nodes
  int deviceCount = 0;
  gpu::check_status(gpu::get_device_count(&deviceCount));
  if (deviceCount > 1) {
    gpu::check_status(gpu::set_device(comm.rank() % deviceCount));
  }
#endif

  // if(comm.rank() == 0) {
  //   std::cout << "PID = " << getpid() << std::endl;
  // }
  // bool waitLoop = comm.rank() == 0;
  // while(waitLoop) {
  //   sleep(5);
  // }

  int numRepeats = 1;
  int gridDimSize = 4;
  int numTransforms = 1;
  std::string outputFileName;
  std::string exchName;
  std::string procName;
  double sparsity = 1.0;

  CLI::App app{"fft test"};
  app.add_option("-n", gridDimSize, "Size of symmetric fft grid in each dimension")->required();
  app.add_option("-r", numRepeats, "Number of repeats")->required();
  app.add_option("-o", outputFileName, "Output file name")->required();
  app.add_option("-t", numTransforms, "Number of transforms")->default_val("1");
  app.add_option("-s", sparsity, "Sparsity");
  app.add_set("-e", exchName,
              std::set<std::string>{"all", "compact", "compactFloat", "buffered", "bufferedFloat",
                                    "unbuffered"},
              "Exchange type")
      ->required();
  app.add_set("-p", procName, std::set<std::string>{"cpu", "gpu", "gpu-gpu"}, "Processing unit")
      ->required();
  CLI11_PARSE(app, argc, argv);

  int dimX = gridDimSize;
  int dimY = gridDimSize;
  int dimZ = gridDimSize;

  const int numThreads = omp_get_max_threads();

  const SizeType numLocalXYPlanes =
      (dimZ / comm.size()) + (comm.rank() < dimZ % comm.size() ? 1 : 0);
  int numLocalZSticks = 0;

  std::vector<int> xyzIndices;
  {
    // std::mt19937 randGen(42);
    // std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);
    // create all global x-y index pairs
    std::vector<std::pair<int, int>> xyIndicesGlobal;
    xyIndicesGlobal.reserve(dimX * dimY);
    for (int x = 0; x < static_cast<int>(dimX * sparsity); ++x) {
      for (int y = 0; y < static_cast<int>(dimY); ++y) {
        xyIndicesGlobal.emplace_back(x, y);
      }
    }

    // distribute z-sticks as evenly as possible
    numLocalZSticks = (xyIndicesGlobal.size()) / comm.size() +
                      (comm.rank() < (xyIndicesGlobal.size()) % comm.size() ? 1 : 0);
    const int offset =
        ((xyIndicesGlobal.size()) / comm.size()) * comm.rank() +
        std::min(comm.rank(), static_cast<SizeType>(xyIndicesGlobal.size()) % comm.size());

    // assemble index triplets
    xyzIndices.reserve(numLocalZSticks);
    for (int i = offset; i < offset + numLocalZSticks; ++i) {
      for (int z = 0; z < dimZ; ++z) {
        xyzIndices.push_back(xyIndicesGlobal[i].first);
        xyzIndices.push_back(xyIndicesGlobal[i].second);
        xyzIndices.push_back(z);
      }
    }
  }

  // store full z-sticks values

  const auto executionUnit = procName == "cpu" ? SpfftProcessingUnitType::SPFFT_PU_HOST
                                               : SpfftProcessingUnitType::SPFFT_PU_GPU;
  const auto targetUnit = procName == "gpu-gpu" ? SpfftProcessingUnitType::SPFFT_PU_GPU
                                                : SpfftProcessingUnitType::SPFFT_PU_HOST;

  std::vector<double*> freqValuesPointers(numTransforms);
  std::vector<HostArray<std::complex<double>>> freqValues;
  for (int t = 0; t < numTransforms; ++t) freqValues.emplace_back(xyzIndices.size() / 3);
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  std::vector<GPUArray<double2>> freqValuesGPU;
  for (int t = 0; t < numTransforms; ++t) freqValuesGPU.emplace_back(xyzIndices.size() / 3);

  for (int t = 0; t < numTransforms; ++t) {
    freqValuesPointers[t] = procName == "gpu-gpu"
                                ? reinterpret_cast<double*>(freqValuesGPU[t].data())
                                : reinterpret_cast<double*>(freqValues[t].data());
  }
#else
  for (int t = 0; t < numTransforms; ++t) {
    freqValuesPointers[t] = reinterpret_cast<double*>(freqValues[t].data());
  }
#endif

  if (comm.rank() == 0) {
    std::cout << "Num MPI ranks: " << comm.size() << std::endl;
    std::cout << "Grid size: " << dimX << ", " << dimY << ", " << dimZ << std::endl;
    std::cout << "Proc: " << procName << std::endl;
  }

  if (exchName == "all") {
    run_benchmark(dimX, dimY, dimZ, numLocalZSticks, numLocalXYPlanes, executionUnit, targetUnit,
                  numThreads, SpfftExchangeType::SPFFT_EXCH_BUFFERED, xyzIndices, numRepeats,
                  numTransforms, freqValuesPointers.data());
    run_benchmark(dimX, dimY, dimZ, numLocalZSticks, numLocalXYPlanes, executionUnit, targetUnit,
                  numThreads, SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED, xyzIndices,
                  numRepeats, numTransforms, freqValuesPointers.data());
    run_benchmark(dimX, dimY, dimZ, numLocalZSticks, numLocalXYPlanes, executionUnit, targetUnit,
                  numThreads, SpfftExchangeType::SPFFT_EXCH_UNBUFFERED, xyzIndices, numRepeats,
                  numTransforms, freqValuesPointers.data());
  } else {
    auto exchangeType = SpfftExchangeType::SPFFT_EXCH_DEFAULT;
    if (exchName == "compact") {
      exchangeType = SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED;
    } else if (exchName == "compactFloat") {
      exchangeType = SpfftExchangeType::SPFFT_EXCH_COMPACT_BUFFERED_FLOAT;
    } else if (exchName == "buffered") {
      exchangeType = SpfftExchangeType::SPFFT_EXCH_BUFFERED;
    } else if (exchName == "bufferedFloat") {
      exchangeType = SpfftExchangeType::SPFFT_EXCH_BUFFERED_FLOAT;
    } else if (exchName == "unbuffered") {
      exchangeType = SpfftExchangeType::SPFFT_EXCH_UNBUFFERED;
    }

    run_benchmark(dimX, dimY, dimZ, numLocalZSticks, numLocalXYPlanes, executionUnit, targetUnit,
                  numThreads, exchangeType, xyzIndices, numRepeats, numTransforms,
                  freqValuesPointers.data());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (comm.rank() == 0) {
    HOST_TIMING_PRINT();
    if (!outputFileName.empty()) {
      nlohmann::json j;
      const std::time_t t = std::time(nullptr);
      std::string time(std::ctime(&t));
      time.pop_back();
      j["timings"] = HOST_TIMING_PROCESS_TIMINGS();
#ifdef SPFFT_GPU_DIRECT
      const bool gpuDirectEnabled = true;
#else
      const bool gpuDirectEnabled = false;
#endif

      const bool data_on_gpu = procName == "gpu-gpu";
      j["parameters"] = {{"proc", procName},
                         {"data_on_gpu", data_on_gpu},
                         {"gpu_direct", gpuDirectEnabled},
                         {"num_ranks", comm.size()},
                         {"num_threads", numThreads},
                         {"dim_x", dimX},
                         {"dim_y", dimY},
                         {"dim_z", dimZ},
                         {"exchange_type", exchName},
                         {"num_repeats", numRepeats},
                         {"time", time}};
      std::ofstream file(outputFileName);
      file << std::setw(2) << j;
      file.close();
    }
  }

  return 0;
}
