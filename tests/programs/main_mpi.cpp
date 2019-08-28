/*
#define PRINT_VAR(variable)                                                                        \
  {                                                                                                \
    int mpiRankPrint, mpiPrintSize;                                                                \
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRankPrint);                                                  \
    MPI_Comm_size(MPI_COMM_WORLD, &mpiPrintSize);                                                  \
    for (int mpiRankPrintIdx = 0; mpiRankPrintIdx < mpiPrintSize; ++mpiRankPrintIdx) {             \
      MPI_Barrier(MPI_COMM_WORLD);                                                                 \
      if (mpiRankPrint != mpiRankPrintIdx) continue;                                               \
      std::cout << "rank " << mpiRankPrint << ", " << #variable << " = " << variable << std::endl; \
    }                                                                                              \
  }
*/

#include <fftw3.h>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include "CLI/CLI.hpp"
#include "memory/host_array.hpp"
#include "spfft/grid.hpp"
#include "spfft/transform.hpp"

#define SPFFT_ENABLE_TIMING 1
#include "timing/timing.hpp"

#include <mpi.h>
#include "memory/array_view_utility.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_init_handle.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "memory/gpu_array.hpp"
#endif

#include <unistd.h> // for MPI debugging
using namespace spfft;

static bool enablePrint = false;

auto print_view_3d(const HostArrayView3D<std::complex<double>>& view, std::string label) -> void {
  if (!enablePrint) return;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r != rank) continue;
    std::stringstream stream;
    // stream << std::scientific;
    stream << std::fixed;
    stream << std::setprecision(1);
    stream << " -------------------- " << std::endl;
    stream << "Rank = " << rank << ", " << label << ":" << std::endl;
    for (SizeType idxOuter = 0; idxOuter < view.dim_outer(); ++idxOuter) {
      for (SizeType idxMid = 0; idxMid < view.dim_mid(); ++idxMid) {
        for (SizeType idxInner = 0; idxInner < view.dim_inner(); ++idxInner) {
          const auto& value = view(idxOuter, idxMid, idxInner);
          stream << std::setw(8) << std::right << value.real();
          if (std::signbit(value.imag())) {
            stream << " - ";
          } else {
            stream << " + ";
          }
          stream << std::left << std::setw(8) << std::abs(value.imag());
        }
        stream << " | ";
      }
      stream << std::endl;
    }
    stream << " -------------------- " << std::endl;
    std::cout << stream.str();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

auto print_view_3d(const HostArrayView3D<double>& view, std::string label) -> void {
  if (!enablePrint) return;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r != rank) continue;
    std::stringstream stream;
    // stream << std::scientific;
    stream << std::fixed;
    stream << std::setprecision(1);
    stream << " -------------------- " << std::endl;
    stream << "Rank = " << rank << ", " << label << ":" << std::endl;
    for (SizeType idxOuter = 0; idxOuter < view.dim_outer(); ++idxOuter) {
      for (SizeType idxMid = 0; idxMid < view.dim_mid(); ++idxMid) {
        for (SizeType idxInner = 0; idxInner < view.dim_inner(); ++idxInner) {
          const auto& value = view(idxOuter, idxMid, idxInner);
          stream << std::setw(8) << std::right << value;
        }
        stream << " | ";
      }
      stream << std::endl;
    }
    stream << " -------------------- " << std::endl;
    std::cout << stream.str();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

auto print_view_3d_transposed(const HostArrayView3D<std::complex<double>>& view, std::string label)
    -> void {
  if (!enablePrint) return;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r != rank) continue;
    std::stringstream stream;
    // stream << std::scientific;
    stream << std::fixed;
    stream << std::setprecision(1);
    stream << " -------------------- " << std::endl;
    stream << label << ":" << std::endl;
    for (SizeType idxInner = 0; idxInner < view.dim_inner(); ++idxInner) {
      for (SizeType idxMid = 0; idxMid < view.dim_mid(); ++idxMid) {
        for (SizeType idxOuter = 0; idxOuter < view.dim_outer(); ++idxOuter) {
          const auto& value = view(idxOuter, idxMid, idxInner);
          stream << std::setw(8) << std::right << value.real();
          if (std::signbit(value.imag())) {
            stream << " - ";
          } else {
            stream << " + ";
          }
          stream << std::left << std::setw(8) << std::abs(value.imag());
        }
        stream << " | ";
      }
      stream << std::endl;
    }
    stream << " -------------------- " << std::endl;
    std::cout << stream.str();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

auto print_view_3d_transposed(const HostArrayView3D<double>& view, std::string label) -> void {
  if (!enablePrint) return;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r != rank) continue;
    std::stringstream stream;
    // stream << std::scientific;
    stream << std::fixed;
    stream << std::setprecision(1);
    stream << " -------------------- " << std::endl;
    stream << label << ":" << std::endl;
    for (SizeType idxInner = 0; idxInner < view.dim_inner(); ++idxInner) {
      for (SizeType idxMid = 0; idxMid < view.dim_mid(); ++idxMid) {
        for (SizeType idxOuter = 0; idxOuter < view.dim_outer(); ++idxOuter) {
          const auto& value = view(idxOuter, idxMid, idxInner);
          stream << std::setw(8) << std::right << value;
        }
        stream << " | ";
      }
      stream << std::endl;
    }
    stream << " -------------------- " << std::endl;
    std::cout << stream.str();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  MPIInitHandle initHandle(argc, argv, true);
  MPICommunicatorHandle comm(MPI_COMM_WORLD);

  // if(comm.rank() == 0) {
  //   std::cout << "PID = " << getpid() << std::endl;
  // }
  // bool waitLoop = comm.rank() == 0;
  // while(waitLoop) {
  //   sleep(5);
  // }

  SizeType numRepeats = 1;
  SizeType gridDimSize = 4;
  bool realToComplex = false;
  auto transformType = SpfftTransformType::SPFFT_TRANS_C2C;
  CLI::App app{"Single node fft test"};
  app.add_option("-n", gridDimSize, "Size of symmetric fft grid in each dimension")->required();
  app.add_option("-r", numRepeats, "Number of repeats")->default_val("1");
  app.add_flag("-p", enablePrint, "Enable print");
  app.add_flag("-s", realToComplex, "Enable realToComplex");
  CLI11_PARSE(app, argc, argv);

  if (realToComplex) {
    transformType = SpfftTransformType::SPFFT_TRANS_R2C;
  }

  SizeType dimX = gridDimSize;
  SizeType dimY = gridDimSize;
  SizeType dimZ = gridDimSize;
  // SizeType dimX = 11;
  // SizeType dimY = 12;
  // SizeType dimZ = 13;

  const SizeType numLocalXYPlanes =
      (dimZ / comm.size()) + (comm.rank() == comm.size() - 1 ? dimZ % comm.size() : 0);
  const SizeType numLocalYIndices =
      (dimY / comm.size()) + (comm.rank() == comm.size() - 1 ? dimY % comm.size() : 0);
  const SizeType numLocalZSticks = dimX * numLocalYIndices;

  // create xy indices
  std::vector<int> xyzIndices;
  int freqDimX = dimX;
  if (realToComplex) {
    freqDimX = dimX / 2 + 1;
  }
  for (int x = 0; x < static_cast<int>(freqDimX); ++x) {
    for (int y = 0; y < static_cast<int>(numLocalYIndices); ++y) {
      for (int z = 0; z < static_cast<int>(dimZ); ++z) {
        if (!realToComplex || !(x == 0 && y + comm.rank() * (dimY / comm.size()) >= dimY / 2 + 1)) {
          xyzIndices.push_back(x);
          xyzIndices.push_back(y + comm.rank() * (dimY / comm.size()));
          xyzIndices.push_back(z);
        }
      }
    }
  }

  // create full 3d freq view
  HostArray<std::complex<double>> array1(dimX * dimY * dimZ);
  auto fftwView = create_3d_view(array1, 0, dimX, dimY, dimZ);
  SizeType counter = 0;
  std::mt19937 randGen;
  std::uniform_real_distribution<double> uniformRandDis;
  for (SizeType x = 0; x < dimX; ++x) {
    for (SizeType y = 0; y < dimY; ++y) {
      for (SizeType z = 0; z < dimZ; ++z) {
        if (realToComplex) {
          // fftwView(x, y, z) = std::complex<double>(counter, 0.0);
          fftwView(x, y, z) = std::complex<double>(uniformRandDis(randGen), 0.0);
        } else {
          fftwView(x, y, z) = std::complex<double>(counter, counter);
        }
        ++counter;
      }
    }
  }

  // store full z-sticks values
  HostArray<std::complex<double>> arrayPacked(dimZ * numLocalZSticks);

  auto freqDomainZ = create_3d_view(arrayPacked, 0, 1, numLocalZSticks, dimZ);
  // output initial z stick

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  const auto executionUnit = SpfftProcessingUnitType::SPFFT_PU_GPU;
#else
  const auto executionUnit = SpfftProcessingUnitType::SPFFT_PU_HOST;
#endif

  Grid grid(dimX, dimY, dimZ, numLocalZSticks, numLocalXYPlanes, executionUnit, -1, MPI_COMM_WORLD,
            SpfftExchangeType::SPFFT_EXCH_BUFFERED);

  auto transform = grid.create_transform(
      executionUnit, transformType, dimX, dimY, dimZ, numLocalXYPlanes, xyzIndices.size() / 3,
      SpfftIndexFormatType::SPFFT_INDEX_TRIPLETS, xyzIndices.data());

  if (realToComplex) {
    auto spaceDomainView =
        HostArrayView3D<double>(transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_HOST),
                                numLocalXYPlanes, dimY, dimX, false);
    for (SizeType x = 0; x < dimX; ++x) {
      for (SizeType y = 0; y < dimY; ++y) {
        for (SizeType z = 0; z < numLocalXYPlanes; ++z) {
          spaceDomainView(z, y, x) = fftwView(x, y, z + comm.rank() * (dimZ / comm.size())).real();
        }
      }
    }
    for (SizeType repeat = 0; repeat < numRepeats; ++repeat) {
      print_view_3d(spaceDomainView, "Real init");
      transform.forward(SpfftProcessingUnitType::SPFFT_PU_HOST,
                        reinterpret_cast<double*>(arrayPacked.data()));
      print_view_3d(freqDomainZ, "Freq");
      transform.backward(reinterpret_cast<double*>(arrayPacked.data()),
                         SpfftProcessingUnitType::SPFFT_PU_HOST);
      print_view_3d(spaceDomainView, "Real after back and forth");
    }
  } else {
    SizeType valueIndex = 0;
    for (SizeType i = 0; i < xyzIndices.size(); i += 3, ++valueIndex) {
      arrayPacked(valueIndex) = fftwView(xyzIndices[i], xyzIndices[i + 1], xyzIndices[i + 2]);
    }
    auto spaceDomainView = HostArrayView3D<std::complex<double>>(
        reinterpret_cast<std::complex<double>*>(
            transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_HOST)),
        numLocalXYPlanes, dimY, dimX, false);

    for (SizeType repeat = 0; repeat < numRepeats; ++repeat) {
      print_view_3d(freqDomainZ, "Freq input");
      transform.backward(reinterpret_cast<double*>(arrayPacked.data()),
                         SpfftProcessingUnitType::SPFFT_PU_HOST);
      print_view_3d(spaceDomainView, "Real");
      transform.forward(SpfftProcessingUnitType::SPFFT_PU_HOST,
                        reinterpret_cast<double*>(arrayPacked.data()));
      print_view_3d(freqDomainZ, "Freq after forward and backward");
    }
  }

  // output final z stick

  // calculate reference

  MPI_Barrier(MPI_COMM_WORLD);

  HOST_TIMING_START("FFTW 3d init backward")
  fftw_plan plan3DBackward =
      fftw_plan_dft_3d(dimX, dimY, dimZ, (fftw_complex*)fftwView.data(),
                       (fftw_complex*)fftwView.data(), FFTW_BACKWARD, FFTW_ESTIMATE);
  HOST_TIMING_STOP("FFTW 3d init backward")

  HOST_TIMING_START("FFTW 3d init forward")
  fftw_plan plan3DForward =
      fftw_plan_dft_3d(dimX, dimY, dimZ, (fftw_complex*)fftwView.data(),
                       (fftw_complex*)fftwView.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  HOST_TIMING_STOP("FFTW 3d init forward")

  if (realToComplex) {
    for (SizeType repeat = 0; repeat < numRepeats; ++repeat) {
      print_view_3d_transposed(fftwView, "FFTW ref real");

      HOST_TIMING_START("FFTW 3d forward")
      fftw_execute(plan3DForward);
      HOST_TIMING_STOP("FFTW 3d forward")

      print_view_3d(fftwView, "FFTW freq");

      HOST_TIMING_START("FFTW 3d backward")
      fftw_execute(plan3DBackward);
      HOST_TIMING_STOP("FFTW 3d backward")

      print_view_3d_transposed(fftwView, "FFTW real after back and forth");
    }
  } else {
    for (SizeType repeat = 0; repeat < numRepeats; ++repeat) {
      HOST_TIMING_START("FFTW 3d backward")
      fftw_execute(plan3DBackward);
      HOST_TIMING_STOP("FFTW 3d backward")

      print_view_3d_transposed(fftwView, "FFTW ref real");

      HOST_TIMING_START("FFTW 3d forward")
      fftw_execute(plan3DForward);
      HOST_TIMING_STOP("FFTW 3d forward")
      print_view_3d(fftwView, "FFTW freq after forward and");
    }
  }

  fftw_destroy_plan(plan3DBackward);
  fftw_destroy_plan(plan3DForward);

  MPI_Barrier(MPI_COMM_WORLD);
  if (comm.rank() == 0) {
    HOST_TIMING_PRINT();
  }

  return 0;
}
