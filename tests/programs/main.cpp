#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include "CLI/CLI.hpp"
#include "fft/transform_1d_host.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "spfft/config.h"
#include "timing/timing.hpp"

#include "spfft/grid.hpp"
#include "spfft/transform.hpp"

using namespace spfft;

static bool enablePrint = false;

auto print_view_3d(const HostArrayView3D<std::complex<double>>& view, std::string label) -> void {
  if (!enablePrint) return;
  // std::cout << std::scientific;
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  std::cout << " -------------------- " << std::endl;
  std::cout << label << ":" << std::endl;
  for (SizeType idxOuter = 0; idxOuter < view.dim_outer(); ++idxOuter) {
    for (SizeType idxMid = 0; idxMid < view.dim_mid(); ++idxMid) {
      for (SizeType idxInner = 0; idxInner < view.dim_inner(); ++idxInner) {
        const auto& value = view(idxOuter, idxMid, idxInner);
        std::cout << std::setw(8) << std::right << value.real();
        if (std::signbit(value.imag())) {
          std::cout << " - ";
        } else {
          std::cout << " + ";
        }
        std::cout << std::left << std::setw(6) << std::abs(value.imag());
      }
      std::cout << " | ";
    }
    std::cout << std::endl;
  }
  std::cout << " -------------------- " << std::endl;
}

auto print_view_3d_transposed(const HostArrayView3D<std::complex<double>>& view, std::string label)
    -> void {
  if (!enablePrint) return;
  // std::cout << std::scientific;
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  std::cout << " -------------------- " << std::endl;
  std::cout << label << ":" << std::endl;
  for (SizeType idxInner = 0; idxInner < view.dim_inner(); ++idxInner) {
    for (SizeType idxMid = 0; idxMid < view.dim_mid(); ++idxMid) {
      for (SizeType idxOuter = 0; idxOuter < view.dim_outer(); ++idxOuter) {
        const auto& value = view(idxOuter, idxMid, idxInner);
        std::cout << std::setw(8) << std::right << value.real();
        if (std::signbit(value.imag())) {
          std::cout << " - ";
        } else {
          std::cout << " + ";
        }
        std::cout << std::left << std::setw(6) << std::abs(value.imag());
      }
      std::cout << " | ";
    }
    std::cout << std::endl;
  }
  std::cout << " -------------------- " << std::endl;
}

// #define print_view_3d(...)
// #define print_view_3d_transposed(...)

int main(int argc, char** argv) {
  SizeType numRepeats = 1;
  SizeType gridDimSize = 4;

  CLI::App app{"Single node fft test"};
  app.add_option("-n", gridDimSize, "Size of symmetric fft grid in each dimension")->required();
  app.add_option("-r", numRepeats, "Number of repeats")->default_val("1");
  app.add_flag("-p", enablePrint, "Enable print");
  CLI11_PARSE(app, argc, argv);

  SizeType dimX = gridDimSize;
  SizeType dimY = gridDimSize;
  SizeType dimZ = gridDimSize;
  std::vector<int> xyzIndices;

  for (int x = 0; x < static_cast<int>(dimX); ++x) {
    for (int y = 0; y < static_cast<int>(dimY); ++y) {
      for (int z = 0; z < static_cast<int>(dimZ); ++z) {
        xyzIndices.push_back(x);
        xyzIndices.push_back(y);
        xyzIndices.push_back(z);
      }
    }
  }

  // create full 3d freq view
  HostArray<std::complex<double>> array1(dimX * dimY * dimZ);
  auto fftwView = create_3d_view(array1, 0, dimX, dimY, dimZ);
  SizeType counter = 1;
  for (SizeType x = 0; x < dimX; ++x) {
    for (SizeType y = 0; y < dimY; ++y) {
      for (SizeType z = 0; z < dimZ; ++z) {
        fftwView(x, y, z) = std::complex<double>(counter, counter);
        ++counter;
      }
    }
  }

  // store full z-sticks values
  HostArray<std::complex<double>> arrayPacked(dimZ * dimX * dimY);

  SizeType valueIndex = 0;
  for (SizeType i = 0; i < xyzIndices.size(); i += 3, ++valueIndex) {
    arrayPacked(valueIndex) = fftwView(xyzIndices[i], xyzIndices[i + 1], xyzIndices[i + 2]);
  }
  auto freqDomainZ = create_3d_view(arrayPacked, 0, 1, dimX * dimY, dimZ);

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  const auto executionUnit = SpfftProcessingUnitType::SPFFT_PU_GPU;
#else
  const auto executionUnit = SpfftProcessingUnitType::SPFFT_PU_HOST;
#endif

  Grid grid(dimX, dimY, dimZ, dimX * dimY, executionUnit, -1);

  auto transform = grid.create_transform(
      executionUnit, SpfftTransformType::SPFFT_TRANS_C2C, dimX, dimY, dimZ, dimZ,
      xyzIndices.size() / 3, SpfftIndexFormatType::SPFFT_INDEX_TRIPLETS, xyzIndices.data());

  // output initial z stick
  print_view_3d(freqDomainZ, "Freq input");

  auto spaceDomainView = HostArrayView3D<std::complex<double>>(
      reinterpret_cast<std::complex<double>*>(
          transform.space_domain_data(SpfftProcessingUnitType::SPFFT_PU_HOST)),
      dimZ, dimY, dimX, false);
  for (SizeType repeat = 0; repeat < numRepeats; ++repeat) {
    transform.backward(reinterpret_cast<double*>(arrayPacked.data()),
                       SpfftProcessingUnitType::SPFFT_PU_HOST);
    print_view_3d(spaceDomainView, "Real");
    transform.forward(SpfftProcessingUnitType::SPFFT_PU_HOST,
                      reinterpret_cast<double*>(arrayPacked.data()));
  }

  // output final z stick
  print_view_3d(freqDomainZ, "Freq after forward and");

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

  for (SizeType repeat = 0; repeat < numRepeats; ++repeat) {
    HOST_TIMING_START("FFTW 3d backward")
    fftw_execute(plan3DBackward);
    HOST_TIMING_STOP("FFTW 3d backward")

    print_view_3d_transposed(fftwView, "FFTW ref real");

    HOST_TIMING_START("FFTW 3d forward")
    fftw_execute(plan3DForward);
    HOST_TIMING_STOP("FFTW 3d forward")
  }
  print_view_3d(fftwView, "FFTW freq after forward and");

  fftw_destroy_plan(plan3DBackward);
  fftw_destroy_plan(plan3DForward);

  std::cout << ::spfft::timing::GlobalTimer.process().print(
                   {::rt_graph::Stat::Count, ::rt_graph::Stat::Total, ::rt_graph::Stat::Percentage,
                    ::rt_graph::Stat::Mean, ::rt_graph::Stat::Median, ::rt_graph::Stat::Min,
                    ::rt_graph::Stat::Max})
            << std::endl;
  return 0;
}
