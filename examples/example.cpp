#include <complex>
#include <iostream>
#include <vector>

#include "spfft/spfft.hpp"

int main(int argc, char** argv) {
  const int dimX = 2;
  const int dimY = 2;
  const int dimZ = 2;

  std::cout << "Dimensions: x = " << dimX << ", y = " << dimY << ", z = " << dimZ << std::endl
            << std::endl;

  const int numThreads = -1;  // Use default OpenMP value

  std::vector<std::complex<double>> freqValues;
  freqValues.reserve(dimX * dimY * dimZ);

  std::vector<int> indices;
  indices.reserve(dimX * dimY * dimZ * 3);

  // initialize frequency domain values and indices
  double initValue = 0.0;
  for (int xIndex = 0; xIndex < dimX; ++xIndex) {
    for (int yIndex = 0; yIndex < dimY; ++yIndex) {
      for (int zIndex = 0; zIndex < dimZ; ++zIndex) {
        // init values
        freqValues.emplace_back(initValue, -initValue);

        // add index triplet for value
        indices.emplace_back(xIndex);
        indices.emplace_back(yIndex);
        indices.emplace_back(zIndex);

        initValue += 1.0;
      }
    }
  }

  std::cout << "Input:" << std::endl;
  for (const auto& value : freqValues) {
    std::cout << value.real() << ", " << value.imag() << std::endl;
  }

  // create local Grid. For distributed computations, a MPI Communicator has to be provided
  spfft::Grid grid(dimX, dimY, dimZ, dimX * dimY, SPFFT_PU_HOST, numThreads);

  // create transform
  spfft::Transform transform =
      grid.create_transform(SPFFT_PU_HOST, SPFFT_TRANS_C2C, dimX, dimY, dimZ, dimZ,
                            freqValues.size(), SPFFT_INDEX_TRIPLETS, indices.data());

  // get pointer to space domain data. Alignment is guaranteed to fullfill requirements for
  // std::complex
  std::complex<double>* realValues =
      reinterpret_cast<std::complex<double>*>(transform.space_domain_data(SPFFT_PU_HOST));

  // transform backward
  transform.backward(reinterpret_cast<double*>(freqValues.data()), SPFFT_PU_HOST);

  std::cout << std::endl << "After backward transform:" << std::endl;
  for (int i = 0; i < transform.local_slice_size(); ++i) {
    std::cout << realValues[i].real() << ", " << realValues[i].imag() << std::endl;
  }

  // transform forward
  transform.forward(SPFFT_PU_HOST, reinterpret_cast<double*>(freqValues.data()), SPFFT_NO_SCALING);

  std::cout << std::endl << "After forward transform (without scaling):" << std::endl;
  for (const auto& value : freqValues) {
    std::cout << value.real() << ", " << value.imag() << std::endl;
  }

  return 0;
}
