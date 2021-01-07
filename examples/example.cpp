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

  // Use default OpenMP value
  const int numThreads = -1;

  // Use all elements in this example.
  const int numFrequencyElements = dimX * dimY * dimZ;

  // Slice length in space domain. Equivalent to dimZ for non-distributed case.
  const int localZLength = dimZ;

  // Interleaved complex numbers
  std::vector<double> frequencyElements;
  frequencyElements.reserve(2 * numFrequencyElements);

  // Indices of frequency elements
  std::vector<int> indices;
  indices.reserve(dimX * dimY * dimZ * 3);

  // Initialize frequency domain values and indices
  double initValue = 0.0;
  for (int xIndex = 0; xIndex < dimX; ++xIndex) {
    for (int yIndex = 0; yIndex < dimY; ++yIndex) {
      for (int zIndex = 0; zIndex < dimZ; ++zIndex) {
        // init with interleaved complex numbers
        frequencyElements.emplace_back(initValue);
        frequencyElements.emplace_back(-initValue);

        // add index triplet for value
        indices.emplace_back(xIndex);
        indices.emplace_back(yIndex);
        indices.emplace_back(zIndex);

        initValue += 1.0;
      }
    }
  }

  std::cout << "Input:" << std::endl;
  for (int i = 0; i < numFrequencyElements; ++i) {
    std::cout << frequencyElements[2 * i] << ", " << frequencyElements[2 * i + 1] << std::endl;
  }

  // Create local Grid. For distributed computations, a MPI Communicator has to be provided
  spfft::Grid grid(dimX, dimY, dimZ, dimX * dimY, SPFFT_PU_HOST, numThreads);

  // Create transform.
  // Note: A transform handle can be created without a grid if no resource sharing is desired.
  spfft::Transform transform =
      grid.create_transform(SPFFT_PU_HOST, SPFFT_TRANS_C2C, dimX, dimY, dimZ, localZLength,
                            numFrequencyElements, SPFFT_INDEX_TRIPLETS, indices.data());


  ///////////////////////////////////////////////////
  // Option A: Reuse internal buffer for space domain
  ///////////////////////////////////////////////////

  // Transform backward
  transform.backward(frequencyElements.data(), SPFFT_PU_HOST);

  // Get pointer to buffer with space domain data. Is guaranteed to be castable to a valid
  // std::complex pointer. Using the internal working buffer as input / output can help reduce
  // memory usage.
  double* spaceDomainPtr = transform.space_domain_data(SPFFT_PU_HOST);

  std::cout << std::endl << "After backward transform:" << std::endl;
  for (int i = 0; i < transform.local_slice_size(); ++i) {
    std::cout << spaceDomainPtr[2 * i] << ", " << spaceDomainPtr[2 * i + 1] << std::endl;
  }

  /////////////////////////////////////////////////
  // Option B: Use external buffer for space domain
  /////////////////////////////////////////////////

  std::vector<double> spaceDomainVec(2 * transform.local_slice_size());

  // Transform backward
  transform.backward(frequencyElements.data(), spaceDomainVec.data());

  // Transform forward
  transform.forward(spaceDomainVec.data(), frequencyElements.data(), SPFFT_NO_SCALING);

  // Note: In-place transforms are also supported by passing the same pointer for input and output.

  std::cout << std::endl << "After forward transform (without normalization):" << std::endl;
  for (int i = 0; i < numFrequencyElements; ++i) {
    std::cout << frequencyElements[2 * i] << ", " << frequencyElements[2 * i + 1] << std::endl;
  }

  return 0;
}
