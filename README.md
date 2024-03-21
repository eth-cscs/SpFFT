[![CI](https://github.com/eth-cscs/SpFFT/workflows/CI/badge.svg)](https://github.com/eth-cscs/SpFFT/actions?query=workflow%3ACI)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/spfft.svg?style=flat)](https://anaconda.org/conda-forge/spfft)
[![Documentation](https://readthedocs.org/projects/spfft/badge/?version=latest)](https://spfft.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](https://raw.githubusercontent.com/eth-cscs/SpFFT/master/LICENSE)

# SpFFT
SpFFT - A 3D FFT library for sparse frequency domain data written in C++ with support for MPI, OpenMP, CUDA and ROCm.

Inspired by the need of some computational material science applications with spherical cutoff data in frequency domain, SpFFT provides Fast Fourier Transformations of sparse frequency domain data. For distributed computations with MPI, slab decomposition in space domain and pencil decomposition in frequency domain (sparse data within a pencil / column must be on one rank) is used.

<img src="docs/images/sparse_to_dense.png" alt="" width=70% />

***Fig. 1:*** Illustration of a transform, where data on each MPI rank is identified by color.

### Design Goals
- Sparse frequency domain input
- Reuse of pre-allocated memory
- Support for shifted indexing with centered zero-frequency
- Optional parallelization and GPU acceleration
- Unified interface for calculations on CPUs and GPUs
- Support of Complex-To-Real and Real-To-Complex transforms, where the full hermitian symmetry property is utilized
- C++, C and Fortran interfaces

### Interface Design
To allow for pre-allocation and reuse of memory, the design is based on two classes:

- **Grid**: Provides memory for transforms up to a given size.
- **Transform**: Created with information on sparse input data and is associated with a *Grid*. Maximum size is limited by *Grid* dimensions. Internal reference counting to *Grid* objects guarantee a valid state until *Transform* object destruction.

A transform can be computed in-place and out-of-place. Addtionally, an internally allocated work buffer can optionally be used for input / output of space domain data.

### New Features in v1.0
- Support for externally allocated memory for space domain data including in-place and out-of-place transforms
- Optional asynchronous computation when using GPUs
- Simplified / direct transform handle creation if no resource reuse through grid handles is required

## Documentation
Documentation can be found [here](https://spfft.readthedocs.io/en/latest/).

## Requirements
- C++ Compiler with C++17 support. Supported compilers are:
  - GCC 7 and later
  - Clang 5 and later
  - ICC 19.0 and later
- CMake 3.18 and later (3.21 for ROCm)
- Library providing a FFTW 3.x interface (FFTW3 or Intel MKL)
- For multi-threading: OpenMP support by the compiler
- For compilation with GPU support:
  - CUDA 11.0 and later for Nvidia hardware
  - ROCm 5.0 and later for AMD hardware

## Installation
The build system follows the standard CMake workflow. Example:
```console
mkdir build
cd build
cmake .. -DSPFFT_OMP=ON -DSPFFT_MPI=ON -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
make -j8 install
```

### CMake options
| Option                 | Default | Description                                                  |
|------------------------|---------|--------------------------------------------------------------|
| SPFFT_MPI              | ON      | Enable MPI support                                           |
| SPFFT_OMP              | ON      | Enable multi-threading with OpenMP                           |
| SPFFT_GPU_BACKEND      | OFF     | Select GPU backend. Can be OFF, CUDA or ROCM                 |
| SPFFT_GPU_DIRECT       | OFF     | Use GPU aware MPI with GPUDirect                             |
| SPFFT_SINGLE_PRECISION | OFF     | Enable single precision support                              |
| SPFFT_STATIC           | OFF     | Build as static library                                      |
| SPFFT_FFTW_LIB         | AUTO    | Library providing a FFTW interface. Can be AUTO, MKL or FFTW |
| SPFFT_BUILD_TESTS      | OFF     | Build test executables for developement purposes             |
| SPFFT_INSTALL          | ON      | Add library to install target                                |
| SPFFT_FORTRAN          | OFF     | Build Fortran interface module                               |
| SPFFT_BUNDLED_LIBS     | ON      | Download required libraries for building tests               |

**_NOTE:_**  When compiling with CUDA or ROCM (HIP), the standard `CMAKE_CUDA_ARCHITECTURES` and `CMAKE_HIP_ARCHITECTURES` options should be defined as well. `HIP_HCC_FLAGS` is no longer in use.

## Examples
Further exmples for C++, C and Fortran can be found in the "examples" folder.
```cpp
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
```

## Acknowledgements
This work was supported by:


|![ethz](docs/images/logo_ethz.png) | [**Swiss Federal Institute of Technology in Zurich**](https://www.ethz.ch/) |
|:----:|:----:|
|![cscs](docs/images/logo_cscs.png) | [**Swiss National Supercomputing Centre**](https://www.cscs.ch/)            |
|![max](docs/images/logo_max.png)  | [**MAterials design at the eXascale**](http://www.max-centre.eu) <br> (Horizon2020, grant agreement MaX CoE, No. 824143) |
