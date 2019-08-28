# SpFFT
SpFFT is a library for the computation 3D FFTs with sparse frequency domain data written in C++ with support for MPI, OpenMP, CUDA and ROCm.

It was originally intended for transforms of data with spherical cutoff in frequency domain, as required by some computational material science codes, but was generalized to sparse frequency domain data.


### Design Goals
- Sparse frequency domain input
- Reuse of pre-allocated memory
- Support of negative indexing for frequency domain data
- Unified interface for calculations on CPUs and GPUs
- Support of Complex-To-Real and Real-To-Complex transforms, where the full hermitian symmetry property is utilized. Therefore, there is no redundant frequency domain data, as is usually the case for dense 3D R2C / C2R transforms with libraries such as FFTW.
- C++, C and Fortran interfaces

### Interface Design
To allow for pre-allocation and reuse of memory, the design is based on two classes:

- **Grid**: Allocates memory for transforms up to a given size in each dimension.
- **Transform**: Is created using a *Grid* and can have any size up to the maximum allowed by the *Grid*. A *Transform* holds a counted reference to the underlying *Grid*. Therefore, *Transforms* created from the same *Grid* will share the memory, which is only freed, once the *Grid* and all associated *Transforms* are destroyed.

The user provides memory for storing the sparse frequency domain data, while a *Transform* provides memory for the space domain data. This implies, that executing a *Transform* will override the space domain data of all other *Transforms* associated to the same *Grid*.

## Documentation
Documentation can be found HERE (TODO).

## Requirements
- C++ Compiler with C++11 support
- CMake version 3.8 or greater
- Library providing a FFTW 3.x interface (FFTW3 or Intel MKL)
- For multi-threading: OpenMP support by the compiler
- For GPU support: CUDA or ROCm

## Installation
The build system follows the standard CMake workflow. Example:
```console
mkdir build
cd build
cmake .. -DSPFFT_OMP=ON -DSPFFT_MPI=ON -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
make -j8 install
```

### CMake options
| Option                 | Default | Description                                      |
|------------------------|---------|--------------------------------------------------|
| SPFFT_MPI              | ON      | Enable MPI support                               |
| SPFFT_OMP              | ON      | Enable multi-threading with OpenMP               |
| SPFFT_GPU_BACKEND      | OFF     | Select GPU backend. Can be OFF, CUDA or ROCM     |
| SPFFT_GPU_DIRECT       | OFF     | Use GPU aware MPI with GPUDirect                 |
| SPFFT_SINGLE_PRECISION | OFF     | Enable single precision support                  |
| SPFFT_STATIC           | OFF     | Build as static library                          |
| SPFFT_BUILD_TESTS      | OFF     | Build test executables for developement purposes |
| SPFFT_INSTALL          | ON      | Add library to install target                    |

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

  const int numThreads = -1; // Use default OpenMP value

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
```

## License

```
Copyright (c) 2019 ETH Zurich, Simon Frasch

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
