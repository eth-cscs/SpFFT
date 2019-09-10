Installation
============

Requirements
------------
* C++ Compiler with C++11 support. Supported compilers are:

  * GCC 6 and later
  * Clang 5 and later
  * ICC 18.0 and later


- CMake 3.11 and later
- Library providing a FFTW 3.x interface (FFTW3 or Intel MKL)
- For multi-threading: OpenMP support by the compiler
- For compilation with GPU support:

  * CUDA 9.0 and later for Nvidia hardware
  * ROCm 2.6 and later for AMD hardware


Build
-----

The build system follows the standard CMake workflow. 
Example:

.. code-block:: bash

	mkdir build
	cd build
	cmake .. -DSPFFT_OMP=ON -DSPFFT_MPI=ON -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
	make -j8 install

CMake options
-------------
====================== ======= ================================================
Option                 Default Description
====================== ======= ================================================
SPFFT_MPI              ON      Enable MPI support
SPFFT_OMP              ON      Enable multi-threading with OpenMP
SPFFT_GPU_BACKEND      OFF     Select GPU backend. Can be OFF, CUDA or ROCM
SPFFT_GPU_DIRECT       OFF     Use GPU aware MPI with GPUDirect
SPFFT_SINGLE_PRECISION OFF     Enable single precision support
SPFFT_STATIC           OFF     Build as static library
SPFFT_BUILD_TESTS      OFF     Build test executables for developement purposes
SPFFT_INSTALL          ON      Add library to install target
====================== ======= ================================================
