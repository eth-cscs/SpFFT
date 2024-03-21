Installation
============

Requirements
------------

* C++ Compiler with C++17 support. Supported compilers are:

  * GCC 7 and later

  * Clang 5 and later

  * ICC 19.0 and later

* CMake 3.18 and later (3.21 for ROCm)

* Library providing a FFTW 3.x interface (FFTW3 or Intel MKL)

* For multi-threading: OpenMP support by the compiler

* For compilation with GPU support:

  * CUDA 11.0 and later for Nvidia hardware

  * ROCm 5.0 and later for AMD hardware


Build
-----

The build system follows the standard CMake workflow. 
Example:

.. code-block:: bash

	mkdir build
	cd build
	cmake .. -DSPFFT_OMP=ON -DSPFFT_MPI=ON -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
	make -j8 install


NOTE: When compiling with CUDA or ROCM (HIP), the standard `CMAKE_CUDA_ARCHITECTURES` and `CMAKE_HIP_ARCHITECTURES` options should be defined as well. `HIP_HCC_FLAGS` is no longer in use.


CMake options
-------------
====================== ======= =============================================================
Option                 Default Description
====================== ======= =============================================================
SPFFT_MPI              ON      Enable MPI support
SPFFT_OMP              ON      Enable multi-threading with OpenMP
SPFFT_GPU_BACKEND      OFF     Select GPU backend. Can be OFF, CUDA or ROCM
SPFFT_GPU_DIRECT       OFF     Use GPU aware MPI with GPUDirect
SPFFT_SINGLE_PRECISION OFF     Enable single precision support
SPFFT_STATIC           OFF     Build as static library
SPFFT_FFTW_LIB         AUTO    Library providing a FFTW interface. Can be AUTO, MKL or FFTW
SPFFT_BUILD_TESTS      OFF     Build test executables for developement purposes
SPFFT_INSTALL          ON      Add library to install target
SPFFT_FORTRAN          OFF     Build Fortran interface module
====================== ======= ============================================================
