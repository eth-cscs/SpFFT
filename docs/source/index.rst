.. Copyright (c) 2019, ETH Zurich

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

SpFFT Documentation
===================
| SpFFT is a library for the computation 3D FFTs with sparse frequency domain data written in C++ with support for MPI, OpenMP, CUDA and ROCm.

| It was originally intended for transforms of data with spherical cutoff in frequency domain, as required by some computational material science codes, but was generalized to sparse frequency domain data.


Design Goals
------------

- Sparse frequency domain input
- Reuse of pre-allocated memory
- Support of negative indexing for frequency domain data
- Unified interface for calculations on CPUs and GPUs
- Support of Complex-To-Real and Real-To-Complex transforms, where the full hermitian symmetry property is utilized. Therefore, there is no redundant frequency domain data, as is usually the case for dense 3D R2C / C2R transforms with libraries such as FFTW.
- C++, C and Fortran interfaces

Interface Design
----------------
To allow for pre-allocation and reuse of memory, the design is based on two classes:

- **Grid**: Allocates memory for transforms up to a given size in each dimension.
- **Transform**: Is created using a *Grid* and can have any size up to the maximum allowed by the *Grid*. A *Transform* holds a counted reference to the underlying *Grid*. Therefore, *Transforms* created from the same *Grid* will share the memory, which is only freed, once the *Grid* and all associated *Transforms* are destroyed.

The user provides memory for storing the sparse frequency domain data, while a *Transform* provides memory for the space domain data. This implies, that executing a *Transform* will override the space domain data of all other *Transforms* associated to the same *Grid*.

.. note::
   The creation of Grids and Transforms, as well as the forward and backward execution may entail MPI calls and must be synchronized between all ranks.


.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   examples
   details

.. toctree::
   :maxdepth: 2
   :caption: C++ API REFERENCE:
   :hidden:

   types
   grid
   grid_float
   transform
   transform_float
   multi_transform
   exceptions

.. toctree::
   :maxdepth: 2
   :caption: C API REFERENCE:
   :hidden:

   types
   grid_c
   grid_float_c
   transform_c
   transform_float_c
   multi_transform_c
   errors_c






.. Indices and tables
.. ==================

.. * :ref:`genindex`


