.. Copyright (c) 2019, ETH Zurich

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

SpFFT Documentation
===================
| SpFFT - A 3D FFT library for sparse frequency domain data written in C++ with support for MPI, OpenMP, CUDA and ROCm.

| Inspired by the need of some computational material science applications with spherical cutoff data in frequency domain, SpFFT provides Fast Fourier Transformations of sparse frequency domain data. For distributed computations with MPI, slab decomposition in space domain and pencil decomposition in frequency domain (sparse data within a pencil / column must be on one rank) is used.

.. figure:: ../images/sparse_to_dense.png
   :align: center
   :width: 70%

   Illustration of a transform, where data on each MPI rank is identified by color.

Design Goals
------------

- Sparse frequency domain input
- Reuse of pre-allocated memory
- Support of negative indexing for frequency domain data
- Parallelization and acceleration are optional
- Unified interface for calculations on CPUs and GPUs
- Support of Complex-To-Real and Real-To-Complex transforms, where the full hermitian symmetry property is utilized
- C++, C and Fortran interfaces

Interface Design
----------------
To allow for pre-allocation and reuse of memory, the design is based on two classes:

- **Grid**: Allocates memory for transforms up to a given size in each dimension.
- **Transform**: Is associated with a *Grid* and can have any size up to the *Grid* dimensions. A *Transform* holds a counted reference to the underlying *Grid*. Therefore, *Transforms* created with the same *Grid* share memory, which is only freed, once the *Grid* and all associated *Transforms* are destroyed.

A transform can be computed in-place and out-of-place. Addtionally, an internally allocated work buffer can optionally be used for input / output of space domain data.

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


