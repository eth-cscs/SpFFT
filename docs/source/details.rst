Details
=======

Complex Number Format
---------------------
SpFFT always assumes an interleaved format in double or single precision. The alignment of memory provided for space domain data is guaranteed to fulfill to the requirements for std::complex (for C++11), C complex types and GPU complex types of CUDA or ROCM.


Indexing
--------
The only format for providing the indices of the sparse frequency domain data supported at the moment are index triplets in an interleaved array.
Example: x\ :sub:`1`\ , y\ :sub:`1`\ , z\ :sub:`1`\, x\ :sub:`2`\ , y\ :sub:`2`\ , z\ :sub:`2`\ ...

Indices for a dimension of size *n* must be either in the interval [0, *n* - 1] or [floor(*n*/2) - *n* + 1, floor(*n*/2)].

.. note:: For R2C transforms, the full hermitian symmetry property is exploited. All indices in X must always be in the interval [0, floor(*n*/2)], an some other index combinations (where one or two indices are 0) can be ommitted without loss of information.


Data Distribution
-----------------
| The order and distribution of frequency space elements can have significant impact on performance.
| Z-coloumns must *not* be split between MPI ranks. Locally, elements are best grouped by z-columns and ordered by their z-index within each column.

| The ideal distribution of z-columns between MPI ranks differs for execution on host and GPU.

| For execution on host:
|    Indices of z-columns are ideally continuous in y on each MPI rank.

| For execution on GPU:
|    Indices of z-columns are ideally continuous in x on each MPI rank.

MPI Exchange
------------
The MPI exchange is based on a collective MPI call. The following options are available:

SPFFT_EXCH_BUFFERED
 Exchange with MPI_Alltoall. Requires repacking of data into buffer. Possibly best optimized for large number of ranks by MPI implementations, but does not adjust well to non-uniform data distributions.

SPFFT_EXCH_COMPACT_BUFFERED
  Exchange with MPI_Alltoallv. Requires repacking of data into buffer. Performance is usually close to MPI_alltoall and it adapts well to non-unitform data distributions.

SPFFT_EXCH_UNBUFFERED
  Exchange with MPI_Alltoallw. Does not require repacking of data into buffer (outside of the MPI library). Performance varies widely between systems and MPI implementations. It is generally difficult to optimize for large number of ranks, but may perform best in certain conditions.

| For both *SPFFT_EXCH_BUFFERED* and *SPFFT_EXCH_COMPACT_BUFFERED*, an exchange in single precision can be selected. With transforms in double precision, the number of bytes sent and received is halved. For execution on GPUs without GPUDirect, the data transfer between GPU and host also benefits. This option can provide a significant speedup, but incurs a slight accuracy loss. The double precision values are converted to and from single precision between the transform in z and the transform in x / y, while all actual calculations are still done in the selected precision.


GPU
---
| Saving transfer time between host and GPU is key to good performance for execution with GPUs. Ideally, both input and output is located on GPU memory. If host memory pointers are provided as input or output, it is helpful to use pinned memory through the CUDA or ROCm API.

| If available, GPU aware MPI can be utilized, to safe on the otherwise required transfers between host and GPU in preparation of the MPI exchange. This can greatly impact performance and is enabled by compiling the library with the CMake option SPFFT_GPU_DIRECT set to ON.

.. note:: Additional environment variables may have to be set for some MPI implementations, to allow GPUDirect usage.
