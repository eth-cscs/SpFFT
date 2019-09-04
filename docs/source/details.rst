Details
=======

Complex Number Format
---------------------
SpFFT always assumes an interleaved format in double or single precision. The alignment of memory provided for space domain data is guaranteed to fulfill to the requirements for std::complex (for C++11), C complex types and GPU complex types of CUDA or ROCM.

Indexing
--------
| The three dimensions are referred to as :math:`x, y` and :math:`z`. An element in space domain is addressed in memory as:

| :math:`(z \cdot n_y + y) \cdot n_x + x`

| For now, the only supported format for providing the indices of sparse frequency domain data are index triplets in an interleaved array.
| Example: :math:`x_1, y_1, z_1, x_2, y_2, z_2, ...`

Indices for a dimension of size *n* must be either in the interval :math:`[0, n - 1]` or :math:`\left [ \left \lfloor \frac{n}{2} \right \rfloor - n + 1, \left \lfloor \frac{n}{2} \right \rfloor \right ]`. For Real-To-Complex transforms additional restrictions apply (see next section).

Real-To-Complex Transforms
--------------------------
| The Discrete Fourier Transform :math:`f(x, y, z)` of a real valued function is hermitian:

| :math:`f(x, y, z) = f^*(-x, -y, -z)`

| Due to this property, only about half the frequency domain data is required without loss of information. Therefore, similar to other FFT libraries, all indices in :math:`x` *must* be in the interval  :math:`\left [ 0, \left \lfloor \frac{n}{2} \right \rfloor \right ]`.
To fully utlize the symmetry property, the following steps can be followed:

- Only non-redundent z-coloumns on the y-z plane at :math:`x = 0` have to be provided. A z-coloumn must be complete and can be provided at either :math:`y` or :math:`-y`.
- All redundant values in the z-coloumn at :math:`x = 0`, :math:`y = 0` can be omitted.

Normalization
-------------
Normalization is only available for the forward transform with a scaling factor of :math:`\frac{1}{n_x n_y n_z}`. Applying a forward and backwards transform with scaling enabled will therefore yield identical output (within numerical accuracy).


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
  Exchange with MPI_Alltoallv. Requires repacking of data into buffer. Performance is usually close to MPI_alltoall and it adapts well to non-uniform data distributions.

SPFFT_EXCH_UNBUFFERED
  Exchange with MPI_Alltoallw. Does not require repacking of data into buffer (outside of the MPI library). Performance varies widely between systems and MPI implementations. It is generally difficult to optimize for large number of ranks, but may perform best in certain conditions.

| For both *SPFFT_EXCH_BUFFERED* and *SPFFT_EXCH_COMPACT_BUFFERED*, an exchange in single precision can be selected. With transforms in double precision, the number of bytes sent and received is halved. For execution on GPUs without GPUDirect, the data transfer between GPU and host also benefits. This option can provide a significant speedup, but incurs a slight accuracy loss. The double precision values are converted to and from single precision between the transform in z and the transform in x / y, while all actual calculations are still done in the selected precision.


GPU
---
| Saving transfer time between host and GPU is key to good performance for execution with GPUs. Ideally, both input and output is located on GPU memory. If host memory pointers are provided as input or output, it is helpful to use pinned memory through the CUDA or ROCm API.

| If available, GPU aware MPI can be utilized, to safe on the otherwise required transfers between host and GPU in preparation of the MPI exchange. This can greatly impact performance and is enabled by compiling the library with the CMake option SPFFT_GPU_DIRECT set to ON.

.. note:: Additional environment variables may have to be set for some MPI implementations, to allow GPUDirect usage.
.. note:: The execution of a transform is synchronized with the default stream.
