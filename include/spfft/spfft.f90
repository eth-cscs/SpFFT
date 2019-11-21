
!  Copyright (c) 2019 ETH Zurich, Simon Frasch
!
!  Redistribution and use in source and binary forms, with or without
!  modification, are permitted provided that the following conditions are met:
!
!  1. Redistributions of source code must retain the above copyright notice,
!     this list of conditions and the following disclaimer.
!  2. Redistributions in binary form must reproduce the above copyright
!     notice, this list of conditions and the following disclaimer in the
!     documentation and/or other materials provided with the distribution.
!  3. Neither the name of the copyright holder nor the names of its contributors
!     may be used to endorse or promote products derived from this software
!     without specific prior written permission.
!
!  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
!  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
!  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
!  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
!  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
!  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
!  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!  POSSIBILITY OF SUCH DAMAGE.

module spfft

use iso_c_binding
implicit none

! Constants
integer(c_int), parameter ::                  &
    SPFFT_EXCH_DEFAULT                  = 0,  &
    SPFFT_EXCH_BUFFERED                 = 1,  &
    SPFFT_EXCH_BUFFERED_FLOAT           = 2,  &
    SPFFT_EXCH_COMPACT_BUFFERED         = 3,  &
    SPFFT_EXCH_COMPACT_BUFFERED_FLOAT   = 4,  &
    SPFFT_EXCH_UNBUFFERED               = 5,  &

    SPFFT_PU_HOST                       = 1,  &
    SPFFT_PU_GPU                        = 2,  &

    SPFFT_INDEX_TRIPLETS                = 0,  &

    SPFFT_TRANS_C2C                     = 0,  &
    SPFFT_TRANS_R2C                     = 1,  &

    SPFFT_NO_SCALING                    = 0,  &
    SPFFT_FULL_SCALING                  = 1,  &

    SPFFT_SUCCESS                       = 0,  &
    SPFFT_UNKNOWN_ERROR                 = 1,  &
    SPFFT_INVALID_HANDLE_ERROR          = 2,  &
    SPFFT_OVERFLOW_ERROR                = 3,  &
    SPFFT_ALLOCATION_ERROR              = 4,  &
    SPFFT_INVALID_PARAMETER_ERROR       = 5,  &
    SPFFT_DUPLICATE_INDICES_ERROR       = 6,  &
    SPFFT_INVALID_INDICES_ERROR         = 7,  &
    SPFFT_MPI_SUPPORT_ERROR             = 8,  &
    SPFFT_MPI_ERROR                     = 9,  &
    SPFFT_MPI_PARAMETER_MISMATCH_ERROR  = 10, &
    SPFFT_HOST_EXECUTION_ERROR          = 11, &
    SPFFT_FFTW_ERROR                    = 12, &
    SPFFT_GPU_ERROR                     = 13, &
    SPFFT_GPU_PRECEDING_ERROR           = 14, &
    SPFFT_GPU_SUPPORT_ERROR             = 15, &
    SPFFT_GPU_ALLOCATION_ERROR          = 16, &
    SPFFT_GPU_LAUNCH_ERROR              = 17, &
    SPFFT_GPU_NO_DEVICE_ERROR           = 18, &
    SPFFT_GPU_INVALID_VALUE_ERROR       = 19, &
    SPFFT_GPU_INVALID_DEVICE_PTR_ERROR  = 20, &
    SPFFT_GPU_COPY_ERROR                = 21, &
    SPFFT_GPU_FFT_ERROR                 = 22

interface
  !--------------------------
  !          Grid
  !--------------------------
  integer(c_int) function spfft_grid_create(grid, maxDimX, maxDimY, maxDimZ, &
      maxNumLocalZColumns, processingUnit, maxNumThreads) bind(C)
    use iso_c_binding
    type(c_ptr), intent(out) :: grid
    integer(c_int), value :: maxDimX
    integer(c_int), value :: maxDimY
    integer(c_int), value :: maxDimZ
    integer(c_int), value :: maxNumLocalZColumns
    integer(c_int), value :: processingUnit
    integer(c_int), value :: maxNumThreads
  end function

  integer(c_int) function spfft_grid_create_distributed(grid, maxDimX, maxDimY, maxDimZ, &
      maxNumLocalZColumns, maxLocalZLength, processingUnit, maxNumThreads,&
      comm, exchangeType) bind(C, name='spfft_grid_create_distributed_fortran')
    use iso_c_binding
    type(c_ptr), intent(out) :: grid
    integer(c_int), value :: maxDimX
    integer(c_int), value :: maxDimY
    integer(c_int), value :: maxDimZ
    integer(c_int), value :: maxNumLocalZColumns
    integer(c_int), value :: maxLocalZLength
    integer(c_int), value :: processingUnit
    integer(c_int), value :: maxNumThreads
    integer(c_int), value :: comm
    integer(c_int), value :: exchangeType
  end function

  integer(c_int) function spfft_grid_destroy(grid) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
  end function

  integer(c_int) function spfft_grid_max_dim_x(grid, dimX) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: dimX
  end function

  integer(c_int) function spfft_grid_max_dim_y(grid, dimY) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: dimY
  end function

  integer(c_int) function spfft_grid_max_dim_z(grid, dimZ) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: dimZ
  end function

  integer(c_int) function spfft_grid_max_num_local_z_columns(grid, maxNumLocalZColumns) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: maxNumLocalZColumns
  end function

  integer(c_int) function spfft_grid_max_local_z_length(grid, maxLocalZLength) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: maxLocalZLength
  end function

  integer(c_int) function spfft_grid_processing_unit(grid, processingUnit) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: processingUnit
  end function

  integer(c_int) function spfft_grid_device_id(grid, deviceId) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: deviceId
  end function

  integer(c_int) function spfft_grid_num_threads(grid, numThreads) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: numThreads
  end function

  integer(c_int) function spfft_grid_communicator(grid, comm) &
      bind(C, name="spfft_grid_communicator_fortran")
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: comm
  end function

  !--------------------------
  !        Grid Float
  !--------------------------
  integer(c_int) function spfft_float_grid_create(grid, maxDimX, maxDimY, maxDimZ, &
      maxNumLocalZColumns, processingUnit, maxNumThreads) bind(C)
    use iso_c_binding
    type(c_ptr), intent(out) :: grid
    integer(c_int), value :: maxDimX
    integer(c_int), value :: maxDimY
    integer(c_int), value :: maxDimZ
    integer(c_int), value :: maxNumLocalZColumns
    integer(c_int), value :: processingUnit
    integer(c_int), value :: maxNumThreads
  end function

  integer(c_int) function spfft_float_grid_create_distributed(grid, maxDimX, maxDimY, maxDimZ, &
      maxNumLocalZColumns, maxLocalZLength, processingUnit, maxNumThreads,&
      comm, exchangeType) bind(C, name='spfft_float_grid_create_distributed_fortran')
    use iso_c_binding
    type(c_ptr), intent(out) :: grid
    integer(c_int), value :: maxDimX
    integer(c_int), value :: maxDimY
    integer(c_int), value :: maxDimZ
    integer(c_int), value :: maxNumLocalZColumns
    integer(c_int), value :: maxLocalZLength
    integer(c_int), value :: processingUnit
    integer(c_int), value :: maxNumThreads
    integer(c_int), value :: comm
    integer(c_int), value :: exchangeType
  end function

  integer(c_int) function spfft_float_grid_destroy(grid) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
  end function

  integer(c_int) function spfft_float_grid_max_dim_x(grid, dimX) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: dimX
  end function

  integer(c_int) function spfft_float_grid_max_dim_y(grid, dimY) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: dimY
  end function

  integer(c_int) function spfft_float_grid_max_dim_z(grid, dimZ) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: dimZ
  end function

  integer(c_int) function spfft_float_grid_max_num_local_z_columns(grid, maxNumLocalZColumns) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: maxNumLocalZColumns
  end function

  integer(c_int) function spfft_float_grid_max_local_z_length(grid, maxLocalZLength) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: maxLocalZLength
  end function

  integer(c_int) function spfft_float_grid_processing_unit(grid, processingUnit) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: processingUnit
  end function

  integer(c_int) function spfft_float_grid_device_id(grid, deviceId) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: deviceId
  end function

  integer(c_int) function spfft_float_grid_num_threads(grid, numThreads) bind(C)
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: numThreads
  end function

  integer(c_int) function spfft_float_grid_communicator(grid, comm) &
      bind(C, name="spfft_float_grid_communicator_fortran")
    use iso_c_binding
    type(c_ptr), value :: grid
    integer(c_int), intent(out) :: comm
  end function

  !--------------------------
  !        Transform
  !--------------------------
  integer(c_int) function spfft_transform_create(transform, grid, processingUnit, &
      transformType, dimX, dimY, dimZ, localZLength, numLocalElements, indexFormat, indices) bind(C)
    use iso_c_binding
    type(c_ptr), intent(out) :: transform
    type(c_ptr), value :: grid
    integer(c_int), value :: processingUnit
    integer(c_int), value :: transformType
    integer(c_int), value :: dimX
    integer(c_int), value :: dimY
    integer(c_int), value :: dimZ
    integer(c_int), value :: localZLength
    integer(c_int), value :: numLocalElements
    integer(c_int), value :: indexFormat
    integer(c_int), dimension(*), intent(in) :: indices
  end function

  integer(c_int) function spfft_transform_destroy(transform) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
  end function

  integer(c_int) function spfft_transform_clone(transform, newTransform) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    type(c_ptr), intent(out) :: newTransform
  end function

  integer(c_int) function spfft_transform_backward(transform, input, &
                                  outputLocation) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    complex(c_double), dimension(*), intent(in) :: input
    integer(c_int), value :: outputLocation
  end function

  integer(c_int) function spfft_transform_forward(transform, inputLocation, &
                                  output, scaling) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), value :: inputLocation
    complex(c_double), dimension(*), intent(out) :: output
    integer(c_int), value :: scaling
  end function

  integer(c_int) function  spfft_transform_get_space_domain(transform, &
                                             dataLocation, dataPtr) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), value :: dataLocation
    type(c_ptr), intent(out) :: dataPtr
  end function

  integer(c_int) function spfft_transform_dim_x(transform, dimX) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: dimX
  end function

  integer(c_int) function spfft_transform_dim_y(transform, dimY) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: dimY
  end function

  integer(c_int) function spfft_transform_dim_z(transform, dimZ) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: dimZ
  end function

  integer(c_int) function spfft_transform_local_z_length(transform, localZLength) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: localZLength
  end function

  integer(c_int) function spfft_transform_local_slice_size(transform, size) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: size
  end function

  integer(c_int) function spfft_transform_local_z_offset(transform, offset) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: offset
  end function

  integer(c_int) function spfft_transform_global_size(transform, globalSize) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_long_long), intent(out) :: globalSize
  end function

  integer(c_int) function spfft_transform_num_local_elements(transform, numLocalElements) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: numLocalElements
  end function

  integer(c_int) function spfft_transform_num_global_elements(transform, numGlobalElements) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_long_long), intent(out) :: numGlobalElements
  end function

  integer(c_int) function spfft_transform_device_id(transform, deviceId) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: deviceId
  end function

  integer(c_int) function spfft_transform_num_threads(transform, numThreads) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: numThreads
  end function

  integer(c_int) function spfft_transform_communicator(transform, comm) &
      bind(C, name="spfft_transform_communicator_fortran")
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: comm
  end function

  !--------------------------
  !     Transform Float
  !--------------------------
  integer(c_int) function spfft_float_transform_create(transform, grid, processingUnit, &
      transformType, dimX, dimY, dimZ, localZLength, numLocalElements, indexFormat, indices) bind(C)
    use iso_c_binding
    type(c_ptr), intent(out) :: transform
    type(c_ptr), value :: grid
    integer(c_int), value :: processingUnit
    integer(c_int), value :: transformType
    integer(c_int), value :: dimX
    integer(c_int), value :: dimY
    integer(c_int), value :: dimZ
    integer(c_int), value :: localZLength
    integer(c_int), value :: numLocalElements
    integer(c_int), value :: indexFormat
    integer(c_int), dimension(*), intent(in) :: indices
  end function

  integer(c_int) function spfft_float_transform_destroy(transform) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
  end function

  integer(c_int) function spfft_float_transform_clone(transform, newTransform) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    type(c_ptr), intent(out) :: newTransform
  end function

  integer(c_int) function spfft_float_transform_backward(transform, input, &
                                  outputLocation) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    complex(c_double), dimension(*), intent(in) :: input
    integer(c_int), value :: outputLocation
  end function

  integer(c_int) function spfft_float_transform_forward(transform, inputLocation, &
                                  output, scaling) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), value :: inputLocation
    complex(c_double), dimension(*), intent(out) :: output
    integer(c_int), value :: scaling
  end function

  integer(c_int) function  spfft_float_transform_get_space_domain(transform, &
                                             dataLocation, dataPtr) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), value :: dataLocation
    type(c_ptr), intent(out) :: dataPtr
  end function

  integer(c_int) function spfft_float_transform_dim_x(transform, dimX) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: dimX
  end function

  integer(c_int) function spfft_float_transform_dim_y(transform, dimY) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: dimY
  end function

  integer(c_int) function spfft_float_transform_dim_z(transform, dimZ) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: dimZ
  end function

  integer(c_int) function spfft_float_transform_local_z_length(transform, localZLength) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: localZLength
  end function

  integer(c_int) function spfft_float_transform_local_slice_size(transform, size) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: size
  end function

  integer(c_int) function spfft_float_transform_local_z_offset(transform, offset) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: offset
  end function

  integer(c_int) function spfft_float_transform_global_size(transform, globalSize) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_long_long), intent(out) :: globalSize
  end function

  integer(c_int) function spfft_float_transform_num_local_elements(transform, numLocalElements) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: numLocalElements
  end function

  integer(c_int) function spfft_float_transform_num_global_elements(transform, numGlobalElements) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_long_long), intent(out) :: numGlobalElements
  end function

  integer(c_int) function spfft_float_transform_device_id(transform, deviceId) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: deviceId
  end function

  integer(c_int) function spfft_float_transform_num_threads(transform, numThreads) bind(C)
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: numThreads
  end function

  integer(c_int) function spfft_float_transform_communicator(transform, comm) &
      bind(C, name="spfft_float_transform_communicator_fortran")
    use iso_c_binding
    type(c_ptr), value :: transform
    integer(c_int), intent(out) :: comm
  end function

end interface

end
