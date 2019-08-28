Examples
========

C++
----

.. code-block:: c++

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


C
-
.. code-block:: c

   #include <stdio.h>
   #include <stdlib.h>

   #include "spfft/spfft.h"

   int main(int argc, char** argv) {
     const int dimX = 2;
     const int dimY = 2;
     const int dimZ = 2;

     printf("Dimensions: x = %d, y = %d, z = %d\n\n", dimX, dimY, dimZ);

     const int numThreads = -1; /* Use default OpenMP value */

     double* freqValues = (double*)malloc(2 * sizeof(double) * dimX * dimY * dimZ);

     int* indices = (int*)malloc(3 * sizeof(int) * dimX * dimY * dimZ);

     /* initialize frequency domain values and indices */
     double initValue = 0.0;
     size_t count = 0;
     for (int xIndex = 0; xIndex < dimX; ++xIndex) {
       for (int yIndex = 0; yIndex < dimY; ++yIndex) {
	 for (int zIndex = 0; zIndex < dimZ; ++zIndex, ++count) {
	   /* init values */
	   freqValues[2 * count] = initValue;
	   freqValues[2 * count + 1] = -initValue;

	   /* add index triplet for value */
	   indices[3 * count] = xIndex;
	   indices[3 * count + 1] = yIndex;
	   indices[3 * count + 2] = zIndex;

	   initValue += 1.0;
	 }
       }
     }

     printf("Input:\n");
     for (size_t i = 0; i < dimX * dimY * dimZ; ++i) {
       printf("%f, %f\n", freqValues[2 * i], freqValues[2 * i + 1]);
     }
     printf("\n");

     SpfftError status = 0;

     /* create local Grid. For distributed computations, a MPI Communicator has to be provided */
     SpfftGrid grid;
     status = spfft_grid_create(&grid, dimX, dimY, dimZ, dimX * dimY, SPFFT_PU_HOST, numThreads);
     if (status != SPFFT_SUCCESS) exit(status);

     /* create transform */
     SpfftTransform transform;
     status = spfft_transform_create(&transform, grid, SPFFT_PU_HOST, SPFFT_TRANS_C2C, dimX, dimY,
				     dimZ, dimZ, dimX * dimY * dimZ, SPFFT_INDEX_TRIPLETS, indices);
     if (status != SPFFT_SUCCESS) exit(status);

     /* grid can be safely destroyed after creating all transforms */
     status = spfft_grid_destroy(grid);
     if (status != SPFFT_SUCCESS) exit(status);

     /* get pointer to space domain data. Alignment is guaranteed to fullfill requirements C complex
      types */
     double* realValues;
     status = spfft_transform_get_space_domain(transform, SPFFT_PU_HOST, &realValues);
     if (status != SPFFT_SUCCESS) exit(status);

     /* transform backward */
     status = spfft_transform_backward(transform, freqValues, SPFFT_PU_HOST);
     if (status != SPFFT_SUCCESS) exit(status);

     printf("After backward transform:\n");
     for (size_t i = 0; i < dimX * dimY * dimZ; ++i) {
       printf("%f, %f\n", realValues[2 * i], realValues[2 * i + 1]);
     }
     printf("\n");

     /* transform forward */
     status = spfft_transform_forward(transform, SPFFT_PU_HOST, freqValues, SPFFT_NO_SCALING);
     if (status != SPFFT_SUCCESS) exit(status);

     printf("After forward transform (without scaling):\n");
     for (size_t i = 0; i < dimX * dimY * dimZ; ++i) {
       printf("%f, %f\n", freqValues[2 * i], freqValues[2 * i + 1]);
     }

     /* destroying the final transform will free the associated memory */
     status = spfft_transform_destroy(transform);
     if (status != SPFFT_SUCCESS) exit(status);

     return 0;
   }

Fortran
-------
.. code-block:: fortran

   program main
       use iso_c_binding
       use spfft
       implicit none
       integer :: i, j, k, counter
       integer, parameter :: dimX = 2
       integer, parameter :: dimY = 2
       integer, parameter :: dimZ = 2
       integer, parameter :: maxNumLocalZColumns = dimX * dimY
       integer, parameter :: processingUnit = 1
       integer, parameter :: maxNumThreads = -1
       type(c_ptr) :: grid = c_null_ptr
       type(c_ptr) :: transform = c_null_ptr
       integer :: error = 0
       integer, dimension(dimX * dimY * dimZ * 3):: indices = 0
       complex(C_DOUBLE_COMPLEX), dimension(dimX * dimY * dimZ):: freqValues
       complex(C_DOUBLE_COMPLEX), pointer :: realValues(:,:,:)
       type(c_ptr) :: realValuesPtr


       counter = 0
       do k = 1, dimZ
           do j = 1, dimY
               do i = 1, dimX
                freqValues(counter + 1) = cmplx(counter, counter)
                indices(counter * 3 + 1) = i - 1
                indices(counter * 3 + 2) = j - 1
                indices(counter * 3 + 3) = k - 1
                counter = counter + 1
               end do
           end do
       end do

       ! print input
       print *, "Input:"
       do i = 1, size(freqValues)
            print *, freqValues(i)
       end do


       ! create grid and transform
       error = spfft_grid_create(grid, dimX, dimY, dimZ, maxNumLocalZColumns, processingUnit, maxNumThreads);
       if (error /= 0) stop error
       error = spfft_transform_create(transform, grid, processingUnit, 0, dimX, dimY, dimZ, dimZ, size(freqValues), 0, indices)
       if (error /= 0) stop error

       ! grid can be safely deleted after creating all required transforms
       error = spfft_grid_destroy(grid)
       if (error /= 0) stop error

       ! set space domain array to use memory allocted by the library
       error = spfft_transform_get_space_domain(transform, processingUnit, realValuesPtr)
       if (error /= 0) stop error

       ! transform backward
       error = spfft_transform_backward(transform, freqValues, processingUnit)
       if (error /= 0) stop error


       call c_f_pointer(realValuesPtr, realValues, [dimX,dimY,dimZ])

       print *, ""
       print *, "After backward transform:"
       do k = 1, size(realValues, 3)
           do j = 1, size(realValues, 2)
               do i = 1, size(realValues, 1)
                print *, realValues(i, j, k)
               end do
           end do
       end do

       ! transform forward (will invalidate space domain data)
       error = spfft_transform_forward(transform, processingUnit, freqValues, 0)
       if (error /= 0) stop error

       print *, ""
       print *, "After forward transform (without scaling):"
       do i = 1, size(freqValues)
                print *, freqValues(i)
       end do

       ! destroy transform after use
       ! (will release memory if all transforms from the same grid are destroyed)
       error = spfft_transform_destroy(transform)
       if (error /= 0) stop error

   end

