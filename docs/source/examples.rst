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

     // Use default OpenMP value
     const int numThreads = -1;

     // use all elements in this example.
     const int numFrequencyElements = dimX * dimY * dimZ;

     // Slice length in space domain. Equivalent to dimZ for non-distributed case.
     const int localZLength = dimZ;

     // interleaved complex numbers
     std::vector<double> frequencyElements;
     frequencyElements.reserve(2 * numFrequencyElements);

     // indices of frequency elements
     std::vector<int> indices;
     indices.reserve(3 * numFrequencyElements);

     // initialize frequency domain values and indices
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

     // create local Grid. For distributed computations, a MPI Communicator has to be provided
     spfft::Grid grid(dimX, dimY, dimZ, dimX * dimY, SPFFT_PU_HOST, numThreads);

     // create transform
     spfft::Transform transform =
	 grid.create_transform(SPFFT_PU_HOST, SPFFT_TRANS_C2C, dimX, dimY, dimZ, localZLength,
			       numFrequencyElements, SPFFT_INDEX_TRIPLETS, indices.data());

     // Get pointer to space domain data. Alignment fullfills requirements for std::complex.
     // Can also be read as std::complex elements (guaranteed by the standard to be binary compatible
     // since C++11).
     double* spaceDomain = transform.space_domain_data(SPFFT_PU_HOST);

     // transform backward
     transform.backward(frequencyElements.data(), SPFFT_PU_HOST);

     std::cout << std::endl << "After backward transform:" << std::endl;
     for (int i = 0; i < transform.local_slice_size(); ++i) {
       std::cout << spaceDomain[2 * i] << ", " << spaceDomain[2 * i + 1] << std::endl;
     }

     // transform forward
     transform.forward(SPFFT_PU_HOST, frequencyElements.data(), SPFFT_NO_SCALING);

     std::cout << std::endl << "After forward transform (without scaling):" << std::endl;
     for (int i = 0; i < numFrequencyElements; ++i) {
       std::cout << frequencyElements[2 * i] << ", " << frequencyElements[2 * i + 1] << std::endl;
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

     /* Use default OpenMP value */
     const int numThreads = -1;

     /* use all elements in this example. */
     const int numFrequencyElements = dimX * dimY * dimZ;

     /* Slice length in space domain. Equivalent to dimZ for non-distributed case. */
     const int localZLength = dimZ;

     /* interleaved complex numbers */
     double* frequencyElements = (double*)malloc(2 * sizeof(double) * numFrequencyElements);

     /* indices of frequency elements */
     int* indices = (int*)malloc(3 * sizeof(int) * numFrequencyElements);

     /* initialize frequency domain values and indices */
     double initValue = 0.0;
     size_t count = 0;
     for (int xIndex = 0; xIndex < dimX; ++xIndex) {
       for (int yIndex = 0; yIndex < dimY; ++yIndex) {
	 for (int zIndex = 0; zIndex < dimZ; ++zIndex, ++count) {
	   /* init values */
	   frequencyElements[2 * count] = initValue;
	   frequencyElements[2 * count + 1] = -initValue;

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
       printf("%f, %f\n", frequencyElements[2 * i], frequencyElements[2 * i + 1]);
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
				     dimZ, localZLength, numFrequencyElements, SPFFT_INDEX_TRIPLETS, indices);
     if (status != SPFFT_SUCCESS) exit(status);

     /* grid can be safely destroyed after creating all transforms */
     status = spfft_grid_destroy(grid);
     if (status != SPFFT_SUCCESS) exit(status);

     /* get pointer to space domain data. Alignment is guaranteed to fullfill requirements C complex
      types */
     double* spaceDomain;
     status = spfft_transform_get_space_domain(transform, SPFFT_PU_HOST, &spaceDomain);
     if (status != SPFFT_SUCCESS) exit(status);

     /* transform backward */
     status = spfft_transform_backward(transform, frequencyElements, SPFFT_PU_HOST);
     if (status != SPFFT_SUCCESS) exit(status);

     printf("After backward transform:\n");
     for (size_t i = 0; i < dimX * dimY * dimZ; ++i) {
       printf("%f, %f\n", spaceDomain[2 * i], spaceDomain[2 * i + 1]);
     }
     printf("\n");

     /* transform forward */
     status = spfft_transform_forward(transform, SPFFT_PU_HOST, frequencyElements, SPFFT_NO_SCALING);
     if (status != SPFFT_SUCCESS) exit(status);

     printf("After forward transform (without scaling):\n");
     for (size_t i = 0; i < dimX * dimY * dimZ; ++i) {
       printf("%f, %f\n", frequencyElements[2 * i], frequencyElements[2 * i + 1]);
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
       integer :: errorCode = 0
       integer, dimension(dimX * dimY * dimZ * 3):: indices = 0
       complex(C_DOUBLE_COMPLEX), dimension(dimX * dimY * dimZ):: frequencyElements
       complex(C_DOUBLE_COMPLEX), pointer :: spaceDomain(:,:,:)
       type(c_ptr) :: realValuesPtr


       counter = 0
       do k = 1, dimZ
	   do j = 1, dimY
	       do i = 1, dimX
		frequencyElements(counter + 1) = cmplx(counter, -counter)
		indices(counter * 3 + 1) = i - 1
		indices(counter * 3 + 2) = j - 1
		indices(counter * 3 + 3) = k - 1
		counter = counter + 1
	       end do
	   end do
       end do

       ! print input
       print *, "Input:"
       do i = 1, size(frequencyElements)
	    print *, frequencyElements(i)
       end do


       ! create grid and transform
       errorCode = spfft_grid_create(grid, dimX, dimY, dimZ, maxNumLocalZColumns, processingUnit, maxNumThreads);
       if (errorCode /= SPFFT_SUCCESS) error stop
       errorCode = spfft_transform_create(transform, grid, processingUnit, 0, dimX, dimY, dimZ, dimZ,&
	   size(frequencyElements), SPFFT_INDEX_TRIPLETS, indices)
       if (errorCode /= SPFFT_SUCCESS) error stop

       ! grid can be safely destroyed after creating all required transforms
       errorCode = spfft_grid_destroy(grid)
       if (errorCode /= SPFFT_SUCCESS) error stop

       ! set space domain array to use memory allocted by the library
       errorCode = spfft_transform_get_space_domain(transform, processingUnit, realValuesPtr)
       if (errorCode /= SPFFT_SUCCESS) error stop

       ! transform backward
       errorCode = spfft_transform_backward(transform, frequencyElements, processingUnit)
       if (errorCode /= SPFFT_SUCCESS) error stop


       call c_f_pointer(realValuesPtr, spaceDomain, [dimX,dimY,dimZ])

       print *, ""
       print *, "After backward transform:"
       do k = 1, size(spaceDomain, 3)
	   do j = 1, size(spaceDomain, 2)
	       do i = 1, size(spaceDomain, 1)
		print *, spaceDomain(i, j, k)
	       end do
	   end do
       end do

       ! transform forward (will invalidate space domain data)
       errorCode = spfft_transform_forward(transform, processingUnit, frequencyElements, 0)
       if (errorCode /= SPFFT_SUCCESS) error stop

       print *, ""
       print *, "After forward transform (without scaling):"
       do i = 1, size(frequencyElements)
		print *, frequencyElements(i)
       end do

       ! destroying the final transform will free the associated memory
       errorCode = spfft_transform_destroy(transform)
       if (errorCode /= SPFFT_SUCCESS) error stop

   end
