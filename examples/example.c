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


  /**************************************************
   Option A: Reuse internal buffer for space domain
  ***************************************************/

  /* Get pointer to buffer with space domain data. Is guaranteed to be castable to a valid
     complex type pointer. Using the internal working buffer as input / output can help reduce
     memory usage.*/
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


  /**********************************************
   Option B: Use external buffer for space domain
  ***********************************************/
  spaceDomain = (double*)malloc(2 * sizeof(double) * dimX * dimY * dimZ);

  /* transform backward */
  status = spfft_transform_backward_ptr(transform, frequencyElements, spaceDomain);
  if (status != SPFFT_SUCCESS) exit(status);

  /* transform forward */
  status = spfft_transform_forward_ptr(transform, spaceDomain, frequencyElements, SPFFT_NO_SCALING);
  if (status != SPFFT_SUCCESS) exit(status);

  /* Note: In-place transforms are also supported by passing the same pointer for input and output. */

  printf("After forward transform (without normalization):\n");
  for (size_t i = 0; i < dimX * dimY * dimZ; ++i) {
    printf("%f, %f\n", frequencyElements[2 * i], frequencyElements[2 * i + 1]);
  }

  /* destroying the final transform will free the associated memory */
  status = spfft_transform_destroy(transform);
  if (status != SPFFT_SUCCESS) exit(status);

  free(spaceDomain);
  free(frequencyElements);

  return 0;
}
