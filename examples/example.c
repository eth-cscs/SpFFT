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
