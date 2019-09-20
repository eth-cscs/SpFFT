/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SPFFT_GRID_H
#define SPFFT_GRID_H

#include "spfft/config.h"
#include "spfft/errors.h"
#include "spfft/types.h"

#ifdef SPFFT_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Grid handle.
 */
typedef void* SpfftGrid;

/**
 * Constructor for a local grid.
 *
 * @param[out] grid Handle to grid.
 * @param[in] maxDimX Maximum dimension in x.
 * @param[in] maxDimY Maximum dimension in y.
 * @param[in] maxDimZ Maximum dimension in z.
 * @param[in] maxNumLocalZColumns Maximum number of z-columns in frequency domain.
 * @param[in] processingUnit The processing unit type to prepare for. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU or SPFFT_PU_HOST | SPFFT_PU_GPU.
 * @param[in] maxNumThreads The maximum number of threads, transforms created with this grid are
 * allowed to use. If smaller than 1, the OpenMP default value is used.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_create(SpfftGrid* grid, int maxDimX, int maxDimY, int maxDimZ,
                                          int maxNumLocalZColumns,
                                          SpfftProcessingUnitType processingUnit,
                                          int maxNumThreads);

#ifdef SPFFT_MPI
/**
 * Constructor for a distributed grid.
 *
 * @param[out] grid Handle to grid.
 * @param[in] maxDimX Maximum dimension in x.
 * @param[in] maxDimY Maximum dimension in y.
 * @param[in] maxDimZ Maximum dimension in z.
 * @param[in] maxNumLocalZColumns Maximum number of z-columns in frequency domain of the
 * local MPI rank.
 * @param[in] maxLocalZLength Maximum length in z in space domain for the local MPI rank.
 * @param[in] processingUnit The processing unit type to prepare for. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU or SPFFT_PU_HOST | SPFFT_PU_GPU.
 * @param[in] maxNumThreads The maximum number of threads, transforms created with this grid are
 * allowed to use. If smaller than 1, the OpenMP default value is used.
 * @param[in] comm The MPI communicator to use. Will be duplicated for internal use.
 * @param[in] exchangeType The type of MPI exchange to use. Possible values are
 * SPFFT_EXCH_DEFAULT, SPFFT_EXCH_BUFFERED, SPFFT_EXCH_COMPACT_BUFFERED and SPFFT_EXCH_UNBUFFERED.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_create_distributed(SpfftGrid* grid, int maxDimX, int maxDimY,
                                                      int maxDimZ, int maxNumLocalZColumns,
                                                      int maxLocalZLength,
                                                      SpfftProcessingUnitType processingUnit,
                                                      int maxNumThreads, MPI_Comm comm,
                                                      SpfftExchangeType exchangeType);
#endif

/**
 * Destroy a grid.
 *
 * A grid can be safely destroyed independet from any related transforms. The internal memory
 * is released, once all associated transforms are destroyed as well (through internal reference
 * counting).
 *
 * @param[in] grid Handle to grid.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_destroy(SpfftGrid grid);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] dimX Maximum dimension in x.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_max_dim_x(SpfftGrid grid, int* dimX);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] dimY Maximum dimension in y.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_max_dim_y(SpfftGrid grid, int* dimY);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] dimZ Maximum dimension in z.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_max_dim_z(SpfftGrid grid, int* dimZ);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] maxNumLocalZColumns Maximum number of z-columns in frequency domain of the local MPI
 * rank.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_max_num_local_z_columns(SpfftGrid grid,
                                                           int* maxNumLocalZColumns);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] maxLocalZLength Maximum length in z in space domain of the local MPI rank.
 * rank.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_max_local_z_length(SpfftGrid grid, int* maxLocalZLength);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] processingUnit The processing unit, the grid has prepared for. Can be SPFFT_PU_HOST
 * or SPFFT_PU_GPU or SPFFT_PU_HOST | SPFFT_PU_GPU.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_processing_unit(SpfftGrid grid,
                                                   SpfftProcessingUnitType* processingUnit);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] deviceId The GPU device id used. Returns always 0, if no GPU support is enabled.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_device_id(SpfftGrid grid, int* deviceId);

/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] numThreads The exact number of threads used by transforms created from this grid. May
 * be less than the maximum given to the constructor. Always 1, if not compiled with OpenMP support.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_num_threads(SpfftGrid grid, int* numThreads);

#ifdef SPFFT_MPI
/**
 * Access a grid parameter.
 * @param[in] grid Handle to grid.
 * @param[out] comm The internal MPI communicator.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_grid_communicator(SpfftGrid grid, MPI_Comm* comm);
#endif

#ifdef __cplusplus
}
#endif

#endif
