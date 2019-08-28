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
#ifndef SPFFT_TRANSFORM_FLOAT_H
#define SPFFT_TRANSFORM_FLOAT_H

#include "spfft/config.h"
#include "spfft/errors.h"
#include "spfft/grid_float.h"
#include "spfft/types.h"

#ifdef SPFFT_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Transform handle.
 */
typedef void* SpfftFloatTransform;

/**
 * Creates a single precision transform from a single precision grid handle.
 *
 * @param[out] transform Handle to the transform.
 * @param[in] grid Handle to the grid, with which the transform is created.
 * @param[in] processingUnit The processing unit type to use. Must be either SPFFT_PU_HOST or
 * SPFFT_PU_GPU and be supported by the grid itself.
 * @param[in] transformType The transform type (complex to complex or real to complex). Can be
 * SPFFT_TRANS_C2C or SPFFT_TRANS_R2C.
 * @param[in] dimX The dimension in x. The maximum allowed depends on the grid parameters.
 * @param[in] dimY The dimension in y. The maximum allowed depends on the grid parameters.
 * @param[in] dimZ The dimension in z. The maximum allowed depends on the grid parameters.
 * @param[in] localZLength The length in z in space domain of the local MPI rank.
 * @param[in] numLocalElements The number of elements in frequency domain of the local MPI
 * rank.
 * @param[in] indexFormat The index format. Only SPFFT_INDEX_TRIPLETS currently supported.
 * @param[in] indices Pointer to the frequency indices. Posive and negative indexing is supported.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_create(SpfftFloatTransform* transform, SpfftFloatGrid grid,
                                        SpfftProcessingUnitType processingUnit,
                                        SpfftTransformType transformType, int dimX, int dimY,
                                        int dimZ, int localZLength, int numLocalElements,
                                        SpfftIndexFormatType indexFormat, const int* indices);

/**
 * Destroy a transform.
 *
 * @param[in] transform Handle to the transform.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_destroy(SpfftFloatTransform transform);

/**
 * Clone a transform.
 *
 * @param[in] transform Handle to the transform.
 * @param[out] newTransform Independent transform with the same parameters, but with new underlying
 * grid.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_clone(SpfftFloatTransform transform,
                                       SpfftFloatTransform* newTransform);

/**
 * Execute a forward transform from space domain to frequency domain.
 *
 * @param[in] transform Handle to the transform.
 * @param[in] inputLocation The processing unit, to take the input from. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU (if GPU is set as execution unit).
 * @param[out] output Pointer to memory, where the frequency domain elements are written to. Can
 * be located at Host or GPU memory (if GPU is set as processing unit).
 * @param[in] scaling Controls scaling of output. SPFFT_NO_SCALING to disable or
 * SPFFT_FULL_SCALING to scale by factor 1 / (dim_x() * dim_y() * dim_z()).
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_forward(SpfftFloatTransform transform,
                                         SpfftProcessingUnitType inputLocation, float* output,
                                         SpfftScalingType scaling);

/**
 * Execute a backward transform from frequency domain to space domain.
 *
 * @param[in] transform Handle to the transform.
 * @param[in] input Input data in frequency domain. Must match the indices provided at transform
 * creation. Can be located at Host or GPU memory, if GPU is set as processing unit.
 * @param[in] outputLocation The processing unit, to place the output at. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU (if GPU is set as execution unit).
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_backward(SpfftFloatTransform transform, const float* input,
                                          SpfftProcessingUnitType outputLocation);
/**
 * Provides access to the space domain data.
 *
 * @param[in] transform Handle to the transform.
 * @param[in] dataLocation The processing unit to query for the data. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU (if GPU is set as execution unit).
 * @param[out] data Pointer to space domain data on given processing unit. Alignment is guaranteed
 * to fulfill requirements for std::complex and C language complex types.
 * @throw GenericError SpFFT error. Can be a derived type.
 * @throw std::exception Error from standard library calls. Can be a derived type.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_get_space_domain(SpfftFloatTransform transform,
                                                 SpfftProcessingUnitType dataLocation,
                                                 float** data);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] dimX Dimension in x.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_dim_x(SpfftFloatTransform transform, int* dimX);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] dimY Dimension in y.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_dim_y(SpfftFloatTransform transform, int* dimY);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] dimZ Dimension in z.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_dim_z(SpfftFloatTransform transform, int* dimZ);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] localZLength size in z of the slice in space domain on the local MPI rank.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_local_z_length(SpfftFloatTransform transform, int* localZLength);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] offset Offset in z of the space domain slice held by the local MPI rank.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_local_z_offset(SpfftFloatTransform transform, int* offset);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] numLocalElements Number of local elements in frequency domain.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_num_local_elements(SpfftFloatTransform transform, int* numLocalElements);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] deviceId The GPU device id used. Returns always 0, if no GPU support is enabled.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_device_id(SpfftFloatTransform transform, int* deviceId);

/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] numThreads The exact number of threads used by transforms created from this grid. May
 * be less than the maximum given to the constructor. Always 1, if not compiled with OpenMP support.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_num_threads(SpfftFloatTransform transform, int* numThreads);

#ifdef SPFFT_MPI
/**
 * Access a transform parameter.
 * @param[in] transform Handle to the transform.
 * @param[out] comm The internal MPI communicator.
 * @return Error code or SPFFT_SUCCESS.
 */
SpfftError spfft_float_transform_communicator(SpfftFloatTransform transform, MPI_Comm* comm);
#endif

#ifdef __cplusplus
}
#endif

#endif
