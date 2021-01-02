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

#include "spfft/transform_float.hpp"
#include "spfft/grid_float.hpp"
#include "spfft/transform_float.h"
#include "spfft/transform_internal.hpp"

#ifdef SPFFT_MPI
#include "mpi_util/mpi_communicator_handle.hpp"
#endif

#ifdef SPFFT_SINGLE_PRECISION
namespace spfft {
TransformFloat::TransformFloat(const std::shared_ptr<GridInternal<float>>& grid,
                               SpfftProcessingUnitType processingUnit,
                               SpfftTransformType transformType, int dimX, int dimY, int dimZ,
                               int localZLength, int numLocalElements,
                               SpfftIndexFormatType indexFormat, const int* indices) {
  std::shared_ptr<Parameters> param;
  if (!grid->local()) {
#ifdef SPFFT_MPI
    param.reset(new Parameters(grid->communicator(), transformType, dimX, dimY, dimZ, localZLength,
                               numLocalElements, indexFormat, indices));
#else
    throw MPISupportError();
#endif
  } else {
    param.reset(
        new Parameters(transformType, dimX, dimY, dimZ, numLocalElements, indexFormat, indices));
  }

  transform_.reset(new TransformInternal<float>(processingUnit, grid, std::move(param)));
}

TransformFloat::TransformFloat(int maxNumThreads, SpfftProcessingUnitType processingUnit,
          SpfftTransformType transformType, int dimX, int dimY, int dimZ, int numLocalElements,
          SpfftIndexFormatType indexFormat, const int* indices) {
  if (dimX < 0 || dimY < 0 || dimZ < 0 || numLocalElements < 0 ||
      (!indices && numLocalElements > 0)) {
    throw InvalidParameterError();
  }

  std::shared_ptr<Parameters> param (new Parameters(transformType, dimX, dimY, dimZ, numLocalElements, indexFormat, indices));
  std::shared_ptr<GridInternal<float>> grid(new GridInternal<float>(dimX, dimY, dimZ, param->max_num_z_sticks(), processingUnit, maxNumThreads));

  transform_.reset(
      new TransformInternal<float>(processingUnit, std::move(grid), std::move(param)));
}

#ifdef SPFFT_MPI
TransformFloat::TransformFloat(int maxNumThreads, MPI_Comm comm, SpfftExchangeType exchangeType,
          SpfftProcessingUnitType processingUnit, SpfftTransformType transformType, int dimX,
          int dimY, int dimZ, int localZLength, int numLocalElements,
          SpfftIndexFormatType indexFormat, const int* indices) {
  if (dimX < 0 || dimY < 0 || dimZ < 0 || numLocalElements < 0 ||
      (!indices && numLocalElements > 0)) {
    throw InvalidParameterError();
  }

  std::shared_ptr<Parameters> param(new Parameters(MPICommunicatorHandle(comm), transformType, dimX,
                                                   dimY, dimZ, localZLength, numLocalElements,
                                                   indexFormat, indices));
  std::shared_ptr<GridInternal<float>> grid(
      new GridInternal<float>(dimX, dimY, dimZ, param->max_num_z_sticks(), localZLength,
                               processingUnit, maxNumThreads, comm, exchangeType));

  transform_.reset(
      new TransformInternal<float>(processingUnit, std::move(grid), std::move(param)));
}
#endif

TransformFloat::TransformFloat(std::shared_ptr<TransformInternal<float>> transform)
    : transform_(std::move(transform)) {}

TransformFloat TransformFloat::clone() const {
  return TransformFloat(
      std::shared_ptr<TransformInternal<float>>(new TransformInternal<float>(transform_->clone())));
}

float* TransformFloat::space_domain_data(SpfftProcessingUnitType dataLocation) {
  return transform_->space_domain_data(dataLocation);
}

void TransformFloat::forward(SpfftProcessingUnitType inputLocation, float* output,
                             SpfftScalingType scaling) {
  transform_->forward(inputLocation, output, scaling);
}

void TransformFloat::forward(const float* input, float* output,
                             SpfftScalingType scaling) {
  transform_->forward(input, output, scaling);
}

void TransformFloat::backward(const float* input, SpfftProcessingUnitType outputLocation) {
  transform_->backward(input, outputLocation);
}

void TransformFloat::backward(const float* input, float* ouput) {
  transform_->backward(input, ouput);
}

SpfftTransformType TransformFloat::type() const { return transform_->type(); }

int TransformFloat::dim_x() const { return transform_->dim_x(); }

int TransformFloat::dim_y() const { return transform_->dim_y(); }

int TransformFloat::dim_z() const { return transform_->dim_z(); }

int TransformFloat::local_z_length() const { return transform_->num_local_xy_planes(); }

int TransformFloat::local_z_offset() const { return transform_->local_xy_plane_offset(); }

int TransformFloat::local_slice_size() const { return dim_x() * dim_y() * local_z_length(); }

int TransformFloat::num_local_elements() const { return transform_->num_local_elements(); }

long long int TransformFloat::num_global_elements() const {
  return transform_->num_global_elements();
}

long long int TransformFloat::global_size() const { return transform_->global_size(); }

SpfftProcessingUnitType TransformFloat::processing_unit() const {
  return transform_->processing_unit();
}

int TransformFloat::device_id() const { return transform_->device_id(); }

int TransformFloat::num_threads() const { return transform_->num_threads(); }

#ifdef SPFFT_MPI
MPI_Comm TransformFloat::communicator() const { return transform_->communicator(); }
#endif

}  // namespace spfft

//---------------------
// C API
//---------------------

extern "C" {
SpfftError spfft_float_transform_create(SpfftFloatTransform* transform, SpfftFloatGrid grid,
                                        SpfftProcessingUnitType processingUnit,
                                        SpfftTransformType transformType, int dimX, int dimY,
                                        int dimZ, int localZLength, int numLocalElements,
                                        SpfftIndexFormatType indexFormat, const int* indices) {
  try {
    *transform =
        new spfft::TransformFloat(reinterpret_cast<spfft::GridFloat*>(grid)->create_transform(
            processingUnit, transformType, dimX, dimY, dimZ, localZLength, numLocalElements,
            indexFormat, indices));

  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_create_independent(
    SpfftFloatTransform* transform, int maxNumThreads, SpfftProcessingUnitType processingUnit,
    SpfftTransformType transformType, int dimX, int dimY, int dimZ, int numLocalElements,
    SpfftIndexFormatType indexFormat, const int* indices) {
  try {
    *transform = new spfft::TransformFloat(maxNumThreads, processingUnit, transformType, dimX, dimY,
                                           dimZ, numLocalElements, indexFormat, indices);

  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

#ifdef SPFFT_MPI
SpfftError spfft_float_transform_create_independent_distributed(
    SpfftFloatTransform* transform, int maxNumThreads, MPI_Comm comm,
    SpfftExchangeType exchangeType, SpfftProcessingUnitType processingUnit,
    SpfftTransformType transformType, int dimX, int dimY, int dimZ, int localZLength,
    int numLocalElements, SpfftIndexFormatType indexFormat, const int* indices) {
  try {
    *transform = new spfft::TransformFloat(maxNumThreads, comm, exchangeType, processingUnit,
                                           transformType, dimX, dimY, dimZ, localZLength,
                                           numLocalElements, indexFormat, indices);

  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SPFFT_EXPORT SpfftError spfft_float_transform_create_independent_distributed_fortran(
    SpfftFloatTransform* transform, int maxNumThreads, int commFortran,
    SpfftExchangeType exchangeType, SpfftProcessingUnitType processingUnit,
    SpfftTransformType transformType, int dimX, int dimY, int dimZ, int localZLength,
    int numLocalElements, SpfftIndexFormatType indexFormat, const int* indices) {
  MPI_Comm comm = MPI_Comm_f2c(commFortran);
  return spfft_float_transform_create_independent_distributed(
      transform, maxNumThreads, comm, exchangeType, processingUnit, transformType, dimX, dimY, dimZ,
      localZLength, numLocalElements, indexFormat, indices);
}
#endif



SpfftError spfft_float_transform_destroy(SpfftFloatTransform transform) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spfft::TransformFloat*>(transform);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  transform = nullptr;
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_clone(SpfftFloatTransform transform,
                                       SpfftFloatTransform* newTransform) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *newTransform =
        new spfft::TransformFloat(reinterpret_cast<spfft::TransformFloat*>(transform)->clone());
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_forward(SpfftFloatTransform transform,
                                         SpfftProcessingUnitType inputLocation, float* output,
                                         SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::TransformFloat*>(transform)->forward(inputLocation, output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_forward_ptr(SpfftFloatTransform transform, const float* input,
                                             float* output, SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::TransformFloat*>(transform)->forward(input, output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_backward(SpfftFloatTransform transform, const float* input,
                                          SpfftProcessingUnitType outputLocation) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::TransformFloat*>(transform)->backward(input, outputLocation);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_backward_ptr(SpfftFloatTransform transform, const float* input,
                                              float* output) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::TransformFloat*>(transform)->backward(input, output);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_get_space_domain(SpfftFloatTransform transform,
                                                  SpfftProcessingUnitType dataLocation,
                                                  float** data) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *data = reinterpret_cast<spfft::TransformFloat*>(transform)->space_domain_data(dataLocation);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_dim_x(SpfftFloatTransform transform, int* dimX) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimX = reinterpret_cast<spfft::TransformFloat*>(transform)->dim_x();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_dim_y(SpfftFloatTransform transform, int* dimY) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimY = reinterpret_cast<spfft::TransformFloat*>(transform)->dim_y();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_dim_z(SpfftFloatTransform transform, int* dimZ) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimZ = reinterpret_cast<spfft::TransformFloat*>(transform)->dim_z();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_local_z_length(SpfftFloatTransform transform, int* localZLength) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *localZLength = reinterpret_cast<spfft::TransformFloat*>(transform)->local_z_length();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_local_z_offset(SpfftFloatTransform transform, int* offset) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *offset = reinterpret_cast<spfft::TransformFloat*>(transform)->local_z_offset();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_local_slice_size(SpfftFloatTransform transform, int* size) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *size = reinterpret_cast<spfft::TransformFloat*>(transform)->local_slice_size();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_num_local_elements(SpfftFloatTransform transform,
                                                    int* localZLength) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *localZLength = reinterpret_cast<spfft::TransformFloat*>(transform)->local_z_length();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_num_global_elements(SpfftFloatTransform transform,
                                                     long long int* numGlobalElements) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *numGlobalElements = reinterpret_cast<spfft::TransformFloat*>(transform)->num_global_elements();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_global_size(SpfftFloatTransform transform,
                                             long long int* globalSize) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *globalSize = reinterpret_cast<spfft::TransformFloat*>(transform)->global_size();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_device_id(SpfftFloatTransform transform, int* deviceId) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *deviceId = reinterpret_cast<spfft::TransformFloat*>(transform)->device_id();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_transform_num_threads(SpfftFloatTransform transform, int* numThreads) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *numThreads = reinterpret_cast<spfft::TransformFloat*>(transform)->num_threads();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

#ifdef SPFFT_MPI
SpfftError spfft_float_transform_communicator(SpfftFloatTransform transform, MPI_Comm* comm) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *comm = reinterpret_cast<spfft::TransformFloat*>(transform)->communicator();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SPFFT_EXPORT SpfftError spfft_float_transform_communicator_fortran(SpfftFloatTransform transform,
                                                                   int* commFortran) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *commFortran =
        MPI_Comm_c2f(reinterpret_cast<spfft::TransformFloat*>(transform)->communicator());
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}
#endif

}  // extern C

#endif
