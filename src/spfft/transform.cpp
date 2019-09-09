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

#include "spfft/transform.hpp"
#include "spfft/grid.hpp"
#include "spfft/transform.h"
#include "spfft/transform_internal.hpp"
#include "parameters/parameters.hpp"

namespace spfft {

//---------------------
// Double precision
//---------------------
Transform::Transform(const std::shared_ptr<GridInternal<double>>& grid,
                     SpfftProcessingUnitType processingUnit, SpfftTransformType transformType,
                     int dimX, int dimY, int dimZ, int localZLength, int numLocalElements,
                     SpfftIndexFormatType indexFormat, const int* indices)

{
  if (dimX < 0 || dimY < 0 || dimZ < 0 || localZLength < 0 || numLocalElements < 0 ||
      (!indices && numLocalElements > 0)) {
    throw InvalidParameterError();
  }
  std::shared_ptr<Parameters> param;
  if (!grid->local()) {
#ifdef SPFFT_MPI
    param.reset(new Parameters(grid->communicator(), transformType, dimX,
                                                     dimY, dimZ, localZLength, numLocalElements,
                                                     indexFormat, indices));
#else
    throw MPISupportError();
#endif
  } else {
    param.reset(new Parameters(
        transformType, dimX, dimY, dimZ, numLocalElements, indexFormat, indices));
  }

  transform_.reset(new TransformInternal<double>(processingUnit, grid, std::move(param)));
}

Transform::Transform(std::shared_ptr<TransformInternal<double>> transform)
    : transform_(std::move(transform)) {}

Transform Transform::clone() const {
  return Transform(std::shared_ptr<TransformInternal<double>>(
      new TransformInternal<double>(transform_->clone())));
}

double* Transform::space_domain_data(SpfftProcessingUnitType dataLocation) {
  return transform_->space_domain_data(dataLocation);
}

void Transform::forward(SpfftProcessingUnitType inputLocation, double* output,
                        SpfftScalingType scaling) {
  transform_->forward(inputLocation, output, scaling);
}

void Transform::backward(const double* input, SpfftProcessingUnitType outputLocation) {
  transform_->backward(input, outputLocation);
}

SpfftTransformType Transform::type() const { return transform_->type(); }

int Transform::dim_x() const { return transform_->dim_x(); }

int Transform::dim_y() const { return transform_->dim_y(); }

int Transform::dim_z() const { return transform_->dim_z(); }

int Transform::local_z_length() const { return transform_->num_local_xy_planes(); }

int Transform::local_z_offset() const { return transform_->local_xy_plane_offset(); }

int Transform::local_slice_size() const {
  return dim_x() * dim_y() * local_z_length();
}

int Transform::num_local_elements() const { return transform_->num_local_elements(); }

long long int Transform::num_global_elements() const { return transform_->num_global_elements(); }

long long int Transform::global_size() const { return transform_->global_size(); }

SpfftProcessingUnitType Transform::processing_unit() const { return transform_->processing_unit(); }

int Transform::device_id() const { return transform_->device_id(); }

int Transform::num_threads() const { return transform_->num_threads(); }

#ifdef SPFFT_MPI
MPI_Comm Transform::communicator() const { return transform_->communicator(); }
#endif

} // namespace spfft

//---------------------
// C API
//---------------------

extern "C" {
SpfftError spfft_transform_create(SpfftTransform* transform, SpfftGrid grid,
                                  SpfftProcessingUnitType processingUnit,
                                  SpfftTransformType transformType, int dimX, int dimY, int dimZ,
                                  int localZLength, int numLocalElements,
                                  SpfftIndexFormatType indexFormat, const int* indices) {
  try {
    *transform = new spfft::Transform(reinterpret_cast<spfft::Grid*>(grid)->create_transform(
        processingUnit, transformType, dimX, dimY, dimZ, localZLength, numLocalElements,
        indexFormat, indices));

  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_destroy(SpfftTransform transform) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spfft::Transform*>(transform);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  transform = nullptr;
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_clone(SpfftTransform transform, SpfftTransform* newTransform) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *newTransform = new spfft::Transform(reinterpret_cast<spfft::Transform*>(transform)->clone());
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}


SpfftError spfft_transform_forward(SpfftTransform transform, SpfftProcessingUnitType inputLocation,
                                   double* output, SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::Transform*>(transform)->forward(inputLocation, output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_backward(SpfftTransform transform, const double* input,
                                    SpfftProcessingUnitType outputLocation) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::Transform*>(transform)->backward(input, outputLocation);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_get_space_domain(SpfftTransform transform,
                                           SpfftProcessingUnitType dataLocation, double** data) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *data = reinterpret_cast<spfft::Transform*>(transform)->space_domain_data(dataLocation);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_dim_x(SpfftTransform transform, int* dimX) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimX = reinterpret_cast<spfft::Transform*>(transform)->dim_x();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_dim_y(SpfftTransform transform, int* dimY) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimY = reinterpret_cast<spfft::Transform*>(transform)->dim_y();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_dim_z(SpfftTransform transform, int* dimZ) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimZ = reinterpret_cast<spfft::Transform*>(transform)->dim_z();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_local_z_length(SpfftTransform transform, int* localZLength) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *localZLength = reinterpret_cast<spfft::Transform*>(transform)->local_z_length();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_local_z_offset(SpfftTransform transform, int* offset) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *offset = reinterpret_cast<spfft::Transform*>(transform)->local_z_offset();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_num_local_elements(SpfftTransform transform, int* localZLength) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *localZLength = reinterpret_cast<spfft::Transform*>(transform)->local_z_length();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_num_global_elements(SpfftTransform transform, long long int* numGlobalElements) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *numGlobalElements = reinterpret_cast<spfft::Transform*>(transform)->num_global_elements();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_global_size(SpfftTransform transform, long long int* globalSize) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *globalSize = reinterpret_cast<spfft::Transform*>(transform)->global_size();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_device_id(SpfftTransform transform, int* deviceId) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *deviceId = reinterpret_cast<spfft::Transform*>(transform)->device_id();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_num_threads(SpfftTransform transform, int* numThreads) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *numThreads = reinterpret_cast<spfft::Transform*>(transform)->num_threads();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

#ifdef SPFFT_MPI
SpfftError spfft_transform_communicator(SpfftTransform transform, MPI_Comm* comm) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *comm = reinterpret_cast<spfft::Transform*>(transform)->communicator();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_transform_communicator_fortran(SpfftGrid grid, int* commFortran) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *commFortran = MPI_Comm_c2f(reinterpret_cast<spfft::Transform*>(grid)->communicator());
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}
#endif

} // extern C


