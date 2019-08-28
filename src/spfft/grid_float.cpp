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

#include "spfft/grid_float.hpp"
#include "spfft/grid_float.h"
#include "spfft/grid_internal.hpp"

#ifdef SPFFT_SINGLE_PRECISION
namespace spfft {
GridFloat::GridFloat(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZColumns,
                     SpfftProcessingUnitType processingUnit, int maxNumThreads)
    : grid_(new GridInternal<float>(maxDimX, maxDimY, maxDimZ, maxNumLocalZColumns, processingUnit,
                                    maxNumThreads)) {}

#ifdef SPFFT_MPI
GridFloat::GridFloat(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZColumns,
                     int maxLocalZLength, SpfftProcessingUnitType processingUnit, int maxNumThreads,
                     MPI_Comm comm, SpfftExchangeType exchangeType)
    : grid_(new GridInternal<float>(maxDimX, maxDimY, maxDimZ, maxNumLocalZColumns, maxLocalZLength,
                                    processingUnit, maxNumThreads, comm, exchangeType)) {}
#endif

GridFloat::GridFloat(const GridFloat& grid) : grid_(new GridInternal<float>(*(grid.grid_))) {}

GridFloat& GridFloat::operator=(const GridFloat& grid) {
  grid_.reset(new GridInternal<float>(*(grid.grid_)));
  return *this;
}

TransformFloat GridFloat::create_transform(SpfftProcessingUnitType processingUnit,
                                           SpfftTransformType transformType, int dimX, int dimY,
                                           int dimZ, int localZLength, int numLocalElements,
                                           SpfftIndexFormatType indexFormat,
                                           const int* indices) const {
  return TransformFloat(grid_, processingUnit, transformType, dimX, dimY, dimZ, localZLength,
                        numLocalElements, indexFormat, indices);
}

int GridFloat::max_dim_x() const { return grid_->max_dim_x(); }

int GridFloat::max_dim_y() const { return grid_->max_dim_y(); }

int GridFloat::max_dim_z() const { return grid_->max_dim_z(); }

int GridFloat::max_num_local_z_columns() const { return grid_->max_num_local_z_columns(); }

int GridFloat::max_local_z_length() const { return grid_->max_num_local_xy_planes(); }

SpfftProcessingUnitType GridFloat::processing_unit() const { return grid_->processing_unit(); }

int GridFloat::device_id() const { return grid_->device_id(); }

int GridFloat::num_threads() const { return grid_->num_threads(); }

#ifdef SPFFT_MPI
MPI_Comm GridFloat::communicator() const { return grid_->communicator().get(); }
#endif

} // namespace spfft


//---------------------
// C API
//---------------------

extern "C" {
SpfftError spfft_float_grid_create(SpfftFloatGrid* grid, int maxDimX, int maxDimY, int maxDimZ,
                             int maxNumLocalZSticks, SpfftProcessingUnitType processingUnit,
                             int maxNumThreads) {
  try {
    *grid = new spfft::GridFloat(maxDimX, maxDimY, maxDimZ, maxNumLocalZSticks, processingUnit,
                            maxNumThreads);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

#ifdef SPFFT_MPI
SpfftError spfft_float_grid_create_distributed(SpfftFloatGrid* grid, int maxDimX, int maxDimY, int maxDimZ,
                                         int maxNumLocalZSticks, int maxLocalZLength,
                                         SpfftProcessingUnitType processingUnit, int maxNumThreads,
                                         MPI_Comm comm, SpfftExchangeType exchangeType) {
  try {
    *grid = new spfft::GridFloat(maxDimX, maxDimY, maxDimZ, maxNumLocalZSticks, maxLocalZLength,
                            processingUnit, maxNumThreads, comm, exchangeType);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}
#endif

SpfftError spfft_float_grid_destroy(SpfftFloatGrid grid) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spfft::GridFloat*>(grid);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  grid = nullptr;
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_max_dim_x(SpfftFloatGrid grid, int* dimX) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimX = reinterpret_cast<spfft::GridFloat*>(grid)->max_dim_x();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_max_dim_y(SpfftFloatGrid grid, int* dimY) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimY = reinterpret_cast<spfft::GridFloat*>(grid)->max_dim_y();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_max_dim_z(SpfftFloatGrid grid, int* dimZ) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimZ = reinterpret_cast<spfft::GridFloat*>(grid)->max_dim_z();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_max_num_local_z_columns(SpfftFloatGrid grid, int* maxNumLocalZColumns) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *maxNumLocalZColumns = reinterpret_cast<spfft::GridFloat*>(grid)->max_num_local_z_columns();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_max_local_z_length(SpfftFloatGrid grid, int* maxLocalZLength) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *maxLocalZLength = reinterpret_cast<spfft::GridFloat*>(grid)->max_local_z_length();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_processing_unit(SpfftFloatGrid grid, SpfftProcessingUnitType* processingUnit) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *processingUnit = reinterpret_cast<spfft::GridFloat*>(grid)->processing_unit();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_device_id(SpfftFloatGrid grid, int* deviceId) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *deviceId = reinterpret_cast<spfft::GridFloat*>(grid)->device_id();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_num_threads(SpfftFloatGrid grid, int* numThreads) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *numThreads = reinterpret_cast<spfft::GridFloat*>(grid)->num_threads();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

#ifdef SPFFT_MPI
SpfftError spfft_float_grid_communicator(SpfftFloatGrid grid, MPI_Comm* comm) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *comm = reinterpret_cast<spfft::GridFloat*>(grid)->communicator();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_grid_communicator_fortran(SpfftFloatGrid grid, int* commFortran) {
  if (!grid) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *commFortran = MPI_Comm_c2f(reinterpret_cast<spfft::GridFloat*>(grid)->communicator());
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}
#endif

}

#endif
