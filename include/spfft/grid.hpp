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
#ifndef SPFFT_GRID_HPP
#define SPFFT_GRID_HPP

#include <memory>
#include "spfft/config.h"
#include "spfft/transform.hpp"
#include "spfft/types.h"

#ifdef SPFFT_MPI
#include <mpi.h>
#endif

namespace spfft {

// Forward declaration for internal use
template <typename T>
class SPFFT_NO_EXPORT GridInternal;

/**
 * A Grid, which provides pre-allocated memory for double precision transforms.
 */
class SPFFT_EXPORT Grid {
public:
  /**
   * Constructor for a local grid.
   *
   * @param[in] maxDimX Maximum dimension in x.
   * @param[in] maxDimY Maximum dimension in y.
   * @param[in] maxDimZ Maximum dimension in z.
   * @param[in] maxNumLocalZColumns Maximum number of z-columns in frequency domain.
   * @param[in] processingUnit The processing unit type to prepare for. Can be SPFFT_PU_HOST or
   * SPFFT_PU_GPU or SPFFT_PU_HOST | SPFFT_PU_GPU.
   * @param[in] maxNumThreads The maximum number of threads, transforms created with this grid are
   * allowed to use. If smaller than 1, the OpenMP default value is used.
   * @throw GenericError SpFFT error. Can be a derived type.
   * @throw std::exception Error from standard library calls. Can be a derived type.
   */
  Grid(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZColumns,
       SpfftProcessingUnitType processingUnit, int maxNumThreads);

#ifdef SPFFT_MPI
  /**
   * Constructor for a distributed grid.
   * Thread-safe if MPI thread support is set to MPI_THREAD_MULTIPLE.
   *
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
   * @throw GenericError SpFFT error. Can be a derived type.
   * @throw std::exception Error from standard library calls. Can be a derived type.
   */
  Grid(int maxDimX, int maxDimY, int maxDimZ, int maxNumLocalZColumns, int maxLocalZLength,
       SpfftProcessingUnitType processingUnit, int maxNumThreads, MPI_Comm comm,
       SpfftExchangeType exchangeType);
#endif

  /**
   * Custom copy constructor.
   *
   * Creates a independent copy. Calls MPI functions for the distributed case.
   */
  Grid(const Grid&);

  /**
   * Default move constructor.
   */
  Grid(Grid&&) = default;

  /**
   * Custom copy operator.
   *
   * Creates a independent copy. Calls MPI functions for the distributed case.
   */
  Grid& operator=(const Grid&);

  /**
   * Default move operator.
   */
  Grid& operator=(Grid&&) = default;

  /**
   * Creates a transform from this grid object.
   * Thread-safe if no FFTW calls are executed concurrently.
   *
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
   * @return Transform
   * @throw GenericError SpFFT error. Can be a derived type.
   * @throw std::exception Error from standard library calls. Can be a derived type.
   */
  Transform create_transform(SpfftProcessingUnitType processingUnit,
                             SpfftTransformType transformType, int dimX, int dimY, int dimZ,
                             int localZLength, int numLocalElements,
                             SpfftIndexFormatType indexFormat, const int* indices) const;

  /**
   * Access a grid parameter.
   * @return Maximum dimension in x.
   */
  int max_dim_x() const;

  /**
   * Access a grid parameter.
   * @return Maximum dimension in y.
   */
  int max_dim_y() const;

  /**
   * Access a grid parameter.
   * @return Maximum dimension in z.
   */
  int max_dim_z() const;

  /**
   * Access a grid parameter.
   * @return Maximum number of z-columns in frequency domain of the local MPI rank.
   */
  int max_num_local_z_columns() const;

  /**
   * Access a grid parameter.
   * @return Maximum length in z in space domain of the local MPI rank.
   */
  int max_local_z_length() const;

  /**
   * Access a grid parameter.
   * @return The processing unit, the grid has prepared for. Can be SPFFT_PU_HOST or SPFFT_PU_GPU or
   * SPFFT_PU_HOST | SPFFT_PU_GPU.
   */
  SpfftProcessingUnitType processing_unit() const;

  /**
   * Access a grid parameter.
   * @return The GPU device id used. Always returns 0, if no GPU support is enabled.
   */
  int device_id() const;

  /**
   * Access a grid parameter.
   * @return The exact number of threads used by transforms created from this grid. May be less than
   * the maximum given to the constructor. Always 1, if not compiled with OpenMP support.
   */
  int num_threads() const;

#ifdef SPFFT_MPI
  /**
   * Access a grid parameter.
   * @return The internal MPI communicator.
   */
  MPI_Comm communicator() const;
#endif

private:
  std::shared_ptr<GridInternal<double>> grid_;
};
}  // namespace spfft
#endif
