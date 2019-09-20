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
#ifndef SPFFT_TRANSFORM_HPP
#define SPFFT_TRANSFORM_HPP

#include <memory>
#include "spfft/config.h"
#include "spfft/types.h"

#ifdef SPFFT_MPI
#include <mpi.h>
#endif

namespace spfft {

template <typename T>
class SPFFT_NO_EXPORT TransformInternal;

class SPFFT_EXPORT Grid;

template <typename TransformType>
class SPFFT_NO_EXPORT MultiTransformInternal;

template <typename T>
class SPFFT_NO_EXPORT GridInternal;

/**
 * A transform in double precision with fixed dimensions. Shares memory with other transform created
 * from the same Grid object.
 */
class SPFFT_EXPORT Transform {
public:
  using ValueType = double;

  /**
   * Default copy constructor.
   */
  Transform(const Transform&) = default;

  /**
   * Default move constructor.
   */
  Transform(Transform&&) = default;

  /**
   * Default copy operator.
   */
  Transform& operator=(const Transform&) = default;

  /**
   * Default move operator.
   */
  Transform& operator=(Transform&&) = default;

  /**
   * Clone transform.
   *
   * @return Independent transform with the same parameters, but with new underlying grid.
   */
  Transform clone() const;

  /**
   * Access a transform parameter.
   * @return Type of transform.
   */
  SpfftTransformType type() const;

  /**
   * Access a transform parameter.
   * @return Dimension in x.
   */
  int dim_x() const;

  /**
   * Access a transform parameter.
   * @return Dimension in y.
   */
  int dim_y() const;

  /**
   * Access a transform parameter.
   * @return Dimension in z.
   */
  int dim_z() const;

  /**
   * Access a transform parameter.
   * @return Length in z of the space domain slice held by the local MPI rank.
   */
  int local_z_length() const;

  /**
   * Access a transform parameter.
   * @return Offset in z of the space domain slice held by the local MPI rank.
   */
  int local_z_offset() const;

  /**
   * Access a transform parameter.
   * @return Number of elements in the space domain slice held by the local MPI rank.
   */
  int local_slice_size() const;

  /**
   * Access a transform parameter.
   * @return Global number of elements in space domain. Equals dim_x() * dim_y() * dim_z().
   */
  long long int global_size() const;

  /**
   * Access a transform parameter.
   * @return Number of elements in frequency domain.
   */
  int num_local_elements() const;

  /**
   * Access a transform parameter.
   * @return Global number of elements in frequency domain.
   */
  long long int num_global_elements() const;

  /**
   * Access a transform parameter.
   * @return The processing unit used for calculations. Can be SPFFT_PU_HOST or SPFFT_PU_GPU.
   */
  SpfftProcessingUnitType processing_unit() const;

  /**
   * Access a transform parameter.
   * @return The GPU device id used. Returns always 0, if no GPU support is enabled.
   */
  int device_id() const;

  /**
   * Access a transform parameter.
   * @return The exact number of threads used by transforms created from this grid. May be less than
   * the maximum given to the constructor. Always 1, if not compiled with OpenMP support.
   */
  int num_threads() const;

#ifdef SPFFT_MPI
  /**
   * Access a transform parameter.
   * @return The internal MPI communicator.
   */
  MPI_Comm communicator() const;
#endif

  /**
   * Provides access to the space domain data.
   *
   * @param[in] dataLocation The processing unit to query for the data. Can be SPFFT_PU_HOST or
   * SPFFT_PU_GPU (if GPU is set as execution unit).
   * @return Pointer to space domain data on given processing unit. Alignment is guaranteed to
   * fulfill requirements for std::complex and C language complex types.
   * @throw GenericError SpFFT error. Can be a derived type.
   * @throw std::exception Error from standard library calls. Can be a derived type.
   */
  double* space_domain_data(SpfftProcessingUnitType dataLocation);

  /**
   * Execute a forward transform from space domain to frequency domain.
   *
   * @param[in] inputLocation The processing unit, to take the input from. Can be SPFFT_PU_HOST or
   * SPFFT_PU_GPU (if GPU is set as execution unit).
   * @param[out] output Pointer to memory, where the frequency domain elements are written to. Can
   * be located at Host or GPU memory (if GPU is set as processing unit).
   * @param[in] scaling Controls scaling of output. SPFFT_NO_SCALING to disable or
   * SPFFT_FULL_SCALING to scale by factor 1 / (dim_x() * dim_y() * dim_z()).
   * @throw GenericError SpFFT error. Can be a derived type.
   * @throw std::exception Error from standard library calls. Can be a derived type.
   */
  void forward(SpfftProcessingUnitType inputLocation, double* output,
               SpfftScalingType scaling = SPFFT_NO_SCALING);

  /**
   * Execute a backward transform from frequency domain to space domain.
   *
   * @param[in] input Input data in frequency domain. Must match the indices provided at transform
   * creation. Can be located at Host or GPU memory, if GPU is set as processing unit.
   * @param[in] outputLocation The processing unit, to place the output at. Can be SPFFT_PU_HOST or
   * SPFFT_PU_GPU (if GPU is set as execution unit).
   * @throw GenericError SpFFT error. Can be a derived type.
   * @throw std::exception Error from standard library calls. Can be a derived type.
   */
  void backward(const double* input, SpfftProcessingUnitType outputLocation);

private:
  /*! \cond PRIVATE */
  friend Grid;
  friend MultiTransformInternal<Transform>;

  SPFFT_NO_EXPORT Transform(const std::shared_ptr<GridInternal<double>>& grid,
                            SpfftProcessingUnitType executionUnit, SpfftTransformType transformType,
                            int dimX, int dimY, int dimZ, int localZLength, int numLocalElements,
                            SpfftIndexFormatType dataFormat, const int* indices);

  SPFFT_NO_EXPORT explicit Transform(std::shared_ptr<TransformInternal<double>> transform);

  std::shared_ptr<TransformInternal<double>> transform_;
  /*! \endcond */
};

}  // namespace spfft
#endif
