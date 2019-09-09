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
#ifndef SPFFT_TYPES_H
#define SPFFT_TYPES_H

#include "spfft/config.h"

enum SpfftExchangeType {
  /**
   * Default exchange. Equivalent to SPFFT_EXCH_COMPACT_BUFFERED.
   */
  SPFFT_EXCH_DEFAULT,
  /**
   * Exchange based on MPI_Alltoall.
   */
  SPFFT_EXCH_BUFFERED,
  /**
   * Exchange based on MPI_Alltoall in single precision.
   * Slight accuracy loss for double precision transforms due to conversion to float prior to MPI
   * exchange.
   */
  SPFFT_EXCH_BUFFERED_FLOAT,
  /**
   * Exchange based on MPI_Alltoallv.
   */
  SPFFT_EXCH_COMPACT_BUFFERED,
  /**
   * Exchange based on MPI_Alltoallv in single precision.
   * Slight accuracy loss for double precision transforms due to conversion to float prior to MPI
   * exchange.
   */
  SPFFT_EXCH_COMPACT_BUFFERED_FLOAT,
  /**
   * Exchange based on MPI_Alltoallw.
   */
  SPFFT_EXCH_UNBUFFERED
};

/**
 * Processing unit type
 */
enum SpfftProcessingUnitType {
  /**
   * HOST
   */
  SPFFT_PU_HOST = 1,
  /**
   * GPU
   */
  SPFFT_PU_GPU = 2
};

enum SpfftIndexFormatType {
  /**
   * Triplets of x,y,z frequency indices
   */
  SPFFT_INDEX_TRIPLETS
};

enum SpfftTransformType {
  /**
   * Complex-to-Complex transform
   */
  SPFFT_TRANS_C2C,

  /**
   * Real-to-Complex transform
   */
  SPFFT_TRANS_R2C
};

enum SpfftScalingType {
  /**
   * No scaling
   */
  SPFFT_NO_SCALING,
  /**
   * Full scaling
   */
  SPFFT_FULL_SCALING
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum SpfftExchangeType SpfftExchangeType;
typedef enum SpfftProcessingUnitType SpfftProcessingUnitType;
typedef enum SpfftTransformType SpfftTransformType;
typedef enum SpfftIndexFormatType SpfftIndexFormatType;
typedef enum SpfftScalingType SpfftScalingType;
/*! \endcond */
#endif  // cpp
#endif
