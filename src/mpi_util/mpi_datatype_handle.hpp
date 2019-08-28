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
#ifndef SPFFT_MPI_DATATYPE_HANDLE_HPP
#define SPFFT_MPI_DATATYPE_HANDLE_HPP

#include <mpi.h>
#include <memory>
#include <vector>
#include "mpi_util/mpi_check_status.hpp"
#include "spfft/config.h"

namespace spfft {

// Storage for MPI datatypes
class MPIDatatypeHandle {
public:
  MPIDatatypeHandle() = default;

  // Create custom datatype with ownership
  // Does not call MPI_Type_commit!
  // Can take predifined MPI types such as MPI_DOUBLE, on which MPI_Type_free() will not be called
  // NOTE: Freeing a MPI_Datatype on which this type depends on does not affect this type (see "The
  // MPI core")
  MPIDatatypeHandle(const MPI_Datatype& mpiType) {
    assert(mpiType != MPI_DATATYPE_NULL);
    int numIntegers, numAddresses, numDatatypes, combiner;
    mpi_check_status(
        MPI_Type_get_envelope(mpiType, &numIntegers, &numAddresses, &numDatatypes, &combiner));
    if (combiner != MPI_COMBINER_NAMED && combiner != MPI_COMBINER_DUP) {
      // take ownership and call MPI_Type_free upon release
      type_ = std::shared_ptr<MPI_Datatype>(new MPI_Datatype(mpiType), [](MPI_Datatype* ptr) {
        assert(*ptr != MPI_DATATYPE_NULL);
        MPI_Type_free(ptr);
        delete ptr;
      });
    } else {
      // only copy type descriptor, will not call MPI_Type_free()
      type_ = std::make_shared<MPI_Datatype>(mpiType);
    }
  }

  inline auto get() const -> const MPI_Datatype& {
    assert(type_);
    assert(*type_ != MPI_DATATYPE_NULL);
    return *type_;
  }

  inline auto empty() const noexcept -> bool { return type_ == nullptr; }

  inline static MPIDatatypeHandle create_contiguous(int count, MPI_Datatype oldType) {
    MPI_Datatype newType;
    mpi_check_status(MPI_Type_contiguous(count, oldType, &newType));
    mpi_check_status(MPI_Type_commit(&newType));
    return MPIDatatypeHandle(newType);
  }

  inline static MPIDatatypeHandle create_vector(int count, int blocklength, int stride,
                                                MPI_Datatype oldType) {
    MPI_Datatype newType;
    mpi_check_status(MPI_Type_vector(count, blocklength, stride, oldType, &newType));
    mpi_check_status(MPI_Type_commit(&newType));
    return MPIDatatypeHandle(newType);
  }

  inline static MPIDatatypeHandle create_hindexed(int count, const int arrayOfBlocklengths[],
                                                  const MPI_Aint arrayOfDispls[],
                                                  MPI_Datatype oldType) {
    MPI_Datatype newType;
    mpi_check_status(
        MPI_Type_create_hindexed(count, arrayOfBlocklengths, arrayOfDispls, oldType, &newType));
    mpi_check_status(MPI_Type_commit(&newType));
    return MPIDatatypeHandle(newType);
  }

  inline static MPIDatatypeHandle create_subarray(int ndims, const int arrayOfSizes[],
                                                  const int arrayOfSubsizes[],
                                                  const int arrayOfStarts[], int order,
                                                  MPI_Datatype oldType) {
    MPI_Datatype newType;
    mpi_check_status(MPI_Type_create_subarray(ndims, arrayOfSizes, arrayOfSubsizes, arrayOfStarts,
                                              order, oldType, &newType));
    mpi_check_status(MPI_Type_commit(&newType));
    return MPIDatatypeHandle(newType);
  }

private:
  std::shared_ptr<MPI_Datatype> type_ = nullptr;
};

} // namespace spfft

#endif
