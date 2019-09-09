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
#ifndef SPFFT_MPI_COMMUNICATOR_HANDLE_HPP
#define SPFFT_MPI_COMMUNICATOR_HANDLE_HPP

#include <mpi.h>
#include <cassert>
#include <memory>
#include "mpi_util/mpi_check_status.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"

namespace spfft {

// MPI Communicator, which creates a duplicate at construction time.
// Copies of the object share the same communicator, which is reference counted.
class MPICommunicatorHandle {
public:
  MPICommunicatorHandle() : comm_(new MPI_Comm(MPI_COMM_SELF)), size_(1), rank_(0) {}

  MPICommunicatorHandle(const MPI_Comm& comm) {
    // create copy of communicator
    MPI_Comm newComm;
    mpi_check_status(MPI_Comm_dup(comm, &newComm));

    comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(newComm), [](MPI_Comm* ptr) {
      MPI_Comm_free(ptr);
      delete ptr;
    });

    int sizeInt, rankInt;
    mpi_check_status(MPI_Comm_size(*comm_, &sizeInt));
    mpi_check_status(MPI_Comm_rank(*comm_, &rankInt));

    if (sizeInt < 1 || rankInt < 0) {
      throw MPIError();
    }
    rank_ = static_cast<SizeType>(rankInt);
    size_ = static_cast<SizeType>(sizeInt);
  }

  inline auto get() const -> const MPI_Comm& { return *comm_; }

  inline auto size() const noexcept -> SizeType { return size_; }

  inline auto rank() const noexcept -> SizeType { return rank_; }

private:
  std::shared_ptr<MPI_Comm> comm_ = nullptr;
  SizeType size_ = 1;
  SizeType rank_ = 0;
};

}  // namespace spfft

#endif
