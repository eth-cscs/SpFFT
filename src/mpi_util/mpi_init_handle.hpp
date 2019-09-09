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
#ifndef SPFFT_MPI_INIT_HANDLE_HPP
#define SPFFT_MPI_INIT_HANDLE_HPP

#include <mpi.h>
#include "mpi_util/mpi_check_status.hpp"
#include "spfft/config.h"

namespace spfft {

// MPI Communicator, which creates a duplicate at construction time.
// Copies of the object share the same communicator, which is reference counted.
class MPIInitHandle {
public:
  MPIInitHandle(int& argc, char**& argv, bool callFinalize) : callFinalize_(callFinalize) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      // MPI_Init(&argc, &argv);
      int provided;
      mpi_check_status(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    }
  }

  // unmovable
  MPIInitHandle(const MPIInitHandle& other) = delete;
  MPIInitHandle(MPIInitHandle&& other) = delete;
  auto operator=(const MPIInitHandle& other) -> MPIInitHandle& = delete;
  auto operator=(MPIInitHandle&& other) -> MPIInitHandle& = delete;

  ~MPIInitHandle() {
    if (callFinalize_) {
      MPI_Finalize();
    }
  }

private:
  bool callFinalize_ = false;
};

}  // namespace spfft

#endif
