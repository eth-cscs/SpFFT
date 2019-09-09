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
#ifndef SPFFT_MPI_MATCH_ELEMENTARY_TYPE_HPP
#define SPFFT_MPI_MATCH_ELEMENTARY_TYPE_HPP

#include <mpi.h>
#include "spfft/config.h"

namespace spfft {

template <typename T>
struct MPIMatchElementaryType;

template <>
struct MPIMatchElementaryType<char> {
  inline static auto get() -> MPI_Datatype { return MPI_CHAR; }
};

template <>
struct MPIMatchElementaryType<signed short int> {
  inline static auto get() -> MPI_Datatype { return MPI_SHORT; }
};

template <>
struct MPIMatchElementaryType<signed int> {
  inline static auto get() -> MPI_Datatype { return MPI_INT; }
};

template <>
struct MPIMatchElementaryType<signed long int> {
  inline static auto get() -> MPI_Datatype { return MPI_LONG; }
};

template <>
struct MPIMatchElementaryType<signed long long int> {
  inline static auto get() -> MPI_Datatype { return MPI_LONG_LONG; }
};

template <>
struct MPIMatchElementaryType<signed char> {
  inline static auto get() -> MPI_Datatype { return MPI_SIGNED_CHAR; }
};

template <>
struct MPIMatchElementaryType<unsigned char> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_CHAR; }
};

template <>
struct MPIMatchElementaryType<unsigned short int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_SHORT; }
};

template <>
struct MPIMatchElementaryType<unsigned int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED; }
};

template <>
struct MPIMatchElementaryType<unsigned long int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_LONG; }
};

template <>
struct MPIMatchElementaryType<unsigned long long int> {
  inline static auto get() -> MPI_Datatype { return MPI_UNSIGNED_LONG_LONG; }
};

template <>
struct MPIMatchElementaryType<float> {
  inline static auto get() -> MPI_Datatype { return MPI_FLOAT; }
};

template <>
struct MPIMatchElementaryType<double> {
  inline static auto get() -> MPI_Datatype { return MPI_DOUBLE; }
};

template <>
struct MPIMatchElementaryType<long double> {
  inline static auto get() -> MPI_Datatype { return MPI_LONG_DOUBLE; }
};

}  // namespace spfft

#endif
