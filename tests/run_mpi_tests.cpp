#include <mpi.h>

#include "gtest/gtest.h"
#include "gtest_mpi.hpp"

int main(int argc, char* argv[]) {
  // Initialize MPI before any call to gtest_mpi
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  gtest_mpi::InitGoogleTestMPI(&argc, argv);

  auto status = RUN_ALL_TESTS();

  MPI_Finalize();

  return status;
}
