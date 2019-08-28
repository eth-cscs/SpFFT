#include <mpi.h>
#include "gtest/gtest.h"
#include "gtest_mpi/gtest_mpi.hpp"

int main(int argc, char* argv[]) {
  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  // Add a test envirnment, which will initialize a test communicator
  // (a duplicate of MPI_COMM_WORLD)
  ::testing::AddGlobalTestEnvironment(new gtest_mpi::MPITestEnvironment());

  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener and replace with the custom MPI listener
  delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new gtest_mpi::PrettyMPIUnitTestResultPrinter());

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  MPI_Finalize();

  return exit_code;
}
