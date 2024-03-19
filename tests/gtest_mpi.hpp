#ifndef GTEST_MPI_HPP
#define GTEST_MPI_HPP

#include <gtest/gtest.h>

namespace gtest_mpi {
// Internal helper struct
struct TestGuard {
  void (*func)() = nullptr;

  ~TestGuard() {
    if (func)
      func();
  }
};

// Initialize GoogleTest and MPI functionality. MPI_Init has to called before.
void InitGoogleTestMPI(int *argc, char **argv);

// Create a test guard, which has to be placed in all test cases.
TestGuard CreateTestGuard();

} // namespace gtest_mpi

// Helper macro for creating a test guard within test cases.
#define GTEST_MPI_GUARD auto gtest_mpi_guard__LINE__ = ::gtest_mpi::CreateTestGuard();

#endif
