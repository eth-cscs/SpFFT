
if(SPFFT_BUILD_TESTS)
	cmake_minimum_required(VERSION 3.11 FATAL_ERROR) # git fetch module requires at least 3.11
	set(BUILD_GMOCK OFF CACHE BOOL "")
	set(INSTALL_GTEST OFF CACHE BOOL "")
	mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)
	include(FetchContent)

	# add googletest
	FetchContent_Declare(
	  googletest
	  GIT_REPOSITORY https://github.com/google/googletest.git
	  GIT_TAG        release-1.8.1
	)
	FetchContent_GetProperties(googletest)
	if(NOT googletest_POPULATED)
	  message(STATUS "Downloading Google Test repository...")
	  FetchContent_Populate(googletest)
	endif()
	add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

	# add gtest_mpi
	FetchContent_Declare(
	  gtest_mpi
	  GIT_REPOSITORY https://github.com/AdhocMan/gtest_mpi.git
	  GIT_TAG        v1.0.0
	)
	FetchContent_GetProperties(gtest_mpi)
	if(NOT gtest_mpi_POPULATED)
	  message(STATUS "Downloading Google Test MPI extension repository...")
	  FetchContent_Populate(gtest_mpi)
	endif()
	add_subdirectory(${gtest_mpi_SOURCE_DIR} ${gtest_mpi_BINARY_DIR})

	# add command line parser
	FetchContent_Declare(
	  cli11
	  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
	  GIT_TAG        v1.7.1
	)
	FetchContent_GetProperties(cli11)
	if(NOT cli11_POPULATED)
	  message(STATUS "Downloading CLI11 command line parser repository...")
	  FetchContent_Populate(cli11)
	endif()
	list(APPEND SPFFT_EXTERNAL_INCLUDE_DIRS ${cli11_SOURCE_DIR}/include)

	# add json parser
	set(JSON_Install OFF CACHE BOOL "")
	FetchContent_Declare(
	  json
	  GIT_REPOSITORY https://github.com/nlohmann/json.git
	  GIT_TAG        v3.6.1
	)
	FetchContent_GetProperties(json)
	if(NOT json_POPULATED)
	  message(STATUS "Downloading json repository...")
	  FetchContent_Populate(json)
	endif()
	set(JSON_BuildTests OFF CACHE INTERNAL "")
	add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
	list(APPEND SPFFT_EXTERNAL_LIBS nlohmann_json::nlohmann_json)


	list(APPEND SPFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/tests)

	# benchmark executable
	add_executable(benchmark programs/benchmark.cpp)
	target_link_libraries(benchmark PRIVATE spfft_test ${SPFFT_EXTERNAL_LIBS})
	target_include_directories(benchmark PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})

	# test executables
	add_executable(run_local_tests
		run_local_tests.cpp
		local_tests/test_host_array.cpp
		local_tests/test_disjoint.cpp
		local_tests/test_fftw_prop_hash.cpp
		local_tests/test_local_transform.cpp
		)
	target_link_libraries(run_local_tests PRIVATE gtest_main gtest_mpi)
	target_link_libraries(run_local_tests PRIVATE spfft_test ${SPFFT_EXTERNAL_LIBS})
	target_include_directories(run_local_tests PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})

	if(SPFFT_MPI)
		add_executable(run_mpi_tests
			run_mpi_tests.cpp
			mpi_tests/test_transform.cpp
			mpi_tests/test_multi_transform.cpp
			mpi_tests/test_transpose.cpp
			mpi_tests/test_transpose_gpu.cpp
			)
		target_link_libraries(run_mpi_tests PRIVATE gtest_main gtest_mpi)
		target_link_libraries(run_mpi_tests PRIVATE spfft_test ${SPFFT_EXTERNAL_LIBS})
		target_include_directories(run_mpi_tests PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})
	endif()

endif()
