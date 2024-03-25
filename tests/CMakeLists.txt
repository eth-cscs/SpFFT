
if(SPFFT_BUILD_TESTS)
	cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # FetchContent_MakeAvailable requires at least 3.14

	# update time stamps when using FetchContent
	if(POLICY CMP0135)
		cmake_policy(SET CMP0135 NEW)
	endif()

	set(BUILD_GMOCK OFF CACHE BOOL "")
	set(INSTALL_GTEST OFF CACHE BOOL "")
	mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)
	include(FetchContent)

	# add googletest
	if(SPFFT_BUNDLED_GOOGLETEST)
		FetchContent_Declare(
			googletest
			URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
			URL_MD5 c8340a482851ef6a3fe618a082304cfc
		)
		FetchContent_MakeAvailable(googletest)
	else()
	  find_package(GTest CONFIG REQUIRED)
	endif()
	list(APPEND SPFFT_TEST_LIBRARIES GTest::gtest_main)

	# add command line parser
	if(SPFFT_BUNDLED_CLI11)
		FetchContent_Declare(
			cli11
			URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.3.2.tar.gz
			URL_MD5 b80cb645dee25982110b068b426363ff
		)
		FetchContent_MakeAvailable(cli11)
	else()
		find_package(CLI11 CONFIG REQUIRED)
	endif()
	list(APPEND SPFFT_TEST_LIBRARIES CLI11::CLI11)

  # add json parser
	if(SPFFT_BUNDLED_JSON)
	  FetchContent_Declare(
		json
		URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
		URL_MD5 e8d56bc54621037842ee9f0aeae27746
	  )
	  FetchContent_MakeAvailable(json)
	else()
	  find_package(nlohmann_json CONFIG REQUIRED)
	endif()

	list(APPEND SPFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/tests)

	# benchmark executable
	add_executable(benchmark programs/benchmark.cpp)
	target_link_libraries(benchmark PRIVATE spfft_test ${SPFFT_EXTERNAL_LIBS} CLI11::CLI11 nlohmann_json::nlohmann_json)
	target_include_directories(benchmark PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})

	# test executables
	add_executable(run_local_tests
		run_local_tests.cpp
		local_tests/test_host_array.cpp
		local_tests/test_disjoint.cpp
		local_tests/test_fftw_prop_hash.cpp
		local_tests/test_local_transform.cpp
		)
	target_link_libraries(run_local_tests PRIVATE GTest::gtest_main)
	target_link_libraries(run_local_tests PRIVATE spfft_test ${SPFFT_EXTERNAL_LIBS})
	target_include_directories(run_local_tests PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})

	if(SPFFT_MPI)
		add_executable(run_mpi_tests
			run_mpi_tests.cpp
			gtest_mpi.cpp
			mpi_tests/test_transform.cpp
			mpi_tests/test_multi_transform.cpp
			mpi_tests/test_transpose.cpp
			mpi_tests/test_transpose_gpu.cpp
			)
		target_link_libraries(run_mpi_tests PRIVATE GTest::gtest_main)
		target_link_libraries(run_mpi_tests PRIVATE spfft_test ${SPFFT_EXTERNAL_LIBS})
		target_include_directories(run_mpi_tests PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})
	endif()

endif()
