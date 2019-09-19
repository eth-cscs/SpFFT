cmake_minimum_required(VERSION 3.11 FATAL_ERROR) # 3.8 added CUDA language support without FindCUDA
project(SpFFT LANGUAGES CXX VERSION 0.9.4)

# allow {module}_ROOT variables to be set
if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()

# use INTERFACE_LINK_LIBRARIES property if available
if(POLICY CMP0022)
	cmake_policy(SET CMP0022 NEW)
endif()

# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
		)
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 11)

#add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

# Options
option(SPFFT_STATIC "Compile as static library" OFF)
option(SPFFT_OMP "Compile with OpenMP support" ON)
option(SPFFT_MPI "Compile with MPI support" ON)
option(SPFFT_GPU_DIRECT "Compile with GPU direct (GPU aware MPI) support." OFF)
option(SPFFT_BUILD_TESTS "Build tests" OFF)
option(SPFFT_SINGLE_PRECISION "Enable single precision support" OFF)
option(SPFFT_INSTALL "Enable CMake install commands" ON)

set(SPFFT_GPU_BACKEND "OFF" CACHE STRING "GPU backend")
set_property(CACHE SPFFT_GPU_BACKEND PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)


# set preferred library type
if (SPFFT_STATIC)
	# prefer static over dynamic libraries with the find_library() command by changing the order
	set(CMAKE_FIND_LIBRARY_SUFFIXES_SAVE ${CMAKE_FIND_LIBRARY_SUFFIXES})
	if(APPLE)
		set(CMAKE_FIND_LIBRARY_SUFFIXES .a .tbd .dylib .so)
	elseif(UNIX)
		set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
	endif()
	set(SPFFT_LIBRARY_TYPE STATIC)
else()
	set(SPFFT_LIBRARY_TYPE SHARED)
endif()

set(SPFFT_DEFINITIONS)
set(SPFFT_EXTERNAL_COMPILE_OPTIONS)
set(SPFFT_LIBS)
set(SPFFT_EXTERNAL_LIBS)
set(SPFFT_INTERFACE_LIBS)
set(SPFFT_INCLUDE_DIRS)
set(SPFFT_EXTERNAL_INCLUDE_DIRS)

# Options combination check
set(SPFFT_CUDA OFF)
set(SPFFT_ROCM OFF)
if(SPFFT_GPU_BACKEND)
	if(SPFFT_GPU_BACKEND STREQUAL "CUDA")
		set(SPFFT_CUDA ON)
	elseif(SPFFT_GPU_BACKEND STREQUAL "ROCM")
		set(SPFFT_ROCM ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend option")
	endif()
endif()
mark_as_advanced(SPFFT_CUDA SPFFT_ROCM)

# CUDA
if(SPFFT_CUDA)
	enable_language(CUDA)
	find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	find_library(CUDA_CUFFT_LIBRARY cufft PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	list(APPEND SPFFT_EXTERNAL_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_CUFFT_LIBRARY})
	list(APPEND SPFFT_EXTERNAL_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# ROCM
if(SPFFT_ROCM)
	find_package(ROCM REQUIRED COMPONENTS rocfft)
	list(APPEND SPFFT_EXTERNAL_INCLUDE_DIRS ${ROCM_INCLUDE_DIRS})
	list(APPEND SPFFT_EXTERNAL_LIBS ${ROCM_LIBRARIES})
	list(APPEND SPFFT_EXTERNAL_COMPILE_OPTIONS ${ROCM_DEFINITIONS})
endif()



if(SPFFT_MPI)
	find_package(MPI REQUIRED)
	list(APPEND SPFFT_EXTERNAL_LIBS MPI::MPI_CXX)
	list(APPEND SPFFT_INTERFACE_LIBS ${MPI_CXX_LIBRARIES})
endif()

if(SPFFT_OMP)
	find_package(OpenMP REQUIRED)
	list(APPEND SPFFT_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
	list(APPEND SPFFT_INTERFACE_LIBS ${OpenMP_CXX_LIBRARIES})
endif()

if(SPFFT_GPU_DIRECT)
	message(STATUS "GPU Direct support enabled: Additional environment variables might have to be set before execution. (e.g \"export MPICH_RDMA_ENABLED_CUDA=1\")")
endif()



# Use MKL if available, otherwise require FFTW3
find_package(MKLSequential)
if(MKLSequential_FOUND)
	list(APPEND SPFFT_EXTERNAL_LIBS MKL::Sequential)
	list(APPEND SPFFT_INTERFACE_LIBS ${MKLSequential_LIBRARIES})
else()
	find_package(FFTW REQUIRED)
	list(APPEND SPFFT_EXTERNAL_LIBS FFTW::FFTW)
	list(APPEND SPFFT_INTERFACE_LIBS ${FFTW_LIBRARIES})
	if(SPFFT_SINGLE_PRECISION AND NOT FFTW_FLOAT_FOUND)
		message(FATAL_ERROR "FFTW library with single precision support NOT FOUND. Disable SPFFT_SINGLE_PRECISION or provide path to library.")
	endif()
endif()


if(SPFFT_BUILD_TESTS)
	# enable timing with testing
	set(SPFFT_TIMING ON)
endif()

# generate config.h
configure_file(include/spfft/config.h.in ${PROJECT_BINARY_DIR}/spfft/config.h)

list(APPEND SPFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)
list(APPEND SPFFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
list(APPEND SPFFT_INCLUDE_DIRS ${PROJECT_BINARY_DIR})
list(APPEND SPFFT_EXTERNAL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext)

#############################################################################
# All include dirs and definitions must be set before sub-directory is added!
#############################################################################
add_subdirectory(src)

list(APPEND SPFFT_LIBS spfft)

# add tests for developement
if(SPFFT_BUILD_TESTS)
	add_subdirectory(tests)
endif()

# reset cmake library suffixes
if(SPFFT_STATIC)
	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVE})
endif()
