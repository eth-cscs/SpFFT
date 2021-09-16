cmake_minimum_required(VERSION 3.11 FATAL_ERROR) # 3.11 to avoid issues with OpenMP + CUDA
project(SpFFT LANGUAGES CXX VERSION 1.0.4)
set(SPFFT_SO_VERSION 1)
set(SPFFT_VERSION ${PROJECT_VERSION})

# allow {module}_ROOT variables to be set
if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()

# Initialize CMAKE_CUDA_ARCHITECTURES through nvcc if possible
if(POLICY CMP0104)
	cmake_policy(SET CMP0104 NEW)
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
option(SPFFT_FORTRAN "Compile fortran module" OFF)

set(SPFFT_GPU_BACKEND "OFF" CACHE STRING "GPU backend")
set_property(CACHE SPFFT_GPU_BACKEND PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)

set(SPFFT_FFTW_LIB "AUTO" CACHE STRING "Library providing a FFTW interface")
set_property(CACHE SPFFT_FFTW_LIB PROPERTY STRINGS
	"AUTO" "FFTW" "MKL" "ARMPL"
	)

# Get GNU standard install prefixes
include(GNUInstallDirs)

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

set(SPFFT_EXTERNAL_LIBS)
set(SPFFT_INCLUDE_DIRS)
set(SPFFT_EXTERNAL_INCLUDE_DIRS)
set(SPFFT_EXTERNAL_PKG_PACKAGES)

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

# Fortran
if(SPFFT_FORTRAN)
	enable_language(Fortran)
endif()

# CUDA
if(SPFFT_CUDA)
	enable_language(CUDA)

	if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
		find_package(CUDAToolkit REQUIRED)
	else()
		find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		find_library(CUDA_CUFFT_LIBRARY cufft PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cudart)
			add_library(CUDA::cudart INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY})
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
		if(NOT TARGET CUDA::cufft)
			add_library(CUDA::cufft INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cufft PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUFFT_LIBRARY})
		set_property(TARGET CUDA::cufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()

	list(APPEND SPFFT_EXTERNAL_LIBS CUDA::cudart CUDA::cufft)
endif()

# ROCM
if(SPFFT_ROCM)
	find_package(hip CONFIG REQUIRED)
	find_package(rocfft CONFIG REQUIRED)
	find_package(hipfft CONFIG) # hipfft within rocfft is deprecated. Use separate hipfft if available (not required).

	if(hipfft_FOUND)
		# Issue with rocm 4.1.0: Symlink to rocfft provided hipfft.h in /opt/rocm/include.
		# Workaround: Only use hipfft include directory with hipfft target and place before other hip targets in lib list
		if(HIPFFT_INCLUDE_DIRS)
			set_property(TARGET hip::hipfft PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HIPFFT_INCLUDE_DIRS})
		endif()
		list(APPEND SPFFT_EXTERNAL_LIBS hip::hipfft)
	endif()

	list(APPEND SPFFT_EXTERNAL_LIBS hip::host roc::rocfft)

	# FindHIP module provides compilation command for GPU code
	find_package(HIP MODULE REQUIRED)
	if(NOT HIP_HCC_FLAGS)
		message(STATUS "Using default AMD gpu targets: gfx803, gfx900, gfx906. Set HIP_HCC_FLAGS to override.")
		set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906)
	endif()
endif()

if(SPFFT_MPI)
	find_package(MPI COMPONENTS CXX REQUIRED)
	list(APPEND SPFFT_EXTERNAL_LIBS MPI::MPI_CXX)
endif()

if(SPFFT_OMP)
	find_package(OpenMP COMPONENTS CXX REQUIRED)
	list(APPEND SPFFT_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
endif()

if(SPFFT_GPU_DIRECT)
	message(STATUS "GPU Direct support enabled: Additional environment variables might have to be set before execution. (e.g \"export MPICH_RDMA_ENABLED_CUDA=1\")")
endif()


# FFTW library must be found if not set to AUTO
set(_SPFFT_FIND_FFTW_LIB_OPTION)
if(NOT ${SPFFT_FFTW_LIB} STREQUAL "AUTO")
	set(_SPFFT_FIND_FFTW_LIB_OPTION REQUIRED)
endif()

set(SPFFT_MKL OFF)
set(SPFFT_ARMPL OFF)
set(SPFFT_FFTW OFF)

# Look for MKL first
if(${SPFFT_FFTW_LIB} STREQUAL "AUTO" OR ${SPFFT_FFTW_LIB} STREQUAL "MKL")
	# Use MKL if available, otherwise require FFTW3
	if(UNIX AND NOT APPLE)
		# prefer static MKL in Linux. Together with "-Wl,--exclude-libs,ALL",
		# symbols are not visible for linking afterwards and no conflicts with other MKL versions of other libraries should exist.
		set(_TMP_SAVE ${CMAKE_FIND_LIBRARY_SUFFIXES})
		set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
	endif()
	find_package(MKLSequential ${_SPFFT_FIND_FFTW_LIB_OPTION})
	if(UNIX AND NOT APPLE)
		set(CMAKE_FIND_LIBRARY_SUFFIXES ${_TMP_SAVE})
		unset(_TMP_SAVE)
	endif()
	if(TARGET MKL::Sequential)
		list(APPEND SPFFT_EXTERNAL_LIBS MKL::Sequential)
		list(APPEND SPFFT_EXTERNAL_PKG_PACKAGES mkl-dynamic-lp64-seq)
		set(SPFFT_MKL ON)
	endif()
endif()


# Look for ARM PL
if(NOT SPFFT_MKL AND ${SPFFT_FFTW_LIB} STREQUAL "AUTO" OR ${SPFFT_FFTW_LIB} STREQUAL "ARMPL")
	find_package(ARMPL ${_SPFFT_FIND_FFTW_LIB_OPTION})
	if(TARGET ARM::pl)
		list(APPEND SPFFT_EXTERNAL_LIBS ARM::pl)
		set(SPFFT_ARMPL ON)
	endif()
endif()

# Look for FFTW library if required
if(NOT SPFFT_MKL AND NOT SPFFT_ARMPL)
	find_package(FFTW REQUIRED)
	list(APPEND SPFFT_EXTERNAL_LIBS FFTW::FFTW)
	if(SPFFT_SINGLE_PRECISION)
		find_package(FFTWF REQUIRED)
		list(APPEND SPFFT_EXTERNAL_LIBS FFTWF::FFTWF)
	endif()
	list(APPEND SPFFT_EXTERNAL_PKG_PACKAGES fftw3)
	set(SPFFT_FFTW ON)
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

# add tests for developement
if(SPFFT_BUILD_TESTS)
	add_subdirectory(tests)
endif()

# reset cmake library suffixes
if(SPFFT_STATIC)
	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVE})
endif()
