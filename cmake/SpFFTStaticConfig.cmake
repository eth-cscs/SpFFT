include(CMakeFindDependencyMacro)
macro(find_dependency_components)
	if(${ARGV0}_FOUND AND ${CMAKE_VERSION} VERSION_LESS "3.15.0")
		# find_dependency does not handle new components correctly before 3.15.0
		set(${ARGV0}_FOUND FALSE)
	endif()
	find_dependency(${ARGV})
endmacro()

# options used for building library
set(SPFFT_OMP @SPFFT_OMP@)
set(SPFFT_MPI @SPFFT_MPI@)
set(SPFFT_STATIC @SPFFT_STATIC@)
set(SPFFT_GPU_DIRECT @SPFFT_GPU_DIRECT@)
set(SPFFT_SINGLE_PRECISION @SPFFT_SINGLE_PRECISION@)
set(SPFFT_FFTW_LIB @SPFFT_FFTW_LIB@)
set(SPFFT_GPU_BACKEND @SPFFT_GPU_BACKEND@)
set(SPFFT_CUDA @SPFFT_CUDA@)
set(SPFFT_ROCM @SPFFT_ROCM@)
set(SPFFT_MKL @SPFFT_MKL@)
set(SPFFT_ARMPL @SPFFT_ARMPL@)
set(SPFFT_FFTW @SPFFT_FFTW@)

# make sure CXX is enabled
get_property(_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if(SpFFT_FIND_REQUIRED AND NOT "CXX" IN_LIST _LANGUAGES)
	message(FATAL_ERROR "SpFFT requires CXX language to be enabled for static linking.")
endif()

# Only look for modules we installed and save value
set(_CMAKE_MODULE_PATH_SAVE ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")

if(SPFFT_MKL)
	find_dependency(MKLSequential)
endif()
if(SPFFT_ARMPL)
	find_dependency(ARMPL)
endif()
if(SPFFT_FFTW)
	find_dependency(FFTW)
endif()

if(SPFFT_OMP AND NOT TARGET OpenMP::OpenMP_CXX)
	find_dependency_components(OpenMP COMPONENTS CXX)
endif()

if(SPFFT_MPI AND NOT TARGET MPI::MPI_CXX)
	find_dependency_components(MPI COMPONENTS CXX)
endif()


if(SPFFT_CUDA)
	if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
		find_dependency(CUDAToolkit)
	else()
		enable_language(CUDA)
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
endif()

if(SPFFT_ROCM)
	find_dependency(hip CONFIG)
	find_dependency(rocfft CONFIG)
	find_dependency(hipfft CONFIG)
endif()

set(CMAKE_MODULE_PATH ${_CMAKE_MODULE_PATH_SAVE}) # restore module path

# add version of package
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticConfigVersion.cmake")

# add library target
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticTargets.cmake")

# Make MPI dependency public to compile interface depending on enabled languages
if(SPFFT_MPI)
	if("CXX" IN_LIST _LANGUAGES)
		target_link_libraries(SpFFT::spfft INTERFACE MPI::MPI_CXX)
	endif()

	if("C" IN_LIST _LANGUAGES)
		if(NOT TARGET MPI::MPI_C)
			find_dependency_components(MPI COMPONENTS C)
		endif()
		target_link_libraries(SpFFT::spfft INTERFACE MPI::MPI_C)
	endif()
endif()
