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


# find depencies
include(CMakeFindDependencyMacro)

# Only look for modules we installed and save value
set(_CMAKE_MODULE_PATH_SAVE ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")

if(SPFFT_MKL)
	find_dependency(MKLSequential)
else()
	find_dependency(FFTW)
endif()

if(SPFFT_OMP)
	find_dependency(OpenMP COMPONENTS CXX)
endif()

if(SPFFT_MPI)
	# MPI::MPI_CXX requires CXX Language enabled
	find_dependency(MPI COMPONENTS CXX)
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
endif()

set(CMAKE_MODULE_PATH ${_CMAKE_MODULE_PATH_SAVE}) # restore module path

# add version of package
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticConfigVersion.cmake")

# add library target
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticTargets.cmake")

