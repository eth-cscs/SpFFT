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


# add version of package
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedConfigVersion.cmake")

# add library target
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedTargets.cmake")

# SpFFT only has MPI as public dependency, since the mpi header is
# part of the public header file
if(SPFFT_MPI)
	# only look for MPI if interface for language may be used
	get_property(_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
	if("CXX" IN_LIST _LANGUAGES)
		find_dependency_components(MPI COMPONENTS CXX)
		target_link_libraries(SpFFT::spfft INTERFACE MPI::MPI_CXX)
	endif()

	if("C" IN_LIST _LANGUAGES)
		find_dependency_components(MPI COMPONENTS C)
		target_link_libraries(SpFFT::spfft INTERFACE MPI::MPI_C)
	endif()

	# Fortran interface does not depend on MPI -> no linking for shared library required
endif()
