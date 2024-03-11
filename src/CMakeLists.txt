set(SPFFT_SOURCE_FILES
	memory/aligned_allocation.cpp
	timing/timing.cpp
	timing/rt_graph.cpp
	parameters/parameters.cpp
	execution/execution_host.cpp
	spfft/transform.cpp
	spfft/transform_internal.cpp
	spfft/multi_transform.cpp
	spfft/grid.cpp
	spfft/grid_internal.cpp
	fft/fftw_mutex.cpp
	)

if(SPFFT_SINGLE_PRECISION)
	list(APPEND SPFFT_SOURCE_FILES
		spfft/transform_float.cpp
		spfft/multi_transform_float.cpp
		spfft/grid_float.cpp
		)
endif()

set(SPFFT_GPU_KERNELS)
if(SPFFT_CUDA OR SPFFT_ROCM)
	list(APPEND SPFFT_GPU_KERNELS
		transpose/gpu_kernels/local_transpose_kernels.cu
		compression/gpu_kernels/compression_kernels.cu
		symmetry/gpu_kernels/symmetry_kernels.cu
		transpose/gpu_kernels/buffered_kernels.cu
		transpose/gpu_kernels/compact_buffered_kernels.cu
		)
	list(APPEND SPFFT_SOURCE_FILES
		execution/execution_gpu.cpp
		gpu_util/gpu_fft_api.cpp
		)
	if(SPFFT_MPI)
		list(APPEND SPFFT_SOURCE_FILES
			transpose/transpose_mpi_buffered_gpu.cpp
			transpose/transpose_mpi_compact_buffered_gpu.cpp
			transpose/transpose_mpi_unbuffered_gpu.cpp
			)
	endif()

	if(SPFFT_ROCM)
		set_source_files_properties(${SPFFT_GPU_KERNELS} PROPERTIES LANGUAGE HIP)
	endif()

	list(APPEND SPFFT_SOURCE_FILES ${SPFFT_GPU_KERNELS})
endif()

if(SPFFT_MPI)
	list(APPEND SPFFT_SOURCE_FILES
		transpose/transpose_mpi_buffered_host.cpp
		transpose/transpose_mpi_compact_buffered_host.cpp
		transpose/transpose_mpi_unbuffered_host.cpp
		)
endif()

# Creates library with given name. All common target modifications should be done here.
macro(spfft_create_library _TARGET_NAME)
	# create target
	add_library(${_TARGET_NAME} ${SPFFT_LIBRARY_TYPE} ${SPFFT_SOURCE_FILES})

	# set version
	set_property(TARGET ${_TARGET_NAME} PROPERTY VERSION ${SPFFT_VERSION})
	set_property(TARGET ${_TARGET_NAME} PROPERTY SOVERSION ${SPFFT_SO_VERSION})

	# All .cu files are self-contained. Device linking can have issues with propageted linker flags of other targets like MPI.
	if(SPFFT_CUDA)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
	endif()
	if(SPFFT_ROCM)
		target_compile_options(${_TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
	endif()

	# Don't export any symbols of external static libaries. Only works on linux.
	if(UNIX AND NOT APPLE)
		if(${CMAKE_VERSION} VERSION_LESS "3.13.5") 
			target_link_libraries(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		else()
			target_link_options(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		endif()
	endif()

	target_include_directories(${_TARGET_NAME} PRIVATE ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS})
	target_link_libraries(${_TARGET_NAME} PRIVATE ${SPFFT_EXTERNAL_LIBS})

	target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
	target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)
	if(${SPFFT_FORTRAN}) # Add include directory for fortran module
		target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include/spfft>)
		target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>)
	endif()

endmacro()

# Create library
spfft_create_library(spfft)
set_target_properties(spfft PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

# Create library for testing, which allows linking to internal symbols and has timings enabled.
if(SPFFT_BUILD_TESTS)
	spfft_create_library(spfft_test)
	set_target_properties(spfft_test PROPERTIES VISIBILITY_INLINES_HIDDEN FALSE CXX_VISIBILITY_PRESET default)
	target_compile_options(spfft_test PUBLIC -DSPFFT_STATIC_DEFINE) # disable properties of export header
	# enable internal timings
	target_compile_options(spfft_test PUBLIC -DSPFFT_TIMING)
endif()

# build fortran module
if(SPFFT_FORTRAN)
	add_library(spfft_fortran OBJECT ${PROJECT_SOURCE_DIR}/include/spfft/spfft.f90)
endif()

# generate export header to control symbol visibility
include(GenerateExportHeader)
generate_export_header(spfft)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/spfft_export.h"
	"${PROJECT_BINARY_DIR}/spfft/spfft_export.h"
	COPYONLY
)

# set packge config names
get_target_property(_LIB_TYPE spfft TYPE)
if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
	set(SPFFT_VERSION_FILE "SpFFTStaticConfigVersion.cmake")
	set(SPFFT_CONFIG_FILE "SpFFTStaticConfig.cmake")
	set(SPFFT_TARGETS_FILE "SpFFTStaticTargets.cmake")
else()
	set(SPFFT_VERSION_FILE "SpFFTSharedConfigVersion.cmake")
	set(SPFFT_CONFIG_FILE "SpFFTSharedConfig.cmake")
	set(SPFFT_TARGETS_FILE "SpFFTSharedTargets.cmake")
endif()


# generate cmake package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/${SPFFT_VERSION_FILE}"
	VERSION ${Upstream_VERSION}
	COMPATIBILITY AnyNewerVersion
)
export(TARGETS spfft NAMESPACE SpFFT:: FILE ${PROJECT_BINARY_DIR}/${SPFFT_TARGETS_FILE})
configure_file(${PROJECT_SOURCE_DIR}/cmake/${SPFFT_CONFIG_FILE}
	"${PROJECT_BINARY_DIR}/${SPFFT_CONFIG_FILE}"
	@ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SpFFTConfig.cmake
	"${PROJECT_BINARY_DIR}/SpFFTConfig.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SpFFTConfigVersion.cmake
	"${PROJECT_BINARY_DIR}/SpFFTConfigVersion.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SpFFTTargets.cmake
	"${PROJECT_BINARY_DIR}/SpFFTTargets.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SpFFT.pc.in
	"${PROJECT_BINARY_DIR}/SpFFT.pc"
	@ONLY
)

# installation commands
if(SPFFT_INSTALL)
	install(TARGETS spfft DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT SpFFTTargets)
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/spfft DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.f90")
	install(FILES ${PROJECT_BINARY_DIR}/spfft/config.h "${PROJECT_BINARY_DIR}/spfft/spfft_export.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spfft)
	install(EXPORT SpFFTTargets NAMESPACE SpFFT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpFFT FILE ${SPFFT_TARGETS_FILE})
	install(
	  FILES
		"${PROJECT_BINARY_DIR}/SpFFTConfig.cmake"
		"${PROJECT_BINARY_DIR}/SpFFTTargets.cmake"
		"${PROJECT_BINARY_DIR}/SpFFTConfigVersion.cmake"
		"${PROJECT_BINARY_DIR}/${SPFFT_CONFIG_FILE}"
		"${PROJECT_BINARY_DIR}/${SPFFT_VERSION_FILE}"
	  DESTINATION
	    ${CMAKE_INSTALL_LIBDIR}/cmake/SpFFT
	)

	install(FILES ${PROJECT_BINARY_DIR}/SpFFT.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

	install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SpFFT"
        FILES_MATCHING PATTERN "*.cmake")

	if(SPFFT_FORTRAN)
		install(FILES ${PROJECT_BINARY_DIR}/src/spfft.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spfft)
	endif()
endif()
