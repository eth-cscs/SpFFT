set(SPFFT_SOURCE_FILES
	memory/aligned_allocation.cpp
	timing/timing.cpp
	timing/host_timing.cpp
	parameters/parameters.cpp
	execution/execution_host.cpp
	spfft/transform.cpp
	spfft/transform_internal.cpp
	spfft/multi_transform.cpp
	spfft/grid.cpp
	spfft/grid_internal.cpp
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
		)
	if(SPFFT_MPI)
		list(APPEND SPFFT_SOURCE_FILES
			transpose/transpose_mpi_buffered_gpu.cpp
			transpose/transpose_mpi_compact_buffered_gpu.cpp
			transpose/transpose_mpi_unbuffered_gpu.cpp
			)
	endif()
endif()

if(SPFFT_CUDA)
	list(APPEND SPFFT_SOURCE_FILES ${SPFFT_GPU_KERNELS})
endif()

if(SPFFT_ROCM)
	rocm_generate_objects(ROCM_OBJECTS ROCM_TARGETS ${SPFFT_GPU_KERNELS} INCLUDE_DIRS ${SPFFT_INCLUDE_DIRS} ${SPFFT_EXTERNAL_INCLUDE_DIRS}
		FLAGS -fPIC  -fno-gpu-rdc --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906)
	list(APPEND SPFFT_SOURCE_FILES ${ROCM_OBJECTS})
endif()

if(SPFFT_MPI)
	list(APPEND SPFFT_SOURCE_FILES
		transpose/transpose_mpi_buffered_host.cpp
		transpose/transpose_mpi_compact_buffered_host.cpp
		transpose/transpose_mpi_unbuffered_host.cpp
		)
endif()

add_library(spfft ${SPFFT_LIBRARY_TYPE}
	${SPFFT_SOURCE_FILES}
	)

# add depency on ojbect generating targets for ROCM
if(SPFFT_ROCM)
	add_dependencies(spfft ${ROCM_TARGETS})
endif()

target_compile_options(spfft PRIVATE ${SPFFT_DEFINITIONS} ${SPFFT_EXTERNAL_COMPILE_OPTIONS})
target_include_directories(spfft PRIVATE ${SPFFT_EXTERNAL_INCLUDE_DIRS})
target_include_directories(spfft PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(spfft PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(spfft PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(spfft PRIVATE ${SPFFT_EXTERNAL_LIBS})
target_link_libraries(spfft INTERFACE ${SPFFT_INTERFACE_LIBS})

target_include_directories(spfft INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
target_include_directories(spfft INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)

# set list of interface libries for exporting
if(POLICY CMP0022)
	set_target_properties(spfft PROPERTIES INTERFACE_LINK_LIBRARIES "${SPFFT_INTERFACE_LIBS}")
else()
	set_target_properties(spfft PROPERTIES LINK_INTERFACE_LIBRARIES "${SPFFT_INTERFACE_LIBS}")
endif()


# generate cmake package
include(CMakePackageConfigHelpers)
message(STATUS "project: ${PROJECT_SOURCE_DIR}")
write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/SpFFTConfigVersion.cmake"
	VERSION ${Upstream_VERSION}
	COMPATIBILITY AnyNewerVersion
)
export(TARGETS spfft FILE ${PROJECT_BINARY_DIR}/SpFFTTargets.cmake)
configure_file(${PROJECT_SOURCE_DIR}/cmake/SpFFTConfig.cmake
	"${PROJECT_BINARY_DIR}/SpFFTConfig.cmake"
	@ONLY
)

# installation commands
if(SPFFT_INSTALL)
	install(TARGETS spfft DESTINATION lib EXPORT SpFFTTargets)
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/spfft DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.f90")
	install(FILES ${PROJECT_BINARY_DIR}/spfft/config.h DESTINATION include/spfft)
	install(EXPORT SpFFTTargets DESTINATION lib/cmake)
	install(
	  FILES
		"${PROJECT_BINARY_DIR}/SpFFTConfig.cmake"
		"${PROJECT_BINARY_DIR}/SpFFTConfigVersion.cmake"
	  DESTINATION
		lib/cmake
	)
endif()
