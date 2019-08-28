# options used for building library
set(SPFFT_OMP @SPFFT_OMP@)
set(SPFFT_MPI @SPFFT_MPI@)
set(SPFFT_STATIC @SPFFT_STATIC@)
set(SPFFT_GPU_DIRECT @SPFFT_GPU_DIRECT@)
set(SPFFT_SINGLE_PRECISION @SPFFT_SINGLE_PRECISION@)
set(SPFFT_GPU_BACKEND @SPFFT_GPU_BACKEND@)

# add version of package
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTConfigVersion.cmake")

# add library target
include("${CMAKE_CURRENT_LIST_DIR}/SpFFTTargets.cmake")

