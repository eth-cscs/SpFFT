
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedConfig.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedConfig.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticConfig.cmake")
endif()
