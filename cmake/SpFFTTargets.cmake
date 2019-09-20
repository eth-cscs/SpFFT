
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedTargets.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedTargets.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticTargets.cmake")
endif()
