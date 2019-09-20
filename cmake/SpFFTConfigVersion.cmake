
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedConfigVersion.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/SpFFTSharedConfigVersion.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/SpFFTStaticConfigVersion.cmake")
endif()
