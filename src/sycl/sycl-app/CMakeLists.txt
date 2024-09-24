# Pulled from AdaptiveCpp repo example

cmake_minimum_required (VERSION 3.23)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(sycl-examples)

if(NOT SYCL_IMPLEMENTATION)
  set(SYCL_IMPLEMENTATION AdaptiveCpp)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)

if(SYCL_IMPLEMENTATION MATCHES "AdaptiveCpp")
  find_package(AdaptiveCpp CONFIG REQUIRED)

  if(NOT ACPP_DEBUG_LEVEL)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
      set(ACPP_DEBUG_LEVEL 3 CACHE STRING
        "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
      FORCE)
    else()
      set(ACPP_DEBUG_LEVEL 2 CACHE STRING
        "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
      FORCE)
    endif()
  endif()

  add_compile_definitions(HIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})
elseif(SYCL_IMPLEMENTATION MATCHES "DPCPP")
  find_package(IntelSYCL)
  if(NOT IntelSYCL_FOUND)
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_COMPILER icpx)

    set(ADD_SYCL_FLAGS TRUE)

    # TODO: Make configurable
    #set(ONEAPI_TARGETS "amdgcn-amd-amdhsa,spir64,x86_64,spir64_gen")
    set(ONEAPI_TARGETS "amdgcn-amd-amdhsa")
    set(ONEAPI_BACKEND ${ONEAPI_TARGETS})
    set(ONEAPI_ARCH "gfx1010")

    set(ONEAPI_COMMON_FLAGS "-fsycl -fsycl-targets=${ONEAPI_TARGETS} -Xsycl-target-backend=${ONEAPI_BACKEND} --offload-arch=${ONEAPI_ARCH}")

    set(SYCL_COMPILE_FLAGS "-Wall ${ONEAPI_COMMON_FLAGS}")
    set(SYCL_LINK_FLAGS "${ONEAPI_COMMON_FLAGS}")
  endif()
else()
  message(FATAL_ERROR "Unknown SYCL_IMPLEMENTATION value `${SYCL_IMPLEMENTATION}`")
endif()

add_compile_definitions(SYCL_IMPLEMENTATION="${SYCL_IMPLEMENTATION}")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})

subdirs(fft-demo)

