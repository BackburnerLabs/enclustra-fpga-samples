set(TARGET_NAME vector-demo)

add_executable(${TARGET_NAME} main.cpp)

if(ADD_SYCL_FLAGS)
  set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS}")
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${SYCL_LINK_FLAGS}")
else()
  add_sycl_to_target(TARGET ${TARGET_NAME}
    SOURCES main.cpp fft.cpp
  )
endif()
