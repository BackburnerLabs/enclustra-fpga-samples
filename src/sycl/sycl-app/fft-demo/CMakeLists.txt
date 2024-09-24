add_executable(fft-demo main.cpp fft.cpp)

if(ADD_SYCL_FLAGS)
    set_target_properties(fft-demo PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS}")
    set_target_properties(fft-demo PROPERTIES LINK_FLAGS "${SYCL_LINK_FLAGS}")
else()
  add_sycl_to_target(TARGET fft-demo
    SOURCES main.cpp fft.cpp
  )
endif()
