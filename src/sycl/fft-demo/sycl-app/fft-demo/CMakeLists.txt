add_executable(fft-demo main.cpp fft.cpp)

add_sycl_to_target(TARGET fft-demo
    SOURCES main.cpp fft.cpp
)
