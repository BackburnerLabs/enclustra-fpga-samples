# SYCL demos

This directory contains a few demos of using SYCL to optimize computational
problems compared to running a (naive) computation in plain C++. These demos
don't do anything with the output currently, and are solely used for
benchmarking the different implementations.

## Demo explanations

### matrix-demo

This demo multiplies two matrices of increasing size, each x-by-x where x is a
power of 2. Example output:

```
# units for runtime in nanoseconds per iteration
matrix size, single threaded CPU, sycl GPU, sycl CPU, sycl GPU+CPU
8, 271.75, 827881, 946516, 144053
16, 1724.5, 119323, 5648.25, 73294.1
32, 12283.2, 120260, 5923.75, 39917.7
64, 97308.5, 54326.9, 9356.5, 26422
128, 822172, 57824.8, 15568.2, 27638.1
256, 9.96869e+06, 98352.9, 44574.6, 80638.7
512, 2.07724e+08, 136831, 212631, 200519
1024, SKIP, 650897, 1.36554e+06, 1.37587e+06
2048, SKIP, 8.73713e+06, 1.13174e+07, 8.99299e+06
4096, SKIP, 8.72435e+07, SKIP, SKIP
```

### vector-demo

This demo multiplies each element in two vectors, where each vector is a power
of two in size. Example output:

```
# units for runtime in nanoseconds per iteration
vector size, single threaded CPU, sycl GPU, sycl CPU, sycl GPU+CPU
16, 3.90625, 100743, 39685.3, 14202.2
32, 5.625, 16553.3, 1559.53, 10166.5
64, 11.125, 17320.2, 1241.11, 6348.61
128, 21.6094, 6563.86, 1534.62, 3949.6
256, 39.1406, 10255.2, 674.094, 3512.91
512, 69.3438, 7446.16, 627.75, 3797.2
1024, 121.172, 6659.52, 564.203, 3524.11
2048, 220.734, 8316.72, 853.172, 3598.78
4096, 436.141, 6773.48, 897.641, 3419.3
8192, 824.688, 6556.83, 1116.34, 3505.95
16384, 1704.8, 7394.34, 1402.98, 3564.5
32768, 3281.55, 6814.19, 2385.14, 3714.86
65536, 6513.47, 7392.31, 3094.62, 3785.47
131072, 14437.6, 10690.8, 6968.23, 5322.6
262144, 27485.5, 9648, 11546.9, 5096.16
524288, 53913.4, 11765.6, 8062.48, 6736.07
1048576, 110420, 31565.4, 12570.6, 8955.27
2097152, 237859, 57929.3, 31208.3, 14287.7
4194304, 733084, 102214, 60235.1, 23232.2
8388608, 2.02314e+06, 187570, 122196, 51396.4
16777216, 4.9929e+06, 428661, 679967, 89128.2
33554432, 1.00226e+07, 796754, 1.01425e+06, 165637
67108864, 2.08203e+07, 2.55591e+06, 2.86583e+06, 315917
```

### fft-demo

This demo computes an FFT of increasing size by various means. This example
is a naive implementation of the Cooley-Tukey algorithm, and as such the SYCL
implementation does not have a performance increase over the single-threaded
and multi-threaded implementations (at least on tested hardware). An algorithm
more conducive to massive parallelism would be required.

## Building the demos

These demos can be build using either AdaptiveCPP, or Intel oneAPI/DPCPP

### Environment setup

TODO

The current OneAPI target configuration expects that Codeplay's "oneAPI for AMD
GPU's" plugin is installed.

Note: For running on Arch Linux, the versions of HIP/ROCm available in the
repositories are too new to use with the version of oneAPI in the repositories.
An updated package is available in the `../arch` directory. Building this
package requires that a user named `builduser` is present on the system, and
that the user that runs `makepkg` has access to its home directory.

### Building

#### Intel oneAPI

```
mkdir build; cd build
cmake .. -DSYCL_IMPLEMENTATION=DPCPP
make -j8
```

The OneAPI target currently is geared towards running on an AMD GPU, in
particular an RX 5700 (XT). Running on a different GPU, or on an Intel of nVidia
GPU will require updating the CMakeLists.txt file in the root. More streamlined
support may be added later.

#### AdaptiveCPP

AdaptiveCPP was not as stable in the environment tested, so less testing was done

```
mkdir build; cd build
cmake .. -DSYCL_IMPLEMENTATION=AdaptiveCPP
make -j8
```
