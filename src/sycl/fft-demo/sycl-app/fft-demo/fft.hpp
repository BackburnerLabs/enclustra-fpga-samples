#ifndef FFT_HPP
#define FFT_HPP

#include <complex>
#include <string>
#include <sycl/sycl.hpp>

typedef float fval_t;
typedef std::complex<fval_t> cfval_t;

void gen_data(size_t count, cfval_t *data);

class FFTProvider {
public:
    virtual int fft(size_t count, cfval_t *input, cfval_t *output) = 0;

    virtual std::string ident() = 0;

    /* Get the average number of FFTs per second */
    float benchmark(size_t count, size_t n_points);
};

class FFTCooleyTukeyRecursive : public FFTProvider {
public:
    virtual std::string ident();
    int fft(size_t count, cfval_t *input, cfval_t *output);
};

class FFTCooleyTukeySplitRecursive : public FFTProvider {
public:
    virtual std::string ident();
    int fft(size_t count, cfval_t *input, cfval_t *output);
};

class FFTCooleyTukeyIterative : public FFTProvider {
public:
    virtual std::string ident();
    int fft(size_t count, cfval_t *input, cfval_t *output);
};

class FFTCooleyTukeySplitIterative : public FFTProvider {
public:
    virtual std::string ident();
    int fft(size_t count, cfval_t *input, cfval_t *output);
};

class FFTCooleyTukeyMultithreadedIterative : public FFTProvider {
public:
    virtual std::string ident();
    int fft(size_t count, cfval_t *input, cfval_t *output);
};

#endif
