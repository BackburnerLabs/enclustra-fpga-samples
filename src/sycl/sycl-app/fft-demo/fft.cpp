#include "fft.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

void gen_data(size_t count, cfval_t *data) {
    for (auto i = 0; i < count; i++) {
        fval_t point = sin((fval_t)i * M_PI / 8.) * 2.0;
        point += sin((M_PI / 2.) + (fval_t)i * M_PI / 2.) * 1.0;
        data[i] = cfval_t(point, 0);
    }
}

float FFTProvider::benchmark(size_t count, size_t n_points) {
    auto xt_data = new cfval_t[n_points];
    auto xf_data = new cfval_t[n_points];
    gen_data(n_points, xt_data);

    auto start = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < count; i++) {
        if (this->fft(n_points, xt_data, xf_data)) {
            return 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return (float)count * 1e9f / (float)runtime;
}

static size_t rev_bits(size_t val, size_t bits) {
    /* Ineffecient, just for testiung for now */
    size_t res = 0;
    for(auto i = 0; i < bits; i++) {
        res |= (val & 1) << ((bits - 1) - i);
        val >>= 1;
    }
    return res;
}

static int bit_reverse_iterative(size_t count, const cfval_t *input, cfval_t *output) {
    if ((count == 0) || (count & (count - 1))) {
        /* Not a power of 2 */
        return -1;
    }

    for(auto i = 0; i < count; i++) {
        output[rev_bits(i, __builtin_ctz(count))] = input[i];
    }

    return 0;
}

static std::vector<std::vector<cfval_t>> split_problem(size_t count, const cfval_t *input, size_t split_pow) {
    auto split = 1 << split_pow;

    /* Use bit-reversal to efficiently sort into separate vectors for parallel
     * computation */
    std::vector<std::vector<cfval_t>> sorted(split);

    auto bits = __builtin_ctz(count) - split_pow;
    for (auto i = 0; i < count; i++) {
        auto idx = rev_bits(i, __builtin_ctz(count)) >> bits;
        sorted[idx].push_back(input[i]);
    }

    return sorted;
}

#define USE_SLOW_JOIN 0
static void join_problem(size_t count, cfval_t *data, size_t split_pow) {
#if USE_SLOW_JOIN
    auto buf = new cfval_t[count];
    for (auto i = 0; i < split_pow; i++) {
        auto n_splits = 1 << (split_pow - i);
        auto split_sz = count >> (split_pow - i);

        for (auto j = 0; j < n_splits; j+=2) {
            memset(buf, 0, split_sz * 2 * sizeof(cfval_t));
            auto even = data + split_sz * j;
            auto odd = even + split_sz;

            for (auto k = 0; k < split_sz; k++) {
                auto t = std::exp(std::complex<fval_t>(0., (fval_t)-M_PI * (fval_t)k / (fval_t)split_sz)) * odd[k];
                buf[k] = even[k] + t;
                buf[k + split_sz] = even[k] - t;
            }

            memcpy(even, buf, split_sz * 2 * sizeof(cfval_t));
        }
    }
#else
    for (auto i = __builtin_ctz(count) - split_pow + 1; i <= __builtin_ctz(count); i++) {
        auto m = 1 << i;
        auto cv = std::complex<fval_t>(0, -2. * M_PI / m);
        auto omega_m = std::exp(cv);
        for (auto k = 0; k < count; k += m) {
            auto omega = std::complex<fval_t>(1., 0);
            for (auto j = 0; j < m / 2; j++) {
                auto t = omega * data[k + j + (m/2)];
                auto u = data[k + j];
                data[k + j] = u + t;
                data[k + j + (m/2)] = u - t;
                omega = omega * omega_m;
            }
        }
    }
#endif
}

/*
 * Cooley-Tukey recursive
 */
std::string FFTCooleyTukeyRecursive::ident() {
    return "ct_recur";
}

static int _fft_ct_recur(size_t count, cfval_t *data) {
    if (count <= 1) {
        return 0;
    }

    /* Dumb ineffecient */
    auto even = new std::complex<fval_t>[count / 2];
    auto odd = new std::complex<fval_t>[count / 2];
    for (auto i = 0; i < (count / 2); i++) {
        even[i] = data[i * 2];
        odd[i] = data[i * 2 + 1];
    }

    _fft_ct_recur(count / 2, even);
    _fft_ct_recur(count / 2, odd);

    for (auto i = 0; i < (count / 2); i++) {
        auto t = std::exp(std::complex<fval_t>(0., (fval_t)-2. * (fval_t)M_PI * (fval_t)i / (fval_t)count)) * odd[i];
        data[i] = even[i] + t;
        data[i + count/2] = even[i] - t;
    }

    return 0;
}

int FFTCooleyTukeyRecursive::fft(size_t count, cfval_t *input, cfval_t *output) {
    /* TODO: Better support for in-place algorithms */
    std::memcpy(output, input, sizeof(*input) * count);

    _fft_ct_recur(count, output);

    return 0;
}


/*
 * Cooley-Tukey split-problem recursive
 */
std::string FFTCooleyTukeySplitRecursive::ident() {
    return "ct_spl_rec";
}

int FFTCooleyTukeySplitRecursive::fft(size_t count, cfval_t *input, cfval_t *output) {
    const auto split_pow = 2;

    auto split = split_problem(count, input, split_pow);

    auto base_split_sz = split[0].size();

    /* Sequential for testing */
    for (auto i = 0; i < split.size(); i++) {
        _fft_ct_recur(base_split_sz, split[i].data());
    }

    /* Copy partially solved to output */
    for(auto i = 0; i < split.size(); i++) {
        std::memcpy(output + (i * base_split_sz), split[i].data(), base_split_sz * sizeof(cfval_t));
    }

    join_problem(count, output, split_pow);

    return 0;
}


/*
 * Cooley-Tukey iterative
 */
std::string FFTCooleyTukeyIterative::ident() {
    return "ct_iter";
}

static int _fft_ct_iter(size_t count, cfval_t *input, cfval_t *output) {
    if (bit_reverse_iterative(count, input, output)) {
        std::cerr << "Could not reverse bits!" << std::endl;
        return -1;
    }

    for (auto i = 1; i <= __builtin_ctz(count); i++) {
        auto m = 1 << i;
        auto cv = std::complex<fval_t>(0, -2. * M_PI / m);
        auto omega_m = std::exp(cv);
        for (auto k = 0; k < count; k += m) {
            auto omega = std::complex<fval_t>(1., 0);
            for (auto j = 0; j < m / 2; j++) {
                auto t = omega * output[k + j + (m/2)];
                auto u = output[k + j];
                output[k + j] = u + t;
                output[k + j + (m/2)] = u - t;
                omega = omega * omega_m;
            }
        }
    }
    return 0;
}

int FFTCooleyTukeyIterative::fft(size_t count, cfval_t *input, cfval_t *output) {
    _fft_ct_iter(count, input, output);

    return 0;
}


/*
 * Cooley-Tukey split-problem iterative
 */
std::string FFTCooleyTukeySplitIterative::ident() {
    return "ct_spl_iter";
}

int FFTCooleyTukeySplitIterative::fft(size_t count, cfval_t *input, cfval_t *output) {
    const auto split_pow = 2;

    auto split = split_problem(count, input, split_pow);

    auto base_split_sz = split[0].size();

    for (auto i = 0; i < split.size(); i++) {
        _fft_ct_iter(base_split_sz, split[i].data(), output + (i * base_split_sz));
    }

    join_problem(count, output, split_pow);

    return 0;
}


/*
 * Cooley-Tukey multi-threaded iterative
 */
FFTCooleyTukeyMultithreadedIterative::FFTCooleyTukeyMultithreadedIterative(unsigned split_pow) {
    this->split_pow = split_pow;
}

std::string FFTCooleyTukeyMultithreadedIterative::ident() {
    std::string base = "ct_mt_iter ";
    base += std::to_string(this->split_pow);
    return base;
}

int FFTCooleyTukeyMultithreadedIterative::fft(size_t count, cfval_t *input, cfval_t *output) {
    auto split = split_problem(count, input, this->split_pow);

    auto base_split_sz = split[0].size();

    std::vector<std::thread> threads;

    for (auto i = 0; i < split.size(); i++) {
        threads.emplace_back(std::thread(_fft_ct_iter, base_split_sz, split[i].data(), output + (i * base_split_sz)));
    }

    for (auto &t : threads) {
        t.join();
    }

    join_problem(count, output, split_pow);

    return 0;
}

/*
 * Cooley-Tukey SYCL-parallelized iterative
 */
FFTCooleyTukeySYCLIterative::FFTCooleyTukeySYCLIterative(sycl::queue &queue, unsigned split_pow) :
queue(queue) {
    this->split_pow = split_pow;
}

std::string FFTCooleyTukeySYCLIterative::ident() {
    std::string base = "ct_sycl_iter ";
    base += std::to_string(this->split_pow);
    return base;
}

/* std::complex is not fully supported in SYCL and will not work on GPU targets */
cfval_t cust_exp(cfval_t val) {
    fval_t exp_real = std::exp(val.real());
    return std::complex(exp_real * std::cos(val.imag()), exp_real * std::sin(val.imag()));
}

int FFTCooleyTukeySYCLIterative::fft(size_t count, cfval_t *input, cfval_t *output) {
    auto split = split_problem(count, input, this->split_pow);

    auto base_split_sz = split[0].size();

    sycl::buffer<cfval_t> out_buff{output, sycl::range{count}};
    sycl::buffer<cfval_t> in_buff{input, sycl::range{count}};

    try {
        this->queue.submit([&](auto &h) {
            sycl::accessor out_acc{out_buff, h, sycl::read_write};
            sycl::accessor in_acc{in_buff, h, sycl::read_only};

            h.parallel_for(sycl::range{split.size()}, [=](sycl::id<1> idx) {
                auto base = idx * base_split_sz;

                bit_reverse_iterative(base_split_sz, &in_acc[base], &out_acc[base]);

                for (auto i = 1; i <= __builtin_ctz(base_split_sz); i++) {
                    auto m = 1 << i;
                    auto cv = std::complex<fval_t>(0, -2. * M_PI / m);
                    auto omega_m = cust_exp(cv);
                    for (auto k = 0; k < base_split_sz; k += m) {
                        auto omega = std::complex<fval_t>(1., 0);
                        for (auto j = 0; j < m / 2; j++) {
                            auto t = omega * out_acc[base + k + j + (m/2)];
                            auto u = out_acc[base + k + j];
                            out_acc[base + k + j] = u + t;
                            out_acc[base + k + j + (m/2)] = u - t;
                            omega = omega * omega_m;
                        }
                    }
                }
            });
        }).wait();

        this->queue.throw_asynchronous();
    } catch (const sycl::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }



    join_problem(count, output, split_pow);

    return 0;
}


/*
 * TODO
 */
int fft_v1(sycl::queue &queue, size_t count, cfval_t *input, cfval_t *output) {
    auto input_reversed = new cfval_t[count];

    if (bit_reverse_iterative(count, input, input_reversed)) {
        std::cerr << "Could not reverse bits!" << std::endl;
        return -1;
    }

    try {
        sycl::buffer input_buf{input, sycl::range{count}};
        sycl::buffer output_buf{output, sycl::range{count}};

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor input_acc{input_buf, cgh, sycl::read_only};
            sycl::accessor output_acc{output_buf, cgh, sycl::write_only};
        }).wait();
    } catch (const sycl::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}
