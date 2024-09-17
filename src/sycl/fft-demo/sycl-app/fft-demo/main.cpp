#include <cmath>

#include <sycl/sycl.hpp>
#include <vector>

#include "fft.hpp"

class scalar_add;

static void display_devices();


int main() {

#if 0
    auto fft = new FFTCooleyTukeyStackIterative();

    size_t fft_len = 512;
    auto time_domain = new cfval_t[fft_len];
    auto freq_domain = new cfval_t[fft_len];

    gen_data(fft_len, time_domain);

    /*std::cerr << "x(t): ";
    for (auto i = 0; i < fft_len; i++) {
        std::cerr << time_domain[i];
        if (i != (fft_len - 1)) {
            std::cerr << ",";
        }
    }
    std::cerr << std::endl;*/


    if (fft->fft(fft_len, time_domain, freq_domain)) {
        std::cerr << "Failed to execute FFT" << std::endl;
    }

    for (auto i = 0; i < fft_len / 2; i++) {
        std::cout << std::abs(freq_domain[i]) << std::endl;
    }
#else
    std::vector<FFTProvider*> fft_algos;
    //fft_algos.push_back(new FFTCooleyTukeyRecursive());
    fft_algos.push_back(new FFTCooleyTukeyIterative());
    //fft_algos.push_back(new FFTCooleyTukeySplitRecursive());
    //fft_algos.push_back(new FFTCooleyTukeySplitIterative());
    fft_algos.push_back(new FFTCooleyTukeyMultithreadedIterative());

    std::cout << ", ";
    for (auto i = 0; i < fft_algos.size(); i++) {
        std::cout << fft_algos[i]->ident();
        if (i != (fft_algos.size() - 1)) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    for (auto i = 8; i <= 19; i++) {
        auto n_points = 1 << i;
        std::cout << n_points << ", ";
        for (auto i = 0; i < fft_algos.size(); i++) {
            auto fft_rate = fft_algos[i]->benchmark(256, n_points);
            std::cout << fft_rate;
            if (i != (fft_algos.size() - 1)) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }
#endif


#if 0
    int a = 18, b = 24, r = 0;

    try {
        auto defaultQueue1 = sycl::queue{sycl::default_selector()};

        if (fft_v1(defaultQueue1, fft_len, time_domain, freq_domain)) {
            std::cerr << "Failed to execute FFT" << std::endl;
        }

        std::cout << "Chosen device: "
                  << defaultQueue1.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        {
            auto bufA = sycl::buffer{&a, sycl::range{1}};
            auto bufB = sycl::buffer{&b, sycl::range{1}};
            auto bufR = sycl::buffer{&r, sycl::range{1}};

            defaultQueue1
                .submit([&](sycl::handler &cgh) {
                  auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
                  auto accB = sycl::accessor{bufB, cgh, sycl::read_only};
                  auto accR = sycl::accessor{bufR, cgh, sycl::write_only};

                  cgh.single_task<scalar_add>([=]() { accR[0] = accA[0] + accB[0]; });
                })
                .wait();
        }

        defaultQueue1.throw_asynchronous();
    } catch (const sycl::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    return r != 42;
#else
    return 0;
#endif
}




static void display_devices() {
    for (auto platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices()) {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }
}
