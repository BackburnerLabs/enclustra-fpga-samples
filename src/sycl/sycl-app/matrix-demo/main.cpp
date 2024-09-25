#include <chrono>
#include <cmath>

#include <cstring>
#include <sycl/sycl.hpp>

class scalar_add;

typedef int mtype_t;

/* All matrices are x-by-x to simplify code */
#define MAT_SZ_MIN 8
#define MAT_SZ_MAX 4096
#define MAT_SZ_STEP(sz) ((sz) * 2)

#define REPEAT_COUNT 8

#define SYCL_USE_GPU 1
#define SYCL_USE_CPU 1

#if SYCL_USE_GPU && SYCL_USE_CPU
/* If we have both GPU and CPU, we can run both in parallel - this performs
 * better until we get to 1024+ */
#  define SYCL_USE_X2 1
#else
#  define SYCL_USE_X2 0
#endif

#define MARK_USED(d) { asm volatile ("" :: "g" (d)); }


static void display_devices();
static float matrix_mult_st_cpu(size_t runs, size_t len, const mtype_t *a, const mtype_t *b, mtype_t *out);
static float matrix_mult_sycl(sycl::queue &q, size_t runs, size_t len,const mtype_t *a, const mtype_t *b, mtype_t *out);
#if SYCL_USE_X2
static float matrix_mult_sycl_x2(sycl::queue &q1, sycl::queue &q2, size_t runs, size_t len, const mtype_t *a, const mtype_t *b, mtype_t *out);
#endif



int main() {
    display_devices();

#if SYCL_USE_GPU
    auto sycl_gpu = sycl::queue{sycl::gpu_selector_v};
    std::cerr << "Chosen SYCL GPU device: "
              << sycl_gpu.get_device().get_info<sycl::info::device::name>()
              << std::endl << std::endl;
#endif

#if SYCL_USE_CPU
    auto sycl_cpu = sycl::queue{sycl::cpu_selector_v};

    std::cerr << "Chosen SYCL CPU device: "
              << sycl_cpu.get_device().get_info<sycl::info::device::name>()
              << std::endl << std::endl;
#endif

    /* Setup input and output buffers */
    auto mat_a   = new mtype_t[MAT_SZ_MAX*MAT_SZ_MAX];
    auto mat_b   = new mtype_t[MAT_SZ_MAX*MAT_SZ_MAX];
    auto mat_out = new mtype_t[MAT_SZ_MAX*MAT_SZ_MAX];

    std::cout << "# units for runtime in nanoseconds per iteration" << std::endl;
    std::cout << "matrix size, single threaded CPU"
#if SYCL_USE_GPU
              << ", sycl GPU"
#endif
#if SYCL_USE_CPU
              << ", sycl CPU"
#endif
#if SYCL_USE_X2
              << ", sycl GPU+CPU"
#endif
              << std::endl;

    for (auto len = MAT_SZ_MIN; len <= MAT_SZ_MAX; len = MAT_SZ_STEP(len)) {
        std::cout << len;

        if (len > 512) {
            std::cout << ", SKIP";
        } else {
            auto rt_st_cpu   = matrix_mult_st_cpu(REPEAT_COUNT, len, mat_a, mat_b, mat_out);
            std::cout << ", " << rt_st_cpu;
        }
#if SYCL_USE_GPU
        auto rt_sycl_gpu = matrix_mult_sycl(sycl_gpu, REPEAT_COUNT, len, mat_a, mat_b, mat_out);
        std::cout << ", " << rt_sycl_gpu;
#endif
#if SYCL_USE_CPU
        if (len > 2048) {
            std::cout << ", SKIP";
        } else {
            auto rt_sycl_cpu = matrix_mult_sycl(sycl_cpu, REPEAT_COUNT, len, mat_a, mat_b, mat_out);
            std::cout << ", " << rt_sycl_cpu;
        }
#endif
#if SYCL_USE_X2
        if (len > 2048) {
            std::cout << ", SKIP";
        } else {
            auto rt_sycl_x2 = matrix_mult_sycl_x2(sycl_gpu, sycl_cpu, REPEAT_COUNT, len, mat_a, mat_b, mat_out);
            std::cout << ", " << rt_sycl_x2;
        }
#endif

        std::cout << std::endl;

        /* Without this, the compiler may optimize out our function call */
        MARK_USED(mat_out);
    }

    return 0;
}

static float matrix_mult_st_cpu(size_t runs, size_t len, const mtype_t *a, const mtype_t *b, mtype_t *out) {
    auto start = std::chrono::high_resolution_clock::now();

    for (auto run = 0; run < runs; run++) {
        for (auto i = 0; i < len; i++) {
            for (auto j = 0; j < len; j++) {
                auto sum = 0;
                for (auto k = 0; k < len; k++) {
                    sum += a[i*len + k] * b[k*len + j];
                }
                out[i*len + j] = sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    unsigned long runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return (float)runtime / runs;
}

static void _sycl_enqueue(sycl::queue &q, size_t runs, size_t len, sycl::buffer<mtype_t> &a, sycl::buffer<mtype_t> &b, sycl::buffer<mtype_t> &out) {
    q.submit([&](sycl::handler &cgh) {
        auto accA   = sycl::accessor{a,   cgh, sycl::read_only};
        auto accB   = sycl::accessor{b,   cgh, sycl::read_only};
        auto accOut = sycl::accessor{out, cgh, sycl::read_write, sycl::no_init};

        cgh.parallel_for(sycl::range<2>(len, len), [=](sycl::id<2> idx) {
            auto i = idx[0];
            auto j = idx[1];

            auto sum = 0;
            for (auto k = 0; k < len; k++) {
                sum += accA[i*len + k] * accB[k*len + j];
            }
            accOut[i*len + j] = sum;
        });
    });
}

static float matrix_mult_sycl(sycl::queue &q, size_t runs, size_t len, const mtype_t *a, const mtype_t *b, mtype_t *out) {
    unsigned long runtime = 0;

    try {
        auto bufA   = sycl::buffer{a,   sycl::range{len*len}};
        auto bufB   = sycl::buffer{b,   sycl::range{len*len}};
        auto bufOut = sycl::buffer{out, sycl::range{len*len}};

        auto start = std::chrono::high_resolution_clock::now();

        _sycl_enqueue(q, runs, len, bufA, bufB, bufOut);
        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        q.throw_asynchronous();
    } catch (const sycl::exception &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return (float)runtime / runs;
}

#if SYCL_USE_X2
static float matrix_mult_sycl_x2(sycl::queue &q1, sycl::queue &q2, size_t runs, size_t len, const mtype_t *a, const mtype_t *b, mtype_t *out) {
    auto a_copy   = new mtype_t[len*len];
    auto b_copy   = new mtype_t[len*len];
    auto out_copy = new mtype_t[len*len];
    std::memcpy(a_copy,   a,   len*len * sizeof(mtype_t));
    std::memcpy(b_copy,   b,   len*len * sizeof(mtype_t));
    std::memcpy(out_copy, out, len*len * sizeof(mtype_t));

    unsigned long runtime = 0;

    try {
        auto bufA   = sycl::buffer{a,   sycl::range{len*len}};
        auto bufB   = sycl::buffer{b,   sycl::range{len*len}};
        auto bufOut = sycl::buffer{out, sycl::range{len*len}};

        auto bufACopy   = sycl::buffer{a_copy,   sycl::range{len*len}};
        auto bufBCopy   = sycl::buffer{b_copy,   sycl::range{len*len}};
        auto bufOutCopy = sycl::buffer{out_copy, sycl::range{len*len}};

        auto start = std::chrono::high_resolution_clock::now();

        _sycl_enqueue(q1, runs, len, bufA, bufB, bufOut);
        _sycl_enqueue(q2, runs, len, bufACopy, bufBCopy, bufOutCopy);
        q1.wait();
        q2.wait();

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        q1.throw_asynchronous();
        q2.throw_asynchronous();
    } catch (const sycl::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return (float)runtime / (runs * 2);
}
#endif /* SYCL_USE_X2 */


static void display_devices() {
    std::cerr << "-- Detected SYCL devices --" << std::endl;
    for (auto platform : sycl::platform::get_platforms()) {
        std::cerr << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices()) {
            std::cerr << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }
    std::cerr << std::endl;
}
