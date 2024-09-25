#include <chrono>
#include <cmath>

#include <cstring>
#include <sycl/sycl.hpp>

class scalar_add;

typedef float vtype_t;

#define VEC_SZ_MIN 16
#define VEC_SZ_MAX 1024 * 1024 * 64
#define VEC_SZ_STEP(sz) ((sz) * 2)

#define REPEAT_COUNT 64

#define CPU_MAX_SPLIT 16
#define GPU_MAX_SPLIT 2048

#define SYCL_USE_GPU 1
#define SYCL_USE_CPU 1

#if SYCL_USE_GPU && SYCL_USE_CPU
/* If we have both GPU and CPU, we can run both in parallel */
#  define SYCL_USE_X2 1
#else
#  define SYCL_USE_X2 0
#endif

#define MARK_USED(d) { asm volatile ("" :: "g" (d)); }


static void display_devices();
static float vector_mult_st_cpu(size_t runs, size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out);
static float vector_mult_sycl(sycl::queue &q, size_t runs, size_t len,const vtype_t *a, const vtype_t *b, vtype_t *out);
#if SYCL_USE_X2
static float vector_mult_sycl_x2(sycl::queue &q1, sycl::queue &q2, size_t runs, size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out);
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
    auto vec_a   = new vtype_t[VEC_SZ_MAX];
    auto vec_b   = new vtype_t[VEC_SZ_MAX];
    auto vec_out = new vtype_t[VEC_SZ_MAX];

    std::cout << "# units for runtime in nanoseconds per iteration" << std::endl;
    std::cout << "vector size, single threaded CPU"
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

    for (auto len = VEC_SZ_MIN; len <= VEC_SZ_MAX; len = VEC_SZ_STEP(len)) {
        std::cout << len;

        auto rt_st_cpu   = vector_mult_st_cpu(REPEAT_COUNT, len, vec_a, vec_b, vec_out);
        std::cout << ", " << rt_st_cpu;
#if SYCL_USE_GPU
        auto rt_sycl_gpu = vector_mult_sycl(sycl_gpu, REPEAT_COUNT, len, vec_a, vec_b, vec_out);
        std::cout << ", " << rt_sycl_gpu;
#endif
#if SYCL_USE_CPU
        auto rt_sycl_cpu = vector_mult_sycl(sycl_cpu, REPEAT_COUNT, len, vec_a, vec_b, vec_out);
        std::cout << ", " << rt_sycl_cpu;
#endif
#if SYCL_USE_X2
        auto rt_sycl_x2 = vector_mult_sycl_x2(sycl_gpu, sycl_cpu, REPEAT_COUNT, len, vec_a, vec_b, vec_out);
        std::cout << ", " << rt_sycl_x2;
#endif

        std::cout << std::endl;

        /* Without this, the compiler may optimize out our function call */
        MARK_USED(vec_out);
    }

    return 0;
}

static float vector_mult_st_cpu(size_t runs, size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out) {
    auto start = std::chrono::high_resolution_clock::now();

    for (auto run = 0; run < runs; run++) {
        for (auto i = 0; i < len; i++) {
            out[i] = a[i] * b[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    unsigned long runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return (float)runtime / runs;
}

static void _sycl_enqueue(sycl::queue &q, size_t runs, size_t len, sycl::buffer<vtype_t> &a, sycl::buffer<vtype_t> &b, sycl::buffer<vtype_t> &out) {
    q.submit([&](sycl::handler &cgh) {
        auto accA   = sycl::accessor{a,   cgh, sycl::read_only};
        auto accB   = sycl::accessor{b,   cgh, sycl::read_only};
        auto accOut = sycl::accessor{out, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range<1>(len), [=](sycl::id<1> idx) {
            accOut[idx] = accA[idx] * accB[idx];
        });
    });
}

static float vector_mult_sycl(sycl::queue &q, size_t runs, size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out) {
    unsigned long runtime = 0;

    try {
        auto bufA   = sycl::buffer{a,   sycl::range{len}};
        auto bufB   = sycl::buffer{b,   sycl::range{len}};
        auto bufOut = sycl::buffer{out, sycl::range{len}};

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
static float vector_mult_sycl_x2(sycl::queue &q1, sycl::queue &q2, size_t runs, size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out) {
    auto a_copy   = new vtype_t[len];
    auto b_copy   = new vtype_t[len];
    auto out_copy = new vtype_t[len];
    std::memcpy(a_copy,   a,   len * sizeof(vtype_t));
    std::memcpy(b_copy,   b,   len * sizeof(vtype_t));
    std::memcpy(out_copy, out, len * sizeof(vtype_t));

    unsigned long runtime = 0;

    try {
        auto bufA   = sycl::buffer{a,   sycl::range{len}};
        auto bufB   = sycl::buffer{b,   sycl::range{len}};
        auto bufOut = sycl::buffer{out, sycl::range{len}};

        auto bufACopy   = sycl::buffer{a_copy,   sycl::range{len}};
        auto bufBCopy   = sycl::buffer{b_copy,   sycl::range{len}};
        auto bufOutCopy = sycl::buffer{out_copy, sycl::range{len}};

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
