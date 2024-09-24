#include <chrono>
#include <cmath>

#include <sycl/sycl.hpp>
#include <vector>

class scalar_add;

typedef float vtype_t;

static void display_devices();
static void vector_mult_plain(size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out);

#define VEC_SZ_MIN 16
#define VEC_SZ_MAX 1024 * 1024 * 16
#define VEC_SZ_STEP(sz) ((sz) * 2)

#define REPEAT_COUNT 16

#define CPU_MAX_SPLIT 24
#define GPU_MAX_SPLIT 2048

#define MARK_USED(d) { asm volatile ("" :: "g" (d)); }

int main() {
    display_devices();

    auto sycl_queue = sycl::queue{sycl::default_selector_v};
    //auto sycl_queue = sycl::queue{sycl::cpu_selector_v};
    //auto sycl_queue = sycl::queue{sycl::gpu_selector_v};

    std::cout << "Chosen SYCL device: "
              << sycl_queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    /* Setup input and output buffers */
    auto vec_a   = new vtype_t[VEC_SZ_MAX];
    auto vec_b   = new vtype_t[VEC_SZ_MAX];
    auto vec_out = new vtype_t[VEC_SZ_MAX];

    std::cout << "vector size, plain" << std::endl;

    for (auto len = VEC_SZ_MIN; len <= VEC_SZ_MAX; len = VEC_SZ_STEP(len)) {
        std::cout << len << ", ";

        auto start = std::chrono::high_resolution_clock::now();

        for (auto i = 0; i < REPEAT_COUNT; i++) {
            vector_mult_plain(len, vec_a, vec_b, vec_out);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        /* Without this, the compiler may optimize out our function call */
        MARK_USED(vec_out);

        std::cout << runtime;
        std::cout << std::endl;
    }

    return 0;
}

static void vector_mult_plain(size_t len, const vtype_t *a, const vtype_t *b, vtype_t *out) {
    for (auto i = 0; i < len; i++) {
        out[i] = a[i] * b[i];
    }
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
