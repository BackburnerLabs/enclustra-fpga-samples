#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <time.h>

int counter = 0;
long long ts;
long long last_ts = 0;

SEC("xdp")
int nwpacketscnt(struct xdp_md *ctx) {
    ts = bpf_ktime_get_ns();
    
    if (ts - last_ts > 1000000000) {
        bpf_printk("Number of network packets: %d", counter);
        counter = 0;
        last_ts = ts;
    }

    counter++;
    return XDP_PASS;
}

/* The ebpf verifier requires a GPL if certain third-party features are used */
char LICENSE[] SEC("license") = "Dual BSD/GPL";