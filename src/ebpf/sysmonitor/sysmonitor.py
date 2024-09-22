#!/usr/bin/python3  
from bcc import BPF
from time import sleep

syscall_monitor_program = r"""
/* Define a key/value hashmap, that is accessable from user code */
BPF_HASH(syscall_monitor);

int record_syscall(void *_context) {
    u64 uid;
    u64 counter = 0;
    u64 *ptr;

    /* Attempt to get pointer to user in hashmap */
    uid = bpf_get_current_uid_gid() & 0xFFFFFFFF;
    ptr = syscall_monitor.lookup(&uid);

    /* The ebpf verifier checks that we have a null check before dereferencing
     * the pointer. Try removing the if statement and see what happens.
     */
    if (ptr != 0) {
        /* If user is already in the hash map, use the existing counter value. */
        counter = *ptr;
    }

    counter++;

    /* Add/update the user counter in the hashmap. */
    syscall_monitor.update(&uid, &counter);

    return 0;
}
"""

# Compile the ebpf program and attach "record_syscall" to "execve" syscalls
bpf_handle = BPF(text=syscall_monitor_program)
syscall = bpf_handle.get_syscall_fnname("execve")
bpf_handle.attach_kprobe(event=syscall, fn_name="record_syscall")

# Periodically print a sorted list of users and the number of syscalls they made
while True:
    sleep(1)
    str = ", ".join("ID {}: {}".format(key.value, val.value)
                    for key,val in sorted(bpf_handle["syscall_monitor"].items(),
                                          key=lambda x: x[0].value))
    print(str)