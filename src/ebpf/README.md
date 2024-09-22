# bpftool
This toolsuite is extremely powerful when it comes to loading and debugging ebpf programs. It offers the ability to do many different things, such as: loading programs, listing available programs, dumping memory, modifying program data, viewing traces, etc.
## Loading programs
As shown above in the networkpackets demo, you can load a program into eBPF and pin it to the filesystem with the following command: `bpftool prog load <object_file> /sys/fs/bpf/<program_name>` i.e. `bpftool prog load networkpackets.bpf.o /sys/fs/bpf/networkpackets`.
Then, if a function needs to be attached to a network inferface, it can be done with: `bpftool net attach xdp name <function_name> dev <network_interface>` (i.e. `bpftool net attach xdp name nwpacketscnt dev lo`)<br>
Another option for loading the program, if the above fails or if you want to load it in a single command, is to use ip: `ip link set <network_interface> <type> obj <program_name> sec <type>` (i.e. `ip link set wlo1 xdpgeneric obj ./networkpackets.bpf.o sec xdp`).
## Viewing traces
To view the traces of all running ebpf programs on the system, use: `bpftool prog trace`. Note, this is just listening to the writes to `/sys/kernel/debug/tracing/trace_pipe`, so it would be the same thing as using `cat` on the file. It is also ill-advised to use the trace for passing data in production applications, it should only really be used for debugging purposes.
## List available programs
To list all programs use: `bpftool prog show`. This can be filtered down to looking for a specific program using id, name, or tag:<br>
`bpftool prog show id 72`<br>
`bpftool prog show name hello`<br>
`bpftool prog show tag f1db4e564ad5219a`<br>
## Dumping program memory/data
First, you have to find the map_id of the dataset that you want to view: `bpftool prog show name <function_name>` (i.e. `bpftool prog show name record_syscall`)
This will result in something like the following being returned, where some of the metadata will be different, based on your environment:
```
374: kprobe  name record_syscall  tag de24cf185ee252cd  gpl
        loaded_at 2024-09-22T15:24:49-0500  uid 0
        xlated 208B  jited 124B  memlock 4096B  map_ids 55
        btf_id 303
```
The key point is that in this example, map_id 55 is used by this program for the syscall_monitor hashmap.<br>
To see some information about this map_id, such as the key and value sizes, use: `bpftool map show id <id>`:
```
55: hash  name syscall_monitor  flags 0x0
        key 8B  value 8B  max_entries 10240  memlock 920896B
        btf_id 303
```
In this example, the key and value are 64-bit values.<br>
To get the data currently in the datastructure, it can be dumped with: `bpftool map dump id <id>`:
```
[{
        "key": 0,
        "value": 1877
    },{
        "key": 1000,
        "value": 11909
    }
]
```
To look up the value of a specific key in the hashmap, the following can be used. Though, note that the <key> must be the full width of the key's memory, in LSB. It is space-delimited for each byte. It also is by default decimal input, which can be overridden with the `hex` keyword: `bpftool map lookup id <id> key hex <key>` (i.e.`bpftool map lookup id 55 key hex E8 03 00 00 00 00 00 00`):
```
{
    "key": 1000,
    "value": 12438
}
```
## Modify data
Using the methods above to find the map_id, the following can be used to modify the data located within the hashmap. Note that the value shares the same requirements/behavior as the key: `bpftool map update id <id> key <key> value <value>` (i.e. `bpftool map update id 55 key hex E8 03 00 00 00 00 00 00 value hex FF FF FF FF 0 0 0 0`)
```
{
    "key": 1000,
    "value": 4294967295
}
```
# Demos
## helloworld
Simple python program to prove system functionality with ebpf. This attaches an ebpf program to execve, causing "Hello World" to be printed to a trace file every time a system call is invoked on the system.
## networkpackets
This demo was written with libbpf to show one of the different options you have for devloping ebpf programs. However, it is worth noting that most standard C libraries will not be present. This is due to the restricted instruction set of bpf programs and an attempt to reduce the risk of crashing the kernel. There are a set of header files available for use to offer some of the missing functionality for the bpf programs. What you will notice is that for the C code, it is a bit more involved with compiling and loading the code into the kernel. However, the loading of the bpf program can be automated through the addition of another script to perform the load/unload (similar to python).
- Load program into eBPF and pin it to the filesystem: `bpftool prog load <object_file> /sys/fs/bpf/<program_name>` i.e. `bpftool prog load networkpackets.bpf.o /sys/fs/bpf/networkpackets`.
- Attach a program to a network interface: `bpftool net attach xdp name <function_name> dev <network_interface>` (i.e. `bpftool net attach xdp name nwpacketscnt dev lo`)
- Confirm program is loaded: `bpftool net list` or `ip a show dev <network_interface>` (i.e. `ip a show dev lo`)
- Capture trace information: `bpftool prog trace log` or `cat /sys/kernel/debug/tracing/trace_pipe`. Note: that all ebpf programs report to the same trace file. So, using `/sys/kernel/debug/tracing/trace_pipe` shouldn't be considered for monitoring program performance for production, only for development/debugging.
- Detach program and remove the pin from the filesystem: `bpftool net detach xdp dev <network_interface>` (i.e. `bpftool net detach xdp dev lo`) and `rm /sys/fs/bpf/<program_name>` (i.e. `rm /sys/fs/bpf/networkpackets`)
## sysmonitor
This program is to showcase a monitoring use case, where the ebpf program is setup to log the number of syscalls a user makes. The information is started in a hashmap (key/val data structure) and is accessed in user code to display the current key/values to the console.