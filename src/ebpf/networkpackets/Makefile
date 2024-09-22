TARGETS = networkpackets

all: $(TARGETS)
.PHONY: all

$(TARGETS): %: %.bpf.o

%.bpf.o: %.bpf.c
	clang -target bpf -Wall -I/usr/include/$(shell uname -m)-linux-gnu -g -O2 -o $@ -c $<

clean:
	- rm *.o
	- echo "Make sure to remove any installed programs at /sys/fs/bpf/*"