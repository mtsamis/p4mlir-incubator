#include <linux/types.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/pkt_cls.h>

typedef __u32 counters_key;
typedef __u32 counters_value;

// This is just a rough design
#ifdef P4MLIR_EBPF_USE_COUNTERS
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, counters_key);
    __type(value, counters_value);
    __uint(max_entries, P4MLIR_EBPF_COUNTERS_SIZE);
} counters SEC(".maps");

// Match p4mlir extern name mangling.

// void add(in bit<32> index, in bit<32> value);
void _e_CounterArray_m_add(counters_key key, counters_value amt) {
    counters_value *value = bpf_map_lookup_elem(&counters, &key);
    if (value != NULL) {
        __sync_fetch_and_add(value, amt);
    } else {
        counters_value init_val = amt;
        bpf_map_update_elem(&counters, &key, &init_val, BPF_ANY);
    }
}

// void increment(in bit<32> index);
void _e_CounterArray_m_increment(counters_key key) {
    _e_CounterArray_m_add(key, 1);
}
#endif

// Extern function for main package entry point.
__u32 P4MLIR_EBPF_FILTER_ENTRY(unsigned char *packetStart, __u32 packetLen);

SEC("tc")
int ebpf_filter(struct __sk_buff *skb){
    void* ebpf_packetStart = ((void*)(long)skb->data);
    unsigned char* ebpf_headerStart = ebpf_packetStart;
    // void* ebpf_packetEnd = ((void*)(long)skb->data_end);

    __u32 filterOut = P4MLIR_EBPF_FILTER_ENTRY(ebpf_headerStart, skb->len);

    if (filterOut)
        return TC_ACT_OK;
    else
        return TC_ACT_SHOT;
}

char _license[] SEC("license") = "MIT";
