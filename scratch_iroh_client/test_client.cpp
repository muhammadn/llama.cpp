#include "ggml-rpc.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <server_node_id>\n", argv[0]);
        return 1;
    }
    char endpoint[256];
    snprintf(endpoint, sizeof(endpoint), "%s:0", argv[1]);
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_rpc_get_device_memory(endpoint, 0, &free_mem, &total_mem);
    printf("client: got device memory reply: free=%zu total=%zu\n", free_mem, total_mem);
    if (total_mem == 0 && free_mem == 0) {
        fprintf(stderr, "client: FAIL - looks like the RPC call did not succeed (both zero)\n");
        return 1;
    }
    printf("client: PASS\n");
    return 0;
}
