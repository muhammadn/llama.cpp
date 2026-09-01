// Alternative ggml-rpc transport backed by prime-iroh (QUIC-based P2P,
// node-id addressed, NAT hole-punch/relay fallback via Iroh) instead of
// plain TCP. Mutually exclusive with transport.cpp: exactly one of the two
// is compiled into the ggml-rpc backend (see CMakeLists.txt,
// GGML_RPC_TRANSPORT=tcp|iroh).
//
// Addressing is a placeholder pending a proper rendezvous/coordinator
// service (tracked separately): the RPC "endpoint" string
// (`--rpc host:port`, parsed by parse_endpoint() elsewhere in
// ggml-rpc.cpp) is repurposed as follows, since Iroh has no host:port
// concept of its own:
//   - Server side (`create_server`): `host` is an optional decimal seed
//     used to derive a *stable* node id across restarts (empty -> random
//     identity each run). `port` is ignored. The server prints its own
//     64-hex-char node id to stdout on first accept(); that id is what
//     clients must be told out of band (manually, for now) to connect.
//   - Client side (`connect`): `host` is the server's 64-hex-char node id.
//     `port` is ignored.
//
// RDMA capability upgrade (GGML_RPC_RDMA) is TCP/raw-fd specific and not
// supported here; get_caps()/update_caps() are no-ops.
#include "transport.h"
#include "ggml-impl.h"
#include "prime_iroh.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {
constexpr size_t IROH_NUM_STREAMS = 1;
constexpr size_t IROH_TAG         = 0;
constexpr size_t IROH_CONNECT_RETRIES = 10;
constexpr int    IROH_WAIT_TIMEOUT_S_DEFAULT = 300; // 5 min; GGML_RPC_IROH_TIMEOUT=0 disables it
constexpr int    IROH_WAIT_LOG_INTERVAL_S     = 10;

// Returns the wait timeout in seconds, or -1 to mean "no timeout, wait
// forever" (GGML_RPC_IROH_TIMEOUT=0).
int wait_timeout_seconds() {
    if (const char * s = std::getenv("GGML_RPC_IROH_TIMEOUT")) {
        char * endp = nullptr;
        long   v    = std::strtol(s, &endp, 10);
        if (endp && *endp == '\0') {
            if (v == 0) {
                return -1;
            }
            if (v > 0) {
                return (int) v;
            }
        }
    }
    return IROH_WAIT_TIMEOUT_S_DEFAULT;
}

// Polls `pred` until it returns true, logging progress every
// IROH_WAIT_LOG_INTERVAL_S seconds and giving up after wait_timeout_seconds()
// instead of hanging forever. This matters because address resolution for a
// peer (mapping its node id to actual dialable network paths) is delegated
// entirely to iroh's public n0 pkarr discovery service -- an external
// dependency outside our control that can stall with no error of its own.
// Set GGML_RPC_IROH_TIMEOUT=0 to wait forever instead (still logs progress).
template <typename Pred>
bool wait_with_timeout(Pred pred, const char * what) {
    const int  timeout_s = wait_timeout_seconds();
    const auto start     = std::chrono::steady_clock::now();
    auto       last_log  = start;
    while (!pred()) {
        const auto now        = std::chrono::steady_clock::now();
        const auto elapsed_s  = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (timeout_s >= 0 && elapsed_s >= timeout_s) {
            GGML_LOG_ERROR("iroh transport: timed out after %ds waiting for %s "
                            "(peer discovery via iroh's public relay may be unavailable; "
                            "set GGML_RPC_IROH_TIMEOUT=<seconds>, or 0 to wait forever)\n",
                            timeout_s, what);
            return false;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= IROH_WAIT_LOG_INTERVAL_S) {
            GGML_LOG_INFO("iroh transport: still waiting for %s (%llds elapsed)...\n",
                          what, (long long) elapsed_s);
            last_log = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return true;
}

// Checks peer_id against GGML_RPC_IROH_ALLOWED_PEERS, a comma-separated list
// of node ids. Unset or empty means "allow any peer" (default, unchanged
// behavior).
bool is_peer_allowed(const char * peer_id) {
    const char * allowed = std::getenv("GGML_RPC_IROH_ALLOWED_PEERS");
    if (!allowed || allowed[0] == '\0') {
        return true;
    }
    std::string list(allowed);
    size_t      start = 0;
    while (start <= list.size()) {
        size_t end = list.find(',', start);
        if (end == std::string::npos) {
            end = list.size();
        }
        std::string entry = list.substr(start, end - start);
        size_t      a     = entry.find_first_not_of(" \t");
        if (a != std::string::npos) {
            size_t b = entry.find_last_not_of(" \t");
            if (entry.compare(a, b - a + 1, peer_id) == 0) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}
} // namespace

struct socket_t::impl {
    bool is_listener = false;

    // Listener-only: identity to (re-)construct a node for each accept().
    uint64_t seed     = 0;
    bool     has_seed = false;

    // Listener-only: fired once, from the first accept(), right after the
    // node id is known and before blocking to wait for a peer.
    void (*ready_cb)(const char * identity, void * user_data) = nullptr;
    void * ready_cb_user_data = nullptr;

    // Connected-only (accepted or client-side connect()).
    PrimeIrohNode * node = nullptr;

    impl(bool listener, uint64_t seed_, bool has_seed_)
        : is_listener(listener), seed(seed_), has_seed(has_seed_) {}
    explicit impl(PrimeIrohNode * n) : is_listener(false), node(n) {}

    ~impl() {
        if (node) {
            prime_iroh_node_close(node);
            prime_iroh_node_free(node);
        }
    }

    bool send_data(const void * data, size_t size) {
        if (!node) {
            return false;
        }
        return prime_iroh_node_send(node, IROH_TAG, (const uint8_t *) data, size);
    }

    bool recv_data(void * data, size_t size) {
        if (!node) {
            return false;
        }
        return prime_iroh_node_recv(node, IROH_TAG, (uint8_t *) data, size);
    }

    bool flush() {
        return true; // no framing/coalescing needed over iroh's QUIC streams
    }

    void get_caps(uint8_t * local_caps) {
        std::memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
    }

    void update_caps(const uint8_t * /*remote_caps*/) {
        // Capability-based RDMA upgrade is TCP/raw-fd specific; not applicable.
    }
};

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

bool socket_t::flush() {
    return pimpl->flush();
}

void socket_t::get_caps(uint8_t * local_caps) {
    pimpl->get_caps(local_caps);
}

void socket_t::update_caps(const uint8_t * remote_caps) {
    pimpl->update_caps(remote_caps);
}

void socket_t::set_ready_callback(void (*cb)(const char * identity, void * user_data), void * user_data) {
    pimpl->ready_cb           = cb;
    pimpl->ready_cb_user_data = user_data;
}

socket_ptr socket_t::create_server(const char * host, int /*port*/) {
    bool     has_seed = false;
    uint64_t seed     = 0;
    if (host && host[0] != '\0') {
        char * endp = nullptr;
        unsigned long long v = strtoull(host, &endp, 10);
        if (!endp || *endp != '\0') {
            GGML_LOG_ERROR("iroh transport: server identity must be a decimal seed, got '%s'\n", host);
            return nullptr;
        }
        seed = (uint64_t) v;
        has_seed = true;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(/*listener=*/true, seed, has_seed)));
}

socket_ptr socket_t::accept() {
    if (!pimpl->is_listener) {
        return nullptr;
    }
    PrimeIrohNode * node = prime_iroh_node_new_with_seed(IROH_NUM_STREAMS, pimpl->seed, pimpl->has_seed ? 1 : 0);
    if (!node) {
        GGML_LOG_ERROR("iroh transport: failed to create node\n");
        return nullptr;
    }
    char id[128];
    if (prime_iroh_node_id(node, id, sizeof(id)) > 0) {
        printf("iroh: node id = %s (share with clients via --rpc %s:0)\n", id, id);
        fflush(stdout);
        if (pimpl->ready_cb) {
            pimpl->ready_cb(id, pimpl->ready_cb_user_data);
            pimpl->ready_cb = nullptr; // fire once
        }
    }
    // Block until a peer dials in to us. There is no coordinator yet, so
    // this is a plain wait loop; the peer must already know `id` above.
    if (!wait_with_timeout([node] { return prime_iroh_node_can_recv(node); }, "an inbound peer connection")) {
        prime_iroh_node_free(node);
        return nullptr;
    }
    // A Node only becomes fully usable (is_ready()) once it has a
    // connection in BOTH directions: the inbound one we just received, and
    // an outbound one from us to the peer. Since only the client knows our
    // node id up front, we learn the client's id from the inbound
    // connection and dial it back here.
    char peer_id[128];
    if (prime_iroh_node_remote_id(node, peer_id, sizeof(peer_id)) <= 0) {
        GGML_LOG_ERROR("iroh transport: failed to determine remote peer id\n");
        prime_iroh_node_free(node);
        return nullptr;
    }
    if (!is_peer_allowed(peer_id)) {
        GGML_LOG_ERROR("iroh transport: rejected connection from peer '%s' (not in GGML_RPC_IROH_ALLOWED_PEERS)\n", peer_id);
        prime_iroh_node_free(node);
        return nullptr;
    }
    if (!prime_iroh_node_connect(node, peer_id, IROH_CONNECT_RETRIES)) {
        GGML_LOG_ERROR("iroh transport: failed to connect back to peer '%s'\n", peer_id);
        prime_iroh_node_free(node);
        return nullptr;
    }
    if (!wait_with_timeout([node] { return prime_iroh_node_is_ready(node); }, "connection to become ready")) {
        prime_iroh_node_free(node);
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(node)));
}

socket_ptr socket_t::connect(const char * host, int /*port*/) {
    if (!host || host[0] == '\0') {
        GGML_LOG_ERROR("iroh transport: missing peer node id\n");
        return nullptr;
    }
    PrimeIrohNode * node = prime_iroh_node_new(IROH_NUM_STREAMS);
    if (!node) {
        GGML_LOG_ERROR("iroh transport: failed to create node\n");
        return nullptr;
    }
    if (!prime_iroh_node_connect(node, host, IROH_CONNECT_RETRIES)) {
        GGML_LOG_ERROR("iroh transport: failed to connect to peer '%s'\n", host);
        prime_iroh_node_free(node);
        return nullptr;
    }
    if (!wait_with_timeout([node] { return prime_iroh_node_is_ready(node); }, "connection to become ready")) {
        prime_iroh_node_free(node);
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(node)));
}

bool rpc_transport_init() {
    prime_iroh_init_logging();
    return true;
}

void rpc_transport_shutdown() {
    // No global state to tear down.
}
