#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // Registers a callback fired exactly once, right before this (listening)
    // socket becomes ready to accept a peer connection, with a
    // transport-specific identity string (the iroh node id; unused on TCP,
    // whose endpoint is already known to the caller before create_server()).
    // Must be called on a server socket before its first accept().
    void set_ready_callback(void (*cb)(const char * identity, void * user_data), void * user_data);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
