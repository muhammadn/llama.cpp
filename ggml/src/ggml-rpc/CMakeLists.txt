message(STATUS "Using RPC backend")

set(GGML_RPC_TRANSPORT "tcp" CACHE STRING "ggml-rpc transport backend: tcp or iroh")
set_property(CACHE GGML_RPC_TRANSPORT PROPERTY STRINGS tcp iroh)

if (GGML_RPC_TRANSPORT STREQUAL "iroh")
    # Experimental: P2P transport over PrimeIntellect-ai/prime-iroh (a fork
    # adding a plain C API, see rust/src/capi.rs). Node-id addressing only;
    # there is no coordinator/rendezvous service wired in yet, so peers must
    # be told each other's node id out of band (see transport-iroh.cpp).
    set(GGML_RPC_IROH_ROOT "$ENV{HOME}/Projects/prime-iroh" CACHE PATH
        "Path to the PrimeIntellect-ai/prime-iroh checkout (fork with the C API)")
    if (NOT EXISTS "${GGML_RPC_IROH_ROOT}/Cargo.toml")
        message(FATAL_ERROR "GGML_RPC_IROH_ROOT (${GGML_RPC_IROH_ROOT}) does not look like a prime-iroh checkout")
    endif()
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    set(IROH_LIB "${GGML_RPC_IROH_ROOT}/target/release/${CMAKE_STATIC_LIBRARY_PREFIX}prime_iroh${CMAKE_STATIC_LIBRARY_SUFFIX}")
    add_custom_command(
        OUTPUT "${IROH_LIB}"
        COMMAND ${CARGO_EXECUTABLE} build --release --no-default-features
        WORKING_DIRECTORY "${GGML_RPC_IROH_ROOT}"
        COMMENT "Building prime-iroh C API (cargo build --release --no-default-features)"
        VERBATIM
    )
    add_custom_target(prime_iroh_capi DEPENDS "${IROH_LIB}")

    ggml_add_backend_library(ggml-rpc
                             ggml-rpc.cpp
                             transport-iroh.cpp
                            )
    add_dependencies(ggml-rpc prime_iroh_capi)
    target_include_directories(ggml-rpc PRIVATE "${GGML_RPC_IROH_ROOT}/rust/src")
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_TRANSPORT_IROH)
    target_link_libraries(ggml-rpc PRIVATE "${IROH_LIB}")
    if (NOT WIN32)
        target_link_libraries(ggml-rpc PRIVATE pthread dl m)
    endif()
    message(STATUS "  transport      : iroh (prime-iroh @ ${GGML_RPC_IROH_ROOT})")
    message(STATUS "  RDMA transport : n/a (iroh transport does not support the RDMA capability upgrade)")
else()
    ggml_add_backend_library(ggml-rpc
                             ggml-rpc.cpp
                             transport.cpp
                            )

    if (WIN32)
        target_link_libraries(ggml-rpc PRIVATE ws2_32)
    endif()

    # RDMA auto-detection: Linux RoCE/IB via libibverbs, Apple RDMA-over-Thunderbolt via librdma
    if (APPLE)
        set(RDMA_LIB_NAME rdma)
        set(RDMA_DESC "Apple RDMA-over-Thunderbolt, UC")
    elseif (NOT WIN32)
        set(RDMA_LIB_NAME ibverbs)
        set(RDMA_DESC "auto-detected")
    endif()

    if (RDMA_LIB_NAME)
        find_library(RDMA_LIB ${RDMA_LIB_NAME})
        if (RDMA_LIB)
            option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" ON)
        else()
            option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" OFF)
        endif()
    else()
        set(GGML_RPC_RDMA OFF CACHE BOOL "RDMA not available on this platform" FORCE)
    endif()

    if (GGML_RPC_RDMA)
        if (NOT RDMA_LIB)
            find_library(RDMA_LIB ${RDMA_LIB_NAME} REQUIRED)
        endif()
        target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
        target_link_libraries(ggml-rpc PRIVATE ${RDMA_LIB})
        if (APPLE)
            target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA_APPLE)
            target_sources(ggml-rpc PRIVATE transport-apple.cpp)
        endif()
        message(STATUS "  RDMA transport enabled (${RDMA_DESC})")
    else()
        message(STATUS "  RDMA transport disabled")
    endif()
endif()

