// Vendored copy of the prime-iroh C API (see rust/src/capi.rs in the
// PrimeIntellect-ai/prime-iroh fork at ~/Projects/prime-iroh). Keep this in
// sync with that file if the API changes.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrimeIrohNode PrimeIrohNode;

PrimeIrohNode * prime_iroh_node_new(size_t num_streams);
PrimeIrohNode * prime_iroh_node_new_with_seed(size_t num_streams, uint64_t seed, int has_seed);
void            prime_iroh_node_free(PrimeIrohNode * node);
int             prime_iroh_node_id(const PrimeIrohNode * node, char * buf, size_t buf_len);
int             prime_iroh_node_remote_id(const PrimeIrohNode * node, char * buf, size_t buf_len);
bool            prime_iroh_node_connect(PrimeIrohNode * node, const char * peer_id, size_t num_retries);
bool            prime_iroh_node_is_ready(const PrimeIrohNode * node);
bool            prime_iroh_node_can_send(const PrimeIrohNode * node);
bool            prime_iroh_node_can_recv(const PrimeIrohNode * node);
bool            prime_iroh_node_send(PrimeIrohNode * node, size_t tag, const uint8_t * data, size_t len);
bool            prime_iroh_node_recv(PrimeIrohNode * node, size_t tag, uint8_t * buf, size_t len);
bool            prime_iroh_node_close(PrimeIrohNode * node);
void            prime_iroh_init_logging(void);

#ifdef __cplusplus
}
#endif
