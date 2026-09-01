#!/usr/bin/env python3
"""Rendezvous coordinator for the ggml-rpc iroh transport.

Workers register their iroh node id here (via ggml-rpc-server's
--rpc-coordinator/--rpc-rank flags); the driver fetches the assembled --rpc
connection string once all expected workers have checked in (via
--rpc-coordinator/--rpc-world-size). This only automates the exchange of
node ids - it does not participate in peer discovery or connection setup,
nodes still connect to each other via iroh's normal (pkarr/relay) discovery.

Set COORD_TOKEN in the environment to require a matching
`Authorization: Bearer <token>` header on every request (unset by default:
no auth, same as before).

Each rank may only be registered once per generation: registering a rank
again with the same node id is a harmless no-op (e.g. a retried request),
but registering it with a *different* node id is rejected with 409 -- this
catches two workers accidentally launched with the same RANK. Call
POST /reset before starting a new launch on a coordinator process that is
being reused (e.g. restarting a previous run).

Usage:
    python3 coordinator.py [--host 127.0.0.1] [--port 8765]
"""
import argparse
import http.server
import json
import os
import threading
import time
from urllib.parse import parse_qs, urlparse

# Optional shared secret: if set, all requests must carry an
# `Authorization: Bearer <token>` header matching it. Unset (default) means
# no auth, matching the previous behavior.
TOKEN = os.environ.get("COORD_TOKEN", "")


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.node_ids: dict[int, str] = {}

    def register(self, rank: int, node_id: str) -> str | None:
        """Register a worker's node id for a rank.

        Returns None on success, including an idempotent re-send of the same
        node id. Returns the already-registered node id if this rank was
        already claimed by a *different* node id, so the caller can reject
        the conflict instead of silently overwriting it.
        """
        with self.lock:
            existing = self.node_ids.get(rank)
            if existing is not None and existing != node_id:
                return existing
            self.node_ids[rank] = node_id
            return None

    def reset(self) -> None:
        with self.lock:
            self.node_ids.clear()

    def rpc_arg(self, world_size: int) -> str | None:
        with self.lock:
            ranks = range(1, world_size)
            if not all(r in self.node_ids for r in ranks):
                return None
            return ",".join(f"{self.node_ids[r]}:0" for r in ranks)


state = State()


class Handler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        if not TOKEN:
            return True
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_POST(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/register":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            rank = int(data["rank"])
            node_id = str(data["node_id"])
            conflict = state.register(rank, node_id)
            if conflict is not None:
                self.log_message(
                    "REJECTED rank %d: already registered as %s, got %s (call POST /reset before starting a new launch)",
                    rank, conflict, node_id,
                )
                self._send_json(409, {
                    "error": f"rank {rank} already registered with a different node id",
                    "existing_node_id": conflict,
                })
                return
            self.log_message("registered rank %d -> %s", rank, node_id)
            self._send_json(200, {"ok": True})
        elif self.path == "/reset":
            state.reset()
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not found"})

    def do_GET(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/rpc":
            qs = parse_qs(parsed.query)
            world_size = int(qs.get("world_size", ["1"])[0])
            timeout = float(qs.get("timeout", ["300"])[0])
            deadline = time.monotonic() + timeout
            while True:
                rpc = state.rpc_arg(world_size)
                if rpc is not None:
                    self._send_json(200, {"rpc": rpc})
                    return
                if time.monotonic() >= deadline:
                    self._send_json(504, {"error": f"timed out waiting for {world_size - 1} worker(s) to register"})
                    return
                time.sleep(0.2)
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"coordinator: {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"coordinator: listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
