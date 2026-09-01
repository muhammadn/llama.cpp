# Minimal image for tools/rpc/coordinator.py (the iroh RPC rendezvous
# server). Stdlib-only, no pip dependencies to install.
FROM python:3-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY coordinator.py .

RUN useradd --no-create-home --shell /usr/sbin/nologin coordinator
USER coordinator

EXPOSE 8765

# COORD_TOKEN can be set via `docker run -e COORD_TOKEN=...` to require
# Authorization: Bearer <token> on every request (unset: no auth).
ENTRYPOINT ["python3", "coordinator.py", "--host", "0.0.0.0", "--port", "8765"]
