"""Allowlisting HTTPS proxy for the sandbox setup phase.

Runs in its own container, attached both to the setup container's internal
network (which has no route out) and to a network with internet access. It
only tunnels ``CONNECT`` requests for allowlisted host names on port 443, so
dependency installation can reach the package index and nothing else.

Standalone on purpose (stdlib only): it is copied into the proxy container and
started with ``python proxy.py --allow pypi.org --allow files.pythonhosted.org``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

MAX_HEADER_BYTES = 8192
HEADER_TIMEOUT = 30
CONNECT_TIMEOUT = 30
IDLE_TIMEOUT = 300


def log(*parts: object) -> None:
    print(*parts, flush=True)


def is_allowed(host: str, port: int, allow: frozenset[str]) -> bool:
    return port == 443 and host.lower().rstrip(".") in allow


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await asyncio.wait_for(reader.read(65536), IDLE_TIMEOUT)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (TimeoutError, ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def _reply(writer: asyncio.StreamWriter, status: str) -> None:
    writer.write(f"HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
    try:
        await writer.drain()
    finally:
        writer.close()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, allow: frozenset[str]) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEADER_TIMEOUT)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError, ConnectionError):
        writer.close()
        return
    request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = request_line.split()
    if len(parts) != 3 or parts[0].upper() != "CONNECT":
        log("deny", request_line[:200])
        await _reply(writer, "403 Forbidden")
        return
    host, _, port_text = parts[1].rpartition(":")
    host = host.strip("[]")
    try:
        port = int(port_text)
    except ValueError:
        port = -1
    if not is_allowed(host, port, allow):
        log("deny", f"{host}:{port}")
        await _reply(writer, "403 Forbidden")
        return
    try:
        up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, port), CONNECT_TIMEOUT)
    except (OSError, TimeoutError) as exc:
        log("error", f"{host}:{port}", type(exc).__name__)
        await _reply(writer, "502 Bad Gateway")
        return
    log("allow", f"{host}:{port}")
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))


async def serve(listen: str, port: int, allow: frozenset[str]) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, allow), listen, port, limit=MAX_HEADER_BYTES
    )
    log("listening", f"{listen}:{port}", "allow=" + ",".join(sorted(allow)))
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--allow", action="append", default=[], help="Allowed host name (port 443 only).")
    args = parser.parse_args(argv)
    allow = frozenset(h.lower().rstrip(".") for h in args.allow)
    try:
        asyncio.run(serve(args.listen, args.port, allow))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
