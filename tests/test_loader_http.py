"""HTTP Range redirect preservation, token stripping, and 200-fallback safety tests."""

from __future__ import annotations

import http.server
import socketserver
import threading
from typing import Any
import pytest

from conftest import build_safetensors_header_file
from quantpreflight.loader import CheckpointLoadError, load_checkpoint_metadata


class _QuietTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def test_cross_origin_redirect_preserves_range_and_strips_auth_token() -> None:
    st_bytes = build_safetensors_header_file(
        {"model.embed_tokens.weight": ("BF16", [1024, 512])}
    ) + (b"\x00" * 200_000)

    recorded: dict[str, Any] = {
        "origin_auth": None,
        "cdn_auth": None,
        "cdn_range": None,
    }

    class CDNHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            recorded["cdn_auth"] = self.headers.get("Authorization")
            rng = self.headers.get("Range")
            recorded["cdn_range"] = rng
            if rng and rng.startswith("bytes="):
                s_str, e_str = rng[len("bytes=") :].split("-")
                s, e = int(s_str), int(e_str)
                chunk = st_bytes[s : e + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(st_bytes)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    cdn_srv = _QuietTCPServer(("127.0.0.1", 0), CDNHandler)
    cdn_port = cdn_srv.server_address[1]

    class OriginHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            recorded["origin_auth"] = self.headers.get("Authorization")
            self.send_response(302)
            # Different host:port triggers cross-origin check in _RangePreservingRedirectHandler
            self.send_header("Location", f"http://localhost:{cdn_port}/blob/model.safetensors")
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    origin_srv = _QuietTCPServer(("127.0.0.1", 0), OriginHandler)
    origin_port = origin_srv.server_address[1]

    t1 = threading.Thread(target=cdn_srv.serve_forever, daemon=True)
    t2 = threading.Thread(target=origin_srv.serve_forever, daemon=True)
    t1.start()
    t2.start()

    try:
        meta = load_checkpoint_metadata(
            f"http://127.0.0.1:{origin_port}/repo/model.safetensors",
            token="hf_super_secret_test_token",
        )
        assert "model.embed_tokens.weight" in meta.tensors
        # Auth was sent to origin, stripped on cross-origin redirect to CDN, while Range stayed intact!
        assert recorded["origin_auth"] == "Bearer hf_super_secret_test_token"
        assert recorded["cdn_auth"] is None
        assert recorded["cdn_range"] is not None and recorded["cdn_range"].startswith("bytes=0-")
        assert meta.header_bytes_fetched <= 65536
    finally:
        origin_srv.shutdown()
        cdn_srv.shutdown()
        origin_srv.server_close()
        cdn_srv.server_close()


def test_server_ignoring_nonzero_range_offset_is_rejected() -> None:
    # Header > 32 bytes so a second Range request at offset > 0 is needed if initial window were tiny,
    # or test _http_fetch_bytes directly when server returns HTTP 200 for byte_range=(64, 127)
    from quantpreflight.loader import _http_fetch_bytes

    class BadRangeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * 1024)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    srv = _QuietTCPServer(("127.0.0.1", 0), BadRangeHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(CheckpointLoadError, match="ignored HTTP Range"):
            _http_fetch_bytes(f"http://127.0.0.1:{port}/model.safetensors", byte_range=(64, 127))
    finally:
        srv.shutdown()
        srv.server_close()
