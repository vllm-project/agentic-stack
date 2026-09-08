#!/usr/bin/env python3
"""Replay recorded vLLM turns for coding-harness CI acceptance tests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import yaml


@dataclass(frozen=True)
class ReplayTurn:
    status_code: int
    content_type: str
    body: bytes


def load_turns(path: Path) -> list[ReplayTurn]:
    document = yaml.safe_load(path.read_text())
    return [
        ReplayTurn(
            status_code=turn["response"]["status_code"],
            content_type=turn["response"]["headers"]["content-type"],
            body="".join(turn["response"]["sse"]).encode(),
        )
        for turn in document["turns"]
    ]


def adapt_stream(stream: str, declared_tool_name: str) -> str:
    return stream.replace('"name":"web_search"', f'"name":"{declared_tool_name}"')


def append_capture(path: Path, kind: str, body: dict[str, Any]) -> None:
    with path.open("a") as capture:
        capture.write(json.dumps({"kind": kind, "body": body}, separators=(",", ":")) + "\n")


def load_capture(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _content_blocks(request: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for message in request.get("messages", []):
        content = message.get("content", [])
        if isinstance(content, list):
            blocks.extend(block for block in content if isinstance(block, dict))
    return blocks


def _cache_breakpoint_ttls(request: dict[str, Any]) -> list[str]:
    effective_blocks: list[dict[str, Any]] = []
    effective_blocks.extend(tool for tool in request.get("tools", []) if isinstance(tool, dict))
    system = request.get("system", [])
    if isinstance(system, list):
        effective_blocks.extend(block for block in system if isinstance(block, dict))
    effective_blocks.extend(_content_blocks(request))

    ttls: list[str] = []
    for block in effective_blocks:
        cache_control = block.get("cache_control")
        if isinstance(cache_control, dict):
            ttls.append(cache_control.get("ttl", "5m"))
    return ttls


def validate_capture(records: list[dict[str, Any]], expected_model: str | None = None) -> None:
    messages = [record["body"] for record in records if record["kind"] == "messages"]
    transports = [record["body"] for record in records if record["kind"] == "messages_transport"]
    searches = [record["body"] for record in records if record["kind"] == "search"]
    assert len(messages) == 2, f"expected two Messages rounds, got {len(messages)}"
    assert len(transports) == 2, f"expected transport capture for both Messages rounds, got {len(transports)}"
    assert len(searches) == 1, f"expected one search request, got {len(searches)}"

    if expected_model is not None:
        assert all(message.get("model") == expected_model for message in messages), (
            f"expected Claude Code to request model {expected_model!r}"
        )

    session_ids = set()
    for transport in transports:
        assert transport["path"] == "/v1/messages?beta=true", transport
        headers = transport["headers"]
        assert headers.get("anthropic-version") == "2023-06-01", headers
        assert headers.get("anthropic-beta"), headers
        assert headers.get("x-api-key") == "ci-placeholder", headers
        session_id = headers.get("x-claude-code-session-id")
        assert session_id, headers
        session_ids.add(session_id)
    assert len(session_ids) == 1, "expected one Claude Code session ID across gateway tool rounds"

    system = messages[0].get("system")
    assert isinstance(system, list) and len(system) >= 2, "expected a multi-block system prompt"
    cached_system_blocks = [block for block in system if "cache_control" in block]
    assert len(cached_system_blocks) >= 2, "expected cache_control on at least two system blocks"
    assert messages[1].get("system") == system, "expected the system prompt to survive the gateway tool round"

    cache_ttls = _cache_breakpoint_ttls(messages[0])
    assert len(cache_ttls) <= 4, f"expected at most four cache breakpoints, got {len(cache_ttls)}"
    assert all(ttl in {"1h", "5m"} for ttl in cache_ttls), f"unexpected cache TTLs: {cache_ttls}"
    seen_short_ttl = False
    for ttl in cache_ttls:
        if ttl == "5m":
            seen_short_ttl = True
        assert not (ttl == "1h" and seen_short_ttl), "1h cache breakpoints must precede 5m breakpoints"

    assert any(
        block.get("type") == "text" and "cache_control" in block for block in _content_blocks(messages[0])
    ), "expected cache_control on the user prompt"

    tools = messages[0].get("tools", [])
    web_search = next((tool for tool in tools if tool.get("name") == "WebSearch"), None)
    assert web_search is not None, "expected Claude Code to declare WebSearch"
    assert messages[1].get("tools") == tools, "expected tool declarations to survive the gateway tool round"

    assert any(
        block.get("type") == "tool_result" for block in _content_blocks(messages[1])
    ), "expected a tool_result in the second Messages round"
    assert searches[0].get("query"), "expected a non-empty search query"


PNG_DATA_URL_PREFIX = "data:image/png;base64,"


def _inline_image_urls(request: dict[str, Any]) -> list[str]:
    """Every `input_image` URL in a Responses request, in the order it was sent."""
    urls: list[str] = []
    for item in request.get("input", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                url = part.get("image_url")
                if isinstance(url, str):
                    urls.append(url)
    return urls


def _inline_image_digests(request: dict[str, Any]) -> list[str]:
    """SHA-256 of each inline PNG payload, proving the exact bytes survived the gateway."""
    digests: list[str] = []
    for url in _inline_image_urls(request):
        assert url.startswith(PNG_DATA_URL_PREFIX), (
            f"expected an inline PNG data URL, got {url[:48]!r}"
        )
        payload = base64.b64decode(url[len(PNG_DATA_URL_PREFIX) :], validate=True)
        digests.append(hashlib.sha256(payload).hexdigest())
    return digests


def validate_responses_capture(
    records: list[dict[str, Any]],
    expected_model: str,
    expected_image_sha256: str | None = None,
) -> None:
    responses = [record["body"] for record in records if record["kind"] == "responses"]
    transports = [record["body"] for record in records if record["kind"] == "responses_transport"]
    assert len(responses) == 1, f"expected one Responses request, got {len(responses)}"
    assert len(transports) == 1, f"expected one Responses transport capture, got {len(transports)}"
    assert transports[0]["path"] == "/v1/responses", transports[0]

    request = responses[0]
    assert request.get("model") == expected_model, f"expected requested model {expected_model!r}, got {request.get('model')!r}"
    assert request.get("stream") is True, "expected Codex to request a streaming response"
    assert request.get("input"), "expected a non-empty Responses input"

    if expected_image_sha256 is not None:
        digests = _inline_image_digests(request)
        assert digests, (
            "expected Codex to attach inline image content; a request without an "
            "input_image part means the image was stripped before it reached the gateway"
        )
        assert expected_image_sha256 in digests, (
            f"expected an attached image with sha256 {expected_image_sha256}, got {digests}"
        )


@dataclass
class ReplayState:
    turns: list[ReplayTurn]
    capture_path: Path
    next_turn: int = 0
    #: Served model id for `/v1/models`. Codex reads its image support from the
    #: gateway catalog, which the gateway builds from this upstream listing, so a
    #: replay server that serves no catalog cannot launch Codex at all.
    model: str | None = None

    def __post_init__(self) -> None:
        self.lock = threading.Lock()

    def take_turn(self) -> ReplayTurn | None:
        with self.lock:
            if self.next_turn >= len(self.turns):
                return None
            turn = self.turns[self.next_turn]
            self.next_turn += 1
            return turn


def _declared_web_search_name(request: dict[str, Any]) -> str:
    for tool in request.get("tools", []):
        if tool.get("name") == "WebSearch":
            return "WebSearch"
    return "web_search"


def make_handler(state: ReplayState) -> type[BaseHTTPRequestHandler]:
    class ReplayHandler(BaseHTTPRequestHandler):
        def _send_search_response(self, request: dict[str, Any]) -> None:
            append_capture(state.capture_path, "search", request)
            self._send_json(
                200,
                {
                    "results": {
                        "web": [
                            {
                                "url": "https://www.rust-lang.org/",
                                "title": "Rust",
                                "description": "Rust language release",
                                "snippets": ["Rust 1.89.0 is the latest stable release."],
                            }
                        ],
                        "news": [],
                    },
                    "metadata": {"query": request.get("query", ""), "search_uuid": "ci-search", "latency": 0.0},
                },
            )

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/health":
                self._send_bytes(200, "text/plain", b"")
                return
            if parsed.path == "/v1/models":
                if state.model is None:
                    self.send_error(404)
                    return
                # `image` makes the gateway resolve text+image modalities, which is
                # what stops Codex from stripping image content client-side. This is
                # a transport fixture: no model runs here, the turns are replayed.
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model,
                                "object": "model",
                                "owned_by": "replay",
                                "capabilities": ["image", "reasoning"],
                            }
                        ],
                    },
                )
                return
            if parsed.path == "/v1/search":
                query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                self._send_search_response(query)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            request = self._read_json()
            if request is None:
                return

            if path == "/v1/messages/count_tokens":
                self._send_json(200, {"input_tokens": 512})
                return
            if path == "/v1/search":
                self._send_search_response(request)
                return
            if path == "/v1/responses":
                append_capture(
                    state.capture_path,
                    "responses_transport",
                    {"path": self.path},
                )
                append_capture(state.capture_path, "responses", request)
                turn = state.take_turn()
                if turn is None:
                    self._send_json(409, {"error": {"type": "api_error", "message": "cassette exhausted"}})
                    return
                self._send_bytes(turn.status_code, turn.content_type, turn.body)
                return
            if path != "/v1/messages":
                self.send_error(404)
                return

            append_capture(
                state.capture_path,
                "messages_transport",
                {
                    "path": self.path,
                    "headers": {name.lower(): value for name, value in self.headers.items()},
                },
            )
            append_capture(state.capture_path, "messages", request)
            turn = state.take_turn()
            if turn is None:
                self._send_json(409, {"error": {"type": "api_error", "message": "cassette exhausted"}})
                return
            body = adapt_stream(turn.body.decode(), _declared_web_search_name(request)).encode()
            self._send_bytes(turn.status_code, turn.content_type, body)

        def _read_json(self) -> dict[str, Any] | None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(content_length))
            except (ValueError, json.JSONDecodeError):
                self.send_error(400, "request body must be valid JSON")
                return None
            if not isinstance(body, dict):
                self.send_error(400, "request body must be a JSON object")
                return None
            return body

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            self._send_bytes(status, "application/json", json.dumps(body, separators=(",", ":")).encode())

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ReplayHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--cassette", required=True, type=Path)
    serve.add_argument("--port", required=True, type=int)
    serve.add_argument("--capture", required=True, type=Path)
    serve.add_argument(
        "--model",
        help="serve this model id from /v1/models with image support; omit to serve no catalog",
    )

    assert_capture = subparsers.add_parser("assert-capture")
    assert_capture.add_argument("--capture", required=True, type=Path)
    assert_capture.add_argument("--api", choices=("messages", "responses"), default="messages")
    assert_capture.add_argument("--model", required=True)
    assert_capture.add_argument(
        "--expect-image-sha256",
        help="require an inline PNG with this digest in the captured Responses request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "assert-capture":
        records = load_capture(args.capture)
        if args.api == "responses":
            validate_responses_capture(records, args.model, args.expect_image_sha256)
            responses = sum(record["kind"] == "responses" for record in records)
            transports = sum(record["kind"] == "responses_transport" for record in records)
            print(f"capture valid: responses={responses} transports={transports}")
        else:
            validate_capture(records, args.model)
            messages = sum(record["kind"] == "messages" for record in records)
            transports = sum(record["kind"] == "messages_transport" for record in records)
            searches = sum(record["kind"] == "search" for record in records)
            print(f"capture valid: messages={messages} transports={transports} searches={searches}")
        return

    args.capture.parent.mkdir(parents=True, exist_ok=True)
    args.capture.write_text("")
    state = ReplayState(load_turns(args.cassette), args.capture, model=args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    server.serve_forever()


if __name__ == "__main__":
    main()
