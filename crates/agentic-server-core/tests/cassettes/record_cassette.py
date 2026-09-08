"""
Interactive multi-turn cassette recorder.

Starts an embedded recording proxy between this script and the upstream API,
then drives multi-turn conversations so every request/response is captured
into a YAML cassette.

Wiring:

  [this script] → [embedded proxy:<proxy-port>] → [OpenAI API | vLLM | gateway]
                   (cassette recorded here)

Modes:
  conv        (default) Creates a conversation via POST /v1/conversations, then
              passes conversation id on every turn.
  isolation   Two independent conversations (each with its own conversation id)
              recorded into the same cassette.
  mixed       Creates a conversation; turn 1 uses conversation id, turns 2+
              switch to previous_response_id only (drops conversation id).
  responses   No conversation created. Chains turns purely via
              previous_response_id. Supports --openai, --vllm, and --gateway backends.

Usage:
    python tests/cassettes/record_cassette.py --turns 2 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 3 --mode isolation --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 3 --mode mixed --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 3 --mode conv --branch-from 1 --branch-turn-number 2 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 5 --mode conv --branch-from 1 --branch-turn-number 3 --branch-from 2 --branch-turn-number 5 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 2 --mode responses --vllm http://localhost:8000 --model Qwen/Qwen3-30B-A3B-FP8 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 2 --mode responses --transport websocket --vllm http://localhost:3018 --model Qwen/Qwen3.6-35B-A3B --output path/to/ws-cassette.yaml
    python tests/cassettes/record_cassette.py --turns 2 --mode responses --vllm http://localhost:8000 --model Qwen/Qwen3-30B-A3B-FP8 --max-output-tokens 1024 --no-stream --output path/to/cassette.yaml
    python tests/cassettes/record_cassette.py --turns 1 --mode responses --gateway http://localhost:9000 --model Qwen/Qwen3-30B-A3B-FP8 --no-stream --output path/to/cassette.yaml
"""

import base64
import hashlib
import importlib.util
import json
import logging
import os
import secrets
import socket
import ssl
import struct
import sys
import threading
import time
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

import click
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import AsyncClient
from yaml import dump as yaml_dump, safe_load as yaml_load

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cassette_proxy")

MODEL = "gpt-4o"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7070
TIMEOUT = 60 * 5

EXCLUDED_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
}

RECORDED_HEADERS = {
    "content-type",
    "authorization",
    "user-agent",
    "accept",
    "x-run-id",
}


def _mask_authorization(value: str) -> str:
    if not value:
        return value
    lower = value.lower()
    if lower.startswith("bearer "):
        return "Bearer ***"
    return "***"


def _filter_request_headers(headers) -> dict:
    return {
        k: v if k.lower() != "authorization" else _mask_authorization(v)
        for k, v in headers.items()
        if k.lower() in RECORDED_HEADERS
    }


def _filter_response_headers(headers) -> dict:
    return {
        k: v for k, v in headers.items() if k.lower() not in EXCLUDED_RESPONSE_HEADERS
    }


def _turn_number(output_file: Path) -> int:
    if not output_file.exists():
        return 1
    content = output_file.read_text(encoding="utf-8")
    if not content.strip():
        return 1
    data = yaml_load(content)
    if not data or "turns" not in data:
        return 1
    return len(data["turns"]) + 1


def _append_turn(output_file: Path, turn: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and output_file.stat().st_size > 0:
        data = yaml_load(output_file.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    turns: list = data.get("turns", [])
    turns.append(turn)
    data["turns"] = turns
    with open(output_file, "w", encoding="utf-8") as f:
        yaml_dump(data, f, allow_unicode=True, default_flow_style=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = AsyncClient(timeout=TIMEOUT)
    yield
    await app.state.http_client.aclose()


proxy_app = FastAPI(lifespan=lifespan)


@proxy_app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_request(request: Request, path: str) -> Response:
    http_client: AsyncClient = request.app.state.http_client
    target_host: str = request.app.state.target_host
    output_file: Path = request.app.state.output_file

    turn_num = _turn_number(output_file)
    filename = f"t{turn_num}"

    target_url = f"{target_host}/{path}"
    if str(request.query_params):
        target_url += f"?{request.query_params}"

    raw_body = await request.body()
    parsed_body = json.loads(raw_body.decode("utf-8")) if raw_body else {}

    turn: dict[str, Any] = {
        "filename": filename,
        "request": {
            "method": request.method,
            "path": f"/{path}",
            "query_params": dict(request.query_params),
            "body": parsed_body,
            "headers": _filter_request_headers(request.headers),
        },
        "response": {},
    }

    forward_headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    if parsed_body.get("stream", False):

        async def _stream() -> AsyncGenerator[str, None]:
            async with http_client.stream(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=raw_body,
                timeout=TIMEOUT,
            ) as response:
                yield response  # type: ignore[misc]
                if response.status_code != 200:
                    chunk_str = (await response.aread()).decode()
                    try:
                        turn["response"]["body"] = json.loads(chunk_str)
                    except Exception:
                        turn["response"]["body"] = chunk_str
                    yield chunk_str
                else:
                    sse_events: list[str] = []
                    try:
                        async for line in response.aiter_lines():
                            chunk = f"{line}\n"
                            yield chunk
                            sse_events.append(chunk)
                    except Exception as e:
                        turn["response"]["stream_error"] = (
                            f"{e.__class__.__name__}: {e}"
                        )
                    finally:
                        turn["response"]["sse"] = sse_events
                turn["response"]["status_code"] = response.status_code
                turn["response"]["headers"] = {
                    "content-type": response.headers.get(
                        "content-type", "text/event-stream"
                    )
                }
                _append_turn(output_file, turn)
                print(f"  [recorded turn {turn_num} -> {output_file.name}]")

        agen = _stream()
        upstream = await anext(agen)
        return StreamingResponse(
            agen,
            status_code=upstream.status_code,
            headers=_filter_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    else:
        response = await http_client.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=raw_body,
            timeout=TIMEOUT,
        )
        media_type = response.headers.get("content-type", "application/json")
        body: Any = response.json() if response.status_code == 200 else response.text
        if response.status_code != 200 and "application/json" in media_type:
            try:
                body = json.loads(body)
            except Exception:
                pass
        turn["response"]["body"] = body
        turn["response"]["status_code"] = response.status_code
        turn["response"]["headers"] = {"content-type": media_type}
        _append_turn(output_file, turn)
        print(f"  [recorded turn {turn_num} -> {output_file.name}]")
        return JSONResponse(
            content=body,
            status_code=response.status_code,
            headers=_filter_response_headers(response.headers),
            media_type=media_type,
        )


# ── proxy lifecycle ───────────────────────────────────────────────────────────


def _start_proxy(output_file: Path, target_host: str, port: int, append: bool = False) -> uvicorn.Server:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not append or not output_file.exists():
        output_file.write_text("", encoding="utf-8")
    proxy_app.state.output_file = output_file
    proxy_app.state.target_host = target_host

    config = uvicorn.Config(proxy_app, host=PROXY_HOST, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # TCP-only readiness check — no HTTP request forwarded to upstream
    for _ in range(40):
        try:
            with socket.create_connection((PROXY_HOST, port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.3)

    return server


def _stop_proxy(server: uvicorn.Server) -> None:
    server.should_exit = True
    time.sleep(0.5)


def _create_conversation(client: httpx.Client, proxy_url: str) -> str:
    resp = client.post(f"{proxy_url}/v1/conversations", json={}, timeout=30)
    resp.raise_for_status()
    conv_id = resp.json().get("id")
    print(f"[conversation created: {conv_id}]")
    return conv_id


def _send_nonstreaming(client: httpx.Client, body: dict, proxy_url: str) -> dict | None:
    resp = client.post(f"{proxy_url}/v1/responses", json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    print(f"\n[Response]\n{json.dumps(data, indent=2)}\n")
    return data


def _send_streaming(client: httpx.Client, body: dict, proxy_url: str) -> dict | None:
    response_data = None
    print("\n[Streaming response]")
    with client.stream(
        "POST", f"{proxy_url}/v1/responses", json=body, timeout=300
    ) as resp:
        if resp.status_code != 200:
            # Drain the body fully before raising: the recording proxy is an
            # async generator that only finishes writing this turn once its
            # response is fully consumed. Raising immediately (before reading)
            # aborts the connection and can tear the proxy's generator down
            # before it appends the turn, silently losing this turn's error
            # response from the cassette entirely.
            resp.read()
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            print(line)
            if line.startswith("data:") and line != "data: [DONE]":
                try:
                    payload = json.loads(line[5:].strip())
                    if payload.get("type") == "response.completed":
                        response_data = payload.get("response")
                    elif payload.get("object") == "response" and payload.get("status") == "completed":
                        response_data = payload
                except Exception:
                    pass
    print()
    return response_data


def _send_messages_nonstreaming(client: httpx.Client, body: dict, proxy_url: str) -> dict | None:
    """Send an Anthropic Messages request (non-streaming) and return the message object."""
    resp = client.post(f"{proxy_url}/v1/messages", json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    print(f"\n[Message]\n{json.dumps(data, indent=2)}\n")
    return data


def _send_messages_streaming(client: httpx.Client, body: dict, proxy_url: str) -> dict | None:
    """Send an Anthropic Messages request (streaming) and reconstruct the final message.

    Anthropic SSE sends the message envelope in `message_start`, then mutates it via
    `content_block_start`/`content_block_delta`/`content_block_stop` and `message_delta`.
    We accumulate those into the final message object so callers can chain the next turn.
    """
    message: dict | None = None
    blocks: dict[int, dict] = {}
    print("\n[Streaming message]")
    with client.stream("POST", f"{proxy_url}/v1/messages", json=body, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            print(line)
            try:
                event = json.loads(line[5:].strip())
            except Exception:
                continue
            etype = event.get("type")
            if etype == "message_start":
                message = event.get("message", {})
            elif etype == "content_block_start":
                blocks[event["index"]] = event.get("content_block", {})
            elif etype == "content_block_delta":
                blk = blocks.setdefault(event["index"], {})
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    blk["type"] = blk.get("type", "text")
                    blk["text"] = blk.get("text", "") + delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    blk["type"] = blk.get("type", "tool_use")
                    blk["_partial_json"] = blk.get("_partial_json", "") + delta.get("partial_json", "")
                elif delta.get("type") == "thinking_delta":
                    blk["type"] = blk.get("type", "thinking")
                    blk["thinking"] = blk.get("thinking", "") + delta.get("thinking", "")
            elif etype == "message_delta" and message is not None:
                message.update({k: v for k, v in event.get("delta", {}).items()})
    print()
    if message is not None:
        # Finalize accumulated tool_use input from partial JSON.
        for blk in blocks.values():
            if blk.get("type") == "tool_use" and "_partial_json" in blk:
                raw = blk.pop("_partial_json")
                try:
                    blk["input"] = json.loads(raw) if raw else {}
                except Exception:
                    blk["input"] = {}
        message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


class WebSocketClient:
    """Small stdlib websocket client for cassette recording."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self.sock: socket.socket | ssl.SSLSocket | None = None

    def __enter__(self) -> "WebSocketClient":
        parsed = urlparse(self.url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError(f"websocket URL must use ws:// or wss://, got {self.url}")

        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        raw_sock = socket.create_connection((host, port), timeout=TIMEOUT)
        if parsed.scheme == "wss":
            context = ssl.create_default_context()
            self.sock = context.wrap_socket(raw_sock, server_hostname=host)
        else:
            self.sock = raw_sock

        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        host_header = host if parsed.port is None else f"{host}:{port}"
        request_headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host_header}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        for name, value in self.headers.items():
            request_headers.append(f"{name}: {value}")
        request = "\r\n".join(request_headers) + "\r\n\r\n"
        self.sock.sendall(request.encode("utf-8"))

        response = self._read_http_response()
        status_line, _, header_text = response.partition("\r\n")
        if " 101 " not in status_line:
            raise RuntimeError(f"websocket upgrade failed: {status_line}\n{header_text}")
        accept = _headers_from_text(header_text).get("sec-websocket-accept")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if accept != expected:
            raise RuntimeError("websocket upgrade failed: invalid Sec-WebSocket-Accept")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        try:
            self.send_close()
        finally:
            if self.sock is not None:
                self.sock.close()

    def _read_exact(self, size: int) -> bytes:
        assert self.sock is not None
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.sock.recv(size - len(chunks))
            if not chunk:
                raise EOFError("websocket closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_http_response(self) -> str:
        assert self.sock is not None
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise EOFError("websocket closed during handshake")
            data.extend(chunk)
        return data.decode("iso-8859-1")

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def send_close(self) -> None:
        if self.sock is not None:
            try:
                self._send_frame(0x8, b"")
            except OSError:
                pass

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        assert self.sock is not None
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = secrets.token_bytes(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def receive_text(self) -> str | None:
        message = bytearray()
        while True:
            first, second = self._read_exact(2)
            fin = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length)
            if masked:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))

            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in {0x1, 0x0}:
                message.extend(payload)
                if fin:
                    return message.decode("utf-8")


def _headers_from_text(header_text: str) -> dict[str, str]:
    headers = {}
    for line in header_text.split("\r\n"):
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return headers


def _websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    if parsed.scheme in {"ws", "wss"}:
        root = base_url.rstrip("/")
    elif parsed.scheme == "http":
        root = "ws://" + base_url.rstrip("/")[len("http://"):]
    elif parsed.scheme == "https":
        root = "wss://" + base_url.rstrip("/")[len("https://"):]
    else:
        raise ValueError(f"unsupported websocket base URL: {base_url}")
    return f"{root}/v1/responses"


def _send_websocket(
    body: dict,
    target_base_url: str,
    headers: dict[str, str],
    output_file: Path,
) -> dict | None:
    turn_num = _turn_number(output_file)
    wire_body = dict(body)
    wire_body["type"] = "response.create"
    # WebSocket mode streams by transport; OpenAI's Responses WebSocket API
    # does not use HTTP-only fields such as `stream`.
    wire_body.pop("stream", None)
    wire_body["store"] = True
    websocket_url = _websocket_url(target_base_url)

    turn: dict[str, Any] = {
        "filename": f"t{turn_num}",
        "request": {
            "method": "WEBSOCKET",
            "path": "/v1/responses",
            "query_params": {},
            "body": wire_body,
            "headers": _filter_request_headers(headers),
            "transport": "websocket",
        },
        "response": {
            "status_code": 101,
            "headers": {"transport": "websocket"},
            "websocket": [],
            "sse": [],
        },
    }

    response_data = None
    print("\n[WebSocket response]")
    with WebSocketClient(websocket_url, headers) as ws:
        ws.send_text(json.dumps(wire_body, separators=(",", ":")))
        while True:
            message = ws.receive_text()
            if message is None:
                break
            print(message)
            turn["response"]["websocket"].append(message)
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                continue
            turn["response"]["sse"].append(
                f"data: {json.dumps(event, separators=(',', ':'))}\n"
            )
            event_type = event.get("type")
            if event_type == "response.completed":
                response_data = event.get("response")
                break
            if event_type == "error":
                response_data = event
                break
    turn["response"]["sse"].append("data: [DONE]\n")
    _append_turn(output_file, turn)
    print(f"  [recorded turn {turn_num} -> {output_file.name}]")
    return response_data


def _send(
    client: httpx.Client,
    body: dict,
    stream: bool,
    proxy_url: str,
    transport: str = "http",
    target_base_url: str = "",
    headers: dict[str, str] | None = None,
    output_file: Path | None = None,
) -> dict | None:
    if transport == "websocket":
        if output_file is None:
            raise ValueError("output_file is required for websocket recording")
        return _send_websocket(body, target_base_url, headers or {}, output_file)
    return (
        _send_streaming(client, body, proxy_url)
        if stream
        else _send_nonstreaming(client, body, proxy_url)
    )


def _prompt(label: str) -> str:
    try:
        return input(label).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)


def _load_response_input(path: str | None) -> str | list | None:
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise click.UsageError(f"--input-file is not valid JSON: {error}") from error
    if not isinstance(value, (str, list)):
        raise click.UsageError("--input-file must contain a JSON string or array.")
    return value


def _parse_reasoning(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise click.UsageError(f"--reasoning is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise click.UsageError("--reasoning must contain a JSON object.")
    return value


def _inject_tools(
    body: dict, tools: list | None, tool_choice: Any, parallel_tool_calls: bool | None = None
) -> None:
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        body["parallel_tool_calls"] = parallel_tool_calls


def _extract_tool_calls(response_data: dict | None) -> list[dict]:
    """Extract client-executed function, custom, and shell calls from a response."""
    if not response_data:
        return []
    output = response_data.get("output", [])
    return [
        item
        for item in output
        if item.get("type") in {"function_call", "custom_tool_call", "shell_call"}
    ]


def _build_tool_output_input(
    tool_calls: list[dict],
    tool_outputs: "dict[str, Any] | types.ModuleType",
    user_prompt: str | None,
) -> list[dict]:
    """Build tool output items followed by an optional user message.

    Args:
        tool_calls: function_call, custom_tool_call, or shell_call output items.
        tool_outputs: either
            - a dict mapping tool name -> fake JSON output string (loaded from a
              --tool-outputs *.json* file), matched by name only; or
            - a Python module (loaded from a --tool-outputs *.py* file) whose
              functions are named after each tool. Each pending call invokes the
              matching function with its actual parsed `arguments` as keyword
              arguments -- naturally handling whatever argument types the model
              used (string, number, ...) -- and the JSON-serialized return value
              becomes the output. A function returning `None` omits that call's
              output item entirely, which is how a cassette deliberately tests a
              provider's behavior when the client leaves one specific pending
              call unresolved (e.g. one of two parallel calls to the same tool
              with different arguments) while resolving its sibling(s).
            Shell calls use the key/function `shell`. Its value (or return value
            from `shell(action=...)`) is an object containing the structured
            `output` array and optional `max_output_length`, not a JSON string.
        user_prompt: the next user message (None for tool-output-only turns).

    Returns:
        A list suitable for the `input` field of the next request.
    """
    input_items: list[dict] = []
    for call in tool_calls:
        call_id = call.get("call_id", "")
        if call.get("type") == "shell_call":
            if isinstance(tool_outputs, types.ModuleType):
                fn = getattr(tool_outputs, "shell", None)
                result = fn(action=call["action"]) if fn else None
            else:
                result = tool_outputs.get("shell")
            if result is None:
                continue
            if not isinstance(result, dict) or not isinstance(result.get("output"), list):
                raise click.UsageError("shell tool output must be an object containing an output array")
            item = {"type": "shell_call_output", "call_id": call_id, "output": result["output"]}
            if "max_output_length" in result:
                item["max_output_length"] = result["max_output_length"]
            input_items.append(item)
            continue
        name = call.get("name", "")
        if isinstance(tool_outputs, types.ModuleType):
            fn = getattr(tool_outputs, name, None)
            if fn is None:
                continue
            try:
                kwargs = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError:
                kwargs = {}
            result = fn(**kwargs)
            if result is None:
                continue
            output = result if isinstance(result, str) else json.dumps(result)
        else:
            if name not in tool_outputs:
                continue
            output = tool_outputs[name]
        input_items.append(
            {
                "type": (
                    "custom_tool_call_output"
                    if call.get("type") == "custom_tool_call"
                    else "function_call_output"
                ),
                "call_id": call_id,
                "output": output,
            }
        )
    if user_prompt:
        input_items.append(
            {
                "type": "message",
                "role": "user",
                "content": user_prompt,
            }
        )
    return input_items


def run_conv(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    branches: list[tuple[int, int | None]],
    proxy_url: str,
) -> None:
    conv_id = _create_conversation(client, proxy_url)
    response_ids: dict[int, str] = {}
    # map: branch_turn_number -> branch_from (which turn's response to use as previous)
    branch_map: dict[int, int] = {}
    extra_branches: list[int] = []  # branch_from values with no branch_turn_number
    for branch_from, branch_turn_number in branches:
        if branch_turn_number is not None:
            branch_map[branch_turn_number] = branch_from
        else:
            extra_branches.append(branch_from)

    previous_response_id: str | None = None
    for turn in range(1, turns + 1):
        if turn in branch_map:
            branch_from = branch_map[turn]
            if branch_from not in response_ids:
                raise click.UsageError(
                    f"--branch-from {branch_from} at turn {turn} has no recorded response "
                    f"(available: {sorted(response_ids)})"
                )
            previous_response_id = response_ids[branch_from]
            click.echo(
                f"\n[Branch] turn {turn} chains from turn {branch_from} (response_id={previous_response_id})"
            )
        prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
        body: dict = {"model": model, "input": prompt, "stream": stream, "store": store}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        else:
            body["conversation"] = conv_id
        response_data = _send(client, body, stream, proxy_url)
        response_id = response_data.get("id") if response_data else None
        if response_id:
            response_ids[turn] = response_id
            previous_response_id = response_id

    # branches without a branch_turn_number get one extra turn each
    for b_idx, branch_from in enumerate(extra_branches, start=1):
        if branch_from not in response_ids:
            raise click.UsageError(
                f"Extra branch {b_idx}: --branch-from {branch_from} has no recorded response "
                f"(available: {sorted(response_ids)})"
            )
        branch_resp_id = response_ids[branch_from]
        click.echo(
            f"\n[Extra branch {b_idx}] from turn {branch_from} (response_id={branch_resp_id}), turn {turns + 1}"
        )
        prompt = _prompt(
            f"Turn {turns + 1} (extra branch from turn {branch_from}) — enter prompt: "
        )
        body = {
            "model": model,
            "input": prompt,
            "stream": stream,
            "store": store,
            "previous_response_id": branch_resp_id,
            "conversation": conv_id,
        }
        _send(client, body, stream, proxy_url)


def run_isolation(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    proxy_url: str,
) -> None:
    for conv_label in ("A", "B"):
        click.echo(f"\n--- Conversation {conv_label} ({turns} turns) ---")
        conv_id = _create_conversation(client, proxy_url)
        for turn in range(1, turns + 1):
            prompt = _prompt(
                f"Conv {conv_label} | Turn {turn}/{turns} — enter prompt: "
            )
            body: dict = {
                "model": model,
                "input": prompt,
                "stream": stream,
                "store": store,
                "conversation": conv_id,
            }
            _send(client, body, stream, proxy_url)


def run_store_true_then_store_false(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    proxy_url: str,
) -> None:
    """Turn 1: store=true with conversation_id. Remaining turns: store=false, still pass conversation_id."""
    conv_id = _create_conversation(client, proxy_url)
    for turn in range(1, turns + 1):
        store_turn = turn == 1
        prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
        body: dict = {
            "model": model,
            "input": prompt,
            "stream": stream,
            "store": store_turn,
            "conversation": conv_id,
        }
        _send(client, body, stream, proxy_url)


def run_mixed(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    proxy_url: str,
) -> None:
    conv_id = _create_conversation(client, proxy_url)
    previous_response_id: str | None = None

    for turn in range(1, turns + 1):
        prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
        body: dict = {"model": model, "input": prompt, "stream": stream, "store": store}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        else:
            body["conversation"] = conv_id
        response_data = _send(client, body, stream, proxy_url)
        previous_response_id = response_data.get("id") if response_data else None


def run_messages(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    proxy_url: str,
    tools: list | None,
    tool_choice: Any,
    tool_outputs: dict[str, str] | None,
    max_tokens: int,
) -> None:
    """Record Anthropic Messages turns.

    Messages is stateless: the client resends the full `messages` history each
    turn. Between turns we append the model's assistant `content` and — for any
    `tool_use` block — a matching `tool_result` block (fake output from
    `--tool-outputs`, keyed by tool name), mirroring how a gateway tool loop
    feeds results back. A turn with pending tool_use consumes those outputs
    instead of prompting for a new user message.
    """
    history: list[dict] = []
    for turn in range(1, turns + 1):
        pending_tool_use = [
            b
            for msg in history[-1:]
            if msg.get("role") == "assistant"
            for b in (msg.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if pending_tool_use and tool_outputs:
            # Feed tool_result blocks back for the prior turn's tool_use calls.
            results = []
            for call in pending_tool_use:
                out = tool_outputs.get(call.get("name"), "{}")
                results.append({"type": "tool_result", "tool_use_id": call.get("id"), "content": out})
            history.append({"role": "user", "content": results})
            click.echo(f"  [fed back {len(results)} tool_result(s) for {[c.get('name') for c in pending_tool_use]}]")
        else:
            prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")
            history.append({"role": "user", "content": prompt})

        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": history,
            "stream": stream,
        }
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        message = (
            _send_messages_streaming(client, body, proxy_url)
            if stream
            else _send_messages_nonstreaming(client, body, proxy_url)
        )
        if not message:
            click.echo("  [no message returned — stopping]")
            break
        # Append the assistant turn so the next request carries full history.
        history.append({"role": "assistant", "content": message.get("content", [])})


def run_responses(
    client: httpx.Client,
    turns: int,
    model: str,
    stream: bool,
    store: bool,
    branches: list[tuple[int, int | None]],
    proxy_url: str,
    transport: str = "http",
    target_base_url: str = "",
    headers: dict[str, str] | None = None,
    output_file: Path | None = None,
    tools: list | None = None,
    tool_choice: Any = None,
    tool_outputs: "dict[str, Any] | types.ModuleType | None" = None,
    max_output_tokens: int | None = None,
    reasoning: dict | None = None,
    preset_input: str | list | None = None,
    parallel_tool_calls: bool | None = None,
) -> None:
    response_ids: dict[int, str] = {}
    responses: dict[int, dict] = {}
    branch_map: dict[int, int] = {}
    extra_branches: list[int] = []
    for branch_from, branch_turn_number in branches:
        if branch_turn_number is not None:
            branch_map[branch_turn_number] = branch_from
        else:
            extra_branches.append(branch_from)

    previous_response_id: str | None = None
    last_response: dict | None = None
    for turn in range(1, turns + 1):
        if turn in branch_map:
            branch_from = branch_map[turn]
            if branch_from not in response_ids:
                raise click.UsageError(
                    f"--branch-from {branch_from} at turn {turn} has no recorded response "
                    f"(available: {sorted(response_ids)})"
                )
            previous_response_id = response_ids[branch_from]
            last_response = responses.get(branch_from)
            click.echo(
                f"\n[Branch] turn {turn} chains from turn {branch_from} (response_id={previous_response_id})"
            )
        if preset_input is not None:
            input_value: Any = preset_input
        else:
            prompt = _prompt(f"Turn {turn}/{turns} — enter prompt: ")

            # Inject matching function/custom/shell output items before the user message.
            pending_calls = _extract_tool_calls(last_response) if tool_outputs else []
            if pending_calls and tool_outputs:
                input_value = _build_tool_output_input(
                    pending_calls, tool_outputs, prompt if prompt else None
                )
                click.echo(
                    f"  [injecting {len(pending_calls)} tool output(s) before user message]"
                )
            else:
                input_value = prompt

        body: dict = {"model": model, "input": input_value, "stream": stream, "store": store}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if reasoning is not None:
            body["reasoning"] = reasoning
        if previous_response_id and store:
            body["previous_response_id"] = previous_response_id
        _inject_tools(body, tools, tool_choice, parallel_tool_calls)
        response_data = _send(
            client,
            body,
            stream,
            proxy_url,
            transport,
            target_base_url,
            headers,
            output_file,
        )
        response_id = response_data.get("id") if response_data else None
        previous_response_id = response_id if store else None
        last_response = response_data
        if response_id:
            response_ids[turn] = response_id
            responses[turn] = response_data

    for b_idx, branch_from in enumerate(extra_branches, start=1):
        if branch_from not in response_ids:
            raise click.UsageError(
                f"Extra branch {b_idx}: --branch-from {branch_from} has no recorded response "
                f"(available: {sorted(response_ids)})"
            )
        branch_resp_id = response_ids[branch_from]
        branch_response = responses.get(branch_from)
        click.echo(
            f"\n[Extra branch {b_idx}] from turn {branch_from} (response_id={branch_resp_id}), turn {turns + 1}"
        )
        prompt = _prompt(
            f"Turn {turns + 1} (extra branch from turn {branch_from}) — enter prompt: "
        )

        pending_calls = _extract_tool_calls(branch_response) if tool_outputs else []
        if pending_calls and tool_outputs:
            input_value = _build_tool_output_input(pending_calls, tool_outputs, prompt if prompt else None)
            click.echo(f"  [injecting {len(pending_calls)} tool output(s) before user message]")
        else:
            input_value = prompt

        body = {
            "model": model,
            "input": input_value,
            "stream": stream,
            "store": store,
            "previous_response_id": branch_resp_id,
        }
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if reasoning is not None:
            body["reasoning"] = reasoning
        _inject_tools(body, tools, tool_choice, parallel_tool_calls)
        _send(
            client,
            body,
            stream,
            proxy_url,
            transport,
            target_base_url,
            headers,
            output_file,
        )


# ── main ──────────────────────────────────────────────────────────────────────


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--turns", "-n", required=True, type=int, help="Number of turns to record."
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output cassette YAML path.",
)
@click.option(
    "--mode",
    type=click.Choice(["conv", "isolation", "mixed", "responses", "messages", "store_true_then_store_false"]),
    default="conv",
    show_default=True,
    help="Recording mode.",
)
@click.option(
    "--branch-from",
    type=int,
    multiple=True,
    metavar="TURN",
    help="Rewind to this turn's response (repeatable, one per branch).",
)
@click.option(
    "--branch-turn-number",
    type=int,
    multiple=True,
    metavar="TURN",
    help="First turn number for the corresponding branch (repeatable, pairs with --branch-from).",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    show_default=True,
    help="Use streaming responses.",
)
@click.option(
    "--transport",
    type=click.Choice(["http", "websocket"]),
    default="http",
    show_default=True,
    help="Wire transport to record. websocket is supported for --mode responses.",
)
@click.option(
    "--model", default=MODEL, show_default=True, help="Model name to pass in requests."
)
@click.option(
    "--no-store", is_flag=True, default=False, help="Set store=false in requests."
)
@click.option(
    "--proxy-port",
    type=int,
    default=PROXY_PORT,
    show_default=True,
    help="Local port for the embedded recording proxy.",
)
@click.option(
    "--openai",
    "openai_url",
    metavar="URL",
    default=None,
    help="OpenAI upstream URL (default https://api.openai.com). Reads OPENAI_API_KEY.",
)
@click.option(
    "--vllm",
    "vllm_url",
    metavar="URL",
    default=None,
    help="vLLM upstream URL, e.g. http://localhost:8000 (responses mode only, no auth).",
)
@click.option(
    "--gateway",
    "gateway_url",
    metavar="URL",
    default=None,
    help="agentic-api gateway URL, e.g. http://localhost:9000 (no auth).",
)
@click.option(
    "--tools",
    "tools_file",
    metavar="FILE",
    default=None,
    type=click.Path(exists=True),
    help="Path to a JSON file containing a tools array to inject into every request.",
)
@click.option(
    "--tool-choice",
    "tool_choice_raw",
    metavar="VALUE",
    default=None,
    help='tool_choice value: "auto", "none", "required", or JSON e.g. \'{"type":"function","name":"foo"}\'.',
)
@click.option(
    "--parallel-tool-calls",
    "parallel_tool_calls_raw",
    type=click.Choice(["true", "false"]),
    default=None,
    help="parallel_tool_calls value to send on Responses requests (omit to use the API default).",
)
@click.option(
    "--tool-outputs",
    "tool_outputs_file",
    metavar="FILE",
    default=None,
    type=click.Path(exists=True),
    help="Path to a *.json file mapping tool names to fake output strings, or a *.py file defining "
    "one function per tool name (called with the model's actual parsed arguments; returning None "
    "omits that call's output). When provided, matching function_call_output or "
    "custom_tool_call_output items are injected between turns (required for OpenAI Responses API). "
    "Shell calls use the key/function shell (called with action=...) returning an object with "
    "an output array and optional max_output_length for shell_call_output.",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file containing one Responses input value; requires HTTP --mode responses --turns 1.",
)
@click.option(
    "--reasoning",
    "reasoning_raw",
    metavar="JSON",
    default=None,
    help='Responses reasoning settings as a JSON object, e.g. \'{"effort":"high","summary":"detailed"}\'.',
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=1024,
    show_default=True,
    help="max_output_tokens for Responses requests. Use 0 to omit the field.",
)
@click.option(
    "--append",
    is_flag=True,
    default=False,
    help="Append turns to an existing --output file instead of truncating it first. Lets multiple "
    "independent invocations (e.g. separate conversation branches that each end in a provider error) "
    "accumulate into one cassette. HTTP --mode responses only.",
)
def main(
    turns: int,
    output: str,
    mode: str,
    branch_from: tuple[int, ...],
    branch_turn_number: tuple[int, ...],
    stream: bool,
    transport: str,
    model: str,
    no_store: bool,
    proxy_port: int,
    openai_url: str | None,
    vllm_url: str | None,
    gateway_url: str | None,
    tools_file: str | None,
    tool_choice_raw: str | None,
    parallel_tool_calls_raw: str | None,
    tool_outputs_file: str | None,
    input_file: str | None,
    reasoning_raw: str | None,
    max_output_tokens: int,
    append: bool,
) -> None:
    """Interactive multi-turn cassette recorder (proxy embedded)."""
    if branch_turn_number and not branch_from:
        raise click.UsageError("--branch-turn-number requires --branch-from.")
    if len(branch_turn_number) > len(branch_from):
        raise click.UsageError(
            "More --branch-turn-number values than --branch-from values."
        )
    # Pair each branch-from with its branch-turn-number (None if not provided)
    branches: list[tuple[int, int | None]] = [
        (bf, branch_turn_number[i] if i < len(branch_turn_number) else None)
        for i, bf in enumerate(branch_from)
    ]
    backend_count = sum(bool(url) for url in (openai_url, vllm_url, gateway_url))
    if backend_count > 1:
        raise click.UsageError("--openai, --vllm, and --gateway are mutually exclusive.")
    if vllm_url and mode not in ("responses", "messages"):
        raise click.UsageError(
            f"--vllm is only supported with --mode responses or --mode messages (got --mode {mode})."
        )
    if transport == "websocket" and mode != "responses":
        raise click.UsageError(
            f"--transport websocket is only supported with --mode responses (got --mode {mode})."
        )
    if max_output_tokens < 0:
        raise click.UsageError("--max-output-tokens must be >= 0.")
    if input_file and (mode != "responses" or turns != 1 or branches or transport != "http"):
        raise click.UsageError(
            "--input-file requires HTTP --mode responses --turns 1 without branches."
        )
    if reasoning_raw is not None and mode != "responses":
        raise click.UsageError("--reasoning is only supported with --mode responses.")
    preset_input = _load_response_input(input_file)
    reasoning = _parse_reasoning(reasoning_raw)

    tools: list | None = None
    if tools_file:
        with open(tools_file, encoding="utf-8") as f:
            tools = json.load(f)
        if not isinstance(tools, list):
            raise click.UsageError("--tools file must contain a JSON array.")

    tool_choice: Any = None
    if tool_choice_raw:
        stripped = tool_choice_raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            tool_choice = json.loads(stripped)
        else:
            tool_choice = stripped

    parallel_tool_calls: bool | None = None
    if parallel_tool_calls_raw is not None:
        parallel_tool_calls = parallel_tool_calls_raw == "true"

    tool_outputs: "dict[str, Any] | types.ModuleType | None" = None
    if tool_outputs_file:
        if tool_outputs_file.endswith(".py"):
            spec = importlib.util.spec_from_file_location("cassette_tool_outputs", tool_outputs_file)
            if spec is None or spec.loader is None:
                raise click.UsageError(f"--tool-outputs could not load Python module: {tool_outputs_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tool_outputs = module
            click.echo(f"Tool outputs: python functions from {tool_outputs_file}")
        else:
            with open(tool_outputs_file, encoding="utf-8") as f:
                tool_outputs = json.load(f)
            if not isinstance(tool_outputs, dict):
                raise click.UsageError("--tool-outputs JSON file must contain an object (name -> output string).")
            click.echo(f"Tool outputs: {list(tool_outputs.keys())}")

    if gateway_url:
        target = gateway_url.rstrip("/")
        headers = {}
        backend_label = f"Gateway: {target}"
    elif vllm_url:
        target = vllm_url.rstrip("/")
        headers: dict = {}
        backend_label = f"vLLM:   {target}"
    else:
        target = (openai_url or "https://api.openai.com").rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise click.ClickException(
                "OPENAI_API_KEY environment variable is not set."
            )
        headers = {"Authorization": f"Bearer {api_key}"}
        backend_label = f"OpenAI: {target}"

    output_file = Path(output).resolve()
    proxy_url = f"http://{PROXY_HOST}:{proxy_port}"
    store = not no_store
    if transport == "websocket":
        stream = True
        store = True
    response_max_output_tokens = max_output_tokens or None

    click.echo(
        f"Mode: {mode} | Turns: {turns} | Stream: {stream} | Transport: {transport} | Model: {model} | "
        f"Max output tokens: {response_max_output_tokens or 'backend default'}"
    )
    click.echo(f"Output:  {output_file}")
    click.echo(backend_label)
    if transport == "websocket":
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("", encoding="utf-8")
        click.echo(f"WebSocket: {_websocket_url(target)}")
        with httpx.Client(headers=headers) as client:
            run_responses(
                client,
                turns,
                model,
                stream,
                store,
                branches,
                proxy_url,
                transport,
                target,
                headers,
                output_file,
                tools,
                tool_choice,
                tool_outputs,
                response_max_output_tokens,
                reasoning,
                preset_input,
                parallel_tool_calls,
            )
    else:
        click.echo(f"Proxy:   {proxy_url}  (requests go through here for recording)")
        server = _start_proxy(output_file, target, proxy_port, append=append)
        click.echo(f"Proxy ready on {proxy_url}\n")

        try:
            with httpx.Client(headers=headers) as client:
                if mode == "conv":
                    run_conv(client, turns, model, stream, store, branches, proxy_url)
                elif mode == "isolation":
                    run_isolation(client, turns, model, stream, store, proxy_url)
                elif mode == "mixed":
                    run_mixed(client, turns, model, stream, store, proxy_url)
                elif mode == "responses":
                    run_responses(
                        client,
                        turns,
                        model,
                        stream,
                        store,
                        branches,
                        proxy_url,
                        transport,
                        target,
                        headers,
                        output_file,
                        tools,
                        tool_choice,
                        tool_outputs,
                        response_max_output_tokens,
                        reasoning,
                        preset_input,
                        parallel_tool_calls,
                    )
                elif mode == "messages":
                    run_messages(
                        client,
                        turns,
                        model,
                        stream,
                        proxy_url,
                        tools,
                        tool_choice,
                        tool_outputs,
                        response_max_output_tokens or 1024,
                    )
                elif mode == "store_true_then_store_false":
                    run_store_true_then_store_false(client, turns, model, stream, proxy_url)
        finally:
            _stop_proxy(server)

    click.echo(f"\nAll turns recorded -> {output_file}")


if __name__ == "__main__":
    main()
