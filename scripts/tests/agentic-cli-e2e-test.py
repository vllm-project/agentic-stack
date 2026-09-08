#!/usr/bin/env python3
"""Exercise `agentic run` with local vLLM-shaped and harness-shaped processes."""

from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.request


class MockVllm(http.server.ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), MockVllmHandler)
        self.responses: list[dict] = []
        self.messages: list[dict] = []


class MockVllmHandler(http.server.BaseHTTPRequestHandler):
    server: MockVllm

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/v1/models":
            self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "test-model",
                            "object": "model",
                            "owned_by": "mock-vllm",
                        }
                    ],
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path == "/v1/responses":
            self.server.responses.append(body)
            number = len(self.server.responses)
            response = {
                "id": f"resp_vllm_{number}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body.get("model", "test-model"),
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{number}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "OK"}],
                    }
                ],
                "previous_response_id": body.get("previous_response_id"),
                "store": True,
            }
            self._json(response)
            return
        if self.path == "/v1/messages":
            self.server.messages.append(body)
            self._json(
                {
                    "id": "msg_vllm_1",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", "test-model"),
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
            return
        self.send_error(404)

    def _json(self, value: dict) -> None:
        encoded = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


HARNESS = r'''#!/usr/bin/env python3
import json
import os
import re
import sys
import urllib.request

if "--version" in sys.argv:
    # The Codex launcher asks the harness binary for the client version it reports to the gateway.
    print("codex-cli 0.0.0-e2e")
    raise SystemExit(0)

def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected status {response.status}")
        return json.loads(response.read())

mode = os.environ["E2E_HARNESS_MODE"]
result = os.environ["E2E_RESULT"]
model = "test-model"
if mode == "codex":
    config = open(os.path.join(os.environ["CODEX_HOME"], "config.toml")).read()
    base = re.search(r'base_url = "([^"]+)"', config).group(1)
    catalog = json.load(open(os.path.join(os.environ["CODEX_HOME"], "model_catalog.json")))
    entry = catalog["models"][0]
    assert entry["slug"] == model, entry
    # The mock upstream advertises no capabilities, so the gateway resolves text-only.
    assert entry["input_modalities"] == ["text"], entry
    first = post(base + "/responses", {"model": model, "input": "Remember APPLE", "store": True, "stream": False})
    second = post(base + "/responses", {"model": model, "input": "What word?", "previous_response_id": first["id"], "store": True, "stream": False})
    assert second["id"], second
else:
    base = os.environ["ANTHROPIC_BASE_URL"]
    response = post(base + "/v1/messages", {"model": model, "max_tokens": 32, "stream": False, "messages": [{"role": "user", "content": "Remember APPLE"}, {"role": "assistant", "content": "OK"}, {"role": "user", "content": "What word?"}]})
    assert response["content"][0]["text"] == "OK", response
open(result, "w").write(mode + " passed\n")
'''


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_harness(agentic: Path, mode: str, upstream: str, temp: Path, harness: Path) -> None:
    gateway_port = free_port()
    database = temp / f"{mode}.db"
    result = temp / f"{mode}.result"
    environment = os.environ.copy()
    environment.update(
        {
            "AGENTIC_CODEX_BIN": str(harness),
            "AGENTIC_CLAUDE_BIN": str(harness),
            "E2E_HARNESS_MODE": mode,
            "E2E_RESULT": str(result),
        }
    )
    command = [
        str(agentic),
        "run",
        mode,
        "--upstream",
        upstream,
        "--gateway-host",
        "127.0.0.1",
        "--gateway-port",
        str(gateway_port),
        "--database-url",
        f"sqlite://{database}",
        "--skip-llm-ready-check",
        "--",
        "e2e",
    ]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise AssertionError(f"{mode} CLI failed ({completed.returncode})\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    assert result.read_text() == f"{mode} passed\n"


def run_python_source_install(repo: Path, temp: Path, expected_version: str) -> None:
    """Install the Python package from this checkout and exercise its public CLI."""

    environment = temp / "python-source-environment"
    completed = subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(f"creating Python E2E environment failed\n{completed.stdout}\n{completed.stderr}")

    python = environment / "bin" / "python"
    cli = environment / "bin" / "agentic-api"
    install_package = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--constraint",
            str(repo / "python-build-constraints.txt"),
            str(repo),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if install_package.returncode != 0:
        raise AssertionError(
            f"installing agentic-api from source failed\n{install_package.stdout}\n{install_package.stderr}"
        )

    import_check = subprocess.run(
        [str(python), "-c", f"import agentic_api; assert agentic_api.__version__ == {expected_version!r}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if import_check.returncode != 0:
        raise AssertionError(
            f"source-installed agentic_api import failed\n{import_check.stdout}\n{import_check.stderr}"
        )

    version_check = subprocess.run(
        [str(cli), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if version_check.returncode != 0 or version_check.stdout.strip() != f"agentic-api {expected_version}":
        raise AssertionError(
            f"source-installed agentic-api --version failed\n{version_check.stdout}\n{version_check.stderr}"
        )

    doctor_check = subprocess.run(
        [str(cli), "doctor", "--mode", "remote"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if doctor_check.returncode != 0 or "Remote mode health: ok" not in doctor_check.stdout:
        raise AssertionError(
            f"source-installed agentic-api doctor failed\n{doctor_check.stdout}\n{doctor_check.stderr}"
        )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    agentic = Path(os.environ.get("AGENTIC_BIN", repo / "target/debug/agentic"))
    if not agentic.is_file():
        raise SystemExit(f"missing {agentic}; run cargo build --bins first")
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--manifest-path", str(repo / "Cargo.toml")],
        capture_output=True,
        text=True,
        check=True,
    )
    expected_version = next(
        package["version"]
        for package in json.loads(metadata.stdout)["packages"]
        if package["name"] == "agentic-server"
    )
    with tempfile.TemporaryDirectory(prefix="agentic-cli-e2e-") as directory:
        temp = Path(directory)
        run_python_source_install(repo, temp, expected_version)
        harness = temp / "fake-harness"
        harness.write_text(HARNESS)
        harness.chmod(0o755)
        server = MockVllm()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            upstream = f"http://127.0.0.1:{server.server_address[1]}"
            run_harness(agentic, "codex", upstream, temp, harness)
            run_harness(agentic, "claude", upstream, temp, harness)
            assert len(server.responses) == 2, server.responses
            assert len(server.messages) == 1, server.messages
            assert len(server.messages[0]["messages"]) == 3, server.messages[0]
            print("agentic CLI end-to-end tests passed")
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
