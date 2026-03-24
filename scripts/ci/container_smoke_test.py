from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path


def _pick_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def main() -> None:
    import agentic_stack.tools.code_interpreter as code_interpreter

    code_dir = Path(code_interpreter.__file__).resolve().parent
    binary = code_dir / "bin" / "linux" / "x86_64" / "code-interpreter-server"
    if not binary.exists():
        raise SystemExit(f"missing bundled binary: {binary}")
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"bundled binary is not executable: {binary}")

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "pyodide"
        downloads_dir = cache_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)

        # Prevent a huge download during smoke tests:
        # create a dummy tarball file so the server skips fetch() and immediately tries to extract.
        dummy_tarball = downloads_dir / "pyodide-0.29.1.tar.bz2"
        dummy_tarball.write_bytes(b"")

        port = _pick_free_port()
        env = dict(os.environ)
        # Simulate "tar is missing" even if present in the image.
        env["PATH"] = "/nonexistent"

        proc = subprocess.Popen(
            [
                str(binary),
                "--port",
                str(port),
                "--pyodide-cache",
                str(cache_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate(timeout=5)
            raise SystemExit(
                "code interpreter did not exit quickly during tar-missing smoke test"
            ) from None

        if proc.returncode == 0:
            raise SystemExit("expected code interpreter to fail (tar missing), but it exited 0")

        out_lower = out.lower()
        if "tar" not in out_lower or "executable is required" not in out_lower:
            raise SystemExit(f"expected actionable tar error. output:\n{out}")

        print("[smoke] ok: missing-tar error message is actionable")


if __name__ == "__main__":
    main()
