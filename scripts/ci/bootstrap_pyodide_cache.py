from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

PYODIDE_VERSION = "0.29.1"
VERSION_MARKER = ".pyodide_version"


def _default_cache_dir() -> Path:
    override = os.environ.get("VR_PYODIDE_CACHE_DIR", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        base = Path(os.path.expanduser(xdg))
    else:
        base = Path.home() / ".cache"
    return base / "agentic-stacks" / "pyodide"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:  # noqa: S310 (trusted URL; CI/dev only)
        dest.write_bytes(r.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap a Pyodide cache directory for agentic-stacks code interpreter tests."
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--version", type=str, default=PYODIDE_VERSION)
    parser.add_argument(
        "--seed-tarball",
        type=str,
        default=os.environ.get("VR_PYODIDE_SEED_TARBALL", "").strip() or None,
        help="Optional local pyodide-<version>.tar.bz2 to use instead of downloading.",
    )
    args = parser.parse_args()

    cache_dir = (
        Path(os.path.expanduser(args.cache_dir)) if args.cache_dir else _default_cache_dir()
    )
    version = args.version
    tarball_name = f"pyodide-{version}.tar.bz2"
    version_path = cache_dir / VERSION_MARKER

    if version_path.exists() and version_path.read_text(encoding="utf-8").strip() == version:
        print(f"[pyodide] cache already initialized: {cache_dir} ({version})")
        return

    downloads_dir = cache_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = downloads_dir / tarball_name

    if args.seed_tarball:
        seed = Path(os.path.expanduser(args.seed_tarball))
        if not seed.exists():
            raise SystemExit(f"seed tarball not found: {seed}")
        tarball_path.write_bytes(seed.read_bytes())
        print(f"[pyodide] seeded tarball: {tarball_path}")
    else:
        url = f"https://github.com/pyodide/pyodide/releases/download/{version}/{tarball_name}"
        print(f"[pyodide] downloading: {url}")
        _download(url, tarball_path)
        print(f"[pyodide] downloaded: {tarball_path}")

    # Extract into cache dir (matching code interpreter behavior: --strip-components=1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, mode="r:bz2") as tf:
        members = tf.getmembers()
        for m in members:
            parts = Path(m.name).parts
            if len(parts) <= 1:
                continue
            m.name = str(Path(*parts[1:]))  # strip top-level directory
            tf.extract(m, path=cache_dir)  # noqa: S202 (trusted tarball; CI/dev only)

    version_path.write_text(f"{version}\n", encoding="utf-8")
    print(f"[pyodide] extracted to: {cache_dir} ({version})")


if __name__ == "__main__":
    main()
