import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { loadPyodide, type PyodideInterface } from "pyodide";
import { XMLHttpRequest as OriginalXMLHttpRequest } from "xmlhttprequest-ssl";
import type { ExecutionResult, PyodideConfig } from "./types";

// Keep this pinned to the exact `pyodide` NPM dependency version.
// Mismatches will cause `loadPyodide()` to throw:
//   "Pyodide version does not match: 'X' <==> 'Y'".
const PYODIDE_VERSION = "0.29.1";
const TARBALL_NAME = `pyodide-${PYODIDE_VERSION}.tar.bz2`;
const VERSION_MARKER = ".pyodide_version";

function isTarMissingError(error: unknown): boolean {
  const e = error as {
    message?: unknown;
    stderr?: unknown;
    stdout?: unknown;
  };
  const combined = [e?.message, e?.stderr, e?.stdout].map((v) => String(v ?? "").toLowerCase()).join("\n");
  return (
    combined.includes("enoent") || combined.includes("spawn tar") || combined.includes("tar: not found") || combined.includes("command not found")
  );
}

// XMLHttpRequest polyfill setup
const forbiddenRequestHeaders = [
  "accept-charset",
  "accept-encoding",
  "access-control-request-headers",
  "access-control-request-method",
  "connection",
  "content-length",
  "content-transfer-encoding",
  "cookie",
  "cookie2",
  "date",
  "expect",
  "host",
  "keep-alive",
  "origin",
  "referer",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "via",
];

class WrappedXMLHttpRequest extends OriginalXMLHttpRequest {
  constructor(options?: any) {
    super({ ...options, syncPolicy: "enabled" });

    // Patch setRequestHeader on the instance after construction
    const originalSetRequestHeader = this.setRequestHeader;
    this.setRequestHeader = function (header: string, value: string) {
      const normalizedHeader = header.toLowerCase();

      // Skip forbidden headers silently
      if (forbiddenRequestHeaders.includes(normalizedHeader) || normalizedHeader.startsWith("proxy-") || normalizedHeader.startsWith("sec-")) {
        return true;
      }

      return originalSetRequestHeader.call(this, header, value);
    };
  }
}

// Install XMLHttpRequest polyfill globally
(globalThis as any).XMLHttpRequest = WrappedXMLHttpRequest;

export class PyodideManager {
  private pyodide: PyodideInterface | null = null;
  private executionCount = 0;
  private executionLock = false;
  private config: PyodideConfig;
  private activeStdout: string[] | null = null;
  private activeStderr: string[] | null = null;

  constructor(config: PyodideConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    // 1. Setup Pyodide environment
    await this.setupPyodide();

    // 2. Load Pyodide WASM
    if (this.config.verbose) {
      console.log("Loading Pyodide...");
    }
    this.pyodide = await loadPyodide({
      indexURL: this.config.pyodideCache,
      stdout: (msg: unknown) => {
        if (this.activeStdout) {
          const s = String(msg);
          if (!s) {
            return;
          }
          // Pyodide's stdout callback is line-oriented and often omits the trailing newline even when Python
          // `print()` writes one. Preserve "human" formatting by restoring newlines between callback messages.
          this.activeStdout.push(s.endsWith("\n") ? s : `${s}\n`);
        }
      },
      stderr: (msg: unknown) => {
        if (this.activeStderr) {
          const s = String(msg);
          if (!s) {
            return;
          }
          this.activeStderr.push(s.endsWith("\n") ? s : `${s}\n`);
        }
      },
    });

    // 3. Load pre-installed packages
    await this.pyodide.loadPackage([
      "aiohttp",
      "audioop-lts",
      "beautifulsoup4",
      "httpx",
      "matplotlib",
      "numpy",
      "opencv-python",
      "orjson",
      "pandas",
      "Pillow",
      "pyodide-http",
      "pyyaml",
      "regex",
      "requests",
      "ruamel.yaml",
      "scikit-image",
      "simplejson",
      "soundfile",
      "sympy",
      "tiktoken",
    ]);

    // 4. Apply patches for HTTP libraries
    this.pyodide.runPython(`
# Specify matplotlib backend
import matplotlib
matplotlib.use("AGG")

# Patch HTTP libraries
import pyodide_http
pyodide_http.patch_all()

# Patch urllib3 Node.JS detection
import js
js.process.release.name = ""

# Patch httpx
import httpx
from httpx._transports import jsfetch
from httpx._transports.jsfetch import _no_jspi_fallback


def _no_jspi_fallback_patched(request):
    request.url = str(request.url)
    return _no_jspi_fallback(request)


jsfetch._no_jspi_fallback = _no_jspi_fallback_patched

# Patch _pyodide
import _pyodide

_pyodide._base.eval_code = lambda code: code

# Patch ctypes
import ctypes
ctypes.CDLL = None
    `);

    if (this.config.verbose) {
      console.log("Pyodide ready");
    }
  }

  async execute(code: string, resetGlobals: boolean): Promise<ExecutionResult> {
    // Acquire lock (simple semaphore pattern)
    while (this.executionLock) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    this.executionLock = true;

    let context = null;
    let status: "success" | "exception" | "error" = "success";
    const startTime = Date.now();
    const stdoutChunks: string[] = [];
    const stderrChunks: string[] = [];

    try {
      if (!this.pyodide) {
        throw new Error("Pyodide not initialized");
      }

      // Create context if reset_globals is enabled
      if (resetGlobals) {
        context = this.pyodide.toPy({});
      }

      // Execute code - treat any output (including errors) as successful execution
      let result: any;
      try {
        this.activeStdout = stdoutChunks;
        this.activeStderr = stderrChunks;

        result = resetGlobals ? await this.pyodide.runPythonAsync(code, { globals: context }) : await this.pyodide.runPythonAsync(code);
      } catch (error: any) {
        // Python error - treat as successful execution, return error message
        // Note: we still want to preserve any stdout/stderr produced before the exception.
        result = this.pyodide.runPython("import sys;repr(sys.last_exc)");
        status = "exception";
      } finally {
        this.activeStdout = null;
        this.activeStderr = null;
      }

      this.executionCount++;

      let resultStr: string | null = null;
      if (result !== undefined) {
        resultStr = result === null ? null : result.toString();
      }

      // Clean up PyProxy
      if (result && typeof result.destroy === "function") {
        result.destroy();
      }
      // Clean up context if needed
      if (context) {
        context.destroy();
        context = null;
      }

      return {
        status,
        stdout: stdoutChunks.join(""),
        stderr: stderrChunks.join(""),
        result: resultStr,
        execution_time_ms: Date.now() - startTime,
      };
    } finally {
      // Always clean up context
      if (context) {
        context.destroy();
      }
      // Release lock
      this.executionLock = false;
    }
  }

  getExecutionCount(): number {
    return this.executionCount;
  }

  private async setupPyodide(): Promise<void> {
    const PYODIDE_ENV_DIR = this.config.pyodideCache;
    const downloadsDir = join(PYODIDE_ENV_DIR, "downloads");
    const tarballPath = join(downloadsDir, TARBALL_NAME);

    // Check if pyodide-env directory already exists
    if (existsSync(PYODIDE_ENV_DIR)) {
      const versionPath = `${PYODIDE_ENV_DIR}/${VERSION_MARKER}`;
      try {
        const installedVersion = readFileSync(versionPath, "utf8").trim();
        if (installedVersion === PYODIDE_VERSION) {
          return;
        }
        console.log(`Pyodide cache version mismatch: '${installedVersion}' != '${PYODIDE_VERSION}', rebuilding cache...`);
      } catch {
        console.log(`Pyodide cache found without version marker, rebuilding cache for '${PYODIDE_VERSION}'...`);
      }
      rmSync(PYODIDE_ENV_DIR, { recursive: true, force: true });
    }

    if (this.config.verbose) {
      console.log(`Pyodide environment not found. Setting up Pyodide ${PYODIDE_VERSION}...`);
    }

    // Fail early with an actionable error instead of downloading hundreds of MB first.
    try {
      await Bun.$`tar --version`.quiet();
    } catch (e: any) {
      if (isTarMissingError(e)) {
        throw new Error(
          "The `tar` executable is required in v1.\n" + "Install it via your system package manager (e.g. `apt-get install -y tar`) and retry.",
        );
      }
      throw e;
    }

    const downloadUrl = `https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/${TARBALL_NAME}`;

    // Ensure cache directories exist before downloading (must never write into CWD).
    mkdirSync(downloadsDir, { recursive: true });

    // Download tarball if not already present
    if (!existsSync(tarballPath)) {
      if (this.config.verbose) {
        console.log(`Downloading ${TARBALL_NAME}...`);
      }
      const response = await fetch(downloadUrl);

      if (!response.ok) {
        throw new Error(`Failed to download Pyodide: ${response.status} ${response.statusText}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      await Bun.write(tarballPath, arrayBuffer);
      if (this.config.verbose) {
        console.log("Download complete.");
      }
    } else if (this.config.verbose) {
      console.log(`${TARBALL_NAME} already exists, skipping download.`);
    }

    // Create pyodide-env directory
    mkdirSync(PYODIDE_ENV_DIR, { recursive: true });

    // Extract tarball using tar command
    if (this.config.verbose) {
      console.log(`Extracting ${TARBALL_NAME}...`);
    }
    try {
      await Bun.$`tar -xjf ${tarballPath} -C ${PYODIDE_ENV_DIR} --strip-components=1`;
    } catch (e: any) {
      // Only special-case the "tar executable missing" scenario. Other extraction failures
      // (corrupt tarball, permission errors, etc.) should surface their real error.
      if (isTarMissingError(e)) {
        throw new Error(
          "Failed to extract Pyodide tarball. The `tar` executable is required in v1.\n" +
            "Install it via your system package manager (e.g. `apt-get install -y tar`) and retry.",
        );
      }
      throw e;
    }
    writeFileSync(`${PYODIDE_ENV_DIR}/${VERSION_MARKER}`, `${PYODIDE_VERSION}\n`, "utf8");
    if (this.config.verbose) {
      console.log("Extraction complete.");
    }
  }
}
