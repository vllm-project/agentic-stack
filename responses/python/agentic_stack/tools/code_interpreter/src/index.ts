import { homedir } from "os";
import { join } from "path";

import { startREPL } from "./repl";
import { startServer } from "./server";

// Parse CLI arguments
function parseArgs() {
  // `bun src/index.ts ...` => Bun.argv == ["bun", "src/index.ts", ...]
  // compiled binary => Bun.argv == ["<exe>", ...]
  const startIndex = Bun.argv.length >= 2 && Bun.argv[1]?.endsWith(".ts") ? 2 : 1;
  const args = Bun.argv.slice(startIndex);
  let resetGlobals = false;
  let pyodideCache = join(homedir(), ".pyodide-env");
  let port: number | null = null;
  let workerCount = 0; // Default 0 = single-threaded

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--reset-globals") {
      resetGlobals = true;
    } else if (args[i] === "--pyodide-cache") {
      i++;
      const cachePath = args[i];
      if (cachePath) {
        // Expand ~ to home directory
        if (cachePath.startsWith("~/")) {
          pyodideCache = join(homedir(), cachePath.slice(2));
        } else if (cachePath === "~") {
          pyodideCache = homedir();
        } else {
          pyodideCache = cachePath;
        }
      }
    } else if (args[i] === "--port") {
      i++;
      const portStr = args[i];
      if (portStr) {
        port = parseInt(portStr, 10);
        if (isNaN(port) || port < 1 || port > 65535) {
          console.error("Error: Invalid port number. Must be between 1 and 65535.");
          process.exit(1);
        }
      } else {
        console.error("Error: --port flag requires a port number.");
        process.exit(1);
      }
    } else if (args[i] === "--workers") {
      i++;
      const workerStr = args[i];
      if (workerStr) {
        workerCount = parseInt(workerStr, 10);
        if (isNaN(workerCount) || workerCount < 0) {
          console.error("Error: Invalid worker count. Must be a non-negative integer.");
          process.exit(1);
        }
      } else {
        console.error("Error: --workers flag requires a number.");
        process.exit(1);
      }
    }
  }

  return { resetGlobals, pyodideCache, port, workerCount };
}

const { resetGlobals, pyodideCache, port, workerCount } = parseArgs();

async function main() {
  if (port !== null) {
    // Start server mode
    await startServer({ port, resetGlobals, pyodideCache, workerCount });
  } else {
    if (workerCount > 0) {
      console.error("Error: --workers flag only supported in server mode (use --port).");
      process.exit(1);
    }
    // Start REPL mode
    await startREPL({ resetGlobals, pyodideCache });
  }
}

main();
