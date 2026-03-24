// Standalone worker entrypoint for Bun `--compile` executables.
//
// In source mode, `WorkerPool` loads `./src/worker.ts` directly from disk.
// In compiled mode, the worker script must be explicitly included as an entrypoint
// so Bun can bundle it into the executable's embedded filesystem.
import "./src/worker";
