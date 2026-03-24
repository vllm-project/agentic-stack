import type { ExecutionResult, PyodideConfig } from "./types";

interface WorkerPoolConfig {
  workerCount: number;
  pyodideConfig: PyodideConfig;
}

interface WorkerState {
  worker: Worker;
  ready: boolean;
  workerId: number;
}

interface PendingRequest {
  resolve: (result: ExecutionResult) => void;
  reject: (error: Error) => void;
  timeout: Timer;
}

interface WorkerMessage {
  type: "result" | "error" | "ready";
  workerId?: number;
  id?: string;
  result?: ExecutionResult;
  error?: string;
}

export class WorkerPool {
  private workers: WorkerState[] = [];
  private pendingRequests = new Map<string, PendingRequest>();
  private config: WorkerPoolConfig;
  private nextRequestId = 0;
  private executionCount = 0;

  constructor(config: WorkerPoolConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log(`Initializing ${this.config.workerCount} workers...`);

    // Create all workers in parallel
    const workerPromises: Promise<void>[] = [];

    for (let i = 0; i < this.config.workerCount; i++) {
      const workerPromise = this.createWorker(i);
      workerPromises.push(workerPromise);
    }

    // Wait for all workers to be ready
    await Promise.all(workerPromises);

    console.log(`All ${this.config.workerCount} workers ready`);
  }

  private createWorker(workerId: number): Promise<void> {
    return new Promise((resolve, reject) => {
      // Create worker pointing to `worker.ts`.
      //
      // Notes:
      // - In source mode, we start the server with `cwd=<code_interpreter_dir>`, so "./worker.ts" resolves to
      //   `<code_interpreter_dir>/worker.ts`.
      // - In compiled mode (`bun build --compile`), the worker entrypoint must be explicitly bundled into the
      //   executable, and Bun resolves "./worker.ts" from the embedded filesystem.
      const worker = new Worker("./worker.ts");

      const state: WorkerState = {
        worker,
        ready: false,
        workerId,
      };

      this.workers.push(state);

      // Setup error handler
      worker.addEventListener("error", (error) => {
        console.error(`Worker ${workerId} error:`, error);
        reject(new Error(`Worker ${workerId} failed: ${error.message || error}`));
      });

      // Listen for ready message
      const readyListener = (event: MessageEvent) => {
        const message = event.data as WorkerMessage;
        if (message.type === "ready" && message.workerId === workerId) {
          state.ready = true;
          worker.removeEventListener("message", readyListener);
          resolve();
        } else if (message.type === "error") {
          reject(new Error(message.error || "Worker initialization failed"));
        }
      };

      worker.addEventListener("message", readyListener);

      // Setup permanent message handler for execution messages
      worker.addEventListener("message", (event: MessageEvent) => {
        this.handleWorkerMessage(workerId, event);
      });

      // Send initialization message
      worker.postMessage({
        type: "init",
        config: this.config.pyodideConfig,
        workerId,
      });
    });
  }

  private handleWorkerMessage(workerId: number, event: MessageEvent): void {
    const message = event.data as WorkerMessage;

    if (message.type === "result" && message.id) {
      const pending = this.pendingRequests.get(message.id);
      if (pending) {
        clearTimeout(pending.timeout);
        this.pendingRequests.delete(message.id);
        pending.resolve(message.result!);
      }
    } else if (message.type === "error" && message.id) {
      const pending = this.pendingRequests.get(message.id);
      if (pending) {
        clearTimeout(pending.timeout);
        this.pendingRequests.delete(message.id);
        pending.reject(new Error(message.error || "Worker execution failed"));
      }
    }
    // "ready" messages are handled in createWorker
  }

  execute(code: string, resetGlobals: boolean): Promise<ExecutionResult> {
    return new Promise((resolve, reject) => {
      // Generate unique request ID
      const requestId = `req-${this.nextRequestId++}`;

      // Select worker randomly
      const workerIdx = this.selectWorkerRandom();

      if (workerIdx === -1) {
        reject(new Error("No workers available"));
        return;
      }

      // Create timeout (30 seconds default)
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error("Request timeout"));
      }, this.config.pyodideConfig.timeout || 30000);

      // Store pending request
      this.pendingRequests.set(requestId, { resolve, reject, timeout });

      // Send to worker
      this.workers[workerIdx].worker.postMessage({
        type: "execute",
        id: requestId,
        code,
        resetGlobals,
      });

      this.executionCount++;
    });
  }

  private selectWorkerRandom(): number {
    // Random selection (as per requirements)
    const readyWorkers = this.workers.map((state, idx) => ({ idx, state })).filter(({ state }) => state.ready);

    if (readyWorkers.length === 0) {
      return -1;
    }

    const randomIndex = Math.floor(Math.random() * readyWorkers.length);
    return readyWorkers[randomIndex].idx;
  }

  pyodideLoaded(): boolean {
    // Returns true only if ALL workers are ready
    return this.workers.length > 0 && this.workers.every((w) => w.ready);
  }

  getExecutionCount(): number {
    return this.executionCount;
  }

  terminate(): void {
    for (const state of this.workers) {
      state.worker.terminate();
    }
    this.workers = [];

    // Reject all pending requests
    for (const [id, pending] of this.pendingRequests) {
      clearTimeout(pending.timeout);
      pending.reject(new Error("WorkerPool terminated"));
    }
    this.pendingRequests.clear();
  }
}
