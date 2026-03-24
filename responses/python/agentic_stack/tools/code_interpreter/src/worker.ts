declare var self: Worker;

import { PyodideManager } from "./pyodide-manager";
import type { ExecutionResult, PyodideConfig } from "./types";

// Worker-specific message types
interface WorkerRequest {
  type: "execute";
  id: string;
  code: string;
  resetGlobals: boolean;
}

interface WorkerResponse {
  type: "result" | "error" | "ready";
  workerId?: number;
  id?: string;
  result?: ExecutionResult;
  error?: string;
}

interface WorkerInitMessage {
  type: "init";
  config: PyodideConfig;
  workerId: number;
}

let pyodideManager: PyodideManager | null = null;
let workerId: number = -1;

// CRITICAL: Add message listener to keep worker's event loop alive
self.addEventListener("message", async (event: MessageEvent) => {
  const message = event.data;

  if (message.type === "init") {
    await handleInit(message as WorkerInitMessage);
  } else if (message.type === "execute") {
    await handleExecute(message as WorkerRequest);
  }
});

async function handleInit(message: WorkerInitMessage): Promise<void> {
  try {
    workerId = message.workerId;
    console.log(`Worker ${workerId}: Initializing Pyodide...`);

    pyodideManager = new PyodideManager(message.config);
    await pyodideManager.initialize();

    console.log(`Worker ${workerId}: Pyodide ready`);

    // Notify main thread that worker is ready
    const response: WorkerResponse = {
      type: "ready",
      workerId,
    };
    self.postMessage(response);
  } catch (error: any) {
    console.error(`Worker ${workerId}: Initialization failed:`, error);
    const response: WorkerResponse = {
      type: "error",
      error: error.message || "Worker initialization failed",
    };
    self.postMessage(response);
  }
}

async function handleExecute(message: WorkerRequest): Promise<void> {
  try {
    if (!pyodideManager) {
      throw new Error("PyodideManager not initialized");
    }

    // Execute using existing PyodideManager logic (includes semaphore)
    const result = await pyodideManager.execute(message.code, message.resetGlobals);

    // Send result back to main thread
    const response: WorkerResponse = {
      type: "result",
      id: message.id,
      result,
    };
    self.postMessage(response);
  } catch (error: any) {
    // Send error back to main thread
    const response: WorkerResponse = {
      type: "error",
      id: message.id,
      error: error.message || "Execution failed",
    };
    self.postMessage(response);
  }
}
