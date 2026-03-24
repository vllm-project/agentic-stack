import type { Subprocess } from "bun";
import { afterAll, expect, test } from "bun:test";
import type { ExecutionResult, HealthResponse } from "./types";

const TEST_PORT = 8765;
const SERVER_URL = `http://localhost:${TEST_PORT}`;
let serverProc: Subprocess | null = null;

// Helper function to poll health endpoint until Pyodide is loaded
async function waitForServerReady(timeoutMs = 30000): Promise<void> {
  const startTime = Date.now();
  const pollInterval = 1000; // Check every second

  while (Date.now() - startTime < timeoutMs) {
    try {
      const res = await fetch(`${SERVER_URL}/health`);
      if (res.ok) {
        const health = (await res.json()) as HealthResponse;
        if (health.pyodide_loaded) {
          console.log("Server is ready!");
          return;
        }
      }
    } catch (error) {
      // Server not ready yet, continue polling
    }

    await Bun.sleep(pollInterval);
  }

  throw new Error("Server did not become ready within timeout period");
}

test("server integration tests", async () => {
  // Start server in background
  console.log("Starting server...");
  serverProc = Bun.spawn(["bun", "src/index.ts", "--port", String(TEST_PORT)], {
    stdout: "inherit",
    stderr: "inherit",
  });

  try {
    // Wait for server to be ready (poll health endpoint)
    await waitForServerReady(30000);

    // Test 1: Health endpoint
    console.log("\nTest 1: Health endpoint");
    const healthRes = await fetch(`${SERVER_URL}/health`);
    expect(healthRes.status).toBe(200);
    const health = (await healthRes.json()) as HealthResponse;
    expect(health.status).toBe("healthy");
    expect(health.pyodide_loaded).toBe(true);
    expect(health.uptime_seconds).toBeGreaterThanOrEqual(0);
    expect(health.execution_count).toBe(0);
    console.log("✓ Health endpoint test passed");

    // Test 2: Execute simple Python code
    console.log("\nTest 2: Execute simple Python code (1+1)");
    const execRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: "x = 1 + 1; x" }),
    });
    expect(execRes.status).toBe(200);
    const result = (await execRes.json()) as ExecutionResult;
    expect(result.status).toBe("success");
    expect(result.stdout).toBe("");
    expect(result.stderr).toBe("");
    expect(result.result).toBe("2");
    expect(result.execution_time_ms).toBeGreaterThan(0);
    console.log("✓ Simple execution test passed");

    // Test 2b: Capture stdout from print(...) and return final expression separately
    console.log("\nTest 2b: Capture stdout + final expression");
    const printRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: 'print("P1"); print("P2"); 2+2' }),
    });
    expect(printRes.status).toBe(200);
    const printResult = (await printRes.json()) as ExecutionResult;
    expect(printResult.status).toBe("success");
    expect(printResult.stdout).toBe("P1\nP2\n");
    expect(printResult.stderr).toBe("");
    expect(printResult.result).toBe("4");
    console.log("✓ stdout capture test passed");

    // Test 3: Execute code with error (treated as successful execution)
    console.log("\nTest 3: Execute code with error");
    const errorRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: "raise ValueError('test error')" }),
    });
    expect(errorRes.status).toBe(200);
    const error = (await errorRes.json()) as ExecutionResult;
    expect(error.status).toBe("exception");
    expect(typeof error.stdout).toBe("string");
    expect(typeof error.stderr).toBe("string");
    expect(error.result).not.toBe(null);
    expect(error.result!).toContain("ValueError");
    expect(error.result!).toContain("test error");
    console.log("✓ Error handling test passed");

    // Test 4: Execute code with exit()
    console.log("\nTest 4: Execute code with exit()");
    const exitRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: "exit()" }),
    });
    expect(exitRes.status).toBe(200);
    const exitResult = (await exitRes.json()) as ExecutionResult;
    expect(exitResult.status).toBe("exception");
    expect(exitResult.result).not.toBe(null);
    expect(exitResult.result!).toContain("SystemExit");
    console.log("✓ exit() handling test passed");

    // Test 5: Invalid request body
    console.log("\nTest 5: Invalid request body");
    const invalidRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ invalid: "field" }),
    });
    expect(invalidRes.status).toBe(400);
    const invalidError = (await invalidRes.json()) as {
      status: string;
      error: string;
    };
    expect(invalidError.status).toBe("error");
    expect(invalidError.error).toContain("code");
    console.log("✓ Invalid request test passed");

    // Test 6: Test reset_globals parameter
    console.log("\nTest 6: Test reset_globals parameter");

    // Set a variable (assignment returns null)
    const setVarRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: "x = 42" }),
    });
    expect(setVarRes.status).toBe(200);
    const setResult = (await setVarRes.json()) as ExecutionResult;
    expect(setResult.result).toBe(null);

    // Access it (should work with persistent globals)
    const accessVarRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: "x" }),
    });
    expect(accessVarRes.status).toBe(200);
    const accessResult = (await accessVarRes.json()) as ExecutionResult;
    expect(accessResult.status).toBe("success");
    expect(accessResult.result).toBe("42");

    // Try with reset_globals (should return NameError as output)
    const resetRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: "x", reset_globals: true }),
    });
    expect(resetRes.status).toBe(200);
    const resetError = (await resetRes.json()) as ExecutionResult;
    expect(resetError.status).toBe("success");
    expect(resetError.result).not.toBe(null);
    expect(resetError.result!).toContain("NameError");
    console.log("✓ reset_globals test passed");

    // Test 7: Test multiline code
    console.log("\nTest 7: Test multiline code");
    const multilineRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code: `
              def x():
                  return 300

              x()
              `,
      }),
    });
    expect(multilineRes.status).toBe(200);
    const multilineResult = (await multilineRes.json()) as ExecutionResult;
    expect(multilineResult.status).toBe("success");
    expect(multilineResult.result).toBe("300");
    console.log("✓ Multiline code test passed");

    // Test 8: Test requests library
    console.log("\nTest 8: Test requests library");
    const requestsRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code: "import requests; response = requests.get('https://httpbin.org/json'); response.status_code",
      }),
    });
    expect(requestsRes.status).toBe(200);
    const requestsResult = (await requestsRes.json()) as ExecutionResult;
    expect(requestsResult.status).toBe("success");
    expect(requestsResult.result).toBe("200");
    console.log("✓ requests library test passed");

    // Test 9: Test httpx library
    console.log("\nTest 9: Test httpx library");
    const httpxRes = await fetch(`${SERVER_URL}/python`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code: "import httpx; response = httpx.get('https://httpbin.org/json'); response.status_code",
      }),
    });
    expect(httpxRes.status).toBe(200);
    const httpxResult = (await httpxRes.json()) as ExecutionResult;
    expect(httpxResult.status).toBe("success");
    expect(httpxResult.result).toBe("200");
    console.log("✓ httpx library test passed");

    // Verify execution count increased
    const finalHealthRes = await fetch(`${SERVER_URL}/health`);
    const finalHealth = (await finalHealthRes.json()) as HealthResponse;
    expect(finalHealth.execution_count).toBeGreaterThan(0);
    console.log(`\n✓ All tests passed! Total executions: ${finalHealth.execution_count}`);
  } finally {
    // Kill server process
    if (serverProc) {
      console.log("\nCleaning up server process...");
      serverProc.kill();
    }
  }
}, 120000); // 2 minute timeout for entire test suite

afterAll(() => {
  if (serverProc) {
    serverProc.kill();
  }
});
