import { expect, test } from "bun:test";

test("compiled REPL integration test", async () => {
  const fs = require("fs");
  const path = require("path");

  // Build the compiled binary
  console.log("Building compiled REPL...");
  const buildProc = Bun.spawn(["bun", "run", "build"], {
    cwd: process.cwd(),
    stdout: "inherit",
    stderr: "inherit",
  });
  const buildExitCode = await buildProc.exited;
  expect(buildExitCode).toBe(0);
  console.log("Build completed");

  // Test the compiled binary
  console.log("Testing compiled REPL...");
  const binaryPath = path.join(process.cwd(), "woma");
  expect(fs.existsSync(binaryPath)).toBe(true);

  const replProc = Bun.spawn([binaryPath], {
    cwd: process.cwd(),
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
  });

  let allOutput = "";
  let lastPromptIndex = 0;

  // Start reading output in background
  const reader = replProc.stdout.getReader();
  const decoder = new TextDecoder();

  // Background task to continuously read stdout
  const readTask = (async () => {
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        allOutput += text;
      }
    } catch (e) {
      // Stream closed, that's ok
    }
  })();

  // Helper to wait for a new prompt
  const waitForPrompt = async (timeoutMs: number = 5000): Promise<string> => {
    const startTime = Date.now();
    const startIndex = lastPromptIndex;

    while (Date.now() - startTime < timeoutMs) {
      const newPromptIndex = allOutput.indexOf(">>> ", lastPromptIndex);
      if (newPromptIndex > lastPromptIndex) {
        const output = allOutput.substring(startIndex, newPromptIndex);
        lastPromptIndex = newPromptIndex + 4; // Skip past ">>> "
        return output;
      }
      await Bun.sleep(100);
    }

    // Return what we have so far
    return allOutput.substring(startIndex);
  };

  // Wait for initial prompt
  await waitForPrompt(60 * 1000); // 1 minute timeout
  console.log("REPL started, sending test commands...");

  // Test 1: 1+1
  replProc.stdin.write("1+1\n");
  await Bun.sleep(100); // Give it a moment to process
  const output1 = await waitForPrompt();
  console.log("Test 1+1 output:", output1);
  expect(output1).toContain("2");

  // Test 2: httpx.get
  replProc.stdin.write(
    'httpx.get("https://raw.githubusercontent.com/EmbeddedLLM/JamAIBase/refs/heads/main/services/api/tests/files/txt/weather.txt").text\n',
  );
  await Bun.sleep(100); // Give it a moment to process
  const output2 = await waitForPrompt(30000); // Longer timeout for HTTP request
  console.log("Test httpx output:", output2);
  expect(output2).toContain("Temperature in Kuala Lumpur is 27 degrees celsius");

  // Cleanup: close the process
  replProc.kill();
  await readTask.catch(() => {}); // Wait for read task to finish

  console.log("Compiled REPL integration test passed!");
}, 180000); // 3 minute timeout for full integration test
