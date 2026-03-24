import readline from "readline";

import { PyodideManager } from "./pyodide-manager";

interface REPLConfig {
  resetGlobals: boolean;
  pyodideCache: string;
}

export async function startREPL(config: REPLConfig) {
  try {
    // Initialize PyodideManager
    const manager = new PyodideManager({
      pyodideCache: config.pyodideCache,
      verbose: true,
    });
    await manager.initialize();

    console.log("\nInteractive Mode: Enter Python code to evaluate. (Ctrl+C to exit)");

    // Set up the Readline interface
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: ">>> ",
    });

    rl.prompt();

    // Handle input line-by-line
    rl.on("line", async (line) => {
      const code = line.trim();

      if (code) {
        try {
          const result = await manager.execute(code, config.resetGlobals);

          // Print result (includes both successful outputs and errors)
          if (result.result !== null) {
            console.log(result.result);
          }
        } catch (error: any) {
          console.error(error);
        }
      }

      rl.prompt();
    });

    // Handle process exit
    rl.on("close", () => {
      console.log("\nExiting...");
      process.exit(0);
    });
  } catch (error) {
    console.error("Fatal Initialization Error:");
    console.error(error);
    process.exit(1);
  }
}
