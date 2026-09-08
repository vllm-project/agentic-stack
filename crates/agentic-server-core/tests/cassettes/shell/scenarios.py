"""Local-shell recording inputs and validation; command outputs are simulated.

No model-generated command is executed. Only the fixed commands below have
client output fixtures. API requests/responses are captured by record_cassette.py.
Wire format: https://developers.openai.com/api/docs/guides/tools-shell#local-shell-mode
"""

import copy
import json
import sys
from pathlib import Path

import yaml


def exited(stdout: str = "", stderr: str = "", code: int = 0) -> dict:
    return {"stdout": stdout, "stderr": stderr, "outcome": {"type": "exit", "exit_code": code}}


COMMAND_OUTPUTS = {
    "printf 'SHELL_OK\\n'": exited(stdout="SHELL_OK\n"),
    "printf 'SHELL_ERROR\\n' >&2; exit 7": exited(stderr="SHELL_ERROR\n", code=7),
    "printf 'SHELL_ERROR\\n' >&2": exited(stderr="SHELL_ERROR\n"),
    "exit 7": exited(code=7),
    "sleep 2": {"stdout": "", "stderr": "", "outcome": {"type": "timeout"}},
}
SCENARIOS = {
    "success": ["printf 'SHELL_OK\\n'"],
    "nonzero-exit": ["printf 'SHELL_ERROR\\n' >&2; exit 7"],
    "timeout": ["sleep 2"],
    "multiple-commands": ["printf 'SHELL_OK\\n'", "printf 'SHELL_ERROR\\n' >&2; exit 7", "sleep 2"],
}
FOLLOW_UP = (
    "Use the shell output above without calling any more tools. For each command, "
    "report its stdout, stderr, and exit code or timeout outcome in order."
)


def first_prompt(scenario: str) -> str:
    return (
        "The local shell environment is Linux bash. Use the shell tool exactly once "
        "with this exact action, preserving the commands array and its order: "
        + json.dumps({"commands": SCENARIOS[scenario], "timeout_ms": 1000, "max_output_length": 4096})
        + ". Keep semicolon-separated statements in the same command string. "
        "Wait for the client to return shell_call_output before interpreting the results."
    )


def shell(action: dict) -> dict:
    """--tool-outputs callback: return structured fixtures for the actual commands."""
    commands = action.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("Expected a nonempty shell commands array")
    for command in commands:
        if command not in COMMAND_OUTPUTS:
            raise ValueError(f"No simulated shell output for command: {command!r}")
    result = {"output": [copy.deepcopy(COMMAND_OUTPUTS[command]) for command in commands]}
    if action.get("max_output_length") is not None:
        result["max_output_length"] = action["max_output_length"]
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def completed_response(turn: dict, streaming: bool) -> dict:
    response = turn["response"]
    require(response.get("status_code") == 200, f"HTTP recording failed: {response.get('status_code')}")
    if streaming:
        events = [
            json.loads(line[5:].strip())
            for raw in response.get("sse", [])
            for line in raw.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        require(
            not any(event.get("type") in {"error", "response.failed", "response.incomplete"} for event in events),
            "Recording contains a failed or incomplete streaming event",
        )
        completed = [event["response"] for event in events if event.get("type") == "response.completed"]
        require(len(completed) == 1, "Expected one response.completed event")
        body = completed[0]
        calls = [item for item in body.get("output", []) if item.get("type") == "shell_call"]
        for call in calls:
            for event_type in ("response.output_item.added", "response.output_item.done"):
                require(
                    any(
                        event.get("type") == event_type
                        and event.get("item", {}).get("type") == "shell_call"
                        and event["item"].get("id") == call.get("id")
                        for event in events
                    ),
                    f"Missing {event_type} for shell_call",
                )
            added = next(event for event in events if event.get("type") == "response.output_item.added"
                         and event.get("item", {}).get("id") == call["id"])
            require(added["item"]["action"]["commands"] == [], "Shell item must start with empty commands")
            index = added["output_index"]
            commands = []
            done = []
            for event in events:
                if event.get("output_index") != index:
                    continue
                kind = event.get("type", "")
                if not kind.startswith("response.shell_call_command."):
                    continue
                command_index = event.get("command_index")
                require(type(command_index) is int and command_index >= 0, "Invalid command_index")
                if kind == "response.shell_call_command.added":
                    require(command_index == len(commands), "Command added out of order")
                    commands.append(event["command"])
                    done.append(False)
                else:
                    require(command_index < len(commands) and not done[command_index], "No active shell command")
                    if kind == "response.shell_call_command.delta":
                        commands[command_index] += event["delta"]
                    elif kind == "response.shell_call_command.done":
                        require(event["command"] == commands[command_index], "Command done differs from deltas")
                        done[command_index] = True
            require(commands == call["action"]["commands"] and all(done), "Missing or incomplete shell command events")
    else:
        body = response.get("body", {})
    require(body.get("status") == "completed", "Response did not complete")
    require(bool(body.get("id")), "Response has no ID")
    return body


def validate(document: dict, scenario: str, streaming: bool) -> None:
    turns = document.get("turns", [])
    require(len(turns) == 2, "Expected exactly two recorded requests")
    requests = [turn["request"]["body"] for turn in turns]
    for request in requests:
        require(request.get("stream") is streaming, "Wrong stream mode")
        require(request.get("store") is True, "Stored responses are required for continuation")
        require(
            request.get("tools") == [{"type": "shell", "environment": {"type": "local"}}],
            "Expected the documented local shell tool declaration",
        )
    responses = [completed_response(turn, streaming) for turn in turns]
    require(requests[0].get("input") == first_prompt(scenario), "Wrong first prompt")
    require(requests[1].get("previous_response_id") == responses[0]["id"], "Broken previous_response_id chain")
    calls = [item for item in responses[0].get("output", []) if item.get("type") == "shell_call"]
    require(len(calls) == 1, "Expected exactly one shell_call in the first response")
    call = calls[0]
    require(bool(call.get("call_id")), "Missing shell call_id")
    # Some models split the stderr/exit script into two commands. Preserve that
    # actual action and supply one accurate fixture per command, including exit 0
    # for printf and exit 7 for the separate exit command.
    expected_commands = SCENARIOS[scenario]
    split_commands = [
        part
        for command in expected_commands
        for part in (
            ["printf 'SHELL_ERROR\\n' >&2", "exit 7"]
            if command == "printf 'SHELL_ERROR\\n' >&2; exit 7"
            else [command]
        )
    ]
    require(
        call["action"].get("commands") in [expected_commands, split_commands],
        "Model did not emit the requested commands or the supported split stderr/exit commands",
    )
    require(call["action"].get("timeout_ms") == 1000, "Model did not emit the requested timeout_ms")
    require(call["action"].get("max_output_length") == 4096, "Model did not emit the requested max_output_length")
    expected_output = {"type": "shell_call_output", "call_id": call["call_id"], **shell(call["action"])}
    require(
        requests[1].get("input") == [expected_output, {"type": "message", "role": "user", "content": FOLLOW_UP}],
        "Continuation must contain matching structured shell output followed by the user message",
    )
    output = responses[1].get("output", [])
    require(
        not any(item.get("type") in {"shell_call", "function_call", "custom_tool_call"} for item in output),
        "Second response requested another tool call instead of interpreting the output",
    )
    text = "".join(
        part.get("text", "")
        for item in output if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    )
    require(bool(text.strip()), "Second response has no final assistant text")


if __name__ == "__main__":
    if sys.argv[1] == "prompts":
        print(first_prompt(sys.argv[2]))
        print(FOLLOW_UP)
    elif sys.argv[1] == "validate":
        try:
            validate(yaml.safe_load(Path(sys.argv[2]).read_text()), sys.argv[3], sys.argv[4] == "--stream")
        except (ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"ERROR: {sys.argv[2]}: {error}") from error
    else:
        raise SystemExit("Usage: scenarios.py prompts SCENARIO | validate FILE SCENARIO --stream/--no-stream")
