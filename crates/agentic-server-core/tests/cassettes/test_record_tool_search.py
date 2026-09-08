"""Focused offline tests for client tool-search cassette recording."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

import record_cassette


RETURNED_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
        "defer_loading": True,
    },
    {
        "type": "namespace",
        "name": "travel",
        "description": "Travel tools.",
        "tools": [
            {
                "type": "function",
                "name": "get_timezone",
                "description": "Get the time zone for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
                "defer_loading": True,
            }
        ],
    },
]

FLAT_TIMEZONE_NAME = "agentic_ns__travel__get_timezone"
OPENAI_TOOL_CHOICES = [
    "required",
    {"type": "function", "name": "get_weather"},
    "auto",
    "none",
]
GATEWAY_TOOL_CHOICES = [
    "required",
    {"type": "function", "name": "get_weather"},
    {"type": "function", "namespace": "travel", "name": "get_timezone"},
    "none",
]


class RecordToolSearchTests(unittest.TestCase):
    def test_mixed_catalog_fixture_loads_one_function_and_one_namespace_member(self) -> None:
        fixture_dir = Path(__file__).parent / "tool_search"
        initial = json.loads((fixture_dir / "openai_tools.json").read_text(encoding="utf-8"))
        returned = json.loads((fixture_dir / "returned_tools.json").read_text(encoding="utf-8"))
        normalized_initial = json.loads(
            (fixture_dir / "vllm_initial_tools.json").read_text(encoding="utf-8")
        )
        normalized_loaded = json.loads(
            (fixture_dir / "vllm_tools_after_search.json").read_text(encoding="utf-8")
        )
        openai_choices = json.loads(
            (fixture_dir / "openai_tool_choice_sequence.json").read_text(encoding="utf-8")
        )
        gateway_choices = json.loads(
            (fixture_dir / "gateway_tool_choice_sequence.json").read_text(encoding="utf-8")
        )
        ordinary = [tool for tool in initial if tool["type"] == "function"]
        namespaces = [tool for tool in initial if tool["type"] == "namespace"]
        self.assertEqual(
            [tool["name"] for tool in ordinary],
            ["get_weather", "get_exchange_rate", "search_hotels"],
        )
        self.assertTrue(all(tool["defer_loading"] is True for tool in ordinary))
        self.assertEqual(len(namespaces), 1)
        self.assertEqual(namespaces[0]["name"], "travel")
        self.assertEqual(
            [member["name"] for member in namespaces[0]["tools"]],
            ["get_timezone", "get_coordinates", "calculate_distance"],
        )
        self.assertTrue(
            all(member["defer_loading"] is True for member in namespaces[0]["tools"])
        )

        self.assertEqual([tool["type"] for tool in returned], ["function", "namespace"])
        self.assertEqual(returned[0]["name"], "get_weather")
        self.assertEqual(returned[1]["name"], "travel")
        self.assertEqual([member["name"] for member in returned[1]["tools"]], ["get_timezone"])

        self.assertEqual([tool["name"] for tool in normalized_initial], ["tool_search"])
        self.assertEqual(
            [tool["name"] for tool in normalized_loaded],
            ["tool_search", "get_weather", FLAT_TIMEZONE_NAME],
        )
        normalized_names = {tool["name"] for tool in normalized_loaded}
        self.assertTrue(
            normalized_names.isdisjoint(
                {"get_exchange_rate", "search_hotels", "get_coordinates", "calculate_distance"}
            )
        )
        self.assertEqual(openai_choices, OPENAI_TOOL_CHOICES)
        self.assertEqual(gateway_choices, GATEWAY_TOOL_CHOICES)

    def test_gateway_cli_profile_accepts_public_store_false_manual_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tools = directory_path / "tools.json"
            outputs = directory_path / "outputs.json"
            returned = directory_path / "returned.json"
            choices = directory_path / "choices.json"
            capture = directory_path / "capture.yaml"
            tools.write_text(json.dumps([{"type": "tool_search", "execution": "client"}]), encoding="utf-8")
            outputs.write_text(json.dumps({"get_weather": "sunny"}), encoding="utf-8")
            returned.write_text(json.dumps(RETURNED_TOOLS), encoding="utf-8")
            choices.write_text(json.dumps(GATEWAY_TOOL_CHOICES), encoding="utf-8")

            with (
                mock.patch.object(record_cassette, "_start_proxy", return_value=object()),
                mock.patch.object(record_cassette, "_stop_proxy"),
                mock.patch.object(record_cassette, "run_responses") as run_responses,
            ):
                result = CliRunner().invoke(
                    record_cassette.main,
                    [
                        "--mode", "responses",
                        "--turns", "4",
                        "--gateway", "http://gateway.test",
                        "--model", "test-model",
                        "--no-stream",
                        "--no-store",
                        "--manual-item-replay",
                        "--tools", str(tools),
                        "--tool-outputs", str(outputs),
                        "--tool-search-output-tools", str(returned),
                        "--tool-choice-sequence", str(choices),
                        "--output", str(capture),
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIs(run_responses.call_args.kwargs["manual_item_replay"], True)
        self.assertEqual(run_responses.call_args.kwargs["tool_choice_sequence"], GATEWAY_TOOL_CHOICES)
        self.assertIs(run_responses.call_args.args[4], False, "gateway profile must set store=false")

    def test_gateway_websocket_cli_profile_accepts_stored_tool_search_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tools = directory_path / "tools.json"
            outputs = directory_path / "outputs.json"
            returned = directory_path / "returned.json"
            choices = directory_path / "choices.json"
            capture = directory_path / "capture.yaml"
            tools.write_text(json.dumps([{"type": "tool_search", "execution": "client"}]), encoding="utf-8")
            outputs.write_text(json.dumps({"get_weather": "sunny"}), encoding="utf-8")
            returned.write_text(json.dumps(RETURNED_TOOLS), encoding="utf-8")
            choices.write_text(json.dumps(GATEWAY_TOOL_CHOICES), encoding="utf-8")

            with (
                mock.patch.object(record_cassette, "_start_proxy") as start_proxy,
                mock.patch.object(record_cassette, "run_responses") as run_responses,
            ):
                result = CliRunner().invoke(
                    record_cassette.main,
                    [
                        "--mode", "responses",
                        "--turns", "4",
                        "--gateway", "http://gateway.test",
                        "--transport", "websocket",
                        "--model", "test-model",
                        "--stream",
                        "--tools", str(tools),
                        "--tool-outputs", str(outputs),
                        "--tool-search-output-tools", str(returned),
                        "--tool-choice-sequence", str(choices),
                        "--output", str(capture),
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        start_proxy.assert_not_called()
        self.assertIs(run_responses.call_args.args[4], True)
        self.assertEqual(run_responses.call_args.args[7], "websocket")
        self.assertEqual(run_responses.call_args.kwargs["tool_choice_sequence"], GATEWAY_TOOL_CHOICES)

    def test_tool_search_cli_requires_one_tool_choice_per_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tools = directory_path / "tools.json"
            outputs = directory_path / "outputs.json"
            returned = directory_path / "returned.json"
            short_choices = directory_path / "short-choices.json"
            capture = directory_path / "capture.yaml"
            tools.write_text(json.dumps([{"type": "tool_search", "execution": "client"}]), encoding="utf-8")
            outputs.write_text(json.dumps({"get_weather": "sunny"}), encoding="utf-8")
            returned.write_text(json.dumps(RETURNED_TOOLS), encoding="utf-8")
            short_choices.write_text(json.dumps(GATEWAY_TOOL_CHOICES[:-1]), encoding="utf-8")
            common_args = [
                "--mode", "responses",
                "--turns", "4",
                "--gateway", "http://gateway.test",
                "--model", "test-model",
                "--tools", str(tools),
                "--tool-outputs", str(outputs),
                "--tool-search-output-tools", str(returned),
                "--output", str(capture),
            ]

            missing = CliRunner().invoke(record_cassette.main, common_args)
            short = CliRunner().invoke(
                record_cassette.main,
                [*common_args, "--tool-choice-sequence", str(short_choices)],
            )

        self.assertNotEqual(missing.exit_code, 0)
        self.assertIn("requires --tool-choice-sequence", missing.output)
        self.assertNotEqual(short.exit_code, 0)
        self.assertIn("one JSON value per turn", short.output)

    def test_websocket_handshake_preserves_coalesced_first_frame(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.chunks = [b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n\x81\x02ok"]

            def recv(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        client = record_cassette.WebSocketClient("ws://gateway.test", {})
        client.sock = Socket()

        response = client._read_http_response()

        self.assertTrue(response.endswith("\r\n\r\n"))
        self.assertEqual(client.receive_text(), "ok")

    def test_websocket_recording_stops_on_response_failed(self) -> None:
        failed = {
            "type": "response.failed",
            "response": {
                "id": "resp_failed",
                "status": "failed",
                "error": {"code": "provider_failure", "message": "stopped"},
            },
        }

        class Socket:
            def __init__(self) -> None:
                self.messages = [json.dumps(failed), None]

            def __enter__(self) -> "Socket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def send_text(self, _text: str) -> None:
                return None

            def receive_text(self) -> str | None:
                return self.messages.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.yaml"
            with (
                mock.patch.object(record_cassette, "WebSocketClient", return_value=Socket()),
                mock.patch.object(record_cassette, "_append_turn") as append_turn,
            ):
                response = record_cassette._send_websocket(
                    {"model": "test", "input": "hello"},
                    "http://gateway.test",
                    {},
                    output,
                )

        self.assertEqual(response, failed["response"])
        turn = append_turn.call_args.args[1]
        self.assertEqual(turn["response"]["status_code"], 101)
        self.assertEqual(json.loads(turn["response"]["websocket"][0]), failed)
        self.assertTrue(turn["response"]["sse"][0].startswith("event: response.failed\n"))

    def test_tool_search_outputs_use_public_and_synthetic_wire_shapes(self) -> None:
        public = record_cassette._build_tool_output_input(
            [{"type": "tool_search_call", "call_id": "call_public"}],
            {},
            None,
            RETURNED_TOOLS,
        )
        synthetic = record_cassette._build_tool_output_input(
            [
                {
                    "type": "function_call",
                    "name": "tool_search",
                    "call_id": "call_synthetic",
                }
            ],
            {},
            None,
            RETURNED_TOOLS,
        )

        self.assertEqual(
            public,
            [
                {
                    "type": "tool_search_output",
                    "call_id": "call_public",
                    "execution": "client",
                    "status": "completed",
                    "tools": RETURNED_TOOLS,
                }
            ],
        )
        self.assertEqual(synthetic[0]["type"], "function_call_output")
        self.assertEqual(
            json.loads(synthetic[0]["output"]), {"tools": RETURNED_TOOLS}
        )

    def test_tool_continuations_reject_empty_ids_and_missing_search_tools(self) -> None:
        for call in (
            {"type": "tool_search_call", "call_id": ""},
            {"type": "function_call", "name": "tool_search"},
            {"type": "function_call", "name": "get_weather", "call_id": "   "},
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                record_cassette._build_tool_output_input(
                    [call], {}, None, RETURNED_TOOLS
                )

        with self.assertRaises(ValueError):
            record_cassette._build_tool_output_input(
                [
                    {
                        "type": "tool_search_call",
                        "call_id": "call_without_tools",
                    }
                ],
                {},
                None,
                None,
            )
        with self.assertRaisesRegex(ValueError, "explicit output fixture"):
            record_cassette._build_tool_output_input(
                [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_without_output",
                    }
                ],
                {},
                None,
                RETURNED_TOOLS,
            )

    def test_existing_function_and_custom_outputs_remain_supported(self) -> None:
        continuation = record_cassette._build_tool_output_input(
            [
                {
                    "type": "function_call",
                    "call_id": "call_function",
                    "name": "lookup",
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "call_custom",
                    "name": "raw_echo",
                },
            ],
            {"lookup": "function result", "raw_echo": "custom result"},
            "continue",
            None,
        )
        self.assertEqual(
            [item["type"] for item in continuation],
            ["function_call_output", "custom_tool_call_output", "message"],
        )

    def test_namespaced_and_flattened_calls_use_explicit_output_fixtures(self) -> None:
        public = record_cassette._build_tool_output_input(
            [
                {
                    "type": "function_call",
                    "namespace": "travel",
                    "name": "get_timezone",
                    "call_id": "call_public_timezone",
                }
            ],
            {"get_timezone": "public time zone"},
            None,
            RETURNED_TOOLS,
        )
        normalized = record_cassette._build_tool_output_input(
            [
                {
                    "type": "function_call",
                    "name": FLAT_TIMEZONE_NAME,
                    "call_id": "call_flat_timezone",
                }
            ],
            {FLAT_TIMEZONE_NAME: "normalized time zone"},
            None,
            RETURNED_TOOLS,
        )

        self.assertEqual(public[0]["output"], "public time zone")
        self.assertEqual(normalized[0]["output"], "normalized time zone")

    def test_synthetic_manual_replay_switches_to_loaded_tools(self) -> None:
        initial_tools = [{"type": "function", "name": "tool_search"}]
        next_tools = initial_tools + [
            RETURNED_TOOLS[0],
            {
                **RETURNED_TOOLS[1]["tools"][0],
                "name": FLAT_TIMEZONE_NAME,
                "defer_loading": False,
            },
        ]
        responses = [
            {
                "id": "resp_search",
                "output": [
                    {
                        "type": "function_call",
                        "name": "tool_search",
                        "call_id": "call_search",
                        "status": "completed",
                        "arguments": '{"query":"weather tool"}',
                    }
                ],
            },
            {
                "id": "resp_weather",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_weather",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {
                "id": "resp_timezone",
                "output": [
                    {
                        "type": "function_call",
                        "name": FLAT_TIMEZONE_NAME,
                        "call_id": "call_timezone",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {"id": "resp_final", "output": [{"type": "message"}]},
        ]
        sent_bodies: list[dict] = []

        def fake_send(_client: object, body: dict, *_args: object, **_kwargs: object) -> dict:
            sent_bodies.append(body)
            return responses[len(sent_bodies) - 1]

        with (
            mock.patch.object(
                record_cassette,
                "_prompt",
                side_effect=["find tools", "call weather", "call timezone", "finish"],
            ),
            mock.patch.object(record_cassette, "_send", side_effect=fake_send),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=4,
                model="test-model",
                stream=False,
                store=False,
                branches=[],
                proxy_url="http://unused",
                tools=initial_tools,
                tool_outputs={
                    "get_weather": '{"temperature_c":21}',
                    FLAT_TIMEZONE_NAME: '{"iana_timezone":"Europe/Paris"}',
                },
                tool_search_output_tools=RETURNED_TOOLS,
                tools_after_search=next_tools,
                manual_item_replay=True,
            )

        self.assertEqual(
            [body["tools"] for body in sent_bodies],
            [initial_tools, next_tools, next_tools, next_tools],
        )
        self.assertTrue(all(body["store"] is False for body in sent_bodies))
        self.assertTrue(all("previous_response_id" not in body for body in sent_bodies))

        turn_one_input = sent_bodies[0]["input"]
        turn_two_input = sent_bodies[1]["input"]
        turn_three_input = sent_bodies[2]["input"]
        turn_four_input = sent_bodies[3]["input"]
        self.assertEqual(
            turn_one_input,
            [{"type": "message", "role": "user", "content": "find tools"}],
        )
        self.assertEqual(turn_two_input[: len(turn_one_input)], turn_one_input)
        self.assertEqual(turn_two_input[1]["type"], "function_call")
        self.assertEqual(turn_two_input[1]["call_id"], "call_search")
        self.assertEqual(turn_two_input[2]["type"], "function_call_output")
        self.assertEqual(turn_two_input[2]["call_id"], "call_search")
        self.assertEqual(turn_three_input[: len(turn_two_input)], turn_two_input)
        weather_call = next(
            item
            for item in turn_three_input[len(turn_two_input) :]
            if item.get("type") == "function_call"
        )
        weather_output = next(
            item
            for item in turn_three_input[len(turn_two_input) :]
            if item.get("type") == "function_call_output"
        )
        self.assertEqual(weather_call["call_id"], "call_weather")
        self.assertEqual(weather_output["call_id"], "call_weather")
        self.assertEqual(turn_four_input[: len(turn_three_input)], turn_three_input)
        timezone_call = next(
            item
            for item in turn_four_input[len(turn_three_input) :]
            if item.get("type") == "function_call"
        )
        timezone_output = next(
            item
            for item in turn_four_input[len(turn_three_input) :]
            if item.get("type") == "function_call_output"
        )
        self.assertEqual(timezone_call["name"], FLAT_TIMEZONE_NAME)
        self.assertEqual(timezone_call["call_id"], "call_timezone")
        self.assertEqual(timezone_output["call_id"], "call_timezone")
        self.assertTrue(all(body["parallel_tool_calls"] is False for body in sent_bodies))

    def test_public_linear_responses_flow_keeps_public_top_level_tools(self) -> None:
        public_tools = [
            {"type": "tool_search", "execution": "client"},
            {
                "type": "function",
                "name": "get_weather",
                "defer_loading": True,
            },
            RETURNED_TOOLS[1],
        ]
        responses = [
            {
                "id": "resp_search",
                "output": [
                    {
                        "type": "tool_search_call",
                        "call_id": "call_search",
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": "weather tool"},
                    }
                ],
            },
            {
                "id": "resp_weather",
                "output": [
                    {
                        "type": "function_call",
                        "name": "get_weather",
                        "call_id": "call_weather",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {
                "id": "resp_timezone",
                "output": [
                    {
                        "type": "function_call",
                        "namespace": "travel",
                        "name": "get_timezone",
                        "call_id": "call_timezone",
                        "status": "completed",
                        "arguments": '{"city":"Paris"}',
                    }
                ],
            },
            {"id": "resp_final", "output": [{"type": "message"}]},
        ]
        sent_bodies: list[dict] = []

        def fake_send(_client: object, body: dict, *_args: object, **_kwargs: object) -> dict:
            sent_bodies.append(body)
            return responses[len(sent_bodies) - 1]

        with (
            mock.patch.object(
                record_cassette,
                "_prompt",
                side_effect=["find tools", "call weather", "call timezone", "finish"],
            ),
            mock.patch.object(record_cassette, "_send", side_effect=fake_send),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=4,
                model="test-model",
                stream=False,
                store=True,
                branches=[],
                proxy_url="http://unused",
                tools=public_tools,
                tool_choice_sequence=GATEWAY_TOOL_CHOICES,
                tool_outputs={
                    "get_weather": '{"temperature_c":21}',
                    "get_timezone": '{"iana_timezone":"Europe/Paris"}',
                },
                tool_search_output_tools=RETURNED_TOOLS,
            )

        self.assertEqual(sent_bodies[0]["tools"], public_tools)
        self.assertEqual([body["tool_choice"] for body in sent_bodies], GATEWAY_TOOL_CHOICES)
        self.assertNotIn("tools", sent_bodies[1])
        self.assertNotIn("tools", sent_bodies[2])
        self.assertNotIn("tools", sent_bodies[3])
        public_output = sent_bodies[1]["input"][0]
        self.assertEqual(public_output["type"], "tool_search_output")
        self.assertEqual(public_output["call_id"], "call_search")
        self.assertEqual(public_output["execution"], "client")
        self.assertEqual(public_output["status"], "completed")
        self.assertEqual(public_output["tools"], RETURNED_TOOLS)
        self.assertEqual(sent_bodies[2]["input"][0]["type"], "function_call_output")
        self.assertEqual(sent_bodies[3]["input"][0]["type"], "function_call_output")
        self.assertEqual(sent_bodies[3]["input"][0]["call_id"], "call_timezone")
        self.assertTrue(all(body["parallel_tool_calls"] is False for body in sent_bodies))

    def test_gateway_public_manual_replay_is_store_false_and_omits_tools_after_search(self) -> None:
        public_tools = [
            {"type": "tool_search", "execution": "client"},
            {"type": "function", "name": "get_weather", "defer_loading": True},
            RETURNED_TOOLS[1],
        ]
        responses = [
            {
                "id": "resp_search",
                "output": [{
                    "type": "tool_search_call",
                    "id": "tsc_search",
                    "call_id": "call_search",
                    "execution": "client",
                    "status": "completed",
                    "arguments": {"query": "weather"},
                }],
            },
            {
                "id": "resp_weather",
                "output": [{
                    "type": "function_call",
                    "id": "fc_weather",
                    "name": "get_weather",
                    "call_id": "call_weather",
                    "status": "completed",
                    "arguments": '{"city":"Paris"}',
                }],
            },
            {
                "id": "resp_timezone",
                "output": [{
                    "type": "function_call",
                    "id": "fc_timezone",
                    "namespace": "travel",
                    "name": "get_timezone",
                    "call_id": "call_timezone",
                    "status": "completed",
                    "arguments": '{"city":"Paris"}',
                }],
            },
            {"id": "resp_final", "output": [{"type": "message"}]},
        ]
        sent_bodies: list[dict] = []

        def fake_send(_client: object, body: dict, *_args: object, **_kwargs: object) -> dict:
            sent_bodies.append(body)
            return responses[len(sent_bodies) - 1]

        with (
            mock.patch.object(
                record_cassette,
                "_prompt",
                side_effect=["find", "call weather", "call timezone", "finish"],
            ),
            mock.patch.object(record_cassette, "_send", side_effect=fake_send),
        ):
            record_cassette.run_responses(
                client=object(),
                turns=4,
                model="test-model",
                stream=False,
                store=False,
                branches=[],
                proxy_url="http://unused",
                tools=public_tools,
                tool_choice_sequence=GATEWAY_TOOL_CHOICES,
                tool_outputs={"get_weather": "sunny", "get_timezone": "Europe/Paris"},
                tool_search_output_tools=RETURNED_TOOLS,
                manual_item_replay=True,
            )

        self.assertTrue(all(body["store"] is False for body in sent_bodies))
        self.assertTrue(all("previous_response_id" not in body for body in sent_bodies))
        self.assertEqual(sent_bodies[0]["tools"], public_tools)
        self.assertEqual([body["tool_choice"] for body in sent_bodies], GATEWAY_TOOL_CHOICES)
        self.assertNotIn("tools", sent_bodies[1])
        self.assertNotIn("tools", sent_bodies[2])
        self.assertNotIn("tools", sent_bodies[3])
        self.assertEqual(sent_bodies[1]["input"][1]["type"], "tool_search_call")
        self.assertEqual(sent_bodies[1]["input"][2]["type"], "tool_search_output")
        self.assertEqual(sent_bodies[2]["input"][3]["type"], "message")
        self.assertEqual(sent_bodies[2]["input"][4]["type"], "function_call")
        self.assertEqual(sent_bodies[2]["input"][5]["type"], "function_call_output")
        self.assertEqual(sent_bodies[3]["input"][7]["namespace"], "travel")
        self.assertEqual(sent_bodies[3]["input"][7]["name"], "get_timezone")
        self.assertEqual(sent_bodies[3]["input"][8]["type"], "function_call_output")
        self.assertEqual(sent_bodies[3]["input"][9]["type"], "message")

if __name__ == "__main__":
    unittest.main()
