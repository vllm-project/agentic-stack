# Verifying image support against a live vision model

This guide is the opt-in, manual counterpart to the deterministic image coverage that runs in CI. It records how to
serve a vision model, how to tell the gateway that model accepts images, and the exact commands that prove an image
survives every transport the gateway offers.

The gateway never decodes an image. It carries `input_image` parts and structured tool output through its typed
model, its persistence layer, and both transports. Decoding and preprocessing stay in vLLM, so a live run is checking
transport fidelity plus the model's ability to actually see what arrived.

## What is already proven without a GPU

| Check | Where | Runs in CI |
|---|---|---|
| Mixed text/image ordering, multiple images, tool-output images, continuation, rehydration, `store: false` | `crates/agentic-server/tests/responses_test.rs` | yes |
| The same over WebSocket streaming | `crates/agentic-server/tests/responses_websocket_test.rs` | yes |
| Compaction retains image-bearing user messages | `crates/agentic-server/tests/compaction_test.rs` | yes |
| The real, pinned Codex CLI attaches an inline PNG that reaches the gateway byte for byte | `scripts/codex-smoke.sh` | yes |
| Catalog modality resolution and precedence | `crates/agentic-server/tests/models_test.rs` | yes |

Those use in-process mocks and a replay server; no model runs. What they cannot show is whether a real vision model
renders the image it received. That is what this guide adds.

## The catalog is what decides whether Codex sends an image

Codex reads image support from its **local** model catalog, not from the model. When the entry for the selected model
does not list `image` in `input_modalities`, Codex replaces the attachment client-side before any request is sent. The
gateway then receives text like this in place of the image:

```json
{"type": "input_text", "text": "<image name=[Image #1] path=\"/tmp/red-pixel.png\">"},
{"type": "input_text", "text": "image content omitted because you do not support image input"}
```

That placeholder is the signature of **client-side stripping**. If you see it, the gateway never had the image and no
gateway change can help — fix the catalog. A gateway-side loss looks different: the request reaching vLLM would be
missing an `input_image` part that the client did send.

## 1. Serve a vision model

```console
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4}' \
  --port 5050
```

- **Chat template.** Qwen3-VL ships a multimodal chat template with the model, and vLLM loads it automatically. Pass
  `--chat-template` only to override it; a text-only template silently drops image parts during rendering, which looks
  exactly like a gateway bug.
- **`--limit-mm-per-prompt` spelling changed across vLLM releases** (the older form is `image=4`). Check
  `vllm serve --help` for the version you run, and raise the limit above the number of images a single turn sends.
- Record the vLLM version you used with `python -c 'import vllm; print(vllm.__version__)'` — image handling has moved
  between releases, so a result is only meaningful with the version attached.

Confirm what is served and whether it advertises image support:

```console
curl -s http://127.0.0.1:5050/v1/models | python3 -m json.tool
```

## 2. Tell the gateway the model accepts images

If the upstream entry already carries `capabilities: ["image"]`, the gateway resolves `["text", "image"]` on its own.
Otherwise declare it in `~/.agentic-api/config.toml`:

```toml
[models."Qwen/Qwen3-VL-8B-Instruct"]
input_modalities = ["text", "image"]
```

Resolution order is an explicit local override, then recognized upstream metadata, then a text-only fallback. An
explicit `["text"]` override wins over upstream image metadata, which is how a vision model gets pinned to text for the
control test in step 5. See [Codex integration](../design/codex-integration.md#image-capability-resolution).

Verify the served catalog agrees, using the client version Codex reports:

```console
curl -s "http://127.0.0.1:3000/v1/models?client_version=$(codex --version | awk '{print $NF}')" \
  | python3 -c 'import sys, json; print([(m["slug"], m["input_modalities"]) for m in json.load(sys.stdin)["models"]])'
```

Expected: `[('Qwen/Qwen3-VL-8B-Instruct', ['text', 'image'])]`.

## 3. Codex with an attached image

Make a small, non-sensitive test image with a property you can check in prose:

```console
python3 -c '
import struct, zlib
def chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
rows = b"".join(b"\x00" + (bytes((255, 0, 0)) * 4 + bytes((0, 0, 255)) * 4) for _ in range(8))
png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))
open("/tmp/half-red-half-blue.png", "wb").write(png)'
```

That is an 8x8 PNG, red on the left half and blue on the right. Run Codex through the gateway:

```console
agentic harness codex \
  --gateway-url http://127.0.0.1:3000 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  -- exec --skip-git-repo-check \
  --image=/tmp/half-red-half-blue.png \
  "Name the two colors in the attached image and say which side each is on."
```

A correct answer names red on the left and blue on the right. A model that answers vaguely, or claims it cannot see an
image, is the signal to check step 5 before suspecting the gateway.

Codex also declares a client-executed `view_image` function tool (`{"path": "<file>"}`) for images it finds on disk
during a session. Ask it to look at a file mid-session to exercise that path:

```console
agentic harness codex --gateway-url http://127.0.0.1:3000 --model Qwen/Qwen3-VL-8B-Instruct
> Use view_image on /tmp/half-red-half-blue.png and tell me what you see.
```

The tool result reaches the next inference round as a `function_call_output` whose `output` is an **array** containing
an `input_image` part — not a stringified JSON blob. `test_websocket_view_image_tool_output_reaches_next_round` locks
that shape in CI.

## 4. Direct Responses checks

These bypass Codex, so they isolate the gateway from client behavior. Export the image once:

```console
IMAGE="data:image/png;base64,$(base64 -w0 /tmp/half-red-half-blue.png)"
GATEWAY=http://127.0.0.1:3000
MODEL=Qwen/Qwen3-VL-8B-Instruct
```

**Mixed content and ordering** — the answer must reflect both the text and the image:

```console
curl -s "$GATEWAY/v1/responses" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\", \"store\": true, \"stream\": false,
  \"input\": [{\"type\": \"message\", \"role\": \"user\", \"content\": [
    {\"type\": \"input_text\", \"text\": \"What two colors are in this image?\"},
    {\"type\": \"input_image\", \"image_url\": \"$IMAGE\"}
  ]}]}" | python3 -c 'import sys, json; print(json.load(sys.stdin)["output"][0]["content"][0]["text"])'
```

**Continuation** — take the `id` from the response above and confirm the model still remembers the image:

```console
curl -s "$GATEWAY/v1/responses" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\", \"store\": true, \"stream\": false,
  \"previous_response_id\": \"resp_...\",
  \"input\": [{\"type\": \"message\", \"role\": \"user\", \"content\": \"Which side was the blue on?\"}]}"
```

An answer of "the right side" proves the stored image rehydrated into the model's context. The same check works with
`conversation_id` instead of `previous_response_id`.

**Client-executed tool output** — the `view_image` shape, without Codex:

```console
curl -s "$GATEWAY/v1/responses" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\", \"store\": true, \"stream\": false,
  \"tools\": [{\"type\": \"function\", \"name\": \"view_image\", \"description\": \"View a local image.\",
               \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}}}],
  \"input\": [
    {\"type\": \"message\", \"role\": \"user\", \"content\": \"Look at the image.\"},
    {\"type\": \"function_call\", \"call_id\": \"call_1\", \"name\": \"view_image\",
     \"arguments\": \"{\\\"path\\\":\\\"/tmp/half-red-half-blue.png\\\"}\"},
    {\"type\": \"function_call_output\", \"call_id\": \"call_1\", \"output\": [
      {\"type\": \"input_text\", \"text\": \"attached local image path: /tmp/half-red-half-blue.png\"},
      {\"type\": \"input_image\", \"image_url\": \"$IMAGE\"}]}]}"
```

**WebSocket** — Codex uses the WebSocket transport by default, so step 3 already covers it. To drive it directly, send
the same payload as a `response.create` message to `ws://127.0.0.1:3000/v1/responses`.

## 5. The text-only control

Run this whenever an image does not appear to arrive, to separate client stripping from gateway loss. Pin the same
vision model to text in `config.toml`:

```toml
[models."Qwen/Qwen3-VL-8B-Instruct"]
input_modalities = ["text"]
```

Restart the gateway, regenerate any persistent Codex home, and repeat step 3. Expected: Codex sends the
`image content omitted because you do not support image input` placeholder, and the model cannot name the colors. The
curl checks in step 4 are unaffected — the gateway still forwards the image, because modality resolution shapes
`/v1/models` only and never filters request content. That asymmetry is the point: it tells you which side dropped the
image.

## 6. Record the run

Image behavior moves between vLLM releases and Codex releases, so a result without versions is not reusable.

| Field | Value |
|---|---|
| vLLM version | |
| Model and revision | `Qwen/Qwen3-VL-8B-Instruct` @ |
| Chat template | shipped with the model / overridden with |
| Serve flags | |
| Codex version | `codex --version` |
| Gateway commit | `git rev-parse --short HEAD` |
| Step 3 result | |
| Step 4 results | |
| Step 5 control | |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Codex sends `image content omitted because you do not support image input` | The catalog Codex read says text-only | Set `input_modalities = ["text", "image"]`, restart the gateway, and delete any persistent `model_catalog.json` written earlier — `agentic run`/`agentic harness` regenerate a session home per invocation, a pinned `AGENTIC_CODEX_HOME` does not |
| Launch fails with `the gateway model catalog ... returned HTTP 404` | The upstream serves no `/v1/models`, so the gateway has no catalog to transform | Point the gateway at a real upstream; the Codex launcher requires the catalog to resolve the model and its modalities |
| The gateway forwards the image but the model ignores it | A text-only chat template, or `--limit-mm-per-prompt` below the number of images sent | Drop the `--chat-template` override and raise the limit |
| vLLM returns a 400 mentioning multimodal input | The served model is not a vision model | Check `/v1/models`; a text model with an image override in `config.toml` will still be sent images |
| Context-length errors once an image is attached | Image tokens are added to the prompt budget | Raise `--max-model-len` or send fewer images per turn |

Token estimation for image-bearing messages is tracked separately in issue #255; this guide does not cover it.
