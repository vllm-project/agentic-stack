# Web Search Examples

Use the gateway-owned `web_search` built-in to let the model search the web,
open pages, and reuse page text inside the same request.

## Prerequisite

Start a supported entrypoint with a shipped profile:

### `agentic-stacks serve`

```bash
export EXA_API_KEY="your-exa-api-key"  # optional for exa_mcp
agentic-stacks serve \
  --upstream http://127.0.0.1:8000/v1 \
  --web-search-profile exa_mcp
```

### `vllm serve --responses`

```bash
export EXA_API_KEY="your-exa-api-key"  # optional for exa_mcp
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --responses \
  --responses-web-search-profile exa_mcp
```

If `EXA_API_KEY` is omitted, the shipped `exa_mcp` helper uses the anonymous
default Exa MCP URL.

## 1. Minimal Request

```python
response = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Find the official vLLM documentation site."}],
    tools=[{"type": "web_search"}],
)
```

## 2. Return Search Sources

Add `web_search_call.action.sources` to `include` when you want the final
search action sources expanded in the response item.

```python
response = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Find migration notes for vLLM and cite the sources."}],
    tools=[{"type": "web_search"}],
    include=["web_search_call.action.sources"],
)
```

## 3. Streaming Web Search Events

```python
with client.responses.stream(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Search for the latest vLLM release notes."}],
    tools=[{"type": "web_search"}],
    include=["web_search_call.action.sources"],
) as stream:
    for event in stream:
        print(event)
```

Expected event families include:

- `response.web_search_call.in_progress`
- `response.web_search_call.searching`
- `response.web_search_call.completed`

## 4. Search Then Reuse Page Text

`find_in_page` is backed by request-local cached page text. In practice, that
means the model must first open a page before it can search inside it during
the same request.

Prompt pattern:

```text
Search for the vLLM migration notes, open the most relevant page, then find the
section mentioning breaking changes.
```
