---
hide:
  - navigation
  - toc
---

# Welcome to vLLM Agentic API

<p style="text-align:center">
<strong>The stateful and agentic API layer for vLLM</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/agentic-api" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/agentic-api/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/agentic-api/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

vLLM Agentic API provides the stateful APIs needed for real-world agentic applications — managing conversations, tool calls, and multi-turn interactions on top of [vLLM](https://github.com/vllm-project/vllm)'s high-throughput inference engine.

!!! important

    This project is in early development. Follow along and contribute on [GitHub](https://github.com/vllm-project/agentic-api).

## Responses API

Our first milestone is implementing the [Responses API](https://platform.openai.com/docs/api-reference/responses), bringing stateful, agentic capabilities to vLLM. We validate our implementation against the [Open Responses](https://www.openresponses.org/) compatibility test suite.

- **Stateful conversations** — The server manages conversation history via `previous_response_id`, eliminating client-side message tracking
- **Built-in tool use** — Web search, file search, and function calling handled within the API, with the model automatically executing multi-step tool chains
- **Streaming** — Server-sent events for real-time token streaming with structured lifecycle events
- **Background execution** — Fire-and-forget requests that continue processing server-side
- **Compatibility tested** — Validated against the open Responses API compatibility test suite

## Why Agentic API?

vLLM is fast with state-of-the-art serving throughput, PagedAttention, continuous batching, and broad hardware support. But building agentic applications on top of it today requires significant client-side orchestration — managing conversation state, tool call loops, and multi-turn flows.

Agentic API moves that complexity server-side, so you can:

- **Drop in a single API call** instead of building multi-turn orchestration
- **Let the server manage state** instead of tracking conversation history client-side
- **Use familiar APIs** — OpenAI-compatible endpoints backed by vLLM's inference engine
