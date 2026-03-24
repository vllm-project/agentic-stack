# Stateful Conversations

One of the most powerful features of the Responses API is the ability to maintain conversation state on the server.

## Overview

In the standard Chat Completions API, the client is responsible for managing the entire conversation history. Every new request must include the full list of previous messages (`messages=[...]`). This is bandwidth-intensive and requires complex client-side state management.

The Responses API introduces **Statefulness** via the `previous_response_id` parameter.

## How It Works

1. **Initial Request**: You send a request with your initial input (e.g., a user message).
1. **Storage**: The gateway generates a response and stores the _entire conversation context_ (including your input and its output) in its database for final responses (`completed` and `incomplete`, when `store=true`).
1. **Continuation**: When you want to reply, you send _only_ your new input and the `previous_response_id` from the last response.
1. **Rehydration**: The gateway looks up the previous response, reconstructs the full history, and sends it to the model.

### Example Flow

#### Step 1: Start the conversation

```python
# No previous ID provided
response_1 = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "My name is Alice."}]
)

print(response_1.output[0].content)
# "Hello Alice!"

print(response_1.id)
# "resp_01J..."
```

#### Step 2: Continue the conversation

```python
# Pass the ID from Step 1
response_2 = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    previous_response_id=response_1.id,
    input=[{"role": "user", "content": "What is my name?"}]
)

print(response_2.output[0].content)
# "Your name is Alice."
```

## Storage Backends

Statefulness is powered by the **ResponseStore**.

- **Development**: By default, `agentic-stack` uses a local **SQLite** database (`agentic_stack.db`). This works great for local setups and single-machine deployments.
- **Production**: For multi-machine deployments or high-traffic production, you should configure a **PostgreSQL** database.

See [Configuration Reference](../reference/configuration.md) and [Configuration Guide](../deployment/configuration.md) for details.

## Security & Lifecycle

- **Capability-Based Access**: The `response_id` acts as a capability token. Anyone who possesses the ID can continue the conversation. Treat these IDs as secrets (like session tokens).
- **Persistence**: By default, responses are stored indefinitely (or until an expiration policy is configured/implemented).
- **`store` Parameter**: You can control whether a response is stored using the `store` parameter (default: `true`). If `store=false`, the response is not persisted and cannot be retrieved later or used as a `previous_response_id`.
- **Terminal Statuses**: Stored terminal responses include both `completed` and `incomplete`. Non-terminal and failed states are not continuation anchors.
