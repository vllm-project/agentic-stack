from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    llm_api_base: str = Field(
        description="Base URL of the upstream vLLM server (e.g. http://127.0.0.1:8000)."
    )
    openai_api_key: str | None = Field(
        description="API key forwarded to the upstream server."
    )

    gateway_host: str = Field(description="Host address the gateway binds to.")
    gateway_port: int = Field(description="Port the gateway listens on.")
    gateway_workers: int = Field(description="Number of uvicorn worker processes.")

    upstream_ready_timeout_s: float = Field(
        description="Seconds to wait for the upstream to become ready before aborting."
    )
    upstream_ready_interval_s: float = Field(
        description="Polling interval in seconds when waiting for the upstream to become ready."
    )
