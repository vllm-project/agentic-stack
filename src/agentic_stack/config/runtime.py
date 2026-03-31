from dataclasses import dataclass
from typing import Literal

RuntimeMode = Literal["standalone", "integrated"]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # Upstream vLLM server
    llm_api_base: str
    openai_api_key: str | None

    # Gateway process
    gateway_host: str
    gateway_port: int
    gateway_workers: int

    # Startup behaviour
    upstream_ready_timeout_s: float
    upstream_ready_interval_s: float
