from __future__ import annotations

from pydantic import BaseModel


class BuiltinProfileSelection(BaseModel):
    profile: str | None = None
