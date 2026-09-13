"""OAuth DTOs. New module on feat/auth."""

from __future__ import annotations

from pydantic import BaseModel


class AuthorizeUrlResponse(BaseModel):
    """Where the browser should be sent to log in with the provider."""

    authorize_url: str
