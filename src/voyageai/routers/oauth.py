"""OAuth login endpoints (controller/router layer). New on feat/auth.

Thin controllers: build the authorize URL, or handle the provider callback.
Mounted under /api/v1/auth, so paths are /api/v1/auth/oauth/{provider}/...
"""

from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter

from voyageai.schemas.oauth import AuthorizeUrlResponse
from voyageai.schemas.user import TokenResponse
from voyageai.services.oauth_service import oauth_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/oauth/{provider}/login", response_model=AuthorizeUrlResponse)
async def oauth_login(provider: str) -> AuthorizeUrlResponse:
    """Return the provider authorize URL the browser should be sent to."""
    state = secrets.token_urlsafe(16)  # anti-CSRF, echoed back in the callback
    return AuthorizeUrlResponse(
        authorize_url=oauth_service.build_authorize_url(provider, state)
    )


@router.get("/oauth/{provider}/callback", response_model=TokenResponse)
async def oauth_callback(
    provider: str, code: str, state: str | None = None
) -> TokenResponse:
    """Provider redirects here with a code. Exchange it and issue our JWT."""
    return await oauth_service.handle_callback(provider, code)
