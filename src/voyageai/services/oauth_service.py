"""OAuth 2.0 authorization-code login (service layer). New on feat/auth.

The third-party login flow, end to end:
  login -> provider authorize page -> callback with a code
  -> exchange code for access token -> fetch userinfo (email)
  -> find-or-create the user -> issue OUR own JWT (reusing auth_service)

A built-in "mock" provider (routers/mock_provider.py) stands in for Google so
the whole loop runs locally with no real credentials. For a real provider, add
an entry to _PROVIDERS with client_id/secret/endpoints read from settings/.env.
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, status

from voyageai.schemas.user import TokenResponse
from voyageai.services.auth_service import auth_service

logger = logging.getLogger(__name__)

# Demo base for the built-in mock provider. Real providers read from settings.
_MOCK_BASE = "http://127.0.0.1:8000/mock-oauth"

_PROVIDERS: dict[str, dict] = {
    "mock": {
        "authorize_url": f"{_MOCK_BASE}/authorize",
        "token_url": f"{_MOCK_BASE}/token",
        "userinfo_url": f"{_MOCK_BASE}/userinfo",
        "client_id": "demo-client-id",
        "client_secret": "demo-client-secret",
        "redirect_uri": "http://127.0.0.1:8000/api/v1/auth/oauth/mock/callback",
        "scope": "openid email",
    },
    # "google": {  # fill from settings/.env to use the real provider
    #     "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
    #     "token_url": "https://oauth2.googleapis.com/token",
    #     "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
    #     "client_id": settings.google_client_id,
    #     "client_secret": settings.google_client_secret,
    #     "redirect_uri": ".../api/v1/auth/oauth/google/callback",
    #     "scope": "openid email profile",
    # },
}


class OAuthService:
    def _provider(self, name: str) -> dict:
        p = _PROVIDERS.get(name)
        if p is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown OAuth provider: {name}",
            )
        return p

    def build_authorize_url(self, provider: str, state: str) -> str:
        p = self._provider(provider)
        params = {
            "client_id": p["client_id"],
            "redirect_uri": p["redirect_uri"],
            "response_type": "code",
            "scope": p["scope"],
            "state": state,
        }
        return f"{p['authorize_url']}?{urlencode(params)}"

    async def handle_callback(self, provider: str, code: str) -> TokenResponse:
        p = self._provider(provider)
        async with httpx.AsyncClient(timeout=10) as client:
            # 1. exchange the one-time code for an access token
            tok = await client.post(
                p["token_url"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": p["client_id"],
                    "client_secret": p["client_secret"],
                    "redirect_uri": p["redirect_uri"],
                },
            )
            if tok.status_code != 200:
                raise HTTPException(status_code=401, detail="Token exchange failed")
            access_token = tok.json().get("access_token")

            # 2. use the access token to fetch the user's info (email)
            info = await client.get(
                p["userinfo_url"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if info.status_code != 200:
                raise HTTPException(status_code=401, detail="Userinfo request failed")
            email = info.json().get("email")

        if not email:
            raise HTTPException(status_code=401, detail="Provider returned no email")

        # 3. find-or-create the local user and issue OUR own JWT
        return auth_service.login_or_register_oauth(email)


oauth_service = OAuthService()
