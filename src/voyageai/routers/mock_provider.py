"""A fake OAuth provider, DEMO ONLY. New on feat/auth.

Stands in for Google so the OAuth flow runs locally with no real credentials.
Do NOT mount this in production. Mounted under /mock-oauth.
"""

from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/authorize")
async def authorize(redirect_uri: str, state: str = ""):
    """Pretend the user consented; redirect back to the app with a code."""
    code = "mock-auth-code"
    return RedirectResponse(url=f"{redirect_uri}?code={code}&state={state}")


@router.post("/token")
async def token(
    grant_type: str = Form(...),
    code: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
    redirect_uri: str = Form(...),
):
    """Exchange the code for an access token (fake)."""
    return {"access_token": "mock-access-token", "token_type": "bearer"}


@router.get("/userinfo")
async def userinfo():
    """Return the logged-in user's profile (fake)."""
    return {"email": "oauth-user@example.com", "name": "OAuth User"}
