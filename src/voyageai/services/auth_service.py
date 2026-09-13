"""Auth business logic (service layer).

New module added by feat/auth. Contains no HTTP concerns beyond raising
HTTPException for auth failures, mirroring how existing routers surface errors.

Storage is an in-memory dict for now so this slice runs with no external
dependency. Swap it later for a real UserStore modeled on
storage/mongodb.py::MongoDBResultStore.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import HTTPException, status

from voyageai.schemas.user import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)

# Demo-only secret. In production read this from settings/.env, never hardcode.
_JWT_SECRET = "dev-only-change-me-please-use-a-32B+-secret"
_JWT_ALG = "HS256"
_TOKEN_TTL_MINUTES = 60


class AuthService:
    """Register and authenticate users.

    In-memory store keyed by email: {email: {"id", "email", "password_hash"}}.
    """

    def __init__(self) -> None:
        self._users: dict[str, dict] = {}

    def register(self, req: RegisterRequest) -> UserResponse:
        if req.email in self._users:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )
        user_id = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(
            req.password.encode(), bcrypt.gensalt()
        ).decode()
        self._users[req.email] = {
            "id": user_id,
            "email": req.email,
            "password_hash": password_hash,
        }
        logger.info(f"Registered user {req.email}")
        # Note: return a response DTO, the password hash never leaves this layer.
        return UserResponse(id=user_id, email=req.email)

    def login(self, req: LoginRequest) -> TokenResponse:
        user = self._users.get(req.email)
        if user is None or not bcrypt.checkpw(
            req.password.encode(), user["password_hash"].encode()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )
        token = self._issue_token(user["id"])
        return TokenResponse(access_token=token)

    def _issue_token(self, user_id: str) -> str:
        payload = {
            "sub": user_id,
            "exp": datetime.now(timezone.utc)
            + timedelta(minutes=_TOKEN_TTL_MINUTES),
        }
        return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALG)


# Module-level singleton, matching the ai_service import style.
auth_service = AuthService()
