"""Auth API endpoints (controller/router layer).

New module added by feat/auth. Thin controllers: bind request DTO,
call one service method, return the response DTO. No business logic here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from voyageai.schemas.user import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from voyageai.services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(req: RegisterRequest) -> UserResponse:
    """Create a new user and return the safe user view (no password)."""
    return auth_service.register(req)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest) -> TokenResponse:
    """Verify credentials and return an access token, or 401."""
    return auth_service.login(req)
