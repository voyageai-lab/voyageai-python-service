"""Auth DTOs (schemas) for user registration and login.

New module added by feat/auth. Does not modify any existing schema.
The response DTO never carries the password or its hash.
"""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Request DTO for creating a new user."""

    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    """Request DTO for logging in."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Response DTO. Only fields the client should see. No password ever."""

    id: str
    email: EmailStr


class TokenResponse(BaseModel):
    """Response DTO returned on successful login."""

    access_token: str
    token_type: str = "bearer"
