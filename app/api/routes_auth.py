from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.auth import get_current_user, issue_token, validate_login
from app.db.postgres import postgres_manager
from app.db.models import LoginRequest, LoginResponse


router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    user = validate_login(payload.username, payload.password)
    token = issue_token(str(user["id"]), payload.username)
    postgres_manager.issue_auth_token(str(user["id"]), payload.username, token)
    return LoginResponse(token=token, user=user)


@router.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)) -> dict:
    return current_user
