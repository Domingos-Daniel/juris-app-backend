from __future__ import annotations

import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.core.auth import (
    get_current_user_full,
    issue_token,
    validate_login,
    validate_register,
)
from app.db.postgres import postgres_manager
from app.db.models import LoginRequest, LoginResponse

router = APIRouter(tags=["auth"])

# ── Rate limiter (in-memory, per-IP) ──────────────────────────────

_AUTH_WINDOW = 60  # 1 minute window
_AUTH_MAX_ATTEMPTS = 5  # max 5 attempts per minute
_attempts: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window = now - _AUTH_WINDOW
    _attempts[client_ip] = [t for t in _attempts[client_ip] if t > window]
    if len(_attempts[client_ip]) >= _AUTH_MAX_ATTEMPTS:
        retry_after = int(_attempts[client_ip][0] + _AUTH_WINDOW - now) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Muitas tentativas. Tente novamente em {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    _attempts[client_ip].append(now)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Models ────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    name: str
    email: str
    phone: str = ""
    password: str


class UpdateProfileRequest(BaseModel):
    name: str
    email: str
    phone: str = ""


class UpdatePreferencesRequest(BaseModel):
    tone: str = "formal"
    audience: str = "auto"
    detail_level: str = "normal"
    language_style: str = "acessivel"
    response_format: str = "auto"


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/auth/register", response_model=LoginResponse)
async def register(payload: RegisterRequest, request: Request) -> LoginResponse:
    _check_rate_limit(_get_client_ip(request))
    data = validate_register(
        payload.name, payload.email, payload.phone, payload.password
    )
    user_id = postgres_manager.register_user(
        data["name"], data["email"], data["phone"], data["password_hash"]
    )
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao criar conta.",
        )
    user = postgres_manager.get_user_by_id(user_id)
    token = issue_token(user_id, data["email"])
    postgres_manager.issue_auth_token(user_id, data["email"], token)
    return LoginResponse(
        token=token,
        user={
            "id": user_id,
            "name": data["name"],
            "email": data["email"],
            "phone": data["phone"],
        },
    )


@router.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest, request: Request) -> LoginResponse:
    _check_rate_limit(_get_client_ip(request))
    user = validate_login(payload.username, payload.password)
    token = issue_token(str(user["id"]), user.get("email", payload.username))
    postgres_manager.issue_auth_token(
        str(user["id"]), user.get("email", payload.username), token
    )
    return LoginResponse(token=token, user=user)


@router.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user_full)) -> dict:
    return current_user


@router.put("/auth/me")
async def update_profile(
    payload: UpdateProfileRequest, current_user: dict = Depends(get_current_user_full)
) -> dict:
    ok = postgres_manager.update_user_profile(
        current_user["id"], payload.name, payload.email, payload.phone
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao atualizar perfil.",
        )
    user = postgres_manager.get_user_by_id(current_user["id"])
    return {
        "id": user["id"],
        "name": user.get("name"),
        "email": user.get("email"),
        "phone": user.get("phone"),
    }


@router.put("/auth/me/preferences")
async def update_preferences(
    payload: UpdatePreferencesRequest,
    current_user: dict = Depends(get_current_user_full),
) -> dict:
    prefs = payload.model_dump()
    ok = postgres_manager.update_user_preferences(current_user["id"], prefs)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao atualizar preferencias.",
        )
    return prefs


@router.get("/auth/me/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user_full)) -> dict:
    return postgres_manager.get_user_preferences(current_user["id"])
