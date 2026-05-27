from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.db.postgres import postgres_manager


def _simple_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def issue_token(user_id: str | int, username: str) -> str:
    now = utc_now().isoformat()
    return _simple_hash(f"{user_id}:{username}:{now}")


def validate_login(username: str, password: str) -> dict:
    settings = get_settings()
    if username != settings.seeded_username or password != settings.seeded_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais invalidas")

    return {
        "id": postgres_manager.get_default_user_id(),
        "name": postgres_manager.get_default_user_name(),
        "username": settings.seeded_username,
    }


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nao autenticado")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido")

    # Auth basico: qualquer token gerado por login nesta sessao e aceite.
    # Mantemos verificacao minima para fluxo local.
    if len(token) < 32:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido")

    if not postgres_manager.has_auth_token(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido")

    return {
        "id": postgres_manager.get_default_user_id(),
        "name": postgres_manager.get_default_user_name(),
        "username": get_settings().seeded_username,
        "expires_at": (utc_now() + timedelta(hours=8)).isoformat(),
    }
