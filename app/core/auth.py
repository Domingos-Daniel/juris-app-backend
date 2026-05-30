from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta

from fastapi import Header, HTTPException, status

from app.db.postgres import postgres_manager

import logging

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────


def _simple_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_password(password: str) -> str:
    return _simple_hash(f"juris-app:{password}")


def utc_now() -> datetime:
    return datetime.now(UTC)


def issue_token(user_id: str, email: str) -> str:
    now = utc_now().isoformat()
    return _simple_hash(f"{user_id}:{email}:{now}")


# ── Validation ────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?\d{7,15}$")


def validate_register(name: str, email: str, phone: str, password: str) -> dict:
    name = (name or "").strip()
    email = (email or "").strip().lower()
    phone = (phone or "").strip()
    password = (password or "").strip()

    if not name or len(name) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nome invalido (minimo 2 caracteres).",
        )
    if not _EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email invalido."
        )
    if phone and not _PHONE_RE.match(phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Numero de telefone invalido.",
        )
    if len(password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password deve ter pelo menos 6 caracteres.",
        )

    existing = postgres_manager.get_user_by_email(email)
    if existing and not existing.get("is_seeded"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email ja registado."
        )

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "password_hash": _hash_password(password),
    }


# ── Login ─────────────────────────────────────────────────────────


def validate_login(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais invalidas"
        )

    user = postgres_manager.get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais invalidas"
        )

    stored = user.get("password_hash", "")
    incoming = _hash_password(password or "")

    # Accept seeded admin password for default user
    if user.get("is_seeded"):
        from app.core.config import get_settings

        settings = get_settings()
        if password == settings.seeded_password:
            return {
                "id": user["id"],
                "name": user.get("name", ""),
                "email": user.get("email", email),
            }

    if stored != incoming:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais invalidas"
        )

    return {
        "id": user["id"],
        "name": user.get("name", ""),
        "email": user.get("email", email),
    }


# ── Auth guards ──────────────────────────────────────────────────


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Nao autenticado"
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not token or len(token) < 32:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido"
        )

    if not postgres_manager.has_auth_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido"
        )

    return {
        "id": "authenticated",
        "token": token,
        "expires_at": (utc_now() + timedelta(hours=8)).isoformat(),
    }


def get_current_user_full(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Nao autenticado"
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not token or len(token) < 32:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido"
        )

    user_id = postgres_manager.get_user_id_for_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido"
        )

    user = postgres_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Utilizador nao encontrado"
        )

    return {
        "id": user["id"],
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone", ""),
        "ai_preferences": user.get("ai_preferences", {}),
        "expires_at": (utc_now() + timedelta(hours=8)).isoformat(),
    }


def get_ws_current_user(token: str) -> dict | None:
    token = (token or "").strip()
    if not token or len(token) < 32:
        return None
    user_id = postgres_manager.get_user_id_for_token(token)
    if not user_id:
        return None
    user = postgres_manager.get_user_by_id(user_id)
    if not user:
        return None
    return {
        "id": user["id"],
        "name": user.get("name", ""),
        "email": user.get("email", ""),
    }
