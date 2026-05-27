import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os

os.environ["DEFAULT_LLM_PROVIDER"] = "opencode"
os.environ["OPENCODE_MODEL"] = "minimax-m2.5-free"

import pytest

from app.core.auth import validate_login
from app.services.rag.pipeline import rag_pipeline


@pytest.fixture(scope="session")
def user():
    return validate_login("admin", "Admin123@")


@pytest.mark.asyncio
async def test_pipeline_simple_penal(user):
    resp = await rag_pipeline.answer_query(
        "Qual a pena para burla em Angola?",
        provider="opencode",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id=user["id"],
    )
    assert resp.answer_mode in ("grounded", "limited")
    assert isinstance(resp.answer, str)
    assert len(resp.answer) > 50


@pytest.mark.asyncio
async def test_pipeline_clarifying(user):
    resp = await rag_pipeline.answer_query(
        "Preciso de ajuda com a lei",
        provider="opencode",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id=user["id"],
    )
    assert resp.answer_mode in ("clarifying", "grounded", "limited")
    if resp.answer_mode == "clarifying":
        assert len(resp.clarifying_questions) > 0


@pytest.mark.asyncio
async def test_pipeline_laboral(user):
    resp = await rag_pipeline.answer_query(
        "O meu chefe nao me quer pagar o salario, o que faco?",
        provider="opencode",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id=user["id"],
    )
    assert resp.answer_mode in ("grounded", "limited")
    assert len(resp.answer) > 50


@pytest.mark.asyncio
async def test_pipeline_confidence_varies(user):
    resp_burla = await rag_pipeline.answer_query(
        "Qual a pena para burla em Angola?",
        provider="opencode",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id=user["id"],
    )
    resp_familia = await rag_pipeline.answer_query(
        "Como funciona a guarda dos filhos no divorcio em Angola?",
        provider="opencode",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id=user["id"],
    )
    c1 = (resp_burla.confidence or {}).get("score", 0)
    c2 = (resp_familia.confidence or {}).get("score", 0)
    assert c1 >= 0.0
    assert c2 >= 0.0


@pytest.mark.asyncio
async def test_pipeline_no_empty_answers(user):
    perguntas = [
        "Qual a pena para burla?",
        "O que diz a lei sobre despedimento?",
        "Como pedir a segunda via do BI?",
    ]
    for q in perguntas:
        resp = await rag_pipeline.answer_query(
            q,
            provider="opencode",
            conversation_history=[],
            chat_id=None,
            active_document_id=None,
            user_id=user["id"],
        )
        assert resp.answer_mode in ("clarifying", "grounded", "limited")
        if resp.answer_mode != "clarifying":
            assert resp.answer, f"Resposta vazia para: {q}"
