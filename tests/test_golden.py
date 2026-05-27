import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import asyncio
import pytest

os.environ["DEFAULT_LLM_PROVIDER"] = "opencode"
os.environ["OPENCODE_MODEL"] = "minimax-m2.5-free"

from app.core.auth import validate_login
from app.services.rag.pipeline import rag_pipeline


def load_golden():
    return json.loads(
        (Path(__file__).parent / "golden_questions.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def user():
    return validate_login("admin", "Admin123@")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_golden())
async def test_golden(case, user):
    resp = await rag_pipeline.answer_query(
        case["question"],
        provider="opencode",
        conversation_history=[],
        chat_id=None,
        active_document_id=None,
        user_id=user["id"],
    )

    expected = case["expected"]

    if "answer_mode" in expected:
        assert resp.answer_mode == expected["answer_mode"], (
            f"[{case['id']}] Esperado mode={expected['answer_mode']}, obtido={resp.answer_mode}"
        )

    if "answer_mode_in" in expected:
        assert resp.answer_mode in expected["answer_mode_in"], (
            f"[{case['id']}] Mode {resp.answer_mode} não está em {expected['answer_mode_in']}"
        )

    conf = (resp.confidence or {}).get("score", 0)
    min_conf = expected.get("min_confidence", 0)
    assert conf >= min_conf, f"[{case['id']}] Confiança {conf} < mínimo {min_conf}"

    answer_lower = resp.answer.lower()

    for term in expected.get("must_include", []):
        assert term.lower() in answer_lower, (
            f"[{case['id']}] Termo obrigatório '{term}' não encontrado"
        )

    for term in expected.get("must_not_contain", []):
        assert term.lower() not in answer_lower, (
            f"[{case['id']}] Termo proibido '{term}' encontrado na resposta"
        )

    for diploma in expected.get("expected_diplomas", []):
        found = any(
            diploma.lower() in (s.title or "").lower()
            or diploma.lower() in (s.source or "").lower()
            for s in resp.sources
        )
        assert found, f"[{case['id']}] Diploma '{diploma}' não encontrado nas fontes"

    classification = resp.classification or {}
    if "main_branch" in expected:
        assert classification.get("main_branch") == expected["main_branch"], (
            f"[{case['id']}] Branch esperado={expected['main_branch']}, real={classification.get('main_branch')}"
        )

    if "main_branch_in" in expected:
        assert classification.get("main_branch") in expected["main_branch_in"], (
            f"[{case['id']}] Branch {classification.get('main_branch')} não está em {expected['main_branch_in']}"
        )

    assert isinstance(resp.answer, str)
    if resp.answer_mode != "clarifying":
        assert len(resp.answer) > 30, f"[{case['id']}] Resposta muito curta"
