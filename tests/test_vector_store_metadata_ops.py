import json

import pytest

from app.services.rag.vector_store import LegislationVectorStore


@pytest.mark.asyncio
async def test_delete_and_count_by_metadata(tmp_path, monkeypatch):
    store = LegislationVectorStore()
    monkeypatch.setattr(store, "base_path", tmp_path)
    monkeypatch.setattr(store, "data_file", tmp_path / "vectors.json")
    store._cache = [
        {"id": "1", "text": "a", "metadata": {"diploma_slug": "x", "source_scope": "official"}, "embedding": [0.1]},
        {"id": "2", "text": "b", "metadata": {"diploma_slug": "y", "source_scope": "official"}, "embedding": [0.2]},
        {"id": "3", "text": "c", "metadata": {"diploma_slug": "x", "source_scope": "user_upload"}, "embedding": [0.3]},
    ]
    store._persist()

    assert store.count_by_metadata(diploma_slug="x") == 2
    assert store.count_by_metadata(diploma_slug="x", source_scope="official") == 1

    removed = store.delete_by_metadata(diploma_slug="x", source_scope="official")

    assert removed == 1
    payload = json.loads(store.data_file.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert store.count_by_metadata(diploma_slug="x") == 1
