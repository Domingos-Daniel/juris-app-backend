from pathlib import Path

from app.services.pdf.ingestion import _document_signature_matches_text, _source_url_for_document


def test_document_signature_rejects_wrong_bilhete_pdf_text():
    wrong_text = (
        "Lei n.º 4/16: Lei de Autorização Legislativa sobre os Procedimentos e Incentivos a Atribuir "
        "às Descobertas Marginais. Ministério das Finanças."
    )

    assert _document_signature_matches_text("Lei do Bilhete de Identidade (Lei 4/16)", wrong_text) is False


def test_document_signature_rejects_wrong_contencioso_pdf_text():
    wrong_text = (
        "Lei n.º 7/21: Que Altera o Código Comercial. Revoga o artigo 32.º do Código Comercial "
        "e outras disposições."
    )

    assert _document_signature_matches_text("Lei do Contencioso Administrativo (Lei 7/21)", wrong_text) is False


def test_document_signature_rejects_partial_name_collision_for_contencioso():
    misleading_text = (
        "Lei n.º 7/21 de 14 de abril. Que altera o Código Comercial e regula matérias comerciais conexas."
    )

    assert _document_signature_matches_text("Lei do Contencioso Administrativo (Lei 7/21)", misleading_text) is False


def test_document_signature_accepts_expected_bilhete_identity_text():
    expected_text = (
        "Lei n.º 4/16 de 17 de maio. Lei do Bilhete de Identidade. "
        "O bilhete de identidade constitui documento bastante para provar a identidade civil do cidadão."
    )

    assert _document_signature_matches_text("Lei do Bilhete de Identidade (Lei 4/16)", expected_text) is True


def test_document_signature_accepts_expected_contencioso_text():
    expected_text = (
        "Lei n.º 33/22 de 01 de setembro. Código de Processo do Contencioso Administrativo. "
        "O contencioso administrativo regula a impugnação judicial do acto administrativo e os respetivos prazos."
    )

    assert _document_signature_matches_text("Codigo de Processo do Contencioso Administrativo (Lei 33/22)", expected_text) is True


def test_source_url_prefers_cached_lex_pdf_when_available():
    document = {
        "diploma_slug": "lei-bilhete-identidade-4-16",
        "url": "https://faolex.fao.org/docs/pdf/ang155187.pdf",
    }

    source_url = _source_url_for_document(document)

    assert Path("data/catalogs/lex_ao_documents.json").exists()
    assert source_url.startswith("https://files.lex.ao/")
    assert source_url.endswith(".pdf")
