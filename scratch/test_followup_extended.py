"""Extended follow-up classification tests."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.legal.classification import legal_classifier

history_sociedades = [
    "Utilizador: Quais sao os direitos essenciais dos socios minoritarios numa sociedade por quotas?",
    "Assistente: Os direitos essenciais dos socios minoritarios incluem...",
]

history_laboral = [
    "Utilizador: O que acontece se eu for despedido sem aviso previo?",
    "Assistente: De acordo com a Lei Geral do Trabalho...",
]

history_constitucional = [
    "Utilizador: O que diz o artigo 26 da constituicao angolana?",
    "Assistente: O artigo 26 da CRA estabelece...",
]

test_cases = [
    ("me traga mais detalhes", history_sociedades, "comercial", "sociedades"),
    ("pode aprofundar?", history_sociedades, "comercial", "sociedades"),
    ("continue", history_sociedades, "comercial", "sociedades"),
    ("fale mais sobre isso", history_sociedades, "comercial", "sociedades"),
    ("quero saber mais", history_laboral, "laboral", "laboral"),
    ("explique melhor", history_laboral, "laboral", "laboral"),
    ("e quanto aos prazos?", history_laboral, "laboral", "laboral"),
    ("me explica melhor", history_constitucional, "constitucional", "constitucional"),
    ("diga mais", history_constitucional, "constitucional", "constitucional"),
    # Independent question should NOT be follow-up
    ("Quais sao os direitos fundamentais na constituicao?", history_laboral, None, None),
]

passed = 0
failed = 0
for question, history, expected_branch, expected_route in test_cases:
    c = legal_classifier.classify(question, history)
    is_followup = c.specificity == "follow_up"
    
    if expected_branch is None:
        # Should NOT be a follow-up
        ok = not is_followup
        status = "PASS" if ok else "FAIL"
        detail = f"specificity={c.specificity} (expected: NOT follow_up)"
    else:
        ok = is_followup and c.main_branch == expected_branch
        status = "PASS" if ok else "FAIL"
        detail = f"specificity={c.specificity}, main_branch={c.main_branch}, topic_route={c.topic_route}"
    
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] '{question}' -> {detail}")

print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)}")
