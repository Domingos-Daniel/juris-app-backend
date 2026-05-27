"""Deep diagnostic of the follow-up classification & retrieval pipeline."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.legal.classification import (
    legal_classifier, _follow_up_anchor, _looks_like_referential_follow_up,
    _detect_branches, _detect_topic_route, _detect_diplomas,
    _history_question_anchor, _specificity
)

# Simulate the exact conversation the user had
history = [
    "Utilizador: O que diz o artigo 26 da constituicao angolana? Eu queria saber os meus direitos.",
    "Assistente: O artigo 26 da Constituição da República de Angola estabelece que os direitos fundamentais não excluem outros direitos previstos em leis e normas internacionais...",
    "Utilizador: ok, Quais sao os direitos essenciais dos socios minoritarios numa sociedade por quotas?",
    "Assistente: Os direitos essenciais dos sócios minoritários numa sociedade por quotas incluem o direito a participar nos lucros, a participar nas deliberações, a obter informações sobre a vida da sociedade...",
]

follow_up_question = "me traga mais detalhes"

print("=" * 80)
print("DIAGNÓSTICO DO FOLLOW-UP")
print("=" * 80)

# 1. Is it detected as follow-up?
print(f"\n[1] _looks_like_referential_follow_up('{follow_up_question}'): {_looks_like_referential_follow_up(follow_up_question)}")

# 2. What anchor does it find?
anchor = _follow_up_anchor(history)
print(f"\n[2] _follow_up_anchor: '{anchor}'")

# 3. What's the last user question?
last_user = _history_question_anchor(history)
print(f"\n[3] _history_question_anchor (last user msg): '{last_user}'")

# 4. Is the last user msg itself a referential follow-up?
print(f"\n[4] Is last user msg a referential follow-up? {_looks_like_referential_follow_up(last_user)}")

# 5. What branches does the anchor resolve to?
anchor_branches = _detect_branches(anchor)
print(f"\n[5] Branches from anchor: {anchor_branches}")

# 6. What topic route does the anchor resolve to?
anchor_route = _detect_topic_route(anchor)
print(f"\n[6] Topic route from anchor: {anchor_route}")

# 7. What diplomas does the anchor resolve to?
anchor_diplomas = _detect_diplomas(anchor)
print(f"\n[7] Diplomas from anchor: {anchor_diplomas}")

# 8. Full classification
classification = legal_classifier.classify(follow_up_question, history)
print(f"\n[8] FULL CLASSIFICATION:")
print(json.dumps(classification.model_dump(), indent=2, ensure_ascii=False, default=str))

# 9. Now check what specificity is set
specificity = _specificity(follow_up_question, history, _detect_branches(follow_up_question))
print(f"\n[9] Specificity: {specificity}")

# 10. Direct branches from the follow-up question itself
direct_branches = _detect_branches(follow_up_question)
print(f"\n[10] Direct branches from '{follow_up_question}': {direct_branches}")

print("\n" + "=" * 80)
print("VERDICT:")
if classification.main_branch == "comercial" or classification.topic_route == "sociedades":
    print("✅ Classification correctly identifies sociedades/comercial context")
else:
    print(f"❌ Classification WRONG! main_branch={classification.main_branch}, topic_route={classification.topic_route}")
    print("   Expected: main_branch=comercial, topic_route=sociedades")
    print(f"   The anchor '{anchor}' should map to comercial/sociedades but didn't.")
