from __future__ import annotations

from app.services.legal.models import LegalBranch, LegalClassification


BRANCH_TO_DIPLOMA: dict[LegalBranch, str] = {
    "penal": "Código Penal Lei 38/20",
    "laboral": "Lei Geral do Trabalho Lei 12/23",
    "civil": "Código Civil",
    "familia": "Código de Família Lei 1/88",
    "tributario": "Código Geral Tributário Lei 21/14",
    "comercial": "Lei das Sociedades Comerciais Lei 1/04",
    "constitucional": "Constituição da República de Angola 2022",
    "administrativo": "Lei do Contencioso Administrativo Lei 33/22",
    "propriedade": "Lei de Terras Lei 9/04",
}

TOPIC_TO_TERMS: dict[str, str] = {
    "penal_substantivo": "crime tipicidade sanção pena artigos",
    "laboral": "trabalhador salário despedimento indemnização contrato",
    "civil_obrigacoes": "contrato obrigação incumprimento resolução indemnização",
    "familia": "casamento divórcio guarda filhos alimentos",
    "sucessoes": "herança herdeiro testamento partilha inventário",
    "cpp": "prisão preventiva recurso prazo medidas coacção processo penal",
    "cpc": "processo civil acção contestação audiência revelia",
    "contencioso_admin": "acto administrativo impugnação recurso contencioso",
    "identificacao_civil": "bilhete identidade segunda via conservatória",
    "sociedades": "sócio sociedade assembleia deliberação quotas acções",
    "tributario": "imposto autuação prazo prescrição administração tributária",
    "iva": "IVA imposto valor acrescentado factura dedução",
    "terras": "terreno propriedade concessão registo posse",
    "constitucional": "direito fundamental garantia constitucional artigo",
}


class QueryExpander:
    def expand(self, question: str, classification: LegalClassification) -> list[str]:
        variants: list[str] = [question]

        branch = classification.main_branch
        if branch not in ("indeterminado", "misto"):
            diploma = BRANCH_TO_DIPLOMA.get(branch)
            if diploma:
                variants.append(f"{question} {diploma}")

        topic = classification.topic_route
        topic_terms = TOPIC_TO_TERMS.get(topic)
        if topic_terms and branch != "indeterminado":
            variants.append(f"{question} {topic_terms}")

        if branch == "misto":
            for candidate in classification.branch_candidates:
                diploma = BRANCH_TO_DIPLOMA.get(candidate)
                if diploma:
                    variants.append(f"{question} {diploma}")

        return list(dict.fromkeys(variants))


query_expander = QueryExpander()
