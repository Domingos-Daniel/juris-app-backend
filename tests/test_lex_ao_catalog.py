from app.services.catalog.lex_ao_catalog import lex_ao_catalog_service


def test_matches_codigo_familia_from_route_and_signature():
    matched = lex_ao_catalog_service._match_internal_slug(
        title="Lei n.º 1/88 de 20 de fevereiro",
        description="",
        entity_slug="assembleia-do-povo",
        year="1988",
        document_slug="lei-n-o-1-88-de-20-de-fevereiro",
    )

    assert matched == "codigo-familia-lei-1-88"


def test_matches_bilhete_identidade_from_route_and_signature():
    matched = lex_ao_catalog_service._match_internal_slug(
        title="Lei n.º 4/16 de 17 de maio",
        description="",
        entity_slug="assembleia-nacional",
        year="2016",
        document_slug="lei-n-o-4-16-de-17-de-maio",
    )

    assert matched == "lei-bilhete-identidade-4-16"


def test_matches_sociedades_comerciais_from_route_and_signature():
    matched = lex_ao_catalog_service._match_internal_slug(
        title="Lei n.º 1/04 de 13 de fevereiro",
        description="",
        entity_slug="assembleia-nacional",
        year="2004",
        document_slug="lei-n-o-1-04-de-13-de-fevereiro",
    )

    assert matched == "lei-sociedades-comerciais-1-04"


def test_matches_contencioso_administrativo_from_route_and_signature():
    matched = lex_ao_catalog_service._match_internal_slug(
        title="Lei n.º 7/21 de 14 de abril",
        description="",
        entity_slug="assembleia-nacional",
        year="2021",
        document_slug="lei-n-o-7-21-de-14-de-abril",
    )

    assert matched == "lei-contencioso-administrativo-7-21"


def test_does_not_force_match_for_unknown_generic_law():
    matched = lex_ao_catalog_service._match_internal_slug(
        title="Lei n.º 99/25 de 1 de janeiro",
        description="",
        entity_slug="assembleia-nacional",
        year="2025",
        document_slug="lei-n-o-99-25-de-1-de-janeiro",
    )

    assert matched is None
