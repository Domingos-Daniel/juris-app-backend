from __future__ import annotations

import argparse
import json

from app.core.logger import configure_logging
from app.services.catalog import lex_ao_catalog_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    payload = lex_ao_catalog_service.build_catalog(limit=args.limit)
    print(json.dumps(payload["summary"], ensure_ascii=False))
    targets = lex_ao_catalog_service.priority_targets(payload)
    print(json.dumps({"priority_targets": len(targets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
