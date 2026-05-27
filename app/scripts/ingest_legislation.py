from __future__ import annotations

import asyncio

from app.core.logger import configure_logging
from app.services.pdf.ingestion import legislation_ingestion_service


async def _main() -> None:
    summary = await legislation_ingestion_service.ingest_official_documents()
    print(summary.model_dump_json(indent=2))


def main() -> None:
    configure_logging()
    asyncio.run(_main())


if __name__ == "__main__":
    main()
