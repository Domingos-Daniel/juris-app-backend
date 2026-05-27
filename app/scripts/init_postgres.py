from __future__ import annotations

from app.core.logger import configure_logging
from app.db.postgres import postgres_manager


def main() -> None:
    configure_logging()
    postgres_manager.initialize()
    print("Postgres schema initialized successfully.")


if __name__ == "__main__":
    main()
