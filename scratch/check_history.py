import os

import psycopg
from psycopg.rows import dict_row

dsn = os.environ.get("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/tcc")

with psycopg.connect(dsn, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT role, content FROM chat_messages ORDER BY created_at DESC LIMIT 10")
        rows = cur.fetchall()
        for row in reversed(rows):
            print(f"[{row['role']}] {row['content'][:500]}")
