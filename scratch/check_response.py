import os

import psycopg
from psycopg.rows import dict_row

dsn = os.environ.get("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/tcc")

with psycopg.connect(dsn, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT content FROM chat_messages WHERE role='assistant' ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            print(row["content"][:2500])
        else:
            print("No messages found")
