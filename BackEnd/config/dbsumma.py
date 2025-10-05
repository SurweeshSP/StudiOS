import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()
conn_str = os.getenv("DATABASE_URL")

try:
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            username = "test_user"
            password = "test_password"

            cur.execute(
                sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD %s").format(
                    sql.Identifier(username)
                ),
                [password]
            )

            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}").format(
                    sql.Identifier(username),
                    sql.Identifier(username)
                )
            )

            cur.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}.sample_table (id SERIAL PRIMARY KEY, name TEXT)").format(
                    sql.Identifier(username)
                )
            )

        print("User, schema, and table created successfully.")

except Exception as e:
    print(f"Connection failed: {e}")
