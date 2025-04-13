# from langgraph.store.memory import InMemoryStore
import sqlite3


class MemoryStore:
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    value TEXT
                )
            """)
            conn.commit()

    def add(self, memory: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_store (value)
                SELECT ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_store WHERE value = ?
                )
            """, (memory, memory))
            conn.commit()

    def update(self, id: str, new_memory: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memory_store
                SET value = ?
                WHERE id = ?
            """, (new_memory, id))
            conn.commit()

    def delete(self, id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM memory_store
                WHERE id = ?
            """, (id,))
            conn.commit()

    def dump(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, value FROM memory_store")
            rows = cursor.fetchall()
            return {str(row[0]): row[1] for row in rows}
