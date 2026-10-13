import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate

DB_FILE = Path("app/storage/chat_sessions.db")


class SessionStore:
    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def list_sessions(self) -> List[Dict[str, str]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, title FROM sessions ORDER BY updated_at DESC"
            )
            return [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]

    def create_session(self, session_id: str, title: str = "New Chat"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO sessions (session_id, title, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (session_id, title),
            )
            conn.commit()

    def update_session_title(self, session_id: str, new_title: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (new_title, session_id),
            )
            conn.commit()

    def get_messages(self, session_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, sources_json FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            messages = []
            for role, content, sources_json in rows:
                msg = {"role": role, "content": content}
                if sources_json:
                    msg["sources"] = json.loads(sources_json)
                messages.append(msg)
            return messages

    def save_messages(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sources_json = json.dumps(sources) if sources else None
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, sources_json) VALUES (?, ?, ?, ?)",
                (session_id, role, content, sources_json),
            ),
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def cleanup_empty_sessions(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM sessions 
                WHERE session_id NOT IN (SELECT DISTINCT session_id FROM messages)
            """)
            conn.commit()

    def delete_session(self, session_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()


def generate_chat_title(first_query: str, llm) -> str:
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You generate short, clean chat thread titles.\n"
                    "Summarize the user question into a concise 2 to 5 word title.\n"
                    "Do NOT use quotes, punctuation, markdown, or greetings. Output ONLY the title.",
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm
        response = chain.invoke({"question": first_query})
        clean_title = response.content.strip().strip('"').strip("'")
        return clean_title if clean_title else first_query[:25]
    except Exception:
        return first_query[:25].strip() + ("..." if len(first_query) > 25 else "")


session_store = SessionStore()
