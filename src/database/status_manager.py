# src/database/status_manager.py
import sqlite3
import json
from datetime import datetime
from src.config import DB_PATH

class StatusManager:
    """
    A simple manager for a SQLite database to track the status of each query.
    This acts as the central "brain" of the application.
    """
    def __init__(self):
        self.db_path = DB_PATH
        self._conn = None
        self.init_db()

    def _get_connection(self):
        """Establishes and returns a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_db(self):
        """Initializes the database and creates the 'queries' table if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stages TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_pdf_path TEXT,
                error_message TEXT
            )
        """)
        conn.commit()

    def add_query(self, query_id: str):
        """Adds a new query to the database with a 'pending' status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        initial_stages = {
            "evidence_extraction": "pending",
            "model_inference": "pending",
            "pdf_generation": "pending"
        }
        
        try:
            cursor.execute(
                "INSERT INTO queries (query_id, status, stages, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (query_id, "pending", json.dumps(initial_stages), now, now)
            )
            conn.commit()
            print(f"INFO: Query '{query_id}' added to status tracker.")
        except sqlite3.IntegrityError:
            print(f"WARN: Query '{query_id}' already exists in the database.")

    def update_stage_status(self, query_id: str, stage: str, status: str, error_message: str = None):
        """Updates the status of a specific stage for a query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT stages FROM queries WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        if not row:
            print(f"ERROR: Query ID '{query_id}' not found.")
            return

        stages = json.loads(row['stages'])
        stages[stage] = status
        
        # Determine overall status
        overall_status = "processing"
        if status == "failed":
            overall_status = "failed"
        elif all(s == "completed" for s in stages.values()):
            overall_status = "completed"

        cursor.execute(
            """
            UPDATE queries 
            SET status = ?, stages = ?, updated_at = ?, error_message = ?
            WHERE query_id = ?
            """,
            (overall_status, json.dumps(stages), datetime.utcnow().isoformat(), error_message, query_id)
        )
        conn.commit()
        print(f"INFO: Status updated for '{query_id}': Stage '{stage}' -> '{status}'")

    def set_result_path(self, query_id: str, pdf_path: str):
        """Sets the final PDF result path for a query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE queries SET result_pdf_path = ?, updated_at = ? WHERE query_id = ?",
            (pdf_path, datetime.utcnow().isoformat(), query_id)
        )
        conn.commit()

    def get_all_queries(self):
        """Retrieves all queries from the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM queries ORDER BY created_at DESC")
        rows = cursor.fetchall()
        # Convert sqlite3.Row objects to standard dictionaries
        return [dict(row) for row in rows]
        
    def get_query_status(self, query_id: str):
        """Retrieves the status of a single query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM queries WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def reset_query(self, query_id: str):
        """Resets a query's status back to 'pending' for reprocessing."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        initial_stages = {
            "evidence_extraction": "pending",
            "model_inference": "pending",
            "pdf_generation": "pending"
        }
        
        cursor.execute(
            """
            UPDATE queries
            SET status = ?, stages = ?, updated_at = ?, result_pdf_path = NULL, error_message = NULL
            WHERE query_id = ?
            """,
            ("pending", json.dumps(initial_stages), now, query_id)
        )
        conn.commit()
        print(f"INFO: Query '{query_id}' has been reset for reprocessing.")
        
    
    def move_to_trash(self, query_id: str):
        """Marks a query's status as 'trashed' in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute(
            "UPDATE queries SET status = ?, updated_at = ? WHERE query_id = ?",
            ("trashed", now, query_id)
        )
        conn.commit()
        self._conn.close()
        self._conn = None
        
    def delete_permanently(self, query_id: str):
        """Removes a query's record from the database entirely."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM queries WHERE query_id = ?", (query_id,))
        conn.commit()


# Instantiate a global manager so other modules can import it easily
status_manager = StatusManager()