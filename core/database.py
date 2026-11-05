import csv
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.config import DB_PATH, EXPORT_DIR, LOG_COOLDOWN_SECONDS, ensure_directories, logger


@dataclass
class PersonProfile:
    id: int
    name: str
    age: int
    department: str = "General"
    role: str = "Member"
    created_at: str = ""


@dataclass
class RecognitionLogEntry:
    log_id: int
    person_id: int
    name: str
    department: str
    similarity_pct: float
    distance: float
    timestamp: str


class DatabaseManager:
    """Enterprise SQLite repository with schema migration, in-memory profile caching, and audit logging."""

    def __init__(self, db_path: Path = DB_PATH):
        ensure_directories()
        self.db_path = Path(db_path)
        self._profile_cache: Dict[int, PersonProfile] = {}
        self._last_logged_time: Dict[int, float] = {}
        self._init_db()
        self.refresh_cache()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and seamlessly migrate existing schemas if needed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS People (
                    ID INTEGER PRIMARY KEY,
                    Name TEXT NOT NULL,
                    Age INTEGER NOT NULL,
                    Department TEXT DEFAULT 'General',
                    Role TEXT DEFAULT 'Member',
                    CreatedAt TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Backward-compatible migration for existing People table created without Department/Role/CreatedAt
            cursor = conn.execute("PRAGMA table_info(People)")
            existing_cols = {row["name"].lower() for row in cursor.fetchall()}

            if "department" not in existing_cols:
                conn.execute("ALTER TABLE People ADD COLUMN Department TEXT DEFAULT 'General'")
            if "role" not in existing_cols:
                conn.execute("ALTER TABLE People ADD COLUMN Role TEXT DEFAULT 'Member'")
            if "createdat" not in existing_cols:
                conn.execute("ALTER TABLE People ADD COLUMN CreatedAt TEXT DEFAULT ''")

            # Audit / Attendance log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS RecognitionLogs (
                    LogID INTEGER PRIMARY KEY AUTOINCREMENT,
                    PersonID INTEGER NOT NULL,
                    Name TEXT NOT NULL,
                    Department TEXT DEFAULT 'General',
                    SimilarityPct REAL NOT NULL,
                    Distance REAL NOT NULL,
                    Timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def refresh_cache(self) -> Dict[int, PersonProfile]:
        """Load all profiles into memory so multi-person video detection performs zero disk I/O per frame."""
        profiles: Dict[int, PersonProfile] = {}
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT ID, Name, Age, Department, Role, CreatedAt FROM People ORDER BY ID ASC"
                )
                for row in cursor.fetchall():
                    profile = PersonProfile(
                        id=int(row["ID"]),
                        name=str(row["Name"]),
                        age=int(row["Age"]) if row["Age"] is not None else 0,
                        department=str(row["Department"] or "General"),
                        role=str(row["Role"] or "Member"),
                        created_at=str(row["CreatedAt"] or ""),
                    )
                    profiles[profile.id] = profile
        except sqlite3.Error as exc:
            logger.error("Failed to refresh profile cache: %s", exc)

        self._profile_cache = profiles
        return self._profile_cache

    def upsert_person(
        self,
        person_id: int,
        name: str,
        age: int,
        department: str = "General",
        role: str = "Member",
    ) -> PersonProfile:
        """Insert a new person or update an existing record by ID."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = name.strip() or f"User-{person_id}"
        department = department.strip() or "General"
        role = role.strip() or "Member"

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM People WHERE ID = ?", (person_id,))
            exists = cursor.fetchone() is not None

            if exists:
                conn.execute(
                    """
                    UPDATE People
                    SET Name = ?, Age = ?, Department = ?, Role = ?
                    WHERE ID = ?
                    """,
                    (name, age, department, role, person_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO People (ID, Name, Age, Department, Role, CreatedAt)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (person_id, name, age, department, role, now_str),
                )
            conn.commit()

        self.refresh_cache()
        return self._profile_cache[person_id]

    def delete_person(self, person_id: int) -> bool:
        """Delete a person record from the database and update cache."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM People WHERE ID = ?", (person_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        self.refresh_cache()
        return deleted

    def get_person(self, person_id: int) -> Optional[PersonProfile]:
        """Fast O(1) in-memory lookup for real-time video frame recognition."""
        return self._profile_cache.get(person_id)

    def list_people(self) -> List[PersonProfile]:
        """Return all registered people sorted by ID."""
        return list(self._profile_cache.values())

    def record_recognition(
        self,
        person: PersonProfile,
        similarity_pct: float,
        distance: float,
        cooldown_seconds: float = LOG_COOLDOWN_SECONDS,
    ) -> bool:
        """Record a recognition event in the audit log with cooldown deduplication."""
        now_ts = time.time()
        last_ts = self._last_logged_time.get(person.id, 0.0)
        if now_ts - last_ts < cooldown_seconds:
            return False

        self._last_logged_time[person.id] = now_ts
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO RecognitionLogs (PersonID, Name, Department, SimilarityPct, Distance, Timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person.id,
                        person.name,
                        person.department,
                        round(similarity_pct, 1),
                        round(distance, 2),
                        timestamp_str,
                    ),
                )
                conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to log recognition event: %s", exc)
            return False

    def get_recent_logs(self, limit: int = 200) -> List[RecognitionLogEntry]:
        """Fetch recent recognition audit logs."""
        entries: List[RecognitionLogEntry] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT LogID, PersonID, Name, Department, SimilarityPct, Distance, Timestamp
                    FROM RecognitionLogs
                    ORDER BY LogID DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                for row in cursor.fetchall():
                    entries.append(
                        RecognitionLogEntry(
                            log_id=int(row["LogID"]),
                            person_id=int(row["PersonID"]),
                            name=str(row["Name"]),
                            department=str(row["Department"] or "General"),
                            similarity_pct=float(row["SimilarityPct"]),
                            distance=float(row["Distance"]),
                            timestamp=str(row["Timestamp"]),
                        )
                    )
        except sqlite3.Error as exc:
            logger.error("Failed to fetch logs: %s", exc)
        return entries

    def clear_logs(self) -> None:
        """Clear all recognition logs."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM RecognitionLogs")
            conn.commit()
        self._last_logged_time.clear()

    def export_logs_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """Export recognition audit logs to a CSV file."""
        ensure_directories()
        if output_path is None:
            filename = f"recognition_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = EXPORT_DIR / filename

        logs = self.get_recent_logs(limit=10000)
        with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["Log ID", "User ID", "Name", "Department", "Match (%)", "LBPH Distance", "Timestamp"]
            )
            for entry in logs:
                writer.writerow(
                    [
                        entry.log_id,
                        entry.person_id,
                        entry.name,
                        entry.department,
                        f"{entry.similarity_pct:.1f}%",
                        f"{entry.distance:.2f}",
                        entry.timestamp,
                    ]
                )
        return output_path

