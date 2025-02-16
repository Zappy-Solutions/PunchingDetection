# database.py
import sqlite3
import logging
from threading import Lock

db_lock = Lock()

def setup_database(db_path="violations.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Violations (
            UserID TEXT,
            Time TEXT,
            Issue TEXT,
            ImagePath TEXT
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL;")
    logging.info("[INFO] Database connected and table ensured.")
    return conn, cursor