
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "ais_positions.sqlite"


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)

    con.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at_utc TEXT NOT NULL,
            ais_timestamp TEXT,
            mmsi TEXT,
            ship_name TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            sog_knots REAL,
            cog_deg REAL,
            heading_deg REAL,
            message_type TEXT,
            region TEXT,
            raw_json TEXT,
            UNIQUE(
                mmsi, ais_timestamp, latitude, longitude,
                sog_knots, cog_deg
            )
        )
    """)

    # 기존 데이터베이스에도 품질 컬럼을 안전하게 추가
    existing = {
        row[1] for row in con.execute(
            "PRAGMA table_info(positions)"
        ).fetchall()
    }

    migrations = {
        "quality_flag": "TEXT NOT NULL DEFAULT 'unverified'",
        "quality_note": "TEXT",
    }

    for column, definition in migrations.items():
        if column not in existing:
            con.execute(
                f"ALTER TABLE positions ADD COLUMN "
                f"{column} {definition}"
            )

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_positions_received
        ON positions(received_at_utc)
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_positions_mmsi
        ON positions(mmsi)
    """)

    con.commit()
    return con


def insert_position(r):
    con = connect()
    try:
        con.execute("""
            INSERT OR IGNORE INTO positions (
                received_at_utc, ais_timestamp, mmsi, ship_name,
                latitude, longitude, sog_knots, cog_deg, heading_deg,
                message_type, region, raw_json,
                quality_flag, quality_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(
            r.get(k) for k in [
                "received_at_utc", "ais_timestamp", "mmsi", "ship_name",
                "latitude", "longitude", "sog_knots", "cog_deg",
                "heading_deg", "message_type", "region", "raw_json",
                "quality_flag", "quality_note",
            ]
        ))
        con.commit()
    finally:
        con.close()


def read_positions(minutes=60, limit=30000):
    con = connect()
    try:
        return pd.read_sql_query("""
            SELECT *
            FROM positions
            WHERE received_at_utc >= datetime('now', ?)
            ORDER BY received_at_utc DESC
            LIMIT ?
        """, con, params=(f"-{int(minutes)} minutes", int(limit)))
    finally:
        con.close()


def latest_status():
    con = connect()
    try:
        return con.execute("""
            SELECT MAX(received_at_utc), COUNT(*)
            FROM positions
        """).fetchone()
    finally:
        con.close()

def read_all_positions(limit=200000):
    """실시간 및 과거 AIS 기록 전체 조회"""
    con = connect()
    try:
        return pd.read_sql_query("""
            SELECT *
            FROM positions
            ORDER BY received_at_utc DESC
            LIMIT ?
        """, con, params=(int(limit),))
    finally:
        con.close()