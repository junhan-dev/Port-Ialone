
import sqlite3
from pathlib import Path

# 기존 과거 데이터베이스: 원본은 읽기만 함
SOURCE_DB = Path(
    r"C:\서강대학교\7학기\빅데이터종합설계\중간과제\ais.db"
)

# 현재 프로젝트의 데이터베이스
PROJECT_DIR = Path(__file__).resolve().parent
TARGET_DB = PROJECT_DIR / "data" / "ais_positions.sqlite"

if not SOURCE_DB.is_file():
    raise FileNotFoundError(f"과거 DB를 찾을 수 없습니다: {SOURCE_DB}")

if not TARGET_DB.is_file():
    raise FileNotFoundError(f"현재 DB를 찾을 수 없습니다: {TARGET_DB}")

conn = sqlite3.connect(TARGET_DB, timeout=30)
conn.execute("PRAGMA busy_timeout = 30000")

try:
    # 과거 DB를 읽기 전용으로 연결
    conn.execute(
    "ATTACH DATABASE ? AS old",
    (str(SOURCE_DB),)
)

    # 원본과 현재 DB의 구조 확인
    source_count = conn.execute(
        "SELECT COUNT(*) FROM old.pos"
    ).fetchone()[0]

    before_count = conn.execute(
        "SELECT COUNT(*) FROM main.positions"
    ).fetchone()[0]

    print(f"과거 DB 위치 기록: {source_count:,}건")
    print(f"통합 전 현재 기록: {before_count:,}건")

    # 중복을 제외하고 과거 기록을 추가
    cur = conn.execute("""
        INSERT INTO main.positions (
            received_at_utc,
            ais_timestamp,
            mmsi,
            ship_name,
            latitude,
            longitude,
            sog_knots,
            cog_deg,
            heading_deg,
            message_type,
            region,
            raw_json,
            quality_flag,
            quality_note
        )
        SELECT
            p.ts,
            p.ts,
            CAST(p.mmsi AS TEXT),
            NULLIF(TRIM(s.name), ''),
            p.lat,
            p.lon,
            p.sog,
            p.cog,
            NULL,
            'historical_import',
            p.area,
            json_object(
                'source', 'ais.db',
                'ts', p.ts,
                'mmsi', p.mmsi,
                'lat', p.lat,
                'lon', p.lon,
                'sog', p.sog,
                'cog', p.cog,
                'area', p.area
            ),
            'unverified',
            '과거 DB에서 가져온 기록; 위치 품질 미검증'
        FROM old.pos AS p
        LEFT JOIN old.static AS s
            ON p.mmsi = s.mmsi
        WHERE NOT EXISTS (
            SELECT 1
            FROM main.positions AS t
            WHERE t.mmsi = CAST(p.mmsi AS TEXT)
              AND t.ais_timestamp = p.ts
              AND t.latitude = p.lat
              AND t.longitude = p.lon
        )
    """)

    imported = cur.rowcount
    conn.commit()

    after_count = conn.execute(
        "SELECT COUNT(*) FROM main.positions"
    ).fetchone()[0]

    print(f"이번에 추가한 기록: {imported:,}건")
    print(f"통합 후 현재 기록: {after_count:,}건")

    print("\n해역별 과거 기록:")
    rows = conn.execute("""
        SELECT region, COUNT(*)
        FROM main.positions
        WHERE message_type = 'historical_import'
        GROUP BY region
        ORDER BY COUNT(*) DESC
    """).fetchall()

    for region, count in rows:
        print(f"  {region}: {count:,}건")

    print("\n통합 완료. 원본 ais.db는 수정하지 않았습니다.")

except Exception:
    conn.rollback()
    raise

finally:
    conn.close()
