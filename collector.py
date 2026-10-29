
"""Run continuously in a separate terminal: python collector.py"""

import json
import os
import time
import logging

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from websocket import WebSocketApp

from src.regions import aisstream_boxes, region_for
from src.db import insert_position

# 육지 판별 기능이 설치되지 않아도 AIS 수집은 계속하도록 처리
try:
    from global_land_mask import globe
    LAND_CHECK_AVAILABLE = True
except Exception:
    globe = None
    LAND_CHECK_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("AISSTREAM_API_KEY")
URL = "wss://stream.aisstream.io/v0/stream"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


if not API_KEY or API_KEY == "replace_with_your_key":
    raise SystemExit("Set AISSTREAM_API_KEY in .env first.")


def num(x):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except (TypeError, ValueError):
        return None


def assess_position(lat, lon):
    """좌표 품질을 1차 판정한다. 의심 데이터도 삭제하지 않는다."""

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "suspected_land", "좌표가 유효 범위를 벗어남"

    if not LAND_CHECK_AVAILABLE:
        return "unverified", "육지 판별 라이브러리 사용 불가"

    try:
        is_land = bool(globe.is_land(lat, lon))

        if is_land:
            return (
                "suspected_land",
                "육지로 판정됨: 해안선, 항만, 좌표 오류 여부 확인 필요"
            )

        return "ok", "육지 판별상 해상 좌표"

    except Exception as exc:
        logging.warning("Land check failed: %s", exc)
        return "unverified", "육지 판별 중 오류 발생"


def on_open(ws):
    ws.send(json.dumps({
        "APIKey": API_KEY,
        "BoundingBoxes": aisstream_boxes(),
        "FilterMessageTypes": [
            "PositionReport",
            "StandardClassBPositionReport",
            "ExtendedClassBPositionReport",
        ],
    }))
    logging.info("Subscription sent for major maritime monitoring boxes.")


def on_message(ws, message):
    try:
        obj = json.loads(
            message.decode("utf-8")
            if isinstance(message, bytes)
            else message
        )

        typ = obj.get("MessageType", "")

        if typ == "SubscriptionConfirmation":
            logging.info(
                "Subscription confirmed: %s",
                obj.get("Message", {})
            )
            return

        meta = obj.get("MetaData") or {}
        body = (obj.get("Message") or {}).get(typ) or {}

        lat = num(meta.get(
            "latitude",
            meta.get("Latitude", body.get("Latitude"))
        ))
        lon = num(meta.get(
            "longitude",
            meta.get("Longitude", body.get("Longitude"))
        ))

        if lat is None or lon is None:
            return

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            logging.warning(
                "Invalid coordinates skipped: lat=%s lon=%s",
                lat, lon
            )
            return

        quality_flag, quality_note = assess_position(lat, lon)

        mmsi = meta.get("MMSI", body.get("UserID"))

        row = {
            "received_at_utc": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "ais_timestamp": (
                meta.get("time_utc")
                or meta.get("TimeUTC")
                or meta.get("timestamp")
            ),

            "mmsi": str(mmsi) if mmsi is not None else None,
            "ship_name": str(meta.get("ShipName") or "").strip() or None,
            "latitude": lat,
            "longitude": lon,
            "sog_knots": num(body.get("Sog")),
            "cog_deg": num(body.get("Cog")),
            "heading_deg": num(body.get("TrueHeading")),
            "message_type": typ,
            "region": region_for(lat, lon),
            "raw_json": json.dumps(obj, ensure_ascii=False),
            "quality_flag": quality_flag,
            "quality_note": quality_note,
        }

        insert_position(row)

        if quality_flag != "ok":
            logging.warning(
                "Position needs review: MMSI=%s lat=%s lon=%s flag=%s note=%s",
                mmsi, lat, lon, quality_flag, quality_note
            )

    except Exception:
        logging.exception("Failed to process AIS message")


def on_error(ws, error):
    logging.error("WebSocket error: %s", error)


def on_close(ws, code, msg):
    logging.warning("WebSocket closed (%s): %s", code, msg)


def main():
    backoff = 2

    if not LAND_CHECK_AVAILABLE:
        logging.warning(
            "Land checker unavailable; positions will be marked unverified."
        )

    while True:
        app = WebSocketApp(
            URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        app.run_forever(ping_interval=25, ping_timeout=15)

        logging.info("Reconnecting in %s seconds", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 120)


if __name__ == "__main__":
    main()
