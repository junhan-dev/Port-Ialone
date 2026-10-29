
from datetime import datetime, timezone

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

from src.db import read_positions, read_all_positions, latest_status
from src.regions import REGIONS


# --------------------------------------------------
# 1. 페이지 설정 및 공통 함수
# --------------------------------------------------

st.set_page_config(
    page_title="PORT-I | Global Strait AIS Monitor",
    page_icon="🚢",
    layout="wide",
)

st.title("PORT-I | Global Strait AIS Risk Monitor")
st.caption(
    "AIS 기반 해상교통 분석 · GIS 시각화 · AI 이상 항적 탐지"
)

REGION_NAMES = ["전체"] + [r["name"] for r in REGIONS]


def valid_coordinates(df):
    """지도에 표시할 수 있는 유효 좌표만 남긴다."""
    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        return df.iloc[0:0].copy()

    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude"])
    out = out[
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    ]
    return out



def create_map(df, title="AIS vessel positions", color_by=None):
    """AIS 위치를 품질 상태별 색상으로 표시하고 범례를 추가한다."""
    points = valid_coordinates(df)

    if points.empty:
        st.info("지도에 표시할 유효한 AIS 좌표가 없습니다.")
        return

    center = [
        float(points["latitude"].median()),
        float(points["longitude"].median()),
    ]

    m = folium.Map(
        location=center,
        zoom_start=3,
        tiles="OpenStreetMap",
    )

    for region in REGIONS:
        folium.Rectangle(
            bounds=[
                [region["south"], region["west"]],
                [region["north"], region["east"]],
            ],
            color="#888888",
            weight=1,
            fill=False,
            tooltip=f"수집 영역: {region['name']}",
        ).add_to(m)

    color_map = {
        "ok": "#1769ff",
        "suspected_land": "#e53935",
        "unverified": "#f9a825",
        "이상 항적": "#8e44ad",
        "정상 항적": "#1769ff",
    }

    # 품질 상태를 알아보기 쉽게 표시한다.
    def get_label(row):
        if color_by and color_by in row.index:
            value = row[color_by]
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        return "unverified"

    for _, row in points.iterrows():
        name = row.get("ship_name", "Unknown vessel")
        if pd.isna(name) or not str(name).strip():
            name = "Unknown vessel"

        mmsi = row.get("mmsi", "N/A")
        region = row.get("region", "N/A")
        speed = row.get("sog_knots", np.nan)
        course = row.get("cog_deg", np.nan)

        label = get_label(row)
        color = color_map.get(label, "#f9a825")

        popup = (
            f"<b>{name}</b><br>"
            f"MMSI: {mmsi}<br>"
            f"Region: {region}<br>"
            f"Speed: {speed} kn<br>"
            f"Course: {course}<br>"
            f"Label: {label}"
        )

        folium.CircleMarker(
            location=[
                float(row["latitude"]),
                float(row["longitude"]),
            ],
            radius=5 if label == "이상 항적" else 4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{name} | {label}",
            popup=folium.Popup(popup, max_width=320),
        ).add_to(m)


    st.subheader(title)
    st_folium(
        m,
        use_container_width=True,
        height=560,
        returned_objects=[],
    )



def prepare_positions(df):
    """공통 데이터 형식과 날짜·숫자 열을 정리한다."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if "region" in out.columns:
        region_names = {r["id"]: r["name"] for r in REGIONS}
        out["region"] = out["region"].replace(region_names)

    for col in ["latitude", "longitude", "sog_knots", "cog_deg"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["received_at_utc", "ais_timestamp"]:
        if col in out.columns:
            out[col] = pd.to_datetime(
                out[col],
                format="mixed",
                errors="coerce",
                utc=True
            )

    if "received_at_utc" in out.columns:
        out = out.sort_values("received_at_utc")

    return out


def add_movement_features(df):
    """
    MMSI별 연속 위치를 이용해 대략적인 이동 속도를 계산한다.
    AIS 수신 간격과 위치 오차 때문에 근사치로만 해석해야 한다.
    """
    out = prepare_positions(df)

    required = {
        "mmsi", "latitude", "longitude", "received_at_utc"
    }
    if out.empty or not required.issubset(out.columns):
        return out

    out = out.dropna(
        subset=["mmsi", "latitude", "longitude", "received_at_utc"]
    ).copy()

    out = out.sort_values(["mmsi", "received_at_utc"])

    grouped = out.groupby("mmsi", sort=False)

    prev_lat = grouped["latitude"].shift(1)
    prev_lon = grouped["longitude"].shift(1)
    prev_time = grouped["received_at_utc"].shift(1)

    lat1 = np.radians(prev_lat)
    lat2 = np.radians(out["latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(out["longitude"] - prev_lon)

    haversine_a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    haversine_a = haversine_a.clip(0, 1)

    distance_km = 6371.0 * 2 * np.arcsin(np.sqrt(haversine_a))
    elapsed_hours = (
        out["received_at_utc"] - prev_time
    ).dt.total_seconds() / 3600

    out["movement_speed_knots"] = (
        distance_km * 0.539957 / elapsed_hours
    ).where(elapsed_hours > 0)

    if "sog_knots" not in out.columns:
        out["sog_knots"] = np.nan

    if "cog_deg" not in out.columns:
        out["cog_deg"] = np.nan

    out["speed_difference"] = (
        out["sog_knots"] - out["movement_speed_knots"]
    ).abs()

    # 진행 방향은 0도와 360도가 같은 방향이므로 sin/cos로 변환
    angle = np.radians(out["cog_deg"])
    out["course_sin"] = np.sin(angle)
    out["course_cos"] = np.cos(angle)

    # 비현실적으로 큰 값은 위치 오류 또는 수신 간격 문제일 수 있다.
    # 모델 입력을 안정화하기 위해 상한을 둔다.
    out["movement_speed_knots"] = (
        out["movement_speed_knots"].clip(0, 80)
    )
    out["speed_difference"] = out["speed_difference"].clip(0, 80)

    return out


# --------------------------------------------------
# 2. 탭 구성
# --------------------------------------------------


tab_live, tab_index, tab_history, tab_ai = st.tabs(
    [
        "🚢 실시간 모니터링",
        "📈 리스크 지수",
        "📊 과거 데이터 분석",
        "🤖 AI 이상탐지",
    ]
)

# --------------------------------------------------
# 3. 실시간 모니터링
# --------------------------------------------------

with tab_live:
    st.subheader("실시간 AIS 모니터링")

    st_autorefresh(interval=300_000, key="refresh_live_5m")

    with st.sidebar:
        st.markdown("### 실시간 조회 설정")
        live_minutes = st.selectbox(
            "조회 범위",
            [15, 60, 180, 720],
            index=1,
            format_func=lambda x: f"최근 {x}분",
        )
        live_region = st.selectbox(
            "해협 / 지역",
            REGION_NAMES,
            key="live_region",
        )
        live_limit = st.slider(
            "최대 표시 메시지 수",
            min_value=100,
            max_value=10000,
            value=2500,
            step=100,
        )

    live_df = read_positions(live_minutes, live_limit * 5)
    live_df = prepare_positions(live_df)

    last_received, total_count = latest_status()

    if not live_df.empty and live_region != "전체":
        if "region" in live_df.columns:
            live_df = live_df[live_df["region"] == live_region]

    metric_cols = st.columns(4)
    metric_cols[0].metric("조회 메시지", f"{len(live_df):,}")

    if "mmsi" in live_df.columns:
        metric_cols[1].metric(
            "고유 MMSI", f"{live_df['mmsi'].nunique():,}"
        )
    else:
        metric_cols[1].metric("고유 MMSI", "N/A")

    if (
        not live_df.empty
        and "received_at_utc" in live_df.columns
        and "mmsi" in live_df.columns
    ):
        latest_by_ship = live_df.groupby("mmsi").tail(1)
        fresh_count = int(
            (
                datetime.now(timezone.utc)
                - latest_by_ship["received_at_utc"]
            ).dt.total_seconds().le(900).sum()
        )
    else:
        fresh_count = 0

    metric_cols[2].metric("최근 15분 내 수신 선박", f"{fresh_count:,}")
    metric_cols[3].metric("DB 전체 메시지", f"{total_count:,}")

    st.caption(f"마지막 DB 수신 시각(UTC): {last_received or '없음'}")

    if live_df.empty:
        st.warning(
            "선택한 조회 범위에 AIS 데이터가 없습니다. "
            "collector.py 실행 상태 또는 조회 조건을 확인하세요. "
            "과거 데이터 분석과 AI 탭은 별도로 이용할 수 있습니다."
        )
    else:
        if "quality_flag" in live_df.columns:
            st.write("실시간 AIS 품질 상태별 건수")
            st.write(live_df["quality_flag"].value_counts(dropna=False))

            st.info(
                "AIS 품질 표시는 데이터 검토용입니다. "
                "의심 좌표만으로 선박의 의도나 위험 행위를 단정할 수 없습니다."
            )

        latest_cols = [
            c for c in
            ["mmsi", "received_at_utc", "ship_name", "region",
             "latitude", "longitude", "sog_knots", "cog_deg",
             "quality_flag"]
            if c in live_df.columns
        ]

        if "mmsi" in live_df.columns and "received_at_utc" in live_df.columns:
            latest_positions = (
                live_df.sort_values("received_at_utc")
                .groupby("mmsi", dropna=False)
                .tail(1)
            )
        else:
            latest_positions = live_df

        create_map(
            latest_positions,
             "최신 선박 위치",
             color_by="quality_flag",
        )

        st.markdown(
            """
            **지도 범례**

            <span style="color:#1769ff;">●</span> 정상 (ok) &nbsp;&nbsp;
            <span style="color:#e53935;">●</span> 육지 위치 의심 &nbsp;&nbsp;
            <span style="color:#f9a825;">●</span> 미검증 데이터
            """,
            unsafe_allow_html=True,
        )

        st.subheader("최근 AIS 메시지")
        st.dataframe(
            live_df[latest_cols].sort_values(
                "received_at_utc", ascending=False
            ) if "received_at_utc" in latest_cols
            else live_df[latest_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "실시간 조회 데이터 CSV 다운로드",
            data=live_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="porti_live_ais.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# 4. 리스크 지수
# --------------------------------------------------

with tab_index:
    st.subheader("해상 물류 리스크 지수")
    
    st.markdown("### 호르무즈 주변 해상 사건")

    incident_path = "data/maritime_incidents.csv"

    try:
        incident_df = pd.read_csv(incident_path)

        incident_df["event_date"] = pd.to_datetime(
            incident_df["event_date"],
            errors="coerce",
        )

        st.metric(
            "등록된 사건 수",
            f"{len(incident_df):,}건",
        )

        display_incidents = incident_df.sort_values(
            "event_date",
            ascending=False,
        ).rename(
            columns={
                "event_date": "사건 날짜",
                "ship_name": "선박명",
                "imo_number": "IMO 번호",
                "location": "기록된 위치",
                "event_description": "사건 내용",
                "region": "분석 해역",
                "source_url": "출처",
            }
        )

        st.dataframe(
            display_incidents,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "초기 수동 등록 데이터입니다. 전체 사건 목록이 아니며 "
            "보험료 산출에 사용하기 전 추가 검증이 필요합니다."
        )

    except FileNotFoundError:
        st.warning(
            "사건 데이터 파일을 찾을 수 없습니다. "
            "data/maritime_incidents.csv 경로를 확인해 주세요."
        )

    st.caption(
        "초기 분석 화면입니다. 통항량 감소 점수는 "
        "전쟁위험 전체를 의미하지 않습니다."
    )

    index_limit = st.selectbox(
        "분석 데이터 범위",
        [10000, 30000, 100000, 200000],
        index=2,
        format_func=lambda x: f"{x:,}개",
        key="index_history_limit",
    )

    index_raw = read_all_positions(index_limit)
    index_df = prepare_positions(index_raw)

    if index_df.empty:
        st.warning("분석할 AIS 데이터가 없습니다.")
    else:
        index_df = index_df.dropna(
            subset=["received_at_utc", "mmsi", "region"]
        ).copy()

        index_df["date"] = (
            index_df["received_at_utc"].dt.floor("D")
        )

        daily_vessels = (
            index_df.groupby(["date", "region"])["mmsi"]
            .nunique()
            .rename("observed_vessels")
            .reset_index()
        )
        
        st.markdown("### 해상 사건 전후 AIS 관측 변화")

        try:
            event_analysis_df = pd.read_csv(
                "data/maritime_incidents.csv"
            )

            event_analysis_df["event_date"] = pd.to_datetime(
                event_analysis_df["event_date"],
                errors="coerce",
                utc=True,
            ).dt.floor("D")

            # 호르무즈 관련 AIS 데이터만 추출
            hormuz_ais = index_df[
                index_df["region"]
                .astype(str)
                .str.contains("hormuz", case=False, na=False)
            ].copy()

            if hormuz_ais.empty:
                st.warning(
                    "현재 분석 범위에 호르무즈 AIS 데이터가 없습니다."
                )
            else:
                daily_hormuz = (
                    hormuz_ais.groupby("date")["mmsi"]
                    .nunique()
                )

                daily_hormuz.index = pd.to_datetime(
                    daily_hormuz.index,
                    utc=True,
                )

                comparison_rows = []

                for _, event in event_analysis_df.iterrows():
                    event_date = event["event_date"]

                    if pd.isna(event_date):
                        continue

                    pre_dates = pd.date_range(
                        event_date - pd.Timedelta(days=3),
                        event_date - pd.Timedelta(days=1),
                        freq="D",
                    )

                    post_dates = pd.date_range(
                        event_date + pd.Timedelta(days=1),
                        event_date + pd.Timedelta(days=3),
                        freq="D",
                    )

                    pre_values = daily_hormuz.reindex(pre_dates)
                    post_values = daily_hormuz.reindex(post_dates)

                    pre_days = int(pre_values.notna().sum())
                    post_days = int(post_values.notna().sum())

                    pre_avg = pre_values.mean()
                    post_avg = post_values.mean()

                    if (
                        pre_days >= 2
                        and post_days >= 2
                        and pre_avg > 0
                    ):
                        change_pct = (
                            post_avg / pre_avg - 1
                        ) * 100
                        status = "비교 가능"
                    else:
                        change_pct = None
                        status = "데이터 부족"

                    comparison_rows.append({
                        "사건 날짜": event_date.strftime("%Y-%m-%d"),
                        "선박명": event.get("ship_name", ""),
                        "사건 위치": event.get("location", ""),
                        "사건 전 관측일": f"{pre_days}/3",
                        "사건 후 관측일": f"{post_days}/3",
                        "사건 전 평균 선박 수": (
                            round(pre_avg, 1)
                            if pd.notna(pre_avg) else None
                        ),
                        "사건 후 평균 선박 수": (
                            round(post_avg, 1)
                            if pd.notna(post_avg) else None
                        ),
                        "변화율(%)": (
                            round(change_pct, 1)
                            if change_pct is not None else None
                        ),
                        "판정": status,
                    })

                if comparison_rows:
                    event_comparison = pd.DataFrame(
                        comparison_rows
                    )

                    st.dataframe(
                        event_comparison,
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.warning(
                        "분석할 수 있는 사건 날짜가 없습니다."
                    )

                st.caption(
                    "사건 전 3일과 후 3일의 일별 고유 MMSI 수를 비교합니다. "
                    "사건 당일은 제외하며, 전후 각각 최소 2일의 관측 자료가 "
                    "있을 때만 변화율을 계산합니다. 주변 해역 사건도 포함될 "
                    "수 있으며, 관측 선박 수의 변화만으로 사건의 인과효과나 "
                    "실제 통항량 변화를 확정할 수는 없습니다."
                )

        except FileNotFoundError:
            st.warning(
                "data/maritime_incidents.csv 파일을 찾을 수 없습니다."
            )


        
        st.markdown("### 전쟁위험 보험료율 벤치마크")

        try:
            rate_df = pd.read_csv(
                "data/war_risk_rates.csv"
            )

            rate_df["date"] = pd.to_datetime(
                rate_df["date"],
                errors="coerce",
            )

            rate_df = rate_df.dropna(
                subset=[
                    "date",
                    "region",
                    "rate_low_pct",
                    "rate_high_pct",
                ]
            ).copy()

            rate_df["rate_low_pct"] = pd.to_numeric(
                rate_df["rate_low_pct"],
                errors="coerce",
            )
            rate_df["rate_high_pct"] = pd.to_numeric(
                rate_df["rate_high_pct"],
                errors="coerce",
            )

            rate_df = rate_df.dropna(
                subset=["rate_low_pct", "rate_high_pct"]
            )

            if rate_df.empty:
                st.warning(
                    "표시할 수 있는 보험료율 데이터가 없습니다."
                )
            else:
                # 해역별로 구분해 차트 표시
                
                for region_name in sorted(
                    rate_df["region"].dropna().unique()
                ):
                    region_rates = rate_df[
                        rate_df["region"].str.lower()
                        == region_name.lower()
                    ].sort_values("date")

                    if region_rates.empty:
                        continue

                    region_labels = {
                        "hormuz": "호르무즈",
                        "bab_el_mandeb": "밥엘만데브",
                        "malacca": "말라카",
                        "red_sea": "홍해",
                        "suez": "수에즈",
                    }

                    region_label = region_labels.get(
                        region_name.lower(),
                        region_name,
                    )

                    st.markdown(f"#### {region_label}")

                    chart_df = region_rates.set_index("date")[
                        ["rate_low_pct", "rate_high_pct"]
                    ].rename(
                        columns={
                            "rate_low_pct": "최저 요율(%)",
                            "rate_high_pct": "최고 요율(%)",
                        }
                    )

                    st.line_chart(chart_df)
                    st.dataframe(
                        region_rates[
                            [
                                "date",
                                "rate_low_pct",
                                "rate_high_pct",
                                "source",
                                "estimate_type",
                                "notes",
                            ]
                        ].rename(
                            columns={
                                "date": "관측일",
                                "rate_low_pct": "최저 요율(%)",
                                "rate_high_pct": "최고 요율(%)",
                                "source": "출처",
                                "estimate_type": "추정 유형",
                                "notes": "비고",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    
                st.markdown("#### 출처별 요율 기록")

                display_rates = rate_df.rename(
                    columns={
                        "date": "기준일",
                        "region": "해역",
                        "rate_low_pct": "최저 요율(%)",
                        "rate_high_pct": "최고 요율(%)",
                        "source": "출처",
                        "estimate_type": "자료 유형",
                        "notes": "비고",
                    }
                )

                st.dataframe(
                    display_rates,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "요율은 선체가액 대비 항해 1회당 전쟁위험 추가 보험료의 "
                    "보도·추정 벤치마크입니다. 연속 시계열이나 실제 계약 "
                    "요율이 아니며, 자료 유형과 출처별 차이를 고려해야 합니다."
                )

        except FileNotFoundError:
            st.warning(
                "data/war_risk_rates.csv 파일을 찾을 수 없습니다."
            )



        st.markdown("### 일별 관측 선박 수")

        vessel_chart = daily_vessels.pivot(
            index="date",
            columns="region",
            values="observed_vessels",
        ).sort_index()

        st.line_chart(vessel_chart)

        st.caption(
            "관측 선박 수는 해당 날짜에 AIS 데이터에서 확인된 "
            "고유 선박 수입니다. 실제 통항량과는 다를 수 있습니다."
        )

        
        st.markdown("### 해역별 데이터 현황")

        region_summary = (
            index_df.groupby("region")
            .agg(
                관측_건수=("mmsi", "size"),
                고유_선박_수=("mmsi", "nunique"),
                최초_관측일=("received_at_utc", "min"),
                최종_관측일=("received_at_utc", "max"),
                관측_날짜_수=("date", "nunique"),
            )
            .reset_index()
            .rename(columns={"region": "해역"})
        )

        region_summary["최초_관측일"] = (
            region_summary["최초_관측일"].dt.strftime("%Y-%m-%d")
        )
        region_summary["최종_관측일"] = (
            region_summary["최종_관측일"].dt.strftime("%Y-%m-%d")
        )

        region_summary = region_summary.rename(
            columns={
                "관측_건수": "관측 건수",
                "고유_선박_수": "고유 선박 수",
                "최초_관측일": "최초 관측일",
                "최종_관측일": "최종 관측일",
                "관측_날짜_수": "관측 날짜 수",
            }
        )

        st.dataframe(
            region_summary,
            width="stretch",
            hide_index=True,
        )

        
        st.markdown("### 날짜별 AIS 데이터 공백")

        # 날짜·해역별 실제 관측 건수
        daily_coverage = (
            index_df.groupby(["date", "region"])
            .agg(
                관측_건수=("mmsi", "size"),
                고유_선박_수=("mmsi", "nunique"),
            )
            .reset_index()
        )

        # 분석 기간의 모든 날짜와 해역 조합 생성
        coverage_dates = pd.date_range(
            start=index_df["date"].min(),
            end=index_df["date"].max(),
            freq="D",
        )

        coverage_regions = sorted(
            index_df["region"].dropna().unique()
        )

        full_coverage = (
            pd.MultiIndex.from_product(
                [coverage_dates, coverage_regions],
                names=["date", "region"],
            )
            .to_frame(index=False)
        )

        # 실제 데이터와 결합
        full_coverage = full_coverage.merge(
            daily_coverage,
            on=["date", "region"],
            how="left",
        )

        # 데이터가 없는 날짜를 0건으로 표시하되,
        # 실제 통항량이 0이었다는 의미로 해석하지 않음
        full_coverage["관측_건수"] = (
            full_coverage["관측_건수"]
            .fillna(0)
            .astype(int)
        )

        full_coverage["상태"] = full_coverage[
            "관측_건수"
        ].apply(
            lambda x: "데이터 공백" if x == 0 else "관측 데이터 있음"
        )

        gap_df = full_coverage[
            full_coverage["관측_건수"] == 0
        ].copy()

        if gap_df.empty:
            st.success("선택한 분석 기간에 날짜별 데이터 공백이 없습니다.")
        else:
            gap_summary = (
                gap_df.groupby("region")
                .agg(공백_날짜_수=("date", "nunique"))
                .reset_index()
                .rename(columns={"region": "해역"})
            )

            gap_summary["해역별 공백 날짜"] = gap_summary[
                "해역"
            ].map(
                gap_df.groupby("region")["date"].apply(
                    lambda dates: ", ".join(
                        dates.dt.strftime("%Y-%m-%d").tolist()
                    )
                )
            )

            gap_summary = gap_summary.rename(
                columns={"공백_날짜_수": "공백 날짜 수"}
            )

            st.dataframe(
                gap_summary,
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "분석에 포함된 데이터 범위 내에서 관측 기록이 없는 날짜입니다. "
            "실제 통항량이 0이었다는 뜻은 아닙니다."
        )

        
        st.markdown("### 관측 건수 급감 탐지")

        # 날짜별·해역별 AIS 메시지 건수
        daily_messages = (
            index_df.groupby(["date", "region"])
            .size()
            .rename("관측_건수")
            .reset_index()
        )

        # 모든 날짜와 해역 조합 생성
        message_dates = pd.date_range(
            start=index_df["date"].min(),
            end=index_df["date"].max(),
            freq="D",
            tz="UTC",
        )

        daily_messages["date"] = pd.to_datetime(
            daily_messages["date"], utc=True
        )

        message_regions = sorted(
            index_df["region"].dropna().unique()
        )

        message_grid = (
            pd.MultiIndex.from_product(
                [message_dates, message_regions],
                names=["date", "region"],
            )
            .to_frame(index=False)
        )

        message_grid = message_grid.merge(
            daily_messages,
            on=["date", "region"],
            how="left",
        )

        # 여기서 0은 관측 기록이 없다는 의미이며,
        # 실제 통항량이 0이라는 뜻은 아님
        message_grid["관측_건수"] = (
            message_grid["관측_건수"].fillna(0)
        )

        # 해역별 직전 7일 중앙값 계산
        message_grid = message_grid.sort_values(
            ["region", "date"]
        )

        message_grid["기준_관측_건수"] = (
            message_grid.groupby("region")["관측_건수"]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=7, min_periods=3
                ).median()
            )
        )

        # 기준값이 양수인 날짜만 감소율 계산
        message_grid["감소율(%)"] = (
            (
                1
                - message_grid["관측_건수"]
                / message_grid["기준_관측_건수"]
            )
            .where(message_grid["기준_관측_건수"] > 0)
            .clip(lower=0)
            .mul(100)
        )

        # 급감 기준: 직전 기준 대비 50% 이상 감소
        # 데이터 공백은 별도 상태로 구분
        message_grid["상태"] = "정상 범위"

        message_grid.loc[
            message_grid["관측_건수"] == 0,
            "상태",
        ] = "데이터 공백"

        message_grid.loc[
            (
                (message_grid["관측_건수"] > 0)
                & (message_grid["감소율(%)"] >= 50)
            ),
            "상태",
        ] = "관측 건수 급감"

        message_grid.loc[
            message_grid["기준_관측_건수"].isna(),
            "상태",
        ] = "기준 데이터 부족"

        # 최근 날짜부터 확인할 수 있도록 정렬
        alert_df = message_grid[
            message_grid["상태"].isin(
                ["관측 건수 급감", "데이터 공백"]
            )
        ].sort_values("date", ascending=False)

        if alert_df.empty:
            st.info("현재 분석 범위에서 급감 또는 데이터 공백이 탐지되지 않았습니다.")
        else:
            st.dataframe(
                alert_df[
                    [
                        "date",
                        "region",
                        "관측_건수",
                        "기준_관측_건수",
                        "감소율(%)",
                        "상태",
                    ]
                ].rename(
                    columns={
                        "date": "날짜",
                        "region": "해역",
                        "관측_건수": "관측 건수",
                        "기준_관측_건수": "직전 7일 기준값",
                        "감소율(%)": "감소율(%)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "급감은 직전 7일 관측 건수 중앙값보다 50% 이상 감소한 경우입니다. "
            "데이터 수집 장애와 실제 통항량 감소를 구분하는 확정 판정은 아닙니다."
        )



        
        st.markdown("### 통항량 감소 점수")

        # 날짜별·해역별 관측 선박 수를 시계열로 변환
        daily_series = daily_vessels.pivot(
            index="date",
            columns="region",
            values="observed_vessels",
        ).sort_index()

        # 관측 데이터가 없는 날짜는 0으로 간주하지 않음
        full_dates = pd.date_range(
            start=daily_series.index.min(),
            end=daily_series.index.max(),
            freq="D",
            tz="UTC",
        )
        daily_series.index = pd.to_datetime(
            daily_series.index, utc=True
        )
        daily_series = daily_series.reindex(full_dates)

        # 직전 7일의 관측 선박 수 중앙값을 기준으로 비교
        baseline = daily_series.shift(1).rolling(
            window=7, min_periods=3
        ).median()

        # 기준 대비 감소율을 0~100점으로 변환
        decline_score = (
            (1 - daily_series.div(baseline))
            .clip(lower=0, upper=1)
            * 100
        )

        
        st.markdown("### 급감 탐지와 통항량 감소 점수 비교")

        # 날짜·해역별 감소 점수를 표 형태로 변환
        score_table = (
            decline_score
            .rename_axis(index="date", columns="region")
            .stack()
            .rename("통항량 감소 점수")
            .reset_index()
        )

        # 급감 탐지 결과와 점수 결합
        comparison_df = message_grid.merge(
            score_table,
            on=["date", "region"],
            how="left",
        )

        # 급감 또는 데이터 공백이 발생한 날짜만 확인
        comparison_df = comparison_df[
            comparison_df["상태"].isin(
                ["관측 건수 급감", "데이터 공백"]
            )
        ].sort_values("date", ascending=False)

        if comparison_df.empty:
            st.info("비교할 급감 또는 데이터 공백 사례가 없습니다.")
        else:
            display_comparison = comparison_df[
                [
                    "date",
                    "region",
                    "관측_건수",
                    "감소율(%)",
                    "통항량 감소 점수",
                    "상태",
                ]
            ].rename(
                columns={
                    "date": "날짜",
                    "region": "해역",
                    "관측_건수": "관측 건수",
                    "감소율(%)": "관측 건수 감소율(%)",
                }
            )

            st.dataframe(
                display_comparison,
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "두 지표는 서로 다른 기준으로 계산됩니다. "
            "점수가 높더라도 실제 전쟁위험이 높다고 단정할 수 없습니다."
        )

        
        st.markdown("### 통항량 감소 점수 신뢰도")

        # 기본적으로 분석 가능 상태로 설정
        reliability = pd.DataFrame(
            "분석 가능",
            index=daily_series.index,
            columns=daily_series.columns,
        )

        # 당일 관측 데이터가 없는 경우
        reliability = reliability.mask(
            daily_series.isna(),
            "데이터 공백",
        )

        # 직전 7일 기준값을 계산하기에 데이터가 부족한 경우
        reliability = reliability.mask(
            baseline.isna() & daily_series.notna(),
            "기준 데이터 부족",
        )

        # 기준값 대비 관측 선박 수가 50% 이상 감소한 경우
        reliability = reliability.mask(
            (decline_score >= 50)
            & daily_series.notna()
            & baseline.notna(),
            "급감 확인 필요",
        )

        # 날짜·해역별 표로 변환
        reliability_table = (
            reliability
            .rename_axis(index="날짜", columns="해역")
            .stack()
            .rename("판정")
            .reset_index()
        )

        st.dataframe(
            reliability_table.sort_values(
                "날짜",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "분석 가능은 데이터 조건을 충족한다는 뜻이며, "
            "실제 전쟁위험이나 보험료 예측의 정확성이 검증됐다는 뜻은 아닙니다."
        )


        st.line_chart(decline_score)

        st.caption(
            "점수가 높을수록 직전 관측 기간의 기준 수준보다 "
            "관측 선박 수가 크게 감소했다는 뜻입니다. "
            "기준 데이터가 부족하거나 해당 날짜의 관측값이 없으면 "
            "점수를 계산하지 않습니다."
        )

        
        st.markdown("### 위험 신호와 시장 보험료율 비교")
        

        # 호르무즈의 최신 통항량 감소 점수
        
        # 호르무즈의 최신 통항량 감소 점수
        hormuz_column = next(
            (
                col for col in decline_score.columns
                if (
                    "hormuz" in str(col).lower()
                    or "호르무즈" in str(col)
                )
            ),
            None,
        )


        if hormuz_column is not None:
            hormuz_scores = decline_score[hormuz_column].dropna()

            if not hormuz_scores.empty:
                latest_score = float(hormuz_scores.iloc[-1])
                st.metric(
                    "호르무즈 최신 통항량 감소 점수",
                    f"{latest_score:.1f}/100",
                )
            else:
                latest_score = None
                st.warning(
                    "호르무즈 감소 점수를 계산할 데이터가 부족합니다."
                )
        else:
            latest_score = None
            st.warning(
                "현재 분석 범위에 호르무즈 데이터가 없습니다."
            )


        # 시장 요율 자료 읽기
        try:
            benchmark_df = pd.read_csv(
                "data/war_risk_rates.csv"
            )

            benchmark_df["date"] = pd.to_datetime(
                benchmark_df["date"],
                errors="coerce",
            )

            benchmark_df["rate_low_pct"] = pd.to_numeric(
                benchmark_df["rate_low_pct"],
                errors="coerce",
            )

            benchmark_df["rate_high_pct"] = pd.to_numeric(
                benchmark_df["rate_high_pct"],
                errors="coerce",
            )

            hormuz_benchmarks = benchmark_df[
                (benchmark_df["region"].str.lower() == "hormuz")
                & benchmark_df["rate_low_pct"].notna()
                & benchmark_df["rate_high_pct"].notna()
            ].sort_values("date")

            if hormuz_benchmarks.empty:
                st.warning("호르무즈 보험료율 벤치마크가 없습니다.")
            else:
                st.markdown("#### 비교할 시장 벤치마크")

                benchmark_options = {
                    f"{row['date']:%Y-%m-%d} | "
                    f"{row['source']} | "
                    f"{row['rate_low_pct']:g}~{row['rate_high_pct']:g}% "
                    f"({row['estimate_type']})": idx
                    for idx, row in hormuz_benchmarks.iterrows()
                }

                selected_label = st.selectbox(
                    "비교 기준 선택",
                    list(benchmark_options.keys()),
                    index=len(benchmark_options) - 1,
                    key="hormuz_rate_benchmark",
                )

                
                selected_label = st.selectbox(
                    "보험료율 관측치 선택",
                    options=list(benchmark_options.keys()),
                )


                selected_row = hormuz_benchmarks.loc[
                    benchmark_options[selected_label]
                ]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "보도된 최저 요율",
                        f"{selected_row['rate_low_pct']:g}%",
                    )

                with col2:
                    st.metric(
                        "보도된 최고 요율",
                        f"{selected_row['rate_high_pct']:g}%",
                    )

                st.caption(
                    f"출처: {selected_row['source']} · "
                    f"자료 유형: {selected_row['estimate_type']}"
                )

                
                # 선택한 보험료율 관측일과 AIS 감소 점수 연결
                benchmark_date = pd.Timestamp(selected_row["date"])

                if benchmark_date.tzinfo is None:
                    benchmark_date = benchmark_date.tz_localize("UTC")
                else:
                    benchmark_date = benchmark_date.tz_convert("UTC")

                if hormuz_column is not None:
                    score_series = (
                        decline_score[hormuz_column]
                        .dropna()
                        .copy()
                    )

                    score_series.index = pd.to_datetime(
                        score_series.index,
                        utc=True,
                    )

                    # 보험료율 관측일 당일 또는 그 이전의 최신 점수 선택
                    available_scores = score_series[
                        score_series.index <= benchmark_date
                    ]
                    st.write("보험료율 관측일:", benchmark_date)
                    st.write("AIS 점수 시작일:", score_series.index.min())
                    st.write("AIS 점수 마지막 날:", score_series.index.max())
                    st.write("AIS 점수 관측치 수:", len(score_series))

                    if available_scores.empty:
                        st.warning(
                            "선택한 보험료율 관측일 이전의 "
                            "AIS 감소 점수를 찾을 수 없습니다."
                        )
                        if latest_score is not None:
                            st.metric(
                                "참고: 최신 AIS 통항량 감소 점수",
                                f"{latest_score:.1f}/100",
                            )
                            st.caption(
                                "위 점수는 최신 AIS 관측 기준입니다. "
                                "선택한 과거 보험료율 관측일과 동일 시점의 "
                                "값이 아니므로 직접적인 시점 비교에는 사용할 수 없습니다."
                            )
                    else:
                        matched_date = available_scores.index[-1]
                        matched_score = float(
                            available_scores.iloc[-1]
                        )

                        age_days = (
                            benchmark_date.normalize()
                            - matched_date.normalize()
                        ).days

                        if age_days <= 7:
                            st.metric(
                                "보험료율 관측일 기준 AIS 감소 점수",
                                f"{matched_score:.1f}/100",
                            )
                            st.caption(
                                f"보험료율 관측일: "
                                f"{benchmark_date:%Y-%m-%d} · "
                                f"AIS 점수 기준일: "
                                f"{matched_date:%Y-%m-%d}"
                            )
                        else:
                            st.warning(
                                f"가장 최근 AIS 점수는 "
                                f"{matched_date:%Y-%m-%d} 기준으로, "
                                f"보험료율 관측일보다 {age_days}일 이전입니다. "
                                "직접 비교하기에는 데이터 간격이 큽니다."
                            )


                st.caption(
                    "위 요율은 특정 시점에 보도된 시장 벤치마크입니다. "
                    "현재 적용 가능한 보험사 견적이나 개별 선박의 보험료를 "
                    "의미하지 않습니다."
                )

        except FileNotFoundError:
            st.warning("data/war_risk_rates.csv 파일을 찾을 수 없습니다.")


        


# --------------------------------------------------
# 5. 과거 데이터 분석
# --------------------------------------------------

with tab_history:
    st.subheader("과거 AIS 데이터 분석")

    history_limit = st.selectbox(
        "분석할 최대 레코드 수",
        [20000, 50000, 100000, 200000],
        index=3,
        format_func=lambda x: f"{x:,}건",
        key="history_limit",
    )

    raw_hist_df = prepare_positions(read_all_positions(history_limit))

    st.write("원본 조회 행 수:", len(raw_hist_df))

    if not raw_hist_df.empty:
        st.write(
            "원본 데이터 최초 시각:",
            raw_hist_df["received_at_utc"].min()
        )
        st.write(
            "원본 데이터 최종 시각:",
            raw_hist_df["received_at_utc"].max()
        )

    hist_df = prepare_positions(raw_hist_df)

    if hist_df.empty:
        st.warning("분석할 과거 AIS 데이터가 없습니다.")
    else:
        hist_filtered = hist_df.copy()

        # 지역별 비교는 단일 지역 필터를 적용하기 전에 계산
        if (
            "received_at_utc" in hist_filtered.columns
            and "region" in hist_filtered.columns
        ):
            daily_base = hist_filtered.dropna(
                subset=["received_at_utc"]
            ).copy()

            if not daily_base.empty:
                daily_base["date"] = (
                    daily_base["received_at_utc"].dt.floor("D")
                )

                all_days = pd.date_range(
                    daily_base["date"].min().floor("D"),
                    daily_base["date"].max().floor("D"),
                    freq="D",
                )

                region_order = [
                    r["name"] for r in REGIONS
                    if r["name"] in daily_base["region"].dropna().unique()
                ]
                if "Other" in daily_base["region"].dropna().unique():
                    region_order.append("Other")

                st.markdown("#### 해역별 일일 AIS 메시지 수 비교")
                daily_region_messages = (
                    daily_base.groupby(["date", "region"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=all_days, columns=region_order, fill_value=0)
                )
                st.line_chart(daily_region_messages)

                if "mmsi" in daily_base.columns:
                    st.markdown("#### 해역별 일일 관측 선박 수")
                    daily_region_vessels = (
                        daily_base.groupby(["date", "region"])["mmsi"]
                        .nunique()
                        .unstack(fill_value=0)
                        .reindex(
                            index=all_days,
                            columns=region_order,
                            fill_value=0,
                        )
                    )
                    st.line_chart(daily_region_vessels)

                st.markdown("#### 날짜별 해역 관측 공백")
                coverage = daily_region_messages.gt(0).replace(
                    {True: "관측", False: "공백"}
                )
                st.dataframe(
                    coverage,
                    use_container_width=True,
                )
                st.caption(
                    "‘공백’은 해당 날짜에 DB 관측 기록이 없다는 뜻입니다. "
                    "선박 통행이 없었다거나 수집기가 중단됐다고 단정할 수 없습니다."
                )

        if "received_at_utc" in hist_filtered.columns:
            hist_filtered = hist_filtered.dropna(
                subset=["received_at_utc"]
            )

            if not hist_filtered.empty:
                date_values = hist_filtered["received_at_utc"].dropna().dt.date

                min_date=min(date_values)
                max_date=max(date_values)

                st.write("날짜 데이터 확인")
                st.write("최초 날짜:", min_date)
                st.write("최종 날짜:", max_date)
                st.write("날짜별 레코드 수:")
                st.write(
                    hist_filtered.assign(
                        date=hist_filtered["received_at_utc"].dt.date
                    ).groupby("date").size()
                )

                date_range = st.date_input(
                    "분석 기간 (UTC 기준)",
                    value=(min_date,max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="history_dates_v2",
                )
                

                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                    start_date, end_date = date_range
                    hist_filtered = hist_filtered[
                        hist_filtered["received_at_utc"].dt.date.between(
                            start_date, end_date
                        )
                    ]

        if "region" in hist_filtered.columns:
            region_options = [
                r for r in REGION_NAMES
                if r == "전체" or r in hist_filtered["region"].dropna().unique()
            ]
            selected_region = st.selectbox(
                "분석 지역",
                region_options,
                key="history_region",
            )
            if selected_region != "전체":
                hist_filtered = hist_filtered[
                    hist_filtered["region"] == selected_region
                ]

        if hist_filtered.empty:
            st.warning("선택한 기간과 지역에 해당하는 데이터가 없습니다.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("메시지 수", f"{len(hist_filtered):,}")

            if "mmsi" in hist_filtered.columns:
                c2.metric(
                    "고유 MMSI",
                    f"{hist_filtered['mmsi'].nunique():,}",
                )
            else:
                c2.metric("고유 MMSI", "N/A")

            if "region" in hist_filtered.columns:
                c3.metric(
                    "관측 지역 수",
                    f"{hist_filtered['region'].nunique():,}",
                )
            else:
                c3.metric("관측 지역 수", "N/A")

            if "sog_knots" in hist_filtered.columns:
                speed = pd.to_numeric(
                    hist_filtered["sog_knots"], errors="coerce"
                )
                c4.metric(
                    "평균 보고 속도",
                    f"{speed.mean():.2f} kn" if speed.notna().any() else "N/A",
                )
            else:
                c4.metric("평균 보고 속도", "N/A")

            if "received_at_utc" in hist_filtered.columns:
                st.markdown("#### 일별 AIS 메시지 수")
                daily_messages = (
                    hist_filtered.assign(
                        date=hist_filtered["received_at_utc"].dt.strftime("%Y-%m-%d")
                    )
                    .groupby("date")
                    .size()
                    .rename("메시지 수")
                )
                st.line_chart(daily_messages)

                if "mmsi" in hist_filtered.columns:
                    st.markdown("#### 일별 관측 선박 수")
                    daily_vessels = (
                        hist_filtered.assign(
                            date=hist_filtered["received_at_utc"].dt.strftime("%Y-%m-%d")
                        )
                        .groupby("date")["mmsi"]
                        .nunique()
                        .rename("고유 MMSI")
                    )
                    st.line_chart(daily_vessels)

            if "region" in hist_filtered.columns:
                st.markdown("#### 지역별 메시지 분포")
                region_counts = hist_filtered["region"].value_counts()
                st.bar_chart(region_counts)

            if "sog_knots" in hist_filtered.columns:
                st.markdown("#### 보고 속도 분포")
                speed = pd.to_numeric(
                    hist_filtered["sog_knots"], errors="coerce"
                ).dropna()
                speed = speed[speed.between(0, 80)]
                if not speed.empty:
                    hist, bins = np.histogram(speed, bins=20)
                    speed_chart = pd.DataFrame({
                        "속도 구간(노트)": [
                            f"{bins[i]:.1f}–{bins[i + 1]:.1f}"
                            for i in range(len(bins) - 1)
                        ],
                        "메시지 수": hist,
                    }).set_index("속도 구간(노트)")
                    st.bar_chart(speed_chart)

            create_map(hist_filtered, "과거 AIS 위치 분포")

            st.markdown("#### 분석 데이터")
            st.dataframe(
                hist_filtered.sort_values(
                    "received_at_utc", ascending=False
                ) if "received_at_utc" in hist_filtered.columns
                else hist_filtered,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "과거 분석 데이터 CSV 다운로드",
                data=hist_filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name="porti_historical_ais.csv",
                mime="text/csv",
            )


# --------------------------------------------------
# 6. AI 이상 항적 탐지
# --------------------------------------------------
with tab_ai:
    st.subheader("AI 기반 AIS 이상 항적 탐지")

    st.write(
        "Isolation Forest를 사용해 관측 데이터 중 상대적으로 특이한 "
        "속도·진행 방향·이동 패턴을 탐색합니다."
    )

    st.warning(
        "중요: 이상탐지 결과는 통계적 이상치일 뿐, 적대 행위나 "
        "전쟁 위험을 직접 의미하지 않습니다. AIS 수신 공백, 항만의 "
        "저속 운항, 위치 오차 등도 이상치로 분류될 수 있습니다. "
        "현재는 라벨이 확인된 학습 데이터가 없는 탐색적 분석입니다."
    )

    ai_limit = st.selectbox(
        "AI 분석 데이터 수",
        [5000, 10000, 30000, 50000],
        index=1,
        key="ai_limit",
        format_func=lambda x: f"{x:,}건",
    )

    contamination = st.slider(
        "예상 이상치 비율",
        min_value=0.01,
        max_value=0.15,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="모델이 이상치로 표시할 데이터 비율의 설정값입니다. 실제 위험 발생률이 아닙니다.",
    )

    if st.button("AI 이상탐지 실행", type="primary"):
        with st.spinner("AIS 데이터를 정리하고 모델을 학습하는 중입니다..."):
            ai_df = add_movement_features(
                read_all_positions(ai_limit)
            )

            required = ["sog_knots", "course_sin", "course_cos"]
            missing = [c for c in required if c not in ai_df.columns]

            if missing:
                st.error(
                    "AI 분석에 필요한 열이 없습니다: "
                    + ", ".join(missing)
                )
            elif len(ai_df) < 30:
                st.error(
                    "분석 데이터가 너무 적습니다. 최소 30개 이상의 "
                    "AIS 레코드를 확보한 뒤 다시 시도하세요."
                )
            else:
                feature_cols = [
                    "sog_knots",
                    "course_sin",
                    "course_cos",
                    "movement_speed_knots",
                    "speed_difference",
                ]

                for col in feature_cols:
                    if col not in ai_df.columns:
                        ai_df[col] = np.nan

                model_df = ai_df[feature_cols].copy()
                model_df = model_df.replace([np.inf, -np.inf], np.nan)

                # 속도 및 이동 관련 결측치는 중앙값으로 대체한다.
                for col in feature_cols:
                    median = model_df[col].median()
                    if pd.isna(median):
                        median = 0.0
                    model_df[col] = model_df[col].fillna(median)

                # 모델 입력을 제한해 비정상 값의 영향을 완화한다.
                model_df["sog_knots"] = model_df["sog_knots"].clip(0, 80)
                model_df["movement_speed_knots"] = (
                    model_df["movement_speed_knots"].clip(0, 80)
                )
                model_df["speed_difference"] = (
                    model_df["speed_difference"].clip(0, 80)
                )

                model = IsolationForest(
                    n_estimators=200,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                )

                predictions = model.fit_predict(model_df)
                ai_df["anomaly_score"] = model.decision_function(model_df)
                ai_df["anomaly_label"] = np.where(
                    predictions == -1, "이상 항적", "정상 항적"
                )

                st.session_state["porti_ai_results"] = ai_df

    if "porti_ai_results" in st.session_state:
        result = st.session_state["porti_ai_results"].copy()

        anomaly_df = result[result["anomaly_label"] == "이상 항적"].copy()
        normal_df = result[result["anomaly_label"] == "정상 항적"].copy()

        a, b, c = st.columns(3)
        a.metric("분석 레코드", f"{len(result):,}")
        b.metric("이상치 레코드", f"{len(anomaly_df):,}")
        c.metric(
            "모델 이상치 비율",
            f"{len(anomaly_df) / len(result) * 100:.1f}%"
            if len(result) else "N/A",
        )

        st.caption(
            "이상치 비율은 모델 설정과 입력 데이터에 따라 달라집니다. "
            "이를 실제 위험 선박 비율로 해석하면 안 됩니다."
        )

        st.markdown("#### 이상 항적 위치")
        if anomaly_df.empty:
            st.info("현재 설정에서 이상치로 분류된 레코드가 없습니다.")
        else:
            create_map(
                result,
                "AIS 이상치(보라색)와 정상 항적(파란색)",
                color_by="anomaly_label",
            )

            display_cols = [
                c for c in [
                    "received_at_utc", "ais_timestamp", "mmsi",
                    "ship_name", "region", "latitude", "longitude",
                    "sog_knots", "cog_deg", "movement_speed_knots",
                    "speed_difference", "anomaly_score", "anomaly_label",
                    "quality_flag",
                ]
                if c in anomaly_df.columns
            ]

            st.markdown("#### 이상치 상세 목록")
            st.dataframe(
                anomaly_df[display_cols].sort_values(
                    "anomaly_score", ascending=True
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "AI 이상탐지 결과 CSV 다운로드",
                data=result.to_csv(index=False).encode("utf-8-sig"),
                file_name="porti_ai_anomaly_results.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "아직 AI 분석을 실행하지 않았습니다. 위의 "
            "「AI 이상탐지 실행」 버튼을 눌러 분석을 시작하세요."
        )