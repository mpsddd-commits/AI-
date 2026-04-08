import streamlit as st
import pandas as pd
import plotly.express as px
from db import getConn

st.set_page_config(
    page_title="월별 인기 노선 분석",
    layout="wide"
)

st.title("✈️ 월별 인기 항공 노선 분석")

# ----------------------
# 데이터 가져오기
# ----------------------

query = """
SELECT 운항노선, 년도, 월, cnt
FROM db_air.월별인기노선
ORDER BY 년도, 월
"""

df = pd.read_sql(query, getConn())

# ----------------------
# TAB 생성
# ----------------------

tab1, tab2 = st.tabs(["월별 TOP5 노선", "전체 TOP5 노선"])

# ==============================
# TAB 1 : 월별 인기노선
# ==============================

with tab1:

    st.subheader("년도 / 월 선택")

    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox(
            "년도 선택",
            sorted(df["년도"].unique())
        )

    with col2:
        month = st.selectbox(
            "월 선택",
            sorted(df[df["년도"] == year]["월"].unique())
        )

    # 데이터 필터
    df_month = df[(df["년도"] == year) & (df["월"] == month)].sort_values(
        "cnt", ascending=False
    )
    
    st.subheader(f"{year}년 {month}월 인기 노선 TOP5")

    fig = px.bar(
        df_month,
        x="운항노선",
        y="cnt",
        color="운항노선",
        text="cnt"
    )

    fig.update_layout(
        xaxis_title="운항 노선",
        yaxis_title="비행 횟수"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_month, hide_index=True)


# ==============================
# TAB 2 : 전체 인기노선
# ==============================

with tab2:

    st.subheader("전체 기간 인기 노선 TOP5")

    df_top = (
        df.groupby("운항노선")["cnt"]
        .sum()
        .reset_index()
        .sort_values("cnt", ascending=False)
        .head(5)
    )

    fig2 = px.bar(
        df_top,
        x="운항노선",
        y="cnt",
        color="운항노선",
        text="cnt"
    )

    fig2.update_layout(
        xaxis_title="운항 노선",
        yaxis_title="총 비행 횟수"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df_top, hide_index=True)