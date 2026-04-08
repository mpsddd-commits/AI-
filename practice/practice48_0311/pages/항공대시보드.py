import streamlit as st
import pandas as pd
import plotly.express as px
from db import getConn

st.set_page_config(
    page_title="항공 데이터 분석 대시보드",
    layout="wide"
)

st.title("✈️ 항공 데이터 분석 대시보드")

# --------------------------
# 데이터 불러오기
# --------------------------

@st.cache_data
def load_data():

    conn = getConn()

    dep = pd.read_sql("SELECT * FROM db_air.출발지연횟수", conn)
    arr = pd.read_sql("SELECT * FROM db_air.도착지연횟수", conn)
    route = pd.read_sql("SELECT * FROM db_air.월별인기노선", conn)

    return dep, arr, route


dep_df, arr_df, route_df = load_data()

# --------------------------
# TAB 생성
# --------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⏱ 출발 지연 분석",
    "🛬 도착 지연 분석",
    "🛫 월별 인기 노선",
    "🔗 노선 네트워크"
])

# ====================================================
# TAB1 : KPI
# ====================================================

with tab1:

    st.subheader("항공 데이터 요약")

    total_flights = route_df["cnt"].sum()
    total_routes = route_df["운항노선"].nunique()
    avg_delay_dep = dep_df[['delay_1987','delay_1988','delay_1989']].mean().mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("총 비행 횟수", f"{total_flights:,}")
    col2.metric("운항 노선 수", total_routes)
    col3.metric("평균 출발 지연", f"{avg_delay_dep:.2f}")

# ====================================================
# TAB2 : 출발 지연 분석
# ====================================================

with tab2:

    st.subheader("도시별 평균 출발 지연")

    df_delay = dep_df[['도시','delay_1987','delay_1988','delay_1989']].melt(
        id_vars='도시',
        var_name='년도',
        value_name='지연시간'
    )

    df_delay['년도'] = df_delay['년도'].str.replace("delay_","")

    fig = px.bar(
        df_delay,
        x="도시",
        y="지연시간",
        color="년도",
        barmode="stack"
        
    )

    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# TAB3 : 도착 지연 분석
# ====================================================

with tab3:

    st.subheader("도시별 평균 도착 지연")

    df_delay = arr_df[['도시','delay_1987','delay_1988','delay_1989']].melt(
        id_vars='도시',
        var_name='년도',
        value_name='지연시간'
    )

    df_delay['년도'] = df_delay['년도'].str.replace("delay_","")

    fig = px.bar(
        df_delay,
        x="도시",
        y="지연시간",
        color="년도",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# TAB4 : 인기 노선
# ====================================================

with tab4:

    st.subheader("월별 인기 노선")

    col1, col2 = st.columns(2)

    year = col1.selectbox("년도", sorted(route_df["년도"].unique()))
    month = col2.selectbox("월", sorted(route_df[route_df["년도"] == year]["월"].unique()))

    df_month = route_df[(route_df["년도"] == year) &(route_df["월"] == month)].sort_values(
        "cnt", ascending=False)

    fig = px.bar(
        df_month,
        x="운항노선",
        y="cnt",
        color="운항노선",
        text="cnt"
    )

    st.plotly_chart(fig, use_container_width=True)

# ====================================================
# TAB5 : 노선 네트워크
# ====================================================

with tab5:

    st.subheader("도시 간 항공 노선")

    route_split = route_df.copy()

    route_split[['출발','도착']] = route_split["운항노선"].str.split(" → ", expand=True)

    fig = px.sunburst(
        route_split,
        path=["출발","도착"],
        values="cnt"
    )

    st.plotly_chart(fig, use_container_width=True)