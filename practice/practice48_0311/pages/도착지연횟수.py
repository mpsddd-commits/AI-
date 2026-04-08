import streamlit as st
import pandas as pd
from db import getConn
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="도착 지연 분석 대시보드",
    layout="wide"
)

st.title("✈️ 항공 도착 지연 데이터 분석 대시보드")

# -------------------------
# 데이터 불러오기
# -------------------------

query = """
SELECT 도시, cnt_1987, cnt_1988, cnt_1989,
       delay_1987, delay_1988, delay_1989
FROM db_air.도착지연횟수
ORDER BY 도시
"""

df = pd.read_sql(query, getConn())

# -------------------------
# 데이터 변환
# -------------------------

# delay melt
df_delay = df[['도시','delay_1987','delay_1988','delay_1989']].melt(
    id_vars='도시',
    var_name='년도',
    value_name='지연시간'
)

# cnt melt
df_cnt = df[['도시','cnt_1987','cnt_1988','cnt_1989']].melt(
    id_vars='도시',
    var_name='년도',
    value_name='비행횟수'
)

# 년도 정리
df_delay['년도'] = df_delay['년도'].str.replace('delay_','')
df_cnt['년도'] = df_cnt['년도'].str.replace('cnt_','')

# merge
df_final = pd.merge(df_delay, df_cnt, on=['도시','년도'])

# -------------------------
# 도시 정렬 기준 (비행횟수 평균)
# -------------------------

city_order = (
    df_final.groupby("도시")["비행횟수"]
    .mean()
    .sort_values(ascending=False)
    .index
)

# -------------------------
# 도시 필터
# -------------------------

city_list = st.multiselect(
    "도시 선택",
    options=df_final["도시"].unique(),
    default=df_final["도시"].unique()
)

df_filtered = df_final[df_final["도시"].isin(city_list)]

# -------------------------
# 그래프 1 : 도시별 지연시간
# -------------------------

st.subheader("도시별 평균 도착 지연시간")

fig1 = px.bar(
    df_filtered,
    x="도시",
    y="지연시간",
    color="년도",
    barmode="group",
    hover_data=["비행횟수"],
    category_orders={"도시": city_order},
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig1.update_layout(
    xaxis_title="도시",
    yaxis_title="평균 지연시간 (분)",
    title="도시별 평균 출발 지연시간",
    legend_title="년도",
    title_font_size=20,
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=14)
)

fig1.update_traces(
    texttemplate="%{y:.1f}",
    textposition="outside"
)

st.plotly_chart(fig1, use_container_width=True)


# -------------------------
# 그래프 3 : 비행횟수 vs 지연시간
# -------------------------

st.subheader("도시별 지연분포")

fig3 = px.scatter(
    df_filtered,
    x="비행횟수",
    y="지연시간",
    size="비행횟수",
    color="도시",
    hover_data=["년도"],
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig3.update_layout(
    title="비행횟수 vs 지연시간 관계",
    xaxis_title="비행횟수",
    yaxis_title="평균 지연시간",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=14)
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# 그래프 4 : Heatmap (도시 vs 연도)
# -------------------------

st.subheader("Heatmap 도시 vs 연도")

pivot = df_filtered.pivot(
    index="도시",
    columns="년도",
    values="지연시간"
)

pivot = pivot.loc[city_order]

fig4 = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Reds"
)

fig4.update_layout(
    title="도시 vs 연도 평균 지연 Heatmap",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=14)
)
st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# 그래프 5 : Stacked Bar Chart
# -------------------------

st.subheader("Stacked Bar Chart")

fig5 = px.bar(
    df_filtered,
    x="도시",
    y="지연시간",
    color="년도",
    barmode="stack",
    category_orders={"도시": city_order},
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig5.update_layout(
    title="도시별 연도 지연 누적 비교",
    legend_title="년도",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=14)
)


st.plotly_chart(fig5, use_container_width=True)


# -------------------------
# 데이터 테이블
# -------------------------

st.subheader("데이터 테이블")

st.dataframe(df_filtered, hide_index=True)