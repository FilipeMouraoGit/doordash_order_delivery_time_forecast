import streamlit as st
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards
from doordash_challenge.functions.data_processing.Viewer import DataViewer
from doordash_challenge.functions.data_processing.Transformer import ALLOWED_METRICS
from doordash_challenge.functions.data_processing.utils import *


def build_input_sidebar(data_to_plot):
    st.title("Select the desired report specifications")
    existing_food_categories = list(data_to_plot['store_primary_category'].unique())

    market_id = st.selectbox(label="Market_id", options=[1, 2, 3, 4, 5, 6])
    market_id_filtered_value = f'{market_id}.0'
    metric = st.selectbox(label="Metric", options=ALLOWED_METRICS.keys())
    allowed_food_categories = st.multiselect(
        label="Enter the food categories to be considered",
        options=['All'] + existing_food_categories,
        default=['All'],
        help='Select all the Food categories to be considered in the report'
    )
    if 'All' in allowed_food_categories:
        allowed_food_categories = existing_food_categories
    filtered_data_to_plot = data_to_plot.loc[
        data_to_plot[MARKET_ID_COLUMN].eq(market_id_filtered_value) &
        data_to_plot[STORE_CATEGORY].isin(allowed_food_categories)
        ]
    if filtered_data_to_plot.empty:
        return None
    plots_dict = DataViewer.generate_all_streamlit_objects(filtered_data_to_plot, metric=metric)
    return plots_dict


def build_dashboard(plots_dict):
    style_metric_cards(background_color='rgba(255,255,255,0)', border_color='#FFF', border_left_color='#FFF')
    with st.container():
        col1, col2 = st.columns([2, 8])
        col1.image("doordash_challenge/data/doordash_symbol.jpg")
        with col2:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric('Total Revenue', plots_dict['kpis'][TOTAL_REVENUE])
                col2.metric('N° of Transactions', plots_dict['kpis'][TOTAL_TRANSACTIONS])
                col3.metric('Total n° of items sold', plots_dict['kpis'][TOTAL_ITEMS])
                col4.metric('N° of stores in the region', plots_dict['kpis'][TOTAL_NUMBER_OF_STORES])
                col5.metric('N° of food categories in the region', plots_dict['kpis'][DISTINCT_FOOD_CATEGORIES])

            with st.container():
                col6, col7, col8, col9, col10 = st.columns(5)
                col6.metric('Q25 n° of transactions per day', plots_dict['kpis'][P_25_AVG_DAILY_TRANSACTION])
                col7.metric('Q75 n° of transactions per day', plots_dict['kpis'][P_75_AVG_DAILY_TRANSACTION])
                col8.metric('Q95 n° of transactions per day', plots_dict['kpis'][P_95_AVG_DAILY_TRANSACTION])
                col9.metric('Avg items sold per transaction', plots_dict['kpis'][AVG_NUMBER_OF_ITEMS])
                col10.metric('Avg revenue per transaction', plots_dict['kpis'][AVG_REVENUE])

    with st.container():
        col11, col12, col13 = st.columns([0.5, 0.25, 0.25])
        with col11:
            st.plotly_chart(plots_dict[TIME_SERIES_PLOT], use_container_width=True)
        with col12:
            st.plotly_chart(plots_dict[WEEKDAY_PLOT], use_container_width=True)
        with col13:
            st.plotly_chart(plots_dict[TIME_OF_DAY_PLOT], use_container_width=True)

    with st.container():
        col14, col15, col16 = st.columns(3)
        with col14:
            st.plotly_chart(plots_dict[STORE_RANK_PLOT], use_container_width=False)
        with col15:
            st.plotly_chart(plots_dict[CATEGORY_RANK_PLOT], use_container_width=False)
        with col16:
            st.plotly_chart(plots_dict[PROTOCOL_RANK_PLOT], use_container_width=False)


st.set_page_config(layout="wide")
data = pd.read_csv('doordash_challenge/data/grouped_data.csv')
with st.sidebar:
    st_plots_dict = build_input_sidebar(data)
if st_plots_dict is None:
    st.title('Doordash KPIs report - No data available for the parameters chosen')
else:
    st.title('Doordash KPIs report')
    build_dashboard(st_plots_dict)
