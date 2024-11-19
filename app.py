import streamlit as st
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards
from doordash_challenge.functions.data_processing.Viewer import DataViewer
from doordash_challenge.functions.data_processing.Transformer import ALLOWED_METRICS


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
        data_to_plot['market_id'].eq(market_id_filtered_value) &
        data_to_plot['store_primary_category'].isin(allowed_food_categories)
        ]
    if filtered_data_to_plot.empty:
        return None
    plots_dict = DataViewer.generate_all_streamlit_objects(filtered_data_to_plot, metric=metric)
    return plots_dict


def build_dashboard(plots_dict):
    style_metric_cards(background_color='rgba(255,255,255,0)', border_color='#FFF', border_left_color='#FFF')
    with st.container():
        col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])
        with col1:
            st.image("doordash_challenge/data/doordash_symbol.jpg")
        col2.metric('Total Revenue', plots_dict['kpis']['total_revenue'])
        col3.metric('N° of Transactions', plots_dict['kpis']['total_transactions'])
        col4.metric('Total n° of items sold', plots_dict['kpis']['total_items'])
        col5.metric('Number of unique food categories', plots_dict['kpis']['food_categories'])
        col6.metric('Avg items sold per transaction', plots_dict['kpis']['avg_number_of_items'])
        col7.metric('Avg revenue per transaction', plots_dict['kpis']['avg_revenue'])

    with st.container():
        col5, col6, col7 = st.columns([0.5, 0.25, 0.25])
        with col5:
            st.plotly_chart(plots_dict['time_series_plot'], use_container_width=True)
        with col6:
            st.plotly_chart(plots_dict['weekday_plot'], use_container_width=True)
        with col7:
            st.plotly_chart(plots_dict['time_of_day_plot'], use_container_width=True)

    with st.container():
        col7, col8, col10 = st.columns(3)
        with col7:
            st.plotly_chart(plots_dict['store_rank'], use_container_width=False)
        with col8:
            st.plotly_chart(plots_dict['category_rank'], use_container_width=False)
        with col10:
            st.plotly_chart(plots_dict['protocol_rank'], use_container_width=False)

st.set_page_config(layout="wide")
data = pd.read_csv('doordash_challenge/data/grouped_data.csv')
with st.sidebar:
    st_plots_dict = build_input_sidebar(data)
if st_plots_dict is None:
    st.title('Doordash KPIs report - No data available for the parameters chosen')
else:
    st.title('Doordash KPIs report')
    build_dashboard(st_plots_dict)