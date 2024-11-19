from typing import List
import pandas as pd
import numpy as np
from doordash_challenge.functions.data_processing.Transformer import DataTransformer
import plotly.graph_objects as go
import plotly.express as px


class DataViewer:
    @staticmethod
    def plot_time_series(data: pd.DataFrame, date_column: str, metric: str, title: str):
        """
        Given the `data` df group the data with the columns in the `group_columns` list calculating the metrics
        passed in the metrics dictionary with the logic {column_name:['column_to_be_calculated','operation']} ex:
        {'total_spend':['subtotal','sum'], 'n_transactions':['created_at','count'], 'total_items':['items','sum']}
        """
        data_to_plot = DataTransformer.generate_cumulative_time_series(data, date_column, metric)
        convert_to_str_with_sep = np.vectorize(lambda x: f"{x:,}")
        text_legend = convert_to_str_with_sep(np.round(data_to_plot['metric'], 1))
        text_legend_cum = list(convert_to_str_with_sep(np.round(data_to_plot['cum_metric'], 1)))
        text_legend_cum[0] = None
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=data_to_plot[date_column],
            y=data_to_plot['metric'],
            name=f'{date_column}',
            text=text_legend,
            textposition='outside',
            marker_color='#FF3008'
        ))
        fig.add_trace(go.Scatter(
            x=data_to_plot[date_column],
            y=data_to_plot['cum_metric'],
            name='Cumulative',
            line=dict(width=2),
            mode="lines+text",
            text=text_legend_cum,
            textposition="top center",
            marker_color='#0A3AC4'
        ))
        fig.update_layout(
            title={'text': title, 'x': 0.45, 'xanchor': 'center'},
            xaxis_title={'text': f'{date_column}'},
            yaxis_title={'text': 'Metric of Interest'},
            yaxis={'tickformat': ',', 'showticklabels': False},
            template='plotly_dark')
        fig.update_yaxes(tickformat=",")
        return fig
    @staticmethod
    def plot_percentage_distribution(
            data: pd.DataFrame,
            column: str,
            metric: str,
            order_values: List,
            title: str,
            color: str
    ):
        """
        Given the `data` df group the data with the columns in the `group_columns` list calculating the metrics
        passed in the metrics dictionary with the logic {column_name:['column_to_be_calculated','operation']} ex:
        {'total_spend':['subtotal','sum'], 'n_transactions':['created_at','count'], 'total_items':['items','sum']}
        """
        percentage_df = DataTransformer.generate_percentage_group(data, column, metric)
        fig = px.bar(
            percentage_df,
            y=column,
            x='metric',
            category_orders={column: order_values},
            orientation='h',
            color_discrete_sequence=[color]
        )
        fig.update_yaxes(type='category')
        fig.update_traces(texttemplate='%{x:.2s}%', textposition='outside')
        fig.update_layout(
            title={'text': title, 'x': 0.45, 'xanchor': 'center'},
            xaxis_title={'text': None},
            yaxis_title={'text': None},
            template='plotly_dark',
            xaxis={'showticklabels': False},
        )
        return fig

    @staticmethod
    def plot_bar_rank(
        data: pd.DataFrame,
        column: str,
        metric: str,
        title: str,
        color: str,
        rank: int,
    ):
        """
        Given the `data` df group the data with the columns in the `group_columns` list calculating the metrics
        passed in the metrics dictionary with the logic {column_name:['column_to_be_calculated','operation']} ex:
        {'total_spend':['subtotal','sum'], 'n_transactions':['created_at','count'], 'total_items':['items','sum']}
        """
        rank_df = DataTransformer.generate_rank_group(data, column, metric, rank)
        convert_to_str_with_sep = np.vectorize(lambda x: f"{x:,}")
        text_legend = convert_to_str_with_sep(np.round(rank_df['metric'], 1))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=rank_df[column],
            x=rank_df['metric'],
            text=text_legend,
            textposition='outside',
            marker_color=color,
            orientation='h'
        ))
        fig.update_yaxes(type='category')
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title={'text': title, 'x': 0.45, 'xanchor': 'center'},
            xaxis_title={'text': None},
            yaxis_title={'text': None},
            template='plotly_dark',
            xaxis={'showticklabels': False},
        )
        return fig
    @staticmethod
    def generate_all_streamlit_objects(data: pd.DataFrame, metric: str):
        time_series_plot = \
            DataViewer.plot_time_series(data, date_column='week', metric=metric, title='Historical Metric Evolution')
        weekday_plot = DataViewer.plot_percentage_distribution(
            data, column='weekday', metric=metric, title='Metric comparison per weekday', color='#0ACFC5',
            order_values=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        time_of_day_plot = DataViewer.plot_percentage_distribution(
            data, column='time_of_day', metric=metric, order_values=['Morning', 'Afternoon', 'Evening', 'Night'],
            title='Metric comparison per time of the day', color='#F8E500'
        )
        store_rank = DataViewer.plot_bar_rank(
            data, column='store_id', metric=metric, title='Top Stores in the region', color='#005848', rank=10
        )
        category_rank = DataViewer.plot_bar_rank(
            data, column='store_primary_category', metric=metric, title='Top food category', color='#005848', rank=10
        )
        protocol_rank = DataViewer.plot_bar_rank(
            data, column='order_protocol', metric=metric, title='Top protocols', color='#005848', rank=10
        )
        kpis = DataTransformer.get_market_id_kpis(data)
        dict_of_objects = {
            'time_series_plot': time_series_plot,
            'weekday_plot': weekday_plot,
            'time_of_day_plot': time_of_day_plot,
            'store_rank': store_rank,
            'category_rank': category_rank,
            'protocol_rank': protocol_rank,
            'kpis': kpis
        }
        return dict_of_objects


