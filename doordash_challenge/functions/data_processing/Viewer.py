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
            name=f'{date_column} - {metric}',
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
    


