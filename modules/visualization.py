import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                      title: str, x_label: str, y_label: str) -> go.Figure:
    fig = px.line(df, x=x_col, y=y_col, 
                  title=title, 
                  markers=True)
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
                     title: str, x_label: str, y_label: str) -> go.Figure:
    fig = px.bar(df, x=x_col, y=y_col,
                 title=title)
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_histogram(df: pd.DataFrame, column: str, 
                     title: str, x_label: str) -> go.Figure:
    fig = px.histogram(df, x=column, nbins=30, title=title)
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title='Frecuencia',
        hovermode='x',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                        title: str, x_label: str, y_label: str,
                        color_col: str = None) -> go.Figure:
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     title=title)
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_heatmap(data: pd.DataFrame, title: str, 
                   x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='Blues',
        hovertemplate='%{y}: %{x}<br>Ocupación: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500
    )
    
    return fig


def create_box_plot(df: pd.DataFrame, x_col: str, y_col: str,
                    title: str, x_label: str, y_label: str) -> go.Figure:
    fig = px.box(df, x=x_col, y=y_col, title=title)
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=400
    )
    
    return fig


def display_metric_card(label: str, value: float, suffix: str = "", 
                        decimals: int = 2) -> None:
    formatted_value = f"{value:.{decimals}f}{suffix}"
    st.metric(label=label, value=formatted_value)
