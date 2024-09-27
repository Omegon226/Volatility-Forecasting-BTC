import streamlit as st
import plotly.graph_objects as go
from utilsforecast.plotting import plot_series

from logic.forecast_data import get_data_for_forecast_page


st.set_page_config(
    page_title="BTC-USDT Volatility",
    page_icon="📊",
)

st.markdown("# GARCH Forecast")

if st.button("Make Forecast", key="GARCH_forecast", type="primary"):

    df_now, df_forecast, df_forecast_norm, df_btc_usdt = get_data_for_forecast_page(model="garch")

    st.markdown(f"Спрогнозированная волатильность = {df_forecast.iloc[:, 1].std()} σ")

    fig = plot_series(
        df_now,
        forecasts_df=df_forecast,
        engine='plotly',
        level=[95, 90],
        target_col="close_pct_change"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


    fig = plot_series(
        df_btc_usdt,
        forecasts_df=df_forecast_norm,
        engine='plotly',
        level=[95, 90],
        target_col="close"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


    x = df_forecast.iloc[:, 1]

    hist = go.Histogram(
        x=x,
        nbinsx=30,
        marker=dict(
            color='rgba(135, 206, 250, 0.7)',
            line=dict(
                color='rgba(135, 206, 250, 1)',
                width=1
            )
        )
    )
    fig = go.Figure(data=[hist])

    # Обновление макета
    fig.update_layout(
        title='Гистограмма распределения данных',
        xaxis_title='Значение',
        yaxis_title='Частота',
        bargap=0.2,  # Зазор между бинами
        height=400,  # Высота графика в пикселях
        template='plotly_dark',  # Белый фон
        margin = dict(
            l=50,  # Левый отступ
            r=150,  # Правый отступ (увеличен для размещения легенды)
            t=50,  # Верхний отступ
            b=50  # Нижний отступ
        )
    )
    st.plotly_chart(fig, use_container_width=True)
