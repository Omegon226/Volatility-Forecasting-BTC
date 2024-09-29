from utilsforecast.plotting import plot_series
import plotly.graph_objects as go


def return_forecast_plot(df_now, df_forecast):
    fig = plot_series(
        df_now,
        forecasts_df=df_forecast,
        engine='plotly',
        level=[99, 95, 90, 75, 50],
        target_col="close_pct_change"
    )

    fig.update_layout(
        height=400
    )

    return fig


def close_forecast_plot(df_btc_usdt, df_forecast_norm):
    fig = plot_series(
        df_btc_usdt,
        forecasts_df=df_forecast_norm,
        engine='plotly',
        level=[99, 95, 90, 75, 50],
        target_col="close"
    )

    fig.update_layout(
        height=400
    )

    return fig


def return_hist_plot(df_forecast, model):
    if model in ["arch", "garch"]:
        x = df_forecast.iloc[:, 1]
    else:
        x = df_forecast.iloc[:, 2]

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
        margin=dict(
            l=50,  # Левый отступ
            r=150,  # Правый отступ (увеличен для размещения легенды)
            t=50,  # Верхний отступ
            b=50  # Нижний отступ
        )
    )

    return fig
