import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests

def ARCH():
    st.write("ARCH")
    if st.button("Make forecast", key="ARCH_forecast", type="primary"):
        #st.image("https://static.streamlit.io/examples/cat.jpg", width=600)

        x = np.random.randn(200)
        hist_data = [x]
        group_labels = ['Group 1']
        fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1])
        st.plotly_chart(fig, use_container_width=True)

def GARCH():
    st.write("GARCH")
    if st.button("Make forecast", key="GARCH_forecast", type="primary"):
        x = np.random.randn(200)
        y = np.cumsum(x)  # Создаем кумулятивную сумму для получения линии графика

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y))),  # Используем индексы для x
            y=y,
            mode='lines+markers',
            name='Случайная линия'
        ))

        # Настройка макета для размещения легенды справа
        fig.update_layout(
            title='Пример графика с легендой справа',
            xaxis_title='Время',
            yaxis_title='Значение',
            showlegend=True,
            legend=dict(
                x=1.05,  # Горизонтальная позиция легенды (с правой стороны)
                y=1,  # Вертикальная позиция легенды (вверху)
                xanchor='left',  # Привязка левого края легенды к позиции x
                yanchor='top',  # Привязка верхнего края легенды к позиции y
                bgcolor='rgba(255, 255, 255, 0)',  # Фон легенды (прозрачный)
                bordercolor='rgba(0, 0, 0, 0)'  # Граница легенды (прозрачная)
            ),
            margin=dict(
                l=50,  # Левый отступ
                r=150,  # Правый отступ (увеличен для размещения легенды)
                t=50,  # Верхний отступ
                b=50  # Нижний отступ
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.text(f"Прогнозируемая волотильность = {x.std()}")

def KNN():
    st.write("KNN")
    if st.button("Make forecast", key="KNN_forecast", type="primary"):
        st.image("https://static.streamlit.io/examples/cat.jpg", width=600)

def SVR():
    st.write("SVR")
    if st.button("Make forecast", key="SVR_forecast", type="primary"):
        st.image("https://static.streamlit.io/examples/cat.jpg", width=600)

def LightGBM():
    st.write("LightGBM")
    if st.button("Make forecast", key="LightGBM_forecast", type="primary"):
        st.image("https://static.streamlit.io/examples/cat.jpg", width=600)

def test_page():
    tab1, tab2, tab3, tab4 = st.tabs(["Cat", "Dog", "Owl", "BTC-USDT"])

    with tab1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    with tab2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    with tab4:
        st.header("BTC-USDT")

        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": 'BTCUSDT',
            "interval": '1d',
            "limit": 100
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Проверка на успешность запроса
            data = response.json()

            # Преобразование данных в DataFrame
            df = pd.DataFrame(data, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])

            # Преобразование временных меток в читаемый формат
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

            # Преобразование цен и объемов из строк в числовые значения
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                               'Quote Asset Volume', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            # Удаление ненужных столбцов
            df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades']]

            st.dataframe(df, use_container_width=True)

        except Exception as error:
            st.text(error)


if __name__ == "__main__":
    pg = st.navigation({"Forecasting models": [
        st.Page(ARCH),
        st.Page(GARCH),
        st.Page(KNN),
        st.Page(SVR),
        st.Page(LightGBM),
        st.Page(test_page)
    ]})
    
    pg.run()
