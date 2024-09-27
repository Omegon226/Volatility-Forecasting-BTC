import streamlit as st

st.set_page_config(
    page_title="BTC-USDT Volatility",
    page_icon="📊",
)

st.markdown(
    """
    # BTC-USDT Volatility Forecasting with ML
    
    Это открытая система для автоматизации создания прогнозов волатильности
    для курса BTC-USDT, который вычисляется с помощью статистичиских методов 
    (ARCH, GARCH) и методов машинного обучения (KNN, SVR, LightGBM)
    """
)
