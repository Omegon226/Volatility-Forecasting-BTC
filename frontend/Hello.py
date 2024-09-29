import streamlit as st

from styles.page_style import set_page_config_centered, disable_header_and_footer
from styles.sidebar_ref import make_refs_in_sidebar

# Настройка страницы
set_page_config_centered()
make_refs_in_sidebar()
disable_header_and_footer()

st.markdown(
    """
    # BTC-USDT Volatility Forecasting with ML
    
    Это открытая система для автоматизации создания прогнозов волатильности
    для курса BTC-USDT, который вычисляется с помощью статистичиских методов 
    (ARCH, GARCH) и методов машинного обучения (KNN, SVR, LightGBM)
    """
)
