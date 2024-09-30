import streamlit as st

from logic.forecast_data import get_data_for_forecast_page
from logic.plots import return_forecast_plot, close_forecast_plot, return_hist_plot
from styles.page_style import set_page_config_wide, disable_header_and_footer
from styles.sidebar_ref import make_refs_in_sidebar


def make_forecast():
    df_now, df_forecast, df_forecast_norm, df_btc_usdt = get_data_for_forecast_page(model="nbeats")

    st.markdown(f"""
        #### Спрогнозированная волатильность = {df_forecast.iloc[:, 2].std()} σ
    """)
    with st.expander("Справка по расчётам волатильности"):
        st.markdown(r"""
            Для оценки волатильности мы используем **стандартное отклонение** прогнозируемых 
            процентных изменений цены закрытия.

            **Стандартное отклонение** - это мера разброса данных вокруг среднего значения. 
            В данном контексте оно показывает, насколько сильно колеблются прогнозируемые процентные 
            изменения цены относительно своего среднего значения.

            Расчётная формула = $$\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$

            где:

            * $\sigma$ - стандартное отклонение
            * $N$ - количество прогнозируемых значений
            * $x_i$ - i-ое прогнозируемое значение процентного изменения цены закрытия
            * $\bar{x}$ - среднее значение всех прогнозируемых значений процентного изменения цены закрытия 
        """)

    st.markdown(f"""
        ## Прогноз динамики изменения цены закрытия 
    """)
    st.plotly_chart(return_forecast_plot(df_now, df_forecast), use_container_width=True)

    st.markdown(f"""
        ## Прогноз динамики цены закрытия 
    """)
    st.plotly_chart(close_forecast_plot(df_btc_usdt, df_forecast_norm), use_container_width=True)

    st.markdown(f"""
        ## Распределение чисел прогноза изменения цены закрытия
    """)
    st.plotly_chart(return_hist_plot(df_forecast, model="nbeats"), use_container_width=True)


def main():
    # Настройка страницы
    set_page_config_wide()
    make_refs_in_sidebar()
    disable_header_and_footer()

    st.markdown("# NBEATS Forecast")

    st.markdown(r"""
        На этой странице представлены прогнозы, рассчитанные с помощью модели **N-BEATS**.
        N-BEATS — это архитектура нейронной сети, разработанная специально для прогнозирования временных рядов. 
        Модель основана на глубокой стопке полностью связанных блоков (обратных блоков), которые последовательно 
        извлекают различные компоненты временного ряда, такие как тренд и сезонность, без использования заранее 
        заданных предположений о структуре данных. Благодаря своей гибкой и модульной архитектуре, N-BEATS способна 
        эффективно моделировать сложные зависимости и обеспечивать высокую точность прогнозов. 
        Модель не требует специфической настройки под конкретные типы временных рядов, что делает её 
        универсальным инструментом для различных задач прогнозирования. N-BEATS сочетает в себе преимущества 
        традиционных методов и современных нейронных сетей, предоставляя надежные и 
        точные прогнозы для широкого спектра приложений.
    """)
    with st.expander("Дополнительная информация о модели"):
        st.markdown(r"""

        """)

    st.session_state.initialized = False

    # Кнопка для перезапуска
    if st.button("Перезапустить расчёты", key="NBEATS_forecast", type="primary"):
        st.rerun()

    # Выполнение функции при первой загрузке страницы
    if not st.session_state.initialized:
        make_forecast()
        st.session_state.initialized = True


main()
