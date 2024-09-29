import streamlit as st

from logic.forecast_data import get_data_for_forecast_page
from logic.plots import return_forecast_plot, close_forecast_plot, return_hist_plot
from styles.page_style import set_page_config_wide, disable_header_and_footer
from styles.sidebar_ref import make_refs_in_sidebar


def make_forecast():
    df_now, df_forecast, df_forecast_norm, df_btc_usdt = get_data_for_forecast_page(model="garch")

    st.markdown(f"""
        #### Спрогнозированная волатильность = {df_forecast.iloc[:, 1].std()} σ
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
    st.plotly_chart(return_hist_plot(df_forecast, model="garch"), use_container_width=True)


def main():
    # Настройка страницы
    set_page_config_wide()
    make_refs_in_sidebar()
    disable_header_and_footer()

    st.markdown("# GARCH Forecast")

    st.markdown(r"""
        На этой странице представлены прогнозы волатильности, рассчитанные с помощью модели 
        **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)**.  
        Модель GARCH, являющаяся обобщением модели ARCH, используется для 
        прогнозирования будущей дисперсии временного ряда, основываясь как на его прошлых значениях, 
        так и на прошлых значениях самой дисперсии.        
    """)
    with st.expander("Дополнительная информация о модели"):
        st.markdown(r"""
            Формула модели GARCH: $$\sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{q} \beta_j \sigma_{t-j}^2$$
    
            где:
    
            * $\sigma_t^2$ - условная дисперсия в момент времени $t$
            * $\omega$ - константа
            * $p$ - порядок компоненты ARCH (количество лагов прошлых ошибок)
            * $\alpha_i$ - коэффициент, отражающий влияние лаговой ошибки на текущую дисперсию
            * $\epsilon_{t-i}$ - лаговая ошибка модели, описывающая разницу между фактическим и прогнозируемым значением ряда в момент времени $t-i$
            * $q$ - порядок компоненты GARCH (количество лагов прошлых дисперсий)
            * $\beta_j$ - коэффициент, отражающий влияние лаговой дисперсии на текущую дисперсию
        """)

    st.session_state.initialized = False

    # Кнопка для перезапуска
    if st.button("Перезапустить расчёты", key="GARCH_forecast", type="primary"):
        st.rerun()

    # Выполнение функции при первой загрузке страницы
    if not st.session_state.initialized:
        make_forecast()
        st.session_state.initialized = True


main()
