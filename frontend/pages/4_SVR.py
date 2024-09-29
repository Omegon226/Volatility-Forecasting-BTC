import streamlit as st

from logic.forecast_data import get_data_for_forecast_page
from logic.plots import return_forecast_plot, close_forecast_plot, return_hist_plot
from styles.page_style import set_page_config_wide, disable_header_and_footer
from styles.sidebar_ref import make_refs_in_sidebar


def make_forecast():
    df_now, df_forecast, df_forecast_norm, df_btc_usdt = get_data_for_forecast_page(model="svr")

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
    st.plotly_chart(return_hist_plot(df_forecast, model="svr"), use_container_width=True)


def main():
    # Настройка страницы
    set_page_config_wide()
    make_refs_in_sidebar()
    disable_header_and_footer()

    st.markdown("# SVR Forecast")

    st.markdown(r"""
        На этой странице представлены прогнозы волатильности, рассчитанные с помощью модели 
        **Support Vector Regression (SVR)**. KNN - SVR - это метод машинного обучения, используемый для 
        решения задач регрессии. Он основан на поиске гиперплоскости в многомерном пространстве, 
        которая наилучшим образом разделяет данные с учетом установленной пользователем погрешности (ширины ε-трубки). 
        SVR стремится найти такую гиперплоскость, которая максимально близко подходит к как можно большему 
        количеству точек данных, допуская при этом отклонения не более чем на ε. 
    """)
    with st.expander("Дополнительная информация о модели"):
        st.markdown(r"""
            Для решения нелинейных задач регрессии используется **трюк с ядром (kernel trick)**. 
            Ядро - это функция, преобразующая исходные признаки  в  пространство более высокой размерности, 
            где данные могут быть линейно разделимы.

            Функция прогнозирования SVR с ядром:
            
            $$\hat{y}(x) = \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) K(x, x_i) + b$$
            
            где:
            
            * $K(x, x_i)$ - функция ядра, измеряющая сходство между объектами  $x$ и $x_i$
            * $\alpha_i, \alpha_i^*$ -  лагранжевы множители, полученные в результате решения задачи оптимизации
            
            **Популярные функции ядра:**
            
            * **Линейное:**  $K(x, x_i) = x^T x_i$
            * **Полиномиальное:**  $K(x, x_i) = (x^T x_i + 1)^d$
            * **Радиальная базисная функция (RBF):**  $K(x, x_i) = exp(-\gamma ||x - x_i||^2)$ 
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
