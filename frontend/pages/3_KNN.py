import streamlit as st

from logic.forecast_data import get_data_for_forecast_page
from logic.plots import return_forecast_plot, close_forecast_plot, return_hist_plot
from styles.page_style import set_page_config_wide, disable_header_and_footer
from styles.sidebar_ref import make_refs_in_sidebar


def make_forecast():
    df_now, df_forecast, df_forecast_norm, df_btc_usdt = get_data_for_forecast_page(model="knn")

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
    st.plotly_chart(return_hist_plot(df_forecast, model="knn"), use_container_width=True)


def main():
    # Настройка страницы
    set_page_config_wide()
    make_refs_in_sidebar()
    disable_header_and_footer()

    st.markdown("# KNN Forecast")

    st.markdown(r"""
        На этой странице представлены прогнозы волатильности, рассчитанные с помощью модели 
        **K ближайших соседей (KNN)**. KNN - это простой алгоритм машинного обучения, 
        который классифицирует новые данные, находя K наиболее похожих объектов из обучающего набора данных и 
        присваивая новому объекту наиболее распространенную метку класса среди этих K соседей. 
    """)
    with st.expander("Дополнительная информация о модели"):
        st.markdown(r"""
            ### 1. Расстояние между объектами:

            Для определения **K** ближайших соседей необходимо рассчитать расстояние между объектами. 
            
            Обычно используется **Евклидово расстояние**:
            
            $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
            
            где:
            * $x = (x_1, x_2, ..., x_n)$ и $y = (y_1, y_2, ..., y_n)$ - два объекта (векторы признаков)
            * $n$ - количество признаков
            
            ### 2. Выбор K ближайших соседей:
            
            После расчета расстояний до всех объектов обучающей выборки выбираются **K** объектов с 
            наименьшим расстоянием до целевого объекта. 
            
            ### 3. Прогнозирование:
            
            * **Регрессия (предсказание числового значения):**  Среднее значение целевой переменной среди 
            **K** ближайших соседей.
            
            $$\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$
            
            где:
            * $\hat{y}(x)$ - прогнозируемое значение для объекта $x$
            * $N_K(x)$ - множество индексов **K** ближайших соседей объекта $x$
            * $y_i$ - значение целевой переменной для объекта с индексом $i$
            
            * **Классификация (предсказание метки класса):** Наиболее часто встречающаяся метка класса среди 
            **K** ближайших соседей.
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
