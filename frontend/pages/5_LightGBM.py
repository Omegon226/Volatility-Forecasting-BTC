import streamlit as st

from logic.forecast_data import get_data_for_forecast_page
from logic.plots import return_forecast_plot, close_forecast_plot, return_hist_plot
from styles.page_style import set_page_config_wide, disable_header_and_footer
from styles.sidebar_ref import make_refs_in_sidebar


def make_forecast():
    df_now, df_forecast, df_forecast_norm, df_btc_usdt = get_data_for_forecast_page(model="lightgbmregressor")

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
    st.plotly_chart(return_hist_plot(df_forecast, model="lightgbmregressor"), use_container_width=True)


def main():
    # Настройка страницы
    set_page_config_wide()
    make_refs_in_sidebar()
    disable_header_and_footer()

    st.markdown("# LightGBM Forecast")

    st.markdown(r"""
        На этой странице представлены прогнозы волатильности, рассчитанные с помощью модели 
        **LightGBM (Light Gradient Boosting Machine)**. LightGBM - это фреймворк градиентного бустинга, использующий 
        алгоритмы обучения на основе дерева решений. Он известен своей высокой эффективностью и масштабируемостью, 
        особенно для больших наборов данных.
    """)
    with st.expander("Дополнительная информация о модели"):
        st.markdown(r"""
            ### 1. Градиентный бустинг:

            Основная идея градиентного бустинга заключается в последовательном построении деревьев решений, 
            каждое из которых корректирует ошибки предыдущих деревьев. 
            
            На каждой итерации $t$:
            
            * Алгоритм вычисляет **градиент функции потерь** по предсказаниям текущего ансамбля  
            $F_{t-1}(x)$ для каждого объекта $i$:
            
            $$g_i = -\frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}$$
            
            * Строится новое дерево $h_t(x)$, которое аппроксимирует значения градиента:
            
            $$h_t(x)  \approx  g_i$$\end{equation}
            
            *  Ансамбль обновляется с учетом нового дерева:
            
            $$F_t(x) = F_{t-1}(x) + \eta h_t(x)$$
            
            где:
            
            * $L(y_i, F(x_i))$ - функция потерь (например, среднеквадратическая ошибка)
            * $\eta$ -  скорость обучения (learning rate), контролирующая вклад каждого дерева
            
            ### 2. Алгоритм построения дерева в LightGBM:
            
            LightGBM использует **гистограммный алгоритм** для выбора оптимального разделения в узлах дерева. 
            
            *  **Гистограммы:** Значения признаков  дискретизируются и  представляются в виде гистограмм.
            *  **Скорость:** Гистограммный алгоритм позволяет быстро найти оптимальное разделение, так как не требует сортировки  значений признаков.
            *  **Эффективность памяти:**  Гистограммы занимают меньше памяти, чем хранение всех значений признаков.
            
            ### 3.  Регуляризация:
            
            Для предотвращения переобучения  LightGBM использует  **регуляризацию**:
            
            * **L1-регуляризация:**  Добавляет штраф за  сумму абсолютных значений весов листьев дерева.
            * **L2-регуляризация:**  Добавляет штраф за  сумму квадратов  весов листьев дерева.
            
            ### 4.  Прогнозирование:
            
            Прогноз LightGBMRegressor  для нового объекта $x$  - это взвешенная сумма предсказаний всех деревьев в ансамбле:
            
            $$\hat{y}(x) = \sum_{t=1}^{T} \eta h_t(x)$$
        """)

    st.session_state.initialized = False

    # Кнопка для перезапуска
    if st.button("Перезапустить расчёты", key="LightGBM_forecast", type="primary"):
        st.rerun()

    # Выполнение функции при первой загрузке страницы
    if not st.session_state.initialized:
        make_forecast()
        st.session_state.initialized = True


main()
