import streamlit as st
import webbrowser


def open_link(url):
    webbrowser.open_new_tab(url)


def make_refs_in_sidebar():
    with st.sidebar:
        st.write("Панели администрирования:")

        # Добавляем стиль для кнопок-ссылок
        st.markdown("""
                <style>
                .sidebar-button {
                    display: block;
                    width: 100%;
                    padding: 0.5rem 1rem;
                    margin: 0.5rem 0;
                    background-color: #262730;
                    color: white;  
                    text-decoration: none;  
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    border: 1px solid rgba(84, 85, 93, 1);
                    border-radius: 0.25rem;
                    cursor: pointer;
                }
                .sidebar-button:hover {
                    color: rgba(255, 75, 75, 1);  
                    border-color: rgba(255, 75, 75, 1);
                    text-decoration: none;  
                }
                .sidebar-button:visited  {
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)

        # Кнопки-ссылки
        st.markdown("""
            <a href="http://localhost:8012/" target="_blank" class="sidebar-button">AirFlow</a>
            <a href="http://localhost:8010/" target="_blank" class="sidebar-button">InfluxDB</a>
            <a href="http://localhost:5000/" target="_blank" class="sidebar-button">MLFlow</a>
            <a href="http://localhost:8000/docs#/" target="_blank" class="sidebar-button">Backend Swagger</a>
        """, unsafe_allow_html=True)
