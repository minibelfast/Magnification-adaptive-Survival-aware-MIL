import streamlit as st

def add_contact_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='color: #3498db; font-weight: bold; font-size: 1.2em;'>Contact Information</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='margin-top: 10px;'><strong>Xiaoping Liu, Ph.D</strong><br>Email: liuxiaoping@whu.edu.cn</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='margin-top: 10px;'><strong>Xuanyu Wang, B.S</strong><br>Email: wangxuanyu@whu.edu.cn</div>", unsafe_allow_html=True)