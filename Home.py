import streamlit as st
from components.shared import add_contact_info
from pathlib import Path

# 设置页面配置
st.set_page_config(
    page_title="MSAM for GBM Prognostication",
    page_icon="🔬",
    layout="wide"
)

# 添加联系信息到侧边栏
add_contact_info()

# 页面内容
st.title("MSAM: GBM Prognostication Model")

# 模型简介
st.markdown("""
### Model Introduction
MSAM (Magnification-Aware Multi-Instance Attention Model) is a slide-level multiple instance learning (MIL) model for glioblastoma (GBM) prognosis from whole slide images (WSI).

It fuses a global slide representation (Titan features) with patch-level representations (UNI features), then performs magnification-adaptive feature modulation, multi-scale aggregation, and attention-based MIL pooling to produce a slide-level risk score (MSAM). This score can be combined with clinical variables in a Cox model to estimate survival risk and survival curves.
""")

# 显示模型架构图
st.markdown("### Model Architecture")
APP_ROOT = Path(__file__).resolve().parent
st.image(str(APP_ROOT / 'pic' / 'model_architecture.png'), 
         caption='MSAM Model Architecture',
         use_container_width=True)

# 模型特点
st.markdown("""
### Key Modules
- Dual-branch feature projection (Titan / UNI)
- Magnification Adaptive Module
- Multi-scale aggregation (ASPP)
- Channel & Spatial Attention
- Efficient feature refinement (EMA)
- Attention-based MIL pooling and risk prediction
""")

# 添加页脚
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Zhongnan Hospital. All rights reserved.</div>", unsafe_allow_html=True)
