import streamlit as st
from components.shared import add_contact_info

# 设置页面配置
st.set_page_config(
    page_title="MSAM Tutorial (GBM)",
    page_icon="📖",
    layout="wide"
)

# 添加联系信息到侧边栏
add_contact_info()

# 设置页面样式
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .main > div {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .step-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("MSAM Tutorial (GBM)")

# 使用教程
st.markdown("## How to Use MSAM")

with st.container():
    st.markdown("### Step 1: Data Preparation")
    st.markdown("""
    Before using MSAM, please ensure you have:
    - Digital pathology images in WSI format (supporting .svs / .ndpi / .sdpc formats)
    - Clinical information required by your Cox model training dataset (e.g., KPSscore, P53, ATRX)
    """)

with st.container():
    st.markdown("### Step 2: Data Upload and Analysis")
    st.markdown("""
    1. Upload a WSI image on the Analysis page.
    2. Select clinical information using buttons/selectors on the left panel.
    3. Click the "Submit" button to start the analysis.
    """)

# 结果解读
st.markdown("## Understanding the Results")

with st.container():
    st.markdown("### Risk Heatmap")
    st.markdown("""
    The heatmap displays the risk level in different regions of the WSI image:
    - Red areas indicate high-risk regions
    - Blue areas indicate low-risk regions
    """)

with st.container():
    st.markdown("### SHAP Analysis")
    st.markdown("""
    Feature contribution shows how each feature impacts the Cox prediction:
    - Positive values (red) indicate increased risk
    - Negative values (blue) indicate decreased risk
    """)

with st.container():
    st.markdown("### Survival Rate Plot")
    st.markdown("""
    The survival curve shows the predicted survival probability over time (years):
    - The y-axis represents the probability of survival.
    - The x-axis represents time (in years).
    """)

# 常见问题
st.markdown("## FAQ")
with st.expander("Q: How long does it take to process large WSI images?"):
    st.write("Processing time depends on the size of the image and server load, typically taking 3-5 minutes.")

with st.expander("Q: What image formats are supported?"):
    st.write("Currently supports WSI images in .svs / .ndpi / .sdpc formats.")

with st.expander("Q: How to explain risk scoring?"):
    st.write("The Cox model outputs a hazard ratio (HR). HR > 1.0 is shown as High Risk, and HR ≤ 1.0 is shown as Low Risk.")

# 添加页脚
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Zhongnan Hospital. All rights reserved.</div>", unsafe_allow_html=True)
