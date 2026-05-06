import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
import h5py
from PIL import Image
from pathlib import Path
from types import SimpleNamespace

from app_utils.training_schema import coerce_patient_row, infer_training_schema

# 导入项目模块
from utils.eval_utils import initiate_model
from models import get_encoder
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.batch_process_utils import initialize_df
from utils.file_utils import save_hdf5
from components.shared import add_contact_info

# 移除PIL库的图像大小限制
Image.MAX_IMAGE_PIXELS = None

APP_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = Path(os.environ.get("WSI_CONFIG_PATH", str(APP_ROOT / "config" / "default.yaml")))
MODEL_PATH = Path(os.environ.get("WSI_MODEL_PATH", str(APP_ROOT / "weights" / "wsi_model.pth")))
COX_MODEL_PATH = Path(os.environ.get("COX_MODEL_PATH", str(APP_ROOT / "cox_model.pkl")))
TRAIN_CSV_PATH = Path(os.environ.get("TRAIN_CSV_PATH", str(APP_ROOT / "9.1.dataset_train.csv")))
# 设置页面配置
# 在文件开头的st.set_page_config中修改
st.set_page_config(
    page_title="MSAM Analysis (GBM)",
    page_icon="📊",
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
        padding: 0;
        max-width: 100%;
        margin: 0;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1.5rem;
        background-color: #2ecc71;
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    [data-testid="stFileUploader"] {
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        border: 2px dashed #bdc3c7;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stExpander"] > div:first-child {
        border-radius: 0.5rem 0.5rem 0 0;
        background-color: #f8f9fa;
        padding: 1rem;
    }
    h1 {
        color: #2c3e50;
        margin-bottom: 2rem;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    h3 {
        color: #34495e;
        margin: 1.5rem 0 1rem;
        font-size: 1.25rem;
        font-weight: 600;
    }
    [data-testid="stSlider"] {
        padding: 1.5rem 0;
    }
    [data-testid="stSlider"] > div:first-child {
        font-weight: 500;
    }
    .element-container {
        margin-bottom: 1.5rem;
    }
    [data-testid="column"]:first-child {
        background-color: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: 100vh;
    }
    [data-testid="column"]:not(:first-child) {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .shap-importance {
        margin-top: 2rem;
    }
    .nomogram-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .high-risk {
        color: #e74c3c;
    }
    .low-risk {
        color: #2ecc71;
    }
    /* 添加联系信息样式 */
    .contact-info {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.2);
        color: rgba(255,255,255,0.7);
    }
    .contact-info p {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    /* 修改标题样式 */
    [data-testid="stExpander"] > div:first-child {
        background-color: #3498db;
        color: white;
    }
    /* 修改折叠图标颜色 */
    [data-testid="stExpander"] label.st-emotion-cache-1egp7rm {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 设置标题
st.title("MSAM for GBM Prognostication")

# 创建两列布局
left_col, right_col = st.columns([1, 3])

# 左侧列：输入参数
with left_col:
    # 创建一个容器用于输入参数
    with st.container():
        st.markdown("### Input Parameters")
        
        with st.form("input_form"):
            st.markdown("#### Upload WSI Image")
            upload_file = st.file_uploader("", type=["svs","ndpi","sdpc"], 
                help="Upload a WSI image in SVS format", 
                label_visibility='hidden', 
                accept_multiple_files=False)

            schema = infer_training_schema(TRAIN_CSV_PATH)
            clinical_values = {}

            target_features = {"msam", "kpsscore", "p53", "atrx"}
            feature_cols = [c for c in schema.columns if c.name.lower() in target_features]

            kps_min, kps_max = 0, 100
            try:
                df_train = pd.read_csv(TRAIN_CSV_PATH)
                for c in df_train.columns:
                    if str(c).strip().lower() == "kpsscore":
                        s = pd.to_numeric(df_train[c], errors="coerce").dropna()
                        if len(s) > 0:
                            kps_min = int(np.floor(float(s.min())))
                            kps_max = int(np.ceil(float(s.max())))
                        break
            except Exception:
                pass

            for col in feature_cols:
                lname = col.name.lower()
                if lname in {"msam", "gpma"}:
                    continue

                if lname == "kpsscore":
                    st.markdown("#### KPSscore")
                    default_v = schema.defaults.get(col.name, 80)
                    try:
                        default_i = int(round(float(default_v) if isinstance(default_v, (int, float)) else float(str(default_v))))
                    except Exception:
                        default_i = 80
                    default_i = max(kps_min, min(kps_max, default_i))
                    clinical_values[col.name] = st.slider(
                        "",
                        min_value=int(kps_min),
                        max_value=int(kps_max),
                        value=int(default_i),
                        step=10,
                        label_visibility="hidden",
                    )
                    continue

                if lname in {"p53", "atrx"}:
                    st.markdown(f"#### {col.name}")
                    default_v = schema.defaults.get(col.name, 0)
                    try:
                        default_i = int(round(float(default_v) if isinstance(default_v, (int, float)) else float(str(default_v))))
                    except Exception:
                        default_i = 0
                    default_i = 1 if default_i != 0 else 0
                    clinical_values[col.name] = st.radio(
                        "",
                        options=[0, 1],
                        horizontal=True,
                        index=1 if default_i == 1 else 0,
                        label_visibility="hidden",
                    )
                    continue

                st.markdown(f"#### {col.name}")
                if col.kind == "categorical":
                    options = list(col.choices or ())
                    if len(options) <= 6:
                        v = st.radio(
                            "",
                            options=options,
                            horizontal=True,
                            index=options.index(str(col.default)) if str(col.default) in options else 0,
                            label_visibility="hidden",
                        )
                    else:
                        v = st.selectbox(
                            "",
                            options=options,
                            index=options.index(str(col.default)) if str(col.default) in options else 0,
                            label_visibility="hidden",
                        )
                    clinical_values[col.name] = v
                else:
                    default_v = col.default
                    v = st.number_input(
                        "",
                        value=float(default_v) if isinstance(default_v, (int, float)) else 0.0,
                        label_visibility="hidden",
                    )
                    clinical_values[col.name] = v
            
            # 提交按钮
            submit_button = st.form_submit_button("Submit", type="primary")
        
        # 添加一个分隔线
        st.markdown("<hr style='margin: 2rem 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

# 右侧列：结果展示
with right_col:
    # 创建两行两列的布局
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 第一行第一列：原始WSI图像
    with row1_col1:
        raw_wsi_container = st.expander("Raw WSI", expanded=True)
    
    # 第一行第二列：风险热图
    with row1_col2:
        risk_map_container = st.expander("Risk map", expanded=True)
    
    # 第二行第一列：特征贡献度分析
    with row2_col1:
        feature_container = st.expander("Features Contribution", expanded=True)
    
    # 第二行第二列：生存率曲线
    with row2_col2:
        survival_container = st.expander("Survival Rate Plot", expanded=True)
    
    # 第三行：Nomogram
    nomogram_container = st.expander("Nomogram", expanded=True)

# 创建临时目录用于处理上传的文件
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)


# 拟合cox比例风险模型


# 示例：加载数据（确保数据中包含'time'和'event'列）








# 处理函数
def process_wsi(wsi_path):
    # 加载配置文件
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # 准备参数
    patch_args = SimpleNamespace(**config_dict['patching_arguments'])
    data_args = SimpleNamespace(**config_dict['data_arguments'])
    model_args = config_dict['model_arguments']
    model_args.update({'n_classes': config_dict['exp_arguments']['n_classes']})
    model_args = SimpleNamespace(**model_args)
    encoder_args = SimpleNamespace(**config_dict['encoder_arguments'])
    exp_args = SimpleNamespace(**config_dict['exp_arguments'])
    heatmap_args = SimpleNamespace(**config_dict['heatmap_arguments'])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = torch.load(str(MODEL_PATH), map_location=device)
    model.eval()
    
    # 加载特征提取器
    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # 创建临时保存目录
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 定义分割掩码路径
    seg_mask_path = temp_dir / 'seg_mask.pkl'
    
    # 初始化WSI对象
    seg_params = config_dict['segmentation_arguments']
    filter_params = config_dict['filter_arguments']
    wsi_object = initialize_wsi(wsi_path, seg_mask_path=seg_mask_path, seg_params=seg_params, filter_params=filter_params)
    
    # 设置补丁大小和步长
    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    
    # 设置WSI处理参数
    wsi_kwargs = {
        'top_left': None, 
        'bot_right': None, 
        'patch_size': patch_size, 
        'step_size': step_size,
        'custom_downsample': patch_args.custom_downsample, 
        'level': patch_args.patch_level, 
        'use_center_shift': heatmap_args.use_center_shift
    }
    
    # 创建临时保存目录
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 计算特征和热图
    with st.spinner("Processing WSI and generating heatmap..."):
        # 定义特征文件路径
        features_path = output_dir / f"{Path(wsi_path).stem}_features.pt"
        h5_path = output_dir / f"{Path(wsi_path).stem}_features.h5"
        
        if features_path.exists() and h5_path.exists():
            # 如果特征文件已存在，直接加载
            st.info("Loading pre-computed features...")
            features = torch.load(features_path)
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
        else:
            # 如果特征文件不存在，计算并保存
            st.info("Computing features...")
            # 计算特征
            _, _, wsi_object = compute_from_patches(
                wsi_object=wsi_object,
                model=model,
                feature_extractor=feature_extractor,
                img_transforms=img_transforms,
                batch_size=config_dict['exp_arguments']['batch_size'],
                **wsi_kwargs,
                attn_save_path=None,
                feat_save_path=h5_path,
                ref_scores=None
            )
            
            # 保存特征
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            coords = file['coords'][:]
            file.close()

        # 加载特征到设备并进行预测
        features = features.to(device)
        
        # 模型预测
        with torch.no_grad():
            # 处理特征
            features2 = features.expand(1, -1, -1).float()
            features2 = model._fc1(features2)  # [B, n, 512]
            
            # 通过模型层处理
            with torch.no_grad():
                A = features2
                for layer in model.layers:
                    A_ = A
                    A = layer[0](A)
                    A = layer[1](A, rate=model.rate)
                    A = A + A_
                A = model.normA(A)
                B = features2
                for layer in model.layers2:
                    B_ = B
                    B = layer[0](B)
                    B = layer[1](B).squeeze(-1).permute(0, 2, 1)
                    B = B + B_
                B = model.normA(B)
                features2 = model.DFF(A.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1),
                                     B.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
                A = model.classifier(features2)  # [B, n_classes]
                A = A[:, :, 0].unsqueeze(2)
            
            # 获取预测结果
            _, survival, Y_hat, A2, Y_prob = model(features)
            Y_hat = Y_hat.item()
            A = A.view(-1, 1).cpu().numpy()
            
            # 获取概率
            probs, ids = torch.topk(Y_prob, config_dict['exp_arguments']['n_classes'])
            Y_probs = probs[-1].cpu().numpy()
            Y_hats = ids[-1].cpu().numpy()
        
        # 计算风险
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()[0]
        
        # 保存注意力分数
        block_map_save_path = output_dir / "blockmap.h5"
        asset_dict = {'attention_scores': A, 'coords': coords}
        save_hdf5(str(block_map_save_path), asset_dict, mode='w')
        
        # 生成热图
        file = h5py.File(block_map_save_path, 'r')
        scores = file['attention_scores'][:]
        coords = file['coords'][:]
        file.close()
        
        # 生成可视化热图
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_object.level_downsamples[patch_args.patch_level]) * 
                              patch_args.custom_downsample).astype(int))
        
        heatmap = drawHeatmap(
            scores, coords, wsi_path, wsi_object=wsi_object, 
            cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, 
            use_holes=True, binarize=False, vis_level=-1, 
            blank_canvas=False, thresh=-1, patch_size=vis_patch_size, 
            convert_to_percentiles=True
        )
        
        # 保存热图
        heatmap_path = output_dir / "heatmap.png"
        heatmap.save(heatmap_path)
        
        # 返回结果
        return {
            'wsi_object': wsi_object,
            'heatmap': heatmap,
            'heatmap_path': heatmap_path,
            'risk': risk,
            'Y_prob': Y_prob.cpu().numpy(),
            'Y_hat': Y_hat,
            'survival': survival.cpu().numpy()
        }

# 主程序逻辑
if upload_file is not None:
    wsi_path = None
    # 显示原始WSI图像
    with raw_wsi_container:
        try:
            # 保存上传的图像
            wsi_path = temp_dir / upload_file.name
            with open(wsi_path, "wb") as f:
                f.write(upload_file.getbuffer())
            
            # 显示WSI缩略图
            wsi_object = WholeSlideImage(str(wsi_path))
            if str(wsi_path).endswith('.sdpc'):
                thumbnail = wsi_object.wsi.get_thumbnail((1000, 1000))
            else:
                thumbnail = wsi_object.wsi.get_thumbnail((1000, 1000))
            st.image(thumbnail, caption="Raw WSI Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading WSI image: {str(e)}")
    
    # 处理提交
    if submit_button:
        if wsi_path is None:
            st.error("WSI file is not ready. Please re-upload the slide.")
        else:
            try:
                # 处理WSI
                results = process_wsi(str(wsi_path))
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                results = None
            else:
                with risk_map_container:
                    st.image(results['heatmap'], caption="Risk Heatmap", use_container_width=True)
                
                gpma_score = float(results["risk"])

                patient_values = dict(clinical_values)
                for col in schema.columns:
                    if col.name.lower() in {"msam", "gpma"}:
                        patient_values[col.name] = gpma_score

                patient_row = coerce_patient_row(schema, patient_values)
                new_patient = pd.DataFrame([patient_row], columns=list(schema.feature_columns))

                cph = joblib.load(str(COX_MODEL_PATH))
                if hasattr(cph, "predict_partial_hazard"):
                    risk_pred = float(cph.predict_partial_hazard(new_patient).values[0])
                elif hasattr(cph, "predict"):
                    risk_pred = float(np.asarray(cph.predict(new_patient)).reshape(-1)[0])
                else:
                    risk_pred = float("nan")

                times = np.linspace(0.0, 3.0, 301)
                surv_df = None
                if hasattr(cph, "predict_survival_function"):
                    try:
                        surv_df = cph.predict_survival_function(new_patient, times=times)
                    except TypeError:
                        surv_df = cph.predict_survival_function(new_patient)
                
                with feature_container:
                    risk_group = "High Risk" if risk_pred > 1.0 else "Low Risk"
                    risk_class = "high-risk" if risk_pred > 1.0 else "low-risk"
                    
                    st.markdown(
                        f"<div class='nomogram-score {risk_class}'>Risk Group: {risk_group} (HR={risk_pred:.3f}, MSAM={gpma_score:.3f})</div>",
                        unsafe_allow_html=True,
                    )

                    coefs = getattr(cph, "params_", None)
                    coef_map = None
                    if coefs is not None:
                        if hasattr(coefs, "to_dict"):
                            coef_map = coefs.to_dict()
                        elif isinstance(coefs, dict):
                            coef_map = coefs

                    if coef_map:
                        baseline = {k: schema.defaults.get(k, 0.0) for k in coef_map.keys()}

                        def _to_float(x):
                            try:
                                return float(x)
                            except Exception:
                                return 0.0

                        contrib = {k: _to_float(coef_map[k]) * (_to_float(patient_row.get(k, 0.0)) - _to_float(baseline.get(k, 0.0))) for k in coef_map.keys()}
                        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
                        feature_names = [k for k, _ in items]
                        contrib_vals = np.array([v for _, v in items], dtype=float)
                    else:
                        feature_names = list(schema.feature_columns)
                        contrib_vals = np.zeros(len(feature_names), dtype=float)

                    fig, ax = plt.subplots(figsize=(10, 4.8))
                    y_pos = np.arange(len(feature_names))
                    colors = ["red" if x > 0 else "blue" for x in contrib_vals]
                    bars = plt.barh(y_pos, contrib_vals, color=colors)
                    plt.yticks(y_pos, feature_names)
                    plt.xlabel("Feature contribution (relative to training baseline)")
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.3f}",
                            ha='left' if width > 0 else 'right',
                            va='center',
                        )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with survival_container:
                    st.markdown(
                        f"<div class='nomogram-score'>Predicted Hazard Ratio (partial): {risk_pred:.3f}</div>",
                        unsafe_allow_html=True,
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if surv_df is not None:
                        if hasattr(surv_df, "iloc"):
                            s = np.asarray(surv_df.iloc[:, 0], dtype=float)
                            t = np.asarray(surv_df.index, dtype=float)
                        else:
                            s = np.asarray(surv_df, dtype=float).reshape(-1)
                            t = np.linspace(0.0, 3.0, len(s))
                        plt.plot(t, s, "r-", linewidth=2)
                        plt.fill_between(t, s, alpha=0.2, color="red")
                        plt.xlabel("Time (years)")
                        plt.ylabel("Survival Probability")
                        plt.title("Predicted Survival Curve")
                        plt.grid(True, alpha=0.3)
                        plt.ylim(0, 1)
                        for yr in [1.0, 2.0, 3.0]:
                            idx = int(np.argmin(np.abs(t - yr)))
                            plt.scatter([t[idx]], [s[idx]], color="red")
                            plt.annotate(
                                f"{s[idx]:.2f}",
                                (t[idx], s[idx]),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha="center",
                            )
                    else:
                        plt.text(0.5, 0.5, "Survival function not available for this Cox model.", ha="center")
                    
                    st.pyplot(fig)
                
                with nomogram_container:
                    nom_col1, nom_col2 = st.columns(2)
                    
                    with nom_col1:
                        st.subheader("SHAP Importance")
                        st.image(
                            str(APP_ROOT / 'pic' / 'shap_summary_plot.png'),
                            caption='SHAP Summary Plot',
                            use_container_width=True,
                        )
                    
                    with nom_col2:
                        st.subheader("Nomogram")
                        st.image(
                            str(APP_ROOT / 'pic' / 'nomogram_classical.png'),
                            caption='Classical Nomogram',
                            use_container_width=True,
                        )

# 添加页脚
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Zhongnan Hospital. All rights reserved.</div>", unsafe_allow_html=True)
