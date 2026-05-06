# MSAM: GBM Prognostication Model

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)

MSAM (Magnification-Aware Multi-Instance Attention Model) is a deep learning-based application for predicting glioma/GBM prognosis directly from whole-slide histopathology images (WSIs). The system computes a continuous slide-level MSAM score from WSIs and integrates it with clinical variables (**KPSscore, P53, ATRX**) in a Cox proportional hazards model to estimate individualized risk and survival.

## Key Features
- **End-to-end WSI ingestion**: Supports multi-format WSIs (`.svs`, `.ndpi`, `.sdpc`) with thumbnail preview.
- **Configurable patch feature encoders**: Supports multiple pretrained backbones (e.g., ResNet-50, UNI, CONCH; configurable via YAML).
- **MSAM score inference**: Produces a slide-level MSAM risk score and attention maps for spatial visualization.
- **Multimodal Cox prognostic model**: Combines MSAM with **KPSscore, P53, ATRX** to output hazard ratio and survival curve.
- **Visualization & interpretability**: Risk heatmap, feature contribution plot (Cox coefficient-based), and survival probability curve.
- **Caching**: Stores intermediate features to accelerate repeated inference on the same slide.

## Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU is recommended for faster WSI feature extraction and inference
- OpenSlide system libraries are required for `.svs/.ndpi` on most platforms

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MSAMapp.git
   cd MSAMapp
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install OpenSlide (for WSI reading)**
   - **Ubuntu/Debian**: `sudo apt-get install openslide-tools`
   - **CentOS/RedHat**: `sudo yum install openslide`
   - **macOS**: `brew install openslide`
   - **Windows**: install OpenSlide binaries from https://openslide.org/download/

## Usage

1. **Prepare model weights**

This repository does not ship large model checkpoints. Provide the required weights using either local files or environment variables:
- WSI model checkpoint:
  - Place at `weights/wsi_model.pth`, or set:
    ```bash
    export WSI_MODEL_PATH=/absolute/path/to/your_wsi_model.pth
    ```
- UNI / CONCH checkpoints (only if you select these encoders):
  ```bash
  export UNI_CKPT_PATH=/absolute/path/to/pytorch_model.bin
  export CONCH_CKPT_PATH=/absolute/path/to/pytorch_model.bin
  ```

2. **Start the Streamlit app**
   ```bash
   streamlit run Home.py
   ```

3. **Run inference in the UI**
   - Go to the **Analysis** page
   - Upload a WSI (`.svs`, `.ndpi`, `.sdpc`)
   - Input clinical variables (**KPSscore**, **P53**, **ATRX**)
   - Click **Submit**

4. **Interpreting results**
   - **Raw WSI**: thumbnail preview of the uploaded slide
   - **Risk map**: attention-based heatmap highlighting high-risk regions
   - **Features Contribution**: Cox feature contribution plot relative to the training baseline
   - **Survival Rate Plot**: predicted survival curve over time
   - **Nomogram**: static nomogram panel (if provided under `pic/`)

## Configuration

The app reads a YAML config from `WSI_CONFIG_PATH` (default: `config/default.yaml`).

Override config:
```bash
export WSI_CONFIG_PATH=/absolute/path/to/your_config.yaml
```

Other supported environment variables:
- `COX_MODEL_PATH` (default: `./cox_model.pkl`)
- `TRAIN_CSV_PATH` (default: `./9.1.dataset_train.csv`)
- `WSI_MODEL_PATH` (default: `./weights/wsi_model.pth`)

## Project Structure
- `Home.py`: main entry point of the Streamlit app
- `pages/`: Streamlit pages (main inference is in `pages/2_Analysis.py`)
- `app_utils/`: schema inference utilities for clinical inputs
- `components/`: shared UI components
- `wsi_core/`, `vis_utils/`: WSI loading, segmentation, patching, and heatmap rendering
- `models/`: encoder builders and model utilities
- `part/`: MSAM modules (ASPP, EMA, etc.)
- `config/`: GitHub-friendly default config
- `weights/`: local weights directory (not tracked)

## Citation

If you find this project useful, please cite our manuscript:
> **Development of a magnification-adaptive multiple instance learning framework with cross-cohort continual learning**

## Acknowledgement

We thank the investigators and consortia who generated and publicly shared WSIs and clinical data, including TCGA and external validation cohorts used in our study. We also acknowledge the open-source community for foundational libraries (PyTorch, OpenSlide, Streamlit).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

