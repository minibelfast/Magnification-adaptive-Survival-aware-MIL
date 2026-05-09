# MSAM: GBM Prognostication Model


MSAM (Magnification-Aware Multi-Instance Attention Model) is a deep learning-based application for predicting glioma/GBM prognosis directly from whole-slide histopathology images (WSIs). The system computes a continuous slide-level MSAM score from WSIs and integrates it with clinical variables (**KPSscore, P53, ATRX**) in a Cox proportional hazards model to estimate individualized risk and survival.

## Key Features

- **Information Acquisition Module**: Processes multi-format WSIs (e.g., `.svs`, `.ndpi`, `.sdpc`) and clinical data (**KPSscore, P53, ATRX**).
- **Image Processing & Feature Extraction**: Performs tissue segmentation, patching, and configurable feature encoding (e.g., ResNet-50, UNI, CONCH; set via YAML).
- **Risk & Prognosis Prediction**: Produces a slide-level MSAM score and estimates individualized survival risk using a multivariable Cox model.
- **Explainability**: Provides risk heatmaps and a feature contribution plot (Cox coefficient-based) for interpretation.
- **Nomogram**: Displays a nomogram panel (static image, if provided under `pic/`).

## Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU is recommended for faster WSI processing and inference

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MSAMapp.git
   cd MSAMapp
   ```

2. **Create a virtual environment (Optional but recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install OpenSlide (for WSI reading)**
   - **Ubuntu/Debian**: `sudo apt-get install openslide-tools`
   - **CentOS/RedHat**: `sudo yum install openslide`
   - **macOS**: `brew install openslide`
   - **Windows**: download the latest binaries from https://openslide.org/download/

## Usage

1. **Start the Streamlit App**

From the root directory of the project, run:
```bash
streamlit run Home.py
```

2. **Using the Application**

- Navigate to the **Analysis** page via the sidebar.
- **Upload a WSI Image**: upload a WSI (formats: `.svs`, `.ndpi`, `.sdpc`).
- **Input Clinical Data**: select **KPSscore**, **P53**, and **ATRX**.
- Click **Submit**.

3. **Interpreting Results**

- **Raw WSI**: view the thumbnail of the uploaded slide.
- **Risk map**: observe the spatial regions associated with higher risk.
- **Features Contribution**: inspect the contribution of MSAM/KPSscore/P53/ATRX in the Cox model.
- **Survival Rate Plot**: view the predicted survival curve over time.
- **Nomogram**: a graphical summary of the prognostic factors (if provided).

## Project Structure

- `Home.py`: main entry point for the Streamlit web application.
- `pages/`: contains subpages for the application, including the Analysis logic (`pages/2_Analysis.py`).
- `components/`, `vis_utils/`, `wsi_core/`, `utils/`: helper modules for visualization, WSI handling, and core pipeline logic.
- `models/`, `part/`: deep learning modules and custom layers used in the pipeline.
- `config/default.yaml`: GitHub-friendly default configuration file.
- `cox_model.pkl` & `9.1.dataset_train.csv`: Cox weights and training schema used by the app.
- `weights/`: place large model weights here (not tracked by git).

## Citation

If you find this project useful, please cite our manuscript:
> **Development of a magnification-adaptive multiple instance learning framework with cross-cohort continual learning**

## Acknowledgement

We thank the investigators and consortia who generated and publicly shared WSIs and clinical data, including TCGA and external validation cohorts used in our study.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
