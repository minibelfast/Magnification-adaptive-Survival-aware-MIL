# MSAMapp (GBM Prognostication)

MSAMapp is a Streamlit web application for end-to-end prognosis inference from whole-slide histopathology images (WSIs) of glioma/GBM.

It performs:
- WSI upload (`.svs`, `.ndpi`, `.sdpc`)
- Patch feature extraction (e.g., UNI / CONCH; configurable)
- Slide-level MSAM score inference
- Multimodal Cox inference with clinical variables (**KPSscore, P53, ATRX**)
- Visualization: WSI thumbnail, attention heatmap, Cox feature contribution, survival curve

## Repository layout

- `Home.py`: Streamlit home page
- `pages/2_Analysis.py`: main inference + visualization page
- `pages/3_Tutorial.py`: tutorial page
- `config/default.yaml`: default runtime config (relative paths, GitHub-friendly)
- `cox_model.pkl`: Cox model (replace with your own if needed)
- `9.1.dataset_train.csv`: training-data schema used to infer the clinical input fields (replace with your own schema if needed)
- `weights/`: place model weights here (not tracked by git)

## Quick start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Install system dependencies (Linux)

OpenSlide is required for `.svs/.ndpi`:

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools
```

### 3) Provide model weights

This repository does not ship large model checkpoints.

Set one of the following:
- Put your WSI model checkpoint at `weights/wsi_model.pth`, or
- Export an environment variable:

```bash
export WSI_MODEL_PATH=/absolute/path/to/your_wsi_model.pth
```

If you use UNI / CONCH encoders, also provide their checkpoints when required:

```bash
export UNI_CKPT_PATH=/absolute/path/to/pytorch_model.bin
export CONCH_CKPT_PATH=/absolute/path/to/pytorch_model.bin
```

### 4) Run the app

```bash
streamlit run Home.py
```

Open the Analysis page:
1. Upload a WSI file
2. Select clinical variables (KPSscore, P53, ATRX)
3. Click **Submit**

## Configuration

MSAMapp reads a YAML config from `WSI_CONFIG_PATH` (default: `config/default.yaml`).

Override the default config:

```bash
export WSI_CONFIG_PATH=/absolute/path/to/your_config.yaml
```

The app also supports:
- `COX_MODEL_PATH` (default: `./cox_model.pkl`)
- `TRAIN_CSV_PATH` (default: `./9.1.dataset_train.csv`)

## Notes on clinical inputs

The Cox input feature set is inferred from `TRAIN_CSV_PATH`.
By default, the app expects:
- `MSAM` (computed from the uploaded slide)
- `KPSscore` (user input)
- `P53` (user input, 0/1)
- `ATRX` (user input, 0/1)

If your Cox model uses a different schema, update `TRAIN_CSV_PATH` accordingly.

## Disclaimer

This software is provided for research use only and is not a medical device.
Any clinical use must be validated and approved according to local regulations.

