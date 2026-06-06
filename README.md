# Explainable Parkinson's Disease Screening App Using Hand-drawn Spiral and Wave Images

This project is a Streamlit educational demo that classifies hand-drawn spiral and wave images as `Healthy` or `Parkinson's` and visualizes model attention with Grad-CAM.

It is not a diagnosis tool and must not be used for medical decisions.

## Dataset

Kaggle dataset: [kmader/parkinsons-drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)

Expected local dataset location after running `data.py`:

```text
data/raw/parkinsons-drawings/
```

The scanner infers metadata from paths containing:

- `spiral` or `wave`
- `training` or `testing`
- `healthy` or `parkinson`

## Project Structure

```text
.
|-- app.py
|-- data.py
|-- train.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|   |-- 01_dataset_exploration.ipynb
|   |-- 02_baseline_cnn.ipynb
|   `-- 03_gradcam_explainability.ipynb
|-- src/
|   |-- dataset.py
|   |-- evaluate.py
|   |-- gradcam.py
|   |-- model.py
|   `-- train_utils.py
`-- assets/
    |-- sample_images/
    `-- app_screenshots/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Data

Authenticate with Kaggle if needed, then run:

```bash
python data.py
```

This uses `kagglehub.dataset_download("kmader/parkinsons-drawings")` and copies the files into `data/raw/parkinsons-drawings/`.

## Train Models

Train the baseline CNN:

```bash
python train.py --model baseline_cnn --epochs 10
```

Train MobileNetV2 transfer learning:

```bash
python train.py --model mobilenetv2 --epochs 10
```

If pretrained ImageNet weights cannot be downloaded in your environment, use:

```bash
python train.py --model mobilenetv2 --no-pretrained --epochs 10
```

Training outputs:

```text
models/baseline_cnn_parkinsons.pt
models/mobilenetv2_parkinsons.pt
models/metrics.json
```

## Run the App

```bash
streamlit run app.py
```

The app has four pages:

1. Project Overview
2. Dataset Explorer
3. Model Performance
4. Prediction App

## Model Metrics

The training script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC when both classes are present

## Educational Disclaimer

This app demonstrates a machine-learning workflow for image classification and explainability. It is not validated for clinical use, does not diagnose Parkinson's disease, and should not replace professional medical evaluation.
