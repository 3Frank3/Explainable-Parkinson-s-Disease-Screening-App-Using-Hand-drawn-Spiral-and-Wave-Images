from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dataset import IDX_TO_CLASS, discover_image_records, find_examples, get_eval_transform, summarize_records


PROJECT_TITLE = "Explainable Parkinson's Disease Screening App"
DATA_DIR = Path("data/raw/parkinsons-drawings")
MODEL_DIR = Path("models")
DEFAULT_MODEL = MODEL_DIR / "mobilenetv2_parkinsons.pt"


st.set_page_config(
    page_title="Parkinson's Drawing Screening Demo",
    layout="wide",
)


def resolve_data_dir() -> Path:
    if DATA_DIR.exists():
        return DATA_DIR
    return Path("data/raw")


@st.cache_data(show_spinner=False)
def load_records(data_dir: str):
    return discover_image_records(data_dir)


@st.cache_resource(show_spinner=False)
def load_checkpoint(model_path: str):
    from src.model import load_model_from_checkpoint

    return load_model_from_checkpoint(model_path, map_location="cpu")


def caution_box() -> None:
    st.warning(
        "Educational use only. This app is not a medical device, does not diagnose Parkinson's disease, "
        "and should not be used to make health decisions."
    )


def page_overview() -> None:
    st.title(PROJECT_TITLE)
    st.subheader("Using hand-drawn spiral and wave images with explainable deep learning")
    caution_box()

    st.write(
        "This app demonstrates how machine learning can classify hand-drawn spiral and wave images "
        "as Healthy or Parkinson's. It is intended for educational purposes only and should not be "
        "used for medical diagnosis."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Input", "Spiral / Wave image")
    col2.metric("Prediction", "Healthy / Parkinson's")
    col3.metric("Explanation", "Grad-CAM heatmap")

    st.markdown(
        """
        Parkinson's disease is a movement disorder. Handwriting and drawing patterns can reflect
        tremor, motor control changes, and fine motor impairment. This project uses those drawings
        to build a compact image-classification workflow:

        1. Explore the Kaggle Parkinson's drawings dataset.
        2. Train a baseline CNN and a MobileNetV2 transfer-learning model.
        3. Evaluate the models with classification metrics.
        4. Upload a drawing and inspect a Grad-CAM explanation.
        """
    )


def page_dataset_explorer(records) -> None:
    st.title("Dataset Explorer")
    data_dir = resolve_data_dir()
    st.caption(f"Dataset directory: `{data_dir}`")

    if not records:
        st.info("No labeled drawing images were found yet. Run `python data.py` to download the Kaggle dataset.")
        return

    summary_df = pd.DataFrame(summarize_records(records))
    total_images = len(records)
    st.metric("Discovered labeled images", total_images)

    st.subheader("Train/Test Counts")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    pivot = summary_df.pivot_table(
        index=["split", "drawing_type"],
        columns="class",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    st.subheader("Class Distribution")
    st.bar_chart(summary_df, x="class", y="count", color="drawing_type")
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    st.subheader("Example Images")
    examples = [
        ("Healthy Spiral", "spiral", "healthy"),
        ("Parkinson Spiral", "spiral", "parkinson"),
        ("Healthy Wave", "wave", "healthy"),
        ("Parkinson Wave", "wave", "parkinson"),
    ]
    for title, drawing_type, label in examples:
        image_paths = find_examples(records, drawing_type=drawing_type, label=label, limit=4)
        st.markdown(f"**{title}**")
        if image_paths:
            st.image([str(path) for path in image_paths], width=160)
        else:
            st.caption("No example found in the current dataset directory.")


def _metric_value(metrics: dict, key: str):
    value = metrics.get("test_metrics", {}).get(key)
    if value is None:
        return None
    return round(float(value), 4)


def _plot_confusion_matrix(matrix: list[list[int]], title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 3.5))
    matrix_np = np.asarray(matrix)
    image = ax.imshow(matrix_np, cmap="Blues")
    ax.set_xticks([0, 1], ["Healthy", "Parkinson's"])
    ax.set_yticks([0, 1], ["Healthy", "Parkinson's"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for row in range(matrix_np.shape[0]):
        for col in range(matrix_np.shape[1]):
            ax.text(col, row, int(matrix_np[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def page_model_performance() -> None:
    from src.evaluate import load_metrics

    st.title("Model Performance")
    st.write("The project compares a small baseline CNN against MobileNetV2 transfer learning.")

    comparison = pd.DataFrame(
        [
            {
                "Model": "Baseline CNN",
                "Why": "A compact CNN trained from scratch to learn the deep-learning basics.",
            },
            {
                "Model": "MobileNetV2 Transfer Learning",
                "Why": "A lightweight pretrained image model that fits a Streamlit demo workflow.",
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    metrics = load_metrics(MODEL_DIR / "metrics.json")
    if not metrics:
        st.info(
            "No metrics found yet. Train models with `python train.py --model baseline_cnn` and "
            "`python train.py --model mobilenetv2`."
        )
        return

    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Drawing Type": model_metrics.get("drawing_type", "all"),
                "Accuracy": _metric_value(model_metrics, "accuracy"),
                "Precision": _metric_value(model_metrics, "precision"),
                "Recall": _metric_value(model_metrics, "recall"),
                "F1-score": _metric_value(model_metrics, "f1"),
                "ROC-AUC": _metric_value(model_metrics, "roc_auc"),
            }
        )

    st.subheader("Evaluation Metrics")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Confusion Matrices")
    columns = st.columns(min(2, len(metrics)))
    for index, (model_name, model_metrics) in enumerate(metrics.items()):
        matrix = model_metrics.get("test_metrics", {}).get("confusion_matrix")
        if matrix:
            with columns[index % len(columns)]:
                _plot_confusion_matrix(matrix, model_name)


def available_model_paths() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob("*_parkinsons.pt"))


def page_prediction_app() -> None:
    from PIL import Image

    st.title("Prediction App")
    caution_box()

    uploaded_file = st.file_uploader("Upload a spiral or wave image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    drawing_type = st.radio("Drawing type", ["spiral", "wave"], horizontal=True)

    model_paths = available_model_paths()
    if not model_paths and DEFAULT_MODEL.exists():
        model_paths = [DEFAULT_MODEL]

    if not model_paths:
        st.info("No trained model found. Run `python train.py --model mobilenetv2` to create a checkpoint.")
        if uploaded_file is not None:
            st.image(Image.open(uploaded_file).convert("RGB"), caption="Uploaded image", width=320)
        return

    selected_model = st.selectbox("Model checkpoint", model_paths, format_func=lambda path: path.name)

    if uploaded_file is None:
        st.caption("Upload an image to run prediction and Grad-CAM.")
        return

    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Uploaded image", width=340)

    if st.button("Predict", type="primary"):
        import torch
        from src.gradcam import generate_gradcam_overlay
        from src.model import get_gradcam_target_layer

        with st.spinner("Running model inference and Grad-CAM..."):
            model, checkpoint = load_checkpoint(str(selected_model))
            model_name = checkpoint.get("model_name", "mobilenetv2")
            image_size = int(checkpoint.get("image_size", 224))
            trained_drawing_type = checkpoint.get("drawing_type", "all")

            if trained_drawing_type not in {"all", drawing_type}:
                st.warning(
                    f"This checkpoint was trained for `{trained_drawing_type}` drawings, "
                    f"but `{drawing_type}` was selected."
                )

            transform = get_eval_transform(image_size)
            image_tensor = transform(original_image).unsqueeze(0)

            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0)
                class_idx = int(probabilities.argmax().item())
                confidence = float(probabilities[class_idx].item())

            target_layer = get_gradcam_target_layer(model, model_name)
            overlay = generate_gradcam_overlay(
                model=model,
                target_layer=target_layer,
                image_tensor=image_tensor,
                original_image=original_image,
                class_idx=class_idx,
            )

        col1, col2 = st.columns([0.9, 1.1])
        with col1:
            st.metric("Predicted class", IDX_TO_CLASS[class_idx])
            st.metric("Confidence", f"{confidence:.1%}")
            st.caption("Confidence is the model's softmax probability, not clinical certainty.")
        with col2:
            st.image(overlay, caption="Grad-CAM heatmap overlay", width=420)


def main() -> None:
    data_dir = resolve_data_dir()
    records = load_records(str(data_dir))

    page = st.sidebar.radio(
        "Navigation",
        ["Project Overview", "Dataset Explorer", "Model Performance", "Prediction App"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Educational ML demo, not a diagnosis tool.")

    if page == "Project Overview":
        page_overview()
    elif page == "Dataset Explorer":
        page_dataset_explorer(records)
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Prediction App":
        page_prediction_app()


if __name__ == "__main__":
    main()
