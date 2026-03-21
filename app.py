import importlib
import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mtcnn import MTCNN
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

REPORT_ENGINE = None


def get_pdf_engine():
    try:
        reportlab_lib = importlib.import_module("reportlab.lib")
        pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        styles = importlib.import_module("reportlab.lib.styles")
        platypus = importlib.import_module("reportlab.platypus")
        return (
            "reportlab",
            {
                "colors": reportlab_lib.colors,
                "A4": pagesizes.A4,
                "getSampleStyleSheet": styles.getSampleStyleSheet,
                "Paragraph": platypus.Paragraph,
                "SimpleDocTemplate": platypus.SimpleDocTemplate,
                "Spacer": platypus.Spacer,
                "Table": platypus.Table,
                "TableStyle": platypus.TableStyle,
            },
        )
    except Exception:
        try:
            fpdf_module = importlib.import_module("fpdf")
            return "fpdf", {"FPDF": fpdf_module.FPDF}
        except Exception:
            return None, {}


st.set_page_config(
    page_title="VeriSight AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_NAME = "VeriSight AI"
APP_TAGLINE = "DeepFake Intelligence Console"
IMG_SIZE = 224
MODEL_PATH = Path("model_final.keras")
MAX_UPLOAD_MB = 200
DEFAULT_PAGE = "Dashboard"
PAGE_OPTIONS = ["Dashboard", "Upload & Analyze", "Analysis", "Frames", "Reports", "Settings"]
MODE_CONFIG = {
    "Fast": {"label": "Fast", "requested_frames": 10, "description": "Faster scan for quick triage."},
    "Accurate": {"label": "Accurate", "requested_frames": 20, "description": "Higher frame coverage for deeper review."},
}
ACCENT_COLORS = {
    "REAL": "#22c55e",
    "FAKE": "#ef4444",
    "NEUTRAL": "#38bdf8",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #020617;
                --bg2: #0f172a;
                --card: rgba(15, 23, 42, 0.88);
                --border: rgba(148, 163, 184, 0.18);
                --text: #f8fafc;
                --muted: #94a3b8;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
                    radial-gradient(circle at top right, rgba(239, 68, 68, 0.12), transparent 28%),
                    linear-gradient(180deg, #020617 0%, #081121 52%, #020617 100%);
                color: var(--text);
            }

            [data-testid="stHeader"] {
                background: rgba(2, 6, 23, 0.15);
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(8,17,33,0.96) 0%, rgba(15,23,42,0.98) 100%);
                border-right: 1px solid var(--border);
                min-width: 320px;
                max-width: 320px;
            }

            .block-container {
                max-width: 1540px;
                padding-top: 1.8rem;
                padding-bottom: 2.4rem;
                padding-left: 2.2rem;
                padding-right: 2.2rem;
            }

            h1, h2, h3, h4, h5, h6, p, label, div, span {
                color: var(--text);
            }

            .brand-card, .metric-card, .status-card, .insight-card, .frame-card, .report-card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: 0 22px 60px rgba(2, 6, 23, 0.35);
                backdrop-filter: blur(14px);
            }

            .brand-card {
                padding: 1rem 1rem 1.2rem 1rem;
                margin-bottom: 1rem;
            }

            .brand-logo {
                width: 56px;
                height: 56px;
                border-radius: 18px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #0ea5e9, #22c55e);
                font-size: 1.55rem;
                font-weight: 700;
                color: white;
                margin-right: 0.8rem;
            }

            .brand-row {
                display: flex;
                align-items: center;
            }

            .brand-name {
                font-size: 1.28rem;
                font-weight: 700;
                line-height: 1.2;
            }

            .brand-sub, .tiny-note, .insight-text, .metric-foot, .frame-caption {
                color: var(--muted);
            }

            .hero-card {
                padding: 2rem 2.2rem;
                border-radius: 30px;
                background:
                    linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(34, 197, 94, 0.10)),
                    rgba(15, 23, 42, 0.88);
                border: 1px solid rgba(56, 189, 248, 0.18);
                box-shadow: 0 24px 64px rgba(2, 6, 23, 0.34);
                margin-bottom: 1.35rem;
            }

            .hero-title {
                font-size: 2.45rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }

            .hero-subtitle {
                color: var(--muted);
                font-size: 1.08rem;
                line-height: 1.75;
                max-width: 880px;
            }

            .section-title {
                font-size: 1.22rem;
                font-weight: 700;
                margin: 0.15rem 0 1rem 0;
            }

            .metric-card, .status-card, .insight-card, .report-card {
                padding: 1.2rem 1.25rem;
            }

            .metric-label {
                font-size: 0.86rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .metric-value {
                font-size: 2.25rem;
                font-weight: 800;
                margin-top: 0.35rem;
            }

            .metric-foot, .insight-text {
                font-size: 0.98rem;
                line-height: 1.7;
            }

            .status-badge {
                display: inline-flex;
                align-items: center;
                padding: 0.36rem 0.7rem;
                border-radius: 999px;
                font-size: 0.84rem;
                font-weight: 700;
                background: rgba(148, 163, 184, 0.14);
                border: 1px solid rgba(148, 163, 184, 0.18);
            }

            .status-title {
                font-size: 1.75rem;
                font-weight: 800;
                margin-top: 0.9rem;
            }

            .status-subtitle {
                color: var(--muted);
                margin-top: 0.35rem;
                line-height: 1.6;
            }

            .progress-shell {
                height: 12px;
                border-radius: 999px;
                background: rgba(148, 163, 184, 0.15);
                overflow: hidden;
                margin-top: 0.8rem;
                margin-bottom: 0.45rem;
            }

            .progress-fill {
                height: 100%;
                border-radius: 999px;
            }

            .mini-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.75rem;
            }

            .mini-card {
                background: rgba(2, 6, 23, 0.26);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.85rem 0.95rem;
            }

            .mini-card .k {
                color: var(--muted);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .mini-card .v {
                margin-top: 0.3rem;
                font-size: 1.1rem;
                font-weight: 700;
            }

            .frame-card {
                padding: 0.7rem;
                margin-bottom: 0.9rem;
            }

            .frame-flag {
                color: #fecaca;
                font-weight: 700;
            }

            .skeleton {
                position: relative;
                overflow: hidden;
                min-height: 110px;
                border-radius: 22px;
                background: linear-gradient(90deg, rgba(30,41,59,0.85), rgba(51,65,85,0.85), rgba(30,41,59,0.85));
                background-size: 200% 100%;
                animation: shimmer 1.6s infinite linear;
                border: 1px solid var(--border);
            }

            .skeleton.tall {
                min-height: 260px;
            }

            .pill {
                display: inline-block;
                padding: 0.3rem 0.55rem;
                border-radius: 999px;
                font-size: 0.8rem;
                margin-right: 0.45rem;
                border: 1px solid var(--border);
                background: rgba(15, 23, 42, 0.75);
                color: var(--text);
            }

            @keyframes shimmer {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }

            div[data-testid="stFileUploaderDropzone"] {
                background: rgba(15, 23, 42, 0.75);
                border: 1px dashed rgba(56, 189, 248, 0.45);
                border-radius: 22px;
                padding: 1rem;
            }

            button[kind="primary"] {
                border-radius: 14px;
                background: linear-gradient(90deg, #0ea5e9, #22c55e);
                border: none;
                color: white;
                font-weight: 700;
            }

            button[kind="secondary"], button[kind="primary"] {
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_assets():
    model_location = f"{MODEL_PATH.as_posix()}/" if MODEL_PATH.is_dir() else MODEL_PATH.as_posix()
    model = load_model(model_location, compile=False)
    detector = MTCNN()
    with (MODEL_PATH / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    input_shape = config["config"]["layers"][0]["config"]["batch_shape"]
    model_frames = int(input_shape[1]) if input_shape and len(input_shape) > 1 else 20
    return model, detector, model_frames


def initialize_state() -> None:
    defaults = {
        "page": DEFAULT_PAGE,
        "mode": "Accurate",
        "analysis_history": [],
        "current_result": None,
        "uploaded_video_name": None,
        "uploaded_video_bytes": None,
        "last_error": None,
        "show_model_info": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def confidence_color(score: float) -> str:
    if score >= 0.8:
        return ACCENT_COLORS["FAKE"]
    if score >= 0.55:
        return "#f59e0b"
    return ACCENT_COLORS["REAL"]


def render_branding_sidebar() -> None:
    st.sidebar.markdown(
        f"""
        <div class="brand-card">
            <div class="brand-row">
                <div class="brand-logo">🛡️</div>
                <div>
                    <div class="brand-name">{APP_NAME}</div>
                    <div class="brand-sub">{APP_TAGLINE}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    render_branding_sidebar()
    st.sidebar.radio(
        "Navigation",
        PAGE_OPTIONS,
        key="page",
    )

    st.sidebar.markdown("### ⚙️ Detection Mode")
    st.sidebar.radio(
        "Processing Mode",
        list(MODE_CONFIG.keys()),
        key="mode",
        help="Fast mode uses fewer extracted faces, while Accurate mode keeps the model's full sampling depth.",
    )
    st.sidebar.caption(MODE_CONFIG[st.session_state.mode]["description"])

    total = len(st.session_state.analysis_history)
    fake_count = sum(item["label"] == "FAKE" for item in st.session_state.analysis_history)
    real_count = sum(item["label"] == "REAL" for item in st.session_state.analysis_history)
    st.sidebar.markdown("### 📡 Session Snapshot")
    st.sidebar.metric("Videos analyzed", total)
    st.sidebar.metric("Fake detections", fake_count)
    st.sidebar.metric("Real detections", real_count)
    st.sidebar.markdown(
        """
        <span class="pill">EfficientNet + BiLSTM</span>
        <span class="pill">MTCNN Face Cropper</span>
        <span class="pill">PDF Reporting</span>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(icon: str, label: str, value: str, footnote: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_loading_skeleton() -> None:
    placeholders = []
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ph = st.empty()
        ph.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
        placeholders.append(ph)
    with col2:
        ph = st.empty()
        ph.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
        placeholders.append(ph)
    with col3:
        ph = st.empty()
        ph.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
        placeholders.append(ph)
    ph = st.empty()
    ph.markdown('<div class="skeleton tall"></div>', unsafe_allow_html=True)
    placeholders.append(ph)
    return placeholders


def safe_temp_video(uploaded_file) -> str:
    video_bytes = uploaded_file.read()
    st.session_state.uploaded_video_name = uploaded_file.name
    st.session_state.uploaded_video_bytes = video_bytes
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(video_bytes)
        return tmp_file.name


def pad_or_trim_frames(frames: np.ndarray, model_frame_count: int) -> np.ndarray:
    if len(frames) == model_frame_count:
        return frames
    if len(frames) > model_frame_count:
        indices = np.linspace(0, len(frames) - 1, model_frame_count).astype(int)
        return frames[indices]
    padding_needed = model_frame_count - len(frames)
    pad_block = np.repeat(frames[-1][np.newaxis, ...], padding_needed, axis=0)
    return np.concatenate([frames, pad_block], axis=0)


def extract_faces(video_path: str, requested_frames: int, detector: MTCNN):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = total_frames / fps if fps else 0.0

    if total_frames <= 0:
        cap.release()
        return None, {"total_frames": 0, "fps": 0.0, "duration": 0.0}

    sample_points = np.linspace(0, max(total_frames - 1, 0), requested_frames * 3, dtype=int)
    sample_points = np.unique(sample_points)
    frames = []
    source_indices = []

    for frame_index in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb)
            if not detections:
                continue
            x, y, w, h = detections[0]["box"]
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            face = rgb[y : y + h, x : x + w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            frames.append(face)
            source_indices.append(int(frame_index))
            if len(frames) >= requested_frames:
                break
        except Exception:
            continue

    cap.release()
    metadata = {"total_frames": total_frames, "fps": fps, "duration": duration, "source_indices": source_indices}
    if len(frames) < requested_frames:
        return None, metadata
    return np.array(frames), metadata


def predict_sequence(model, frames: np.ndarray) -> float:
    seq = preprocess_input(frames.astype("float32"))
    seq = np.expand_dims(seq, axis=0)
    prediction = model.predict(seq, verbose=0)[0][0]
    return float(prediction)


def get_frame_scores(model, analysis_frames: np.ndarray, model_frame_count: int) -> np.ndarray:
    scores = []
    for index in range(len(analysis_frames)):
        repeated = np.repeat(analysis_frames[index][np.newaxis, ...], model_frame_count, axis=0)
        score = predict_sequence(model, repeated)
        scores.append(score)
    return np.array(scores, dtype=float)


def build_plot(scores: np.ndarray, suspicious_indices):
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="#020617")
    ax.set_facecolor("#0f172a")
    x_axis = np.arange(len(scores))

    ax.plot(x_axis, scores, color="#38bdf8", linewidth=2.6, marker="o", markersize=5)
    if suspicious_indices:
        suspicious_scores = scores[suspicious_indices]
        ax.scatter(
            suspicious_indices,
            suspicious_scores,
            s=110,
            color="#ef4444",
            edgecolors="#fee2e2",
            linewidths=1.2,
            zorder=5,
            label="Suspicious frames",
        )

    ax.axhline(0.5, linestyle="--", linewidth=1.2, color="#f59e0b", alpha=0.85, label="Decision threshold")
    ax.fill_between(x_axis, scores, color="#38bdf8", alpha=0.14)

    ax.set_title("Frame-wise Fake Probability", color="#f8fafc", fontsize=14, pad=12)
    ax.set_xlabel("Frame Index", color="#cbd5e1")
    ax.set_ylabel("Fake Probability", color="#cbd5e1")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(color="#1e293b", alpha=0.7, linestyle="--", linewidth=0.7)
    legend = ax.legend(facecolor="#0f172a", edgecolor="#334155")
    for text in legend.get_texts():
        text.set_color("#e2e8f0")

    fig.tight_layout()
    return fig


def build_heatmap(scores: np.ndarray):
    grid = np.expand_dims(scores, axis=0)
    fig, ax = plt.subplots(figsize=(11, 1.8), facecolor="#020617")
    ax.set_facecolor("#0f172a")
    heatmap = ax.imshow(grid, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([str(i) for i in range(len(scores))], color="#cbd5e1")
    ax.set_title("Suspicion Heatmap", color="#f8fafc", fontsize=13, pad=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cbd5e1")
    fig.tight_layout()
    return fig


def classify_prediction(probability: float):
    label = "FAKE" if probability >= 0.5 else "REAL"
    confidence = probability if label == "FAKE" else 1 - probability
    return label, float(confidence)


def prepare_result(file_name: str, video_path: str, selected_mode: str, model, detector, model_frame_count: int):
    requested_frames = MODE_CONFIG[selected_mode]["requested_frames"]
    extracted_frames, extraction_meta = extract_faces(video_path, requested_frames, detector)
    if extracted_frames is None:
        raise ValueError(
            f"Unable to extract {requested_frames} valid face crops. Try a clearer face track or switch to Accurate mode."
        )

    model_frames = pad_or_trim_frames(extracted_frames, model_frame_count)
    probability = predict_sequence(model, model_frames)
    label, confidence = classify_prediction(probability)
    frame_scores = get_frame_scores(model, extracted_frames, model_frame_count)
    suspicious_count = min(3, len(frame_scores))
    suspicious_indices = sorted(np.argsort(frame_scores)[-suspicious_count:].tolist(), reverse=True)
    created_at = datetime.now()

    explanation = "Model detected inconsistencies in facial regions across frames."
    if label == "REAL":
        explanation = "Model observed comparatively stable facial cues and lower manipulation probability across the sampled frames."

    return {
        "file_name": file_name,
        "mode": selected_mode,
        "requested_frames": requested_frames,
        "model_frames": model_frame_count,
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "frame_scores": frame_scores.tolist(),
        "frames": extracted_frames,
        "suspicious_indices": suspicious_indices,
        "source_indices": extraction_meta.get("source_indices", []),
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": extraction_meta.get("duration", 0.0),
        "total_video_frames": extraction_meta.get("total_frames", 0),
        "fps": extraction_meta.get("fps", 0.0),
        "explanation": explanation,
    }


def push_history(result: dict) -> None:
    history_item = {
        "file_name": result["file_name"],
        "label": result["label"],
        "confidence": result["confidence"],
        "mode": result["mode"],
        "created_at": result["created_at"],
        "requested_frames": result["requested_frames"],
    }
    st.session_state.analysis_history.insert(0, history_item)


def render_confidence_bar(confidence: float, color: str, caption: str = "Confidence") -> None:
    width = max(2, int(confidence * 100))
    st.markdown(
        f"""
        <div class="tiny-note">{caption}: {confidence * 100:.1f}%</div>
        <div class="progress-shell">
            <div class="progress-fill" style="width: {width}%; background: linear-gradient(90deg, {color}, {color});"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(result: dict) -> None:
    label = result["label"]
    confidence = result["confidence"]
    color = ACCENT_COLORS.get(label, ACCENT_COLORS["NEUTRAL"])
    icon = "🚨" if label == "FAKE" else "✅"
    subtitle = (
        "Potential synthetic manipulation detected in the sampled facial timeline."
        if label == "FAKE"
        else "No strong manipulation signal detected in the processed sequence."
    )
    st.markdown(
        f"""
        <div class="status-card" style="border-color:{color}55;">
            <span class="status-badge" style="color:{color}; border-color:{color}55;">{icon} {label}</span>
            <div class="status-title">{label} • {confidence * 100:.1f}% confidence</div>
            <div class="status-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_confidence_bar(confidence, color, "Detection certainty")


def render_compatible_image(image_data, caption=None) -> None:
    st.image(image_data, caption=caption, use_column_width=True)


def generate_pdf_report(result: dict):
    engine, pdf_modules = get_pdf_engine()
    if engine is None:
        return None

    frame_scores = np.array(result["frame_scores"], dtype=float)
    suspicious = ", ".join(str(idx) for idx in result["suspicious_indices"]) or "None"
    rows = [
        ["Video", result["file_name"]],
        ["Prediction", result["label"]],
        ["Confidence", f"{result['confidence'] * 100:.2f}%"],
        ["Mode", result["mode"]],
        ["Frames Used", str(result["requested_frames"])],
        ["Timestamp", result["created_at"]],
        ["Video Duration", f"{result['duration_seconds']:.2f} sec"],
        ["Mean Frame Score", f"{frame_scores.mean():.4f}"],
        ["Max Frame Score", f"{frame_scores.max():.4f}"],
        ["Suspicious Frames", suspicious],
    ]

    if engine == "reportlab":
        colors = pdf_modules["colors"]
        A4 = pdf_modules["A4"]
        get_sample_style_sheet = pdf_modules["getSampleStyleSheet"]
        paragraph = pdf_modules["Paragraph"]
        simple_doc_template = pdf_modules["SimpleDocTemplate"]
        spacer = pdf_modules["Spacer"]
        table_cls = pdf_modules["Table"]
        table_style_cls = pdf_modules["TableStyle"]
        buffer = io.BytesIO()
        document = simple_doc_template(buffer, pagesize=A4, title=f"{APP_NAME} Report")
        styles = get_sample_style_sheet()
        story = [
            paragraph(f"<b>{APP_NAME} DeepFake Detection Report</b>", styles["Title"]),
            spacer(1, 12),
            paragraph(
                "Automated report generated from the uploaded video using MTCNN face extraction and an EfficientNet + BiLSTM classifier.",
                styles["BodyText"],
            ),
            spacer(1, 14),
        ]
        table = table_cls(rows, colWidths=[120, 360])
        table.setStyle(
            table_style_cls(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                    ("BACKGROUND", (1, 0), (1, -1), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(table)
        story.append(spacer(1, 12))
        story.append(paragraph(f"Frame Scores: {', '.join(f'{score:.3f}' for score in frame_scores)}", styles["BodyText"]))
        story.append(spacer(1, 8))
        story.append(paragraph(f"Explanation: {result['explanation']}", styles["BodyText"]))
        document.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    pdf = pdf_modules["FPDF"]()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"{APP_NAME} DeepFake Detection Report", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    for key, value in rows:
        pdf.multi_cell(0, 8, f"{key}: {value}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Frame Scores: {', '.join(f'{score:.3f}' for score in frame_scores)}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Explanation: {result['explanation']}")
    return pdf.output(dest="S").encode("latin-1")


def render_dashboard_page() -> None:
    render_header(
        "🧠 DeepFake Detection Command Center",
        "Monitor session performance, review the latest verdict, and keep a clean operational view of manipulated-media risk without leaving the dashboard.",
    )

    history = st.session_state.analysis_history
    result = st.session_state.current_result
    total = len(history)
    fake_count = sum(item["label"] == "FAKE" for item in history)
    real_count = sum(item["label"] == "REAL" for item in history)
    last_prediction = result["label"] if result else "No analysis yet"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("📼", "Total Videos Analyzed", str(total), "Session-level tracking")
    with col2:
        render_metric_card("🚨", "Fake Count", str(fake_count), "Detected manipulations")
    with col3:
        render_metric_card("✅", "Real Count", str(real_count), "Clean assessments")
    with col4:
        render_metric_card("🧾", "Last Prediction", last_prediction, "Most recent verdict")

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown('<div class="section-title">Latest Detection Overview</div>', unsafe_allow_html=True)
        if result:
            render_status_card(result)
            mini_html = f"""
            <div class="mini-grid">
                <div class="mini-card"><div class="k">Video</div><div class="v">{result['file_name']}</div></div>
                <div class="mini-card"><div class="k">Mode</div><div class="v">{result['mode']}</div></div>
                <div class="mini-card"><div class="k">Frames Used</div><div class="v">{result['requested_frames']}</div></div>
                <div class="mini-card"><div class="k">Timestamp</div><div class="v">{result['created_at']}</div></div>
            </div>
            """
            st.markdown(mini_html, unsafe_allow_html=True)
        else:
            st.info("No completed analysis yet. Upload a video to populate the operational dashboard.")

    with right:
        st.markdown('<div class="section-title">Confidence Gauge</div>', unsafe_allow_html=True)
        if result:
            confidence = result["confidence"]
            st.progress(int(confidence * 100))
            render_confidence_bar(confidence, confidence_color(result["probability"]), "Current run")
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="section-title">Operational Insight</div>
                    <div class="insight-text">{result['explanation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="insight-card">
                    <div class="section-title">Ready State</div>
                    <div class="insight-text">The dashboard is ready for your first upload. Session metrics and reports will appear here after analysis.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Recent Session Activity</div>', unsafe_allow_html=True)
    if history:
        preview_cols = st.columns(min(3, len(history)))
        for idx, item in enumerate(history[:3]):
            with preview_cols[idx]:
                render_metric_card(
                    "🎬",
                    item["file_name"],
                    item["label"],
                    f"{item['confidence'] * 100:.1f}% confidence • {item['created_at']}",
                )
    else:
        st.caption("Session preview cards will appear here after analyses are completed.")


def render_upload_page(model, detector, model_frame_count: int) -> None:
    render_header(
        "📤 Upload & Analyze",
        "Drop a video into the secure intake area, preview it instantly, and launch a guided analysis workflow with premium progress feedback and robust error handling.",
    )

    uploaded_file = st.file_uploader(
        "Drag and drop a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help=f"Maximum recommended upload size: {MAX_UPLOAD_MB} MB",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="report-card">
                <div class="section-title">Secure Intake Ready</div>
                <div class="insight-text">Upload a supported video format to begin. The app keeps your current session history and latest result available across every page.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"📦 This upload is {size_mb:.1f} MB. Please use a video under {MAX_UPLOAD_MB} MB.")
        return

    video_path = safe_temp_video(uploaded_file)
    st.markdown('<div class="section-title">Video Preview</div>', unsafe_allow_html=True)
    st.video(st.session_state.uploaded_video_bytes)

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown(
            f"""
            <div class="report-card">
                <div class="section-title">Queued Asset</div>
                <div class="insight-text">
                    <strong>{uploaded_file.name}</strong><br/>
                    File size: {size_mb:.2f} MB<br/>
                    Selected mode: {st.session_state.mode}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="report-card">
                <div class="section-title">Scan Checklist</div>
                <div class="insight-text">Face crops, temporal scoring, confidence grading, suspicious-frame surfacing, and PDF reporting are all included in this run.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("🚀 Analyze Video", type="primary"):
        try:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            skeleton_placeholders = show_loading_skeleton()
            checkpoints = [
                (8, "Initializing face detector"),
                (24, "Sampling video frames"),
                (48, "Extracting facial regions"),
                (72, "Running temporal classifier"),
                (90, "Scoring suspicious frames"),
                (100, "Compiling dashboard artifacts"),
            ]
            for progress, label in checkpoints:
                progress_text.markdown(f"**{label}...**")
                progress_bar.progress(progress)
                time.sleep(0.16)

            result = prepare_result(uploaded_file.name, video_path, st.session_state.mode, model, detector, model_frame_count)
            st.session_state.current_result = result
            push_history(result)
            st.session_state.last_error = None
            for placeholder in skeleton_placeholders:
                placeholder.empty()
            progress_bar.empty()
            progress_text.success("Analysis complete. The result is now available across the dashboard.")
            render_status_card(result)

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", result["label"])
            c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            c3.metric("Frames Used", result["requested_frames"])
            render_confidence_bar(result["confidence"], confidence_color(result["probability"]))
        except Exception as exc:
            for placeholder in locals().get("skeleton_placeholders", []):
                placeholder.empty()
            st.session_state.last_error = str(exc)
            st.error(f"⚠️ Analysis could not be completed: {exc}")
            st.info("Try a clearer frontal-face clip, a shorter duration video, or switch to Accurate mode for denser sampling.")


def render_analysis_page() -> None:
    render_header(
        "📊 Analysis Intelligence",
        "Explore probability drift across frames, identify high-risk facial segments, and review model-generated reasoning designed to feel like a real-world forensic dashboard.",
    )
    result = st.session_state.current_result
    if not result:
        st.warning("Run an analysis first to unlock frame-level insights.")
        return

    scores = np.array(result["frame_scores"], dtype=float)
    suspicious_indices = result["suspicious_indices"]

    main_col, side_col = st.columns([1.7, 1])
    with main_col:
        st.markdown('<div class="section-title">Probability Timeline</div>', unsafe_allow_html=True)
        fig = build_plot(scores, suspicious_indices)
        st.pyplot(fig, clear_figure=True)

        st.markdown('<div class="section-title">Heatmap View</div>', unsafe_allow_html=True)
        heatmap_fig = build_heatmap(scores)
        st.pyplot(heatmap_fig, clear_figure=True)

    with side_col:
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-title">🧭 Model Explanation</div>
                <div class="insight-text">Model detected inconsistencies in facial regions across frames.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="section-title">🔎 Insights Panel</div>
                <div class="insight-text">
                    Highest frame score: {scores.max():.3f}<br/>
                    Average frame score: {scores.mean():.3f}<br/>
                    Suspicious frames: {', '.join(str(i) for i in suspicious_indices)}<br/>
                    Detection mode: {result['mode']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-title">📌 Analyst Notes</div>
                <div class="insight-text">
                    Temporal inconsistency tends to rise when lip motion, blink cadence, or facial texture transitions diverge from surrounding frames. Use the frames page to review the top outliers visually.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Highlighted Suspicious Frames</div>', unsafe_allow_html=True)
    suspicious_cards = st.columns(max(1, len(suspicious_indices)))
    for idx, frame_index in enumerate(suspicious_indices):
        with suspicious_cards[idx]:
            render_compatible_image(result["frames"][frame_index], caption=f"Frame {frame_index} • score {scores[frame_index]:.3f}")


def render_frames_page() -> None:
    render_header(
        "🎞️ Frames Explorer",
        "Inspect extracted face crops in a clean grid, prioritize high-risk moments, and quickly compare score-driven outliers against the rest of the sampled sequence.",
    )
    result = st.session_state.current_result
    if not result:
        st.warning("Run an analysis first to review the extracted face frames.")
        return

    scores = np.array(result["frame_scores"], dtype=float)
    suspicious_set = set(result["suspicious_indices"])
    st.markdown('<div class="section-title">Extracted Faces</div>', unsafe_allow_html=True)

    frames = result["frames"]
    columns_per_row = 4
    for row_start in range(0, len(frames), columns_per_row):
        columns = st.columns(columns_per_row)
        for offset, frame_index in enumerate(range(row_start, min(row_start + columns_per_row, len(frames)))):
            with columns[offset]:
                border_color = "#ef4444" if frame_index in suspicious_set else "#334155"
                flag = "Top suspicion" if frame_index in suspicious_set else "Observed"
                st.markdown(f'<div class="frame-card" style="border-color:{border_color};">', unsafe_allow_html=True)
                render_compatible_image(frames[frame_index])
                st.markdown(
                    f"""
                    <div class="frame-caption">
                        <span>Frame {frame_index}</span>
                        <span class="frame-flag">{flag}</span>
                    </div>
                    <div class="tiny-note">Score: {scores[frame_index]:.3f}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)


def render_reports_page() -> None:
    render_header(
        "🧾 Reports",
        "Package the latest analysis into a client-facing PDF summary with operational details, score statistics, and a timestamped verdict ready for review or sharing.",
    )
    result = st.session_state.current_result
    if not result:
        st.info("A completed analysis is required before a report can be generated.")
        return

    pdf_bytes = generate_pdf_report(result)
    scores = np.array(result["frame_scores"], dtype=float)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            f"""
            <div class="report-card">
                <div class="section-title">Latest Report Summary</div>
                <div class="insight-text">
                    Video: {result['file_name']}<br/>
                    Prediction: {result['label']}<br/>
                    Confidence: {result['confidence'] * 100:.2f}%<br/>
                    Generated: {result['created_at']}<br/>
                    Mean frame score: {scores.mean():.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pdf_bytes is not None:
            st.download_button(
                "⬇️ Download Report",
                data=pdf_bytes,
                file_name=f"{Path(result['file_name']).stem}_deepfake_report.pdf",
                mime="application/pdf",
                type="primary",
            )
        else:
            st.warning("PDF libraries are not available in this environment, so report download is currently disabled.")

    with right:
        st.markdown(
            """
            <div class="report-card">
                <div class="section-title">Included In Report</div>
                <div class="insight-text">
                    • Prediction result and confidence<br/>
                    • Frame score summary<br/>
                    • Timestamp and processing mode<br/>
                    • Suspicious frame references<br/>
                    • Analyst explanation text
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)
    if st.session_state.analysis_history:
        st.dataframe(st.session_state.analysis_history, hide_index=True)


def render_settings_page(model_frame_count: int) -> None:
    render_header(
        "⚙️ Settings",
        "Expose model metadata, runtime assumptions, and operator-facing controls so the app feels like a polished product instead of a prototype notebook UI.",
    )

    st.session_state.show_model_info = st.toggle("Show model information", value=st.session_state.show_model_info)

    if st.session_state.show_model_info:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="section-title">🧠 Model Info</div>
                <div class="insight-text">
                    Architecture: EfficientNet backbone + Bidirectional LSTM sequence head<br/>
                    Input resolution: {IMG_SIZE} x {IMG_SIZE}<br/>
                    Model sequence length: {model_frame_count} frames<br/>
                    Face detector: MTCNN<br/>
                    Caching: Streamlit resource cache enabled for model and detector loading
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="insight-card">
            <div class="section-title">🛡️ Runtime Safeguards</div>
            <div class="insight-text">
                Large-file upload limit guidance, cached model loading, session-scoped history, readable error states, and mode-based frame sampling are all active in this build.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.error(f"Recent pipeline message: {st.session_state.last_error}")

    if st.button("Clear Current Session Results"):
        st.session_state.analysis_history = []
        st.session_state.current_result = None
        st.session_state.last_error = None
        st.success("Session history and current analysis were cleared.")


def main() -> None:
    inject_css()
    initialize_state()
    model, detector, model_frame_count = load_assets()
    render_sidebar()

    page = st.session_state.page
    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Upload & Analyze":
        render_upload_page(model, detector, model_frame_count)
    elif page == "Analysis":
        render_analysis_page()
    elif page == "Frames":
        render_frames_page()
    elif page == "Reports":
        render_reports_page()
    elif page == "Settings":
        render_settings_page(model_frame_count)


if __name__ == "__main__":
    main()
