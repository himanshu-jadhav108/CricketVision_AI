from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from config import (
    ASSETS_DIR,
    DOCS_DIR,
    EXAMPLES_DIR,
    GRADIO_HOST,
    GRADIO_PORT,
    GRADIO_SHARE,
    LOG_LEVEL,
    LOGO_PATH,
    MODEL_DIR,
    MODEL_PATH,
    ENCODER_PATH,
    PROJECT_NAME,
    REPORTS_DIR,
    SCALER_PATH,
    missing_model_files,
)
from predictor import ShotPredictor

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_production_evaluation_metrics() -> dict[str, str]:
    """Parses metrics dynamically from reports/evaluation.md and docs/training.md."""
    eval_file = REPORTS_DIR / "evaluation.md"
    training_file = DOCS_DIR / "training.md"

    metrics = {
        "overall_acc": "77.53%",
        "cv_score": "78.13%",
        "post_acc": "92.52%",
        "reject_rate": "32.4%",
    }

    if eval_file.exists():
        try:
            eval_text = eval_file.read_text(encoding="utf-8")
            m_raw = re.search(r"Overall Accuracy(?:\s*\(Raw\))?:\s*([0-9.]+%)", eval_text)
            if m_raw:
                metrics["overall_acc"] = m_raw.group(1)

            m_post = re.search(r"Accuracy\s*\(Post-Threshold\):\s*([0-9.]+%)", eval_text)
            if m_post:
                metrics["post_acc"] = m_post.group(1)

            m_rej = re.search(r"Rejected Samples:\s*\d+\s*\(([0-9.]+%)", eval_text)
            if m_rej:
                metrics["reject_rate"] = m_rej.group(1)
        except Exception as err:
            logger.warning("Failed to parse %s: %s", eval_file, err)

    if training_file.exists():
        try:
            train_text = training_file.read_text(encoding="utf-8")
            m_cv = re.search(r"(?:Cross-[Vv]alidation.*?|CV.*?|CV Mean Accuracy:\s*)\*\*([0-9.]+%)\*\*", train_text)
            if m_cv:
                metrics["cv_score"] = m_cv.group(1)
        except Exception as err:
            logger.warning("Failed to parse %s: %s", training_file, err)

    return metrics


PROD_METRICS = load_production_evaluation_metrics()


def get_logo_html() -> str:
    """Encodes the project logo to base64 for seamless, buttonless inline HTML rendering."""
    if LOGO_PATH.exists():
        try:
            with open(LOGO_PATH, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{b64}" class="brand-logo" alt="{PROJECT_NAME}">'
        except Exception:
            return ""
    return ""


LOGO_HTML = get_logo_html()


def get_performance_gallery_items() -> list[tuple[str, str]]:
    """Collects paths and descriptive captions for the Model Performance tab."""
    items = []
    
    cm_path = ASSETS_DIR / "performance" / "eval_confusion_matrix_pct.png"
    if cm_path.exists():
        items.append((
            str(cm_path),
            "Confusion Matrix (%) — Row-normalized classification rate across 6 shot classes (post-threshold @ 0.65).",
        ))

    cm_raw_path = ASSETS_DIR / "performance" / "confusion_matrix.png"
    if cm_raw_path.exists():
        items.append((
            str(cm_raw_path),
            "Confusion Matrix (Counts) — Raw sample test counts across 882 accepted predictions.",
        ))

    f1_path = ASSETS_DIR / "performance" / "eval_per_class_f1.png"
    if f1_path.exists():
        items.append((
            str(f1_path),
            "Per-Class F1 Score — Post-threshold F1 metrics across all 6 production shot classes.",
        ))

    fi_path = ASSETS_DIR / "performance" / "feature_importance.png"
    if fi_path.exists():
        items.append((
            str(fi_path),
            "Feature Importance — Top 20 3D biomechanical features contributing to XGBoost classification.",
        ))

    sweep_path = REPORTS_DIR / "threshold_sweep.png"
    if sweep_path.exists():
        items.append((
            str(sweep_path),
            "Threshold Sensitivity Sweep — Trade-off between post-threshold accuracy and sample rejection rate (0.50 to 0.80).",
        ))

    hist_path = ASSETS_DIR / "performance" / "eval_confidence_histogram.png"
    if hist_path.exists():
        items.append((
            str(hist_path),
            "Confidence Distribution — Winning softmax probability spread on test set with 0.65 guardrail threshold.",
        ))

    dist_path = ASSETS_DIR / "performance" / "eda_class_distribution.png"
    if dist_path.exists():
        items.append((
            str(dist_path),
            "Class Distribution — Sample representation across the 6 active training classes (6,324 total samples).",
        ))

    box_path = ASSETS_DIR / "performance" / "eda_boxplots.png"
    if box_path.exists():
        items.append((
            str(box_path),
            "Joint Angle Variance — 3D elbow, shoulder, and knee angle distributions across shot types.",
        ))

    return items


def models_exist() -> bool:
    return not missing_model_files()


predictor = None


def get_predictor() -> ShotPredictor | None:
    global predictor
    if predictor is None:
        if not models_exist():
            return None
        logger.info("Loading predictor from %s", MODEL_DIR)
        predictor = ShotPredictor(str(MODEL_PATH), str(SCALER_PATH), str(ENCODER_PATH))
    return predictor


# Metadata for rich UI display (Emoji, Human Name, Technical Kinetic Definition)
SHOT_META = {
    'cover_drive': (
        '🏏',
        'Cover Drive',
        'Front-foot drive through the off-side with high lead elbow and forward knee extension.'
    ),
    'pull_shot': (
        '💪',
        'Pull Shot',
        'Aggressive cross-bat hook/pull to short ball; horizontal bat arc with open torso alignment.'
    ),
    'leg_glance_shot': (
        '⬅️',
        'Leg Glance',
        'Subtle wrist deflection off hip/pads to fine leg with closed shoulder face.'
    ),
    'cut_shot': (
        '✂️',
        'Cut Shot',
        'Back-foot square cut through point; elevated wrists and backward center of mass.'
    ),
    'scoop_shot': (
        '🥄',
        'Scoop Shot',
        'Unorthodox ramp loft over wicketkeeper; low crouched stance with vertical bat lift.'
    ),
    'sweep_shot': (
        '🧹',
        'Sweep Shot',
        'Front-knee knelt paddle sweep to the leg side with low rotational center of gravity.'
    ),
    'Uncertain Shot': (
        '🛡️',
        'Uncertain Shot',
        'Confidence below 65% guardrail. Frame kinematics are ambiguous or in mid-transition.'
    ),
}


def format_results(label: str, confidence: float, all_probs: dict[str, float]) -> str:
    """Generates modern, responsive diagnostic HTML for the results card."""
    emoji, name, desc = SHOT_META.get(label, ('🏏', label, ''))
    conf_pct = confidence * 100
    is_uncertain = label == 'Uncertain Shot'

    if is_uncertain:
        accent_color = "#f59e0b"  # Amber warning
        badge_html = """
        <div class="status-badge badge-warning">
            <span class="badge-dot dot-warning"></span>
            <span>Confidence Guardrail Active (&lt; 65.0%)</span>
        </div>
        """
        explanation_html = f"""
        <div class="abstention-banner">
            <div class="abstention-icon">⚠️</div>
            <div class="abstention-text">
                <strong>Selective Abstention:</strong> The model evaluated this posture with <strong>{conf_pct:.1f}% confidence</strong>, 
                falling below the 65.0% threshold. The kinematics may be transitional or partially occluded. 
                Abstaining maintains coaching trust rather than guessing.
            </div>
        </div>
        """
    else:
        accent_color = "#10b981" if conf_pct >= 80 else "#38bdf8"
        badge_html = f"""
        <div class="status-badge badge-success">
            <span class="badge-dot dot-success"></span>
            <span>Verified Posture &bull; {conf_pct:.1f}% Confidence</span>
        </div>
        """
        explanation_html = ""

    # Sort probabilities descending
    sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])

    prob_rows = []
    for shot, prob in sorted_probs:
        sh_em, sh_nm, _ = SHOT_META.get(shot, ('🏏', shot, ''))
        bar_w = prob * 100
        is_winner = (shot == label) and not is_uncertain
        row_cls = "prob-row winner" if is_winner else "prob-row"
        fill_bg = "linear-gradient(90deg, #38bdf8, #818cf8)" if is_winner else "#334155"
        val_color = accent_color if is_winner else "#94a3b8"

        prob_rows.append(f"""
        <div class="{row_cls}">
            <div class="prob-info">
                <span class="prob-name">{'⚡ ' if is_winner else ''}{sh_em} {sh_nm}</span>
                <span class="prob-val" style="color: {val_color};">{bar_w:.1f}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: {bar_w:.1f}%; background: {fill_bg};"></div>
            </div>
        </div>
        """)

    probs_html = "".join(prob_rows)

    return f"""
    <div class="result-card-inner">
        <div class="result-top-bar">
            {badge_html}
            <span class="latency-pill">⚡ Model Latency: &lt; 1ms</span>
        </div>
        
        <div class="result-header">
            <div class="result-kicker">{'PREDICTED SHOT' if not is_uncertain else 'UNCERTAIN POSTURE'}</div>
            <div class="result-title-group">
                <span class="result-emoji">{emoji}</span>
                <h2 class="result-title">{name}</h2>
            </div>
            <p class="result-desc">{desc}</p>
        </div>

        {explanation_html}

        <div class="metric-gauge-card">
            <div class="gauge-ring" style="background: conic-gradient({accent_color} {conf_pct}%, #1e293b 0);">
                <div class="gauge-center">
                    <span class="gauge-pct" style="color: {accent_color};">{conf_pct:.1f}%</span>
                    <span class="gauge-sub">Confidence</span>
                </div>
            </div>
            <div class="gauge-meta">
                <div class="meta-item">
                    <span class="meta-label">Confidence Gate</span>
                    <span class="meta-value">&ge; 65.0%</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Decision Status</span>
                    <span class="meta-value" style="color: {accent_color}; font-weight: 700;">{'ACCEPTED' if not is_uncertain else 'UNCERTAIN'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Selective Accuracy</span>
                    <span class="meta-value">92.52%</span>
                </div>
            </div>
        </div>

        <div class="probs-section">
            <h4 class="probs-heading">Biomechanical Softmax Distribution</h4>
            <div class="probs-list">
                {probs_html}
            </div>
        </div>
    </div>
    """


def classify_shot(image: Image.Image | np.ndarray | None) -> tuple[str, Image.Image | None]:
    """Processes images and returns diagnostic results HTML and annotated pose."""
    if image is None:
        return """
        <div class="status-callout callout-warning">
            <span class="callout-icon">📸</span>
            <div>
                <strong>No Image Provided</strong><br>
                Please upload an image of a cricket batsman or select one of the test examples below.
            </div>
        </div>
        """, None

    pred_instance = get_predictor()
    if pred_instance is None:
        return """
        <div class="status-callout callout-error">
            <span class="callout-icon">❌</span>
            <div>
                <strong>Model Artifact Missing</strong><br>
                Ensure xgboost_v1.pkl, scaler.pkl, and label_encoder.pkl exist in models/.
            </div>
        </div>
        """, None

    try:
        result = pred_instance.predict(image)
        if not result['success']:
            return f"""
            <div class="status-callout callout-warning">
                <span class="callout-icon">⚠️</span>
                <div>
                    <strong>Pose Detection Incomplete</strong><br>
                    {result['error']}
                </div>
            </div>
            """, None

        annotated_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)
        output_html_content = format_results(result['label'], result['confidence'], result['all_probs'])
        return output_html_content, annotated_pil

    except Exception as exc:
        logger.exception("Prediction failed")
        return f"""
        <div class="status-callout callout-error">
            <span class="callout-icon">⚠️</span>
            <div>
                <strong>Diagnostic Error</strong><br>
                Pipeline exception: {exc}
            </div>
        </div>
        """, None


# --- MODERN SPORTS-TECH CSS & RESPONSIVE DESIGN SYSTEM ---
CSS = """
/* Reset & Typography */
* { box-sizing: border-box !important; }
html, body {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #080c14 !important;
    color: #e2e8f0 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: 100vw !important;
    overflow-x: hidden !important;
    -webkit-text-size-adjust: 100% !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1260px !important;
    margin: 0 auto !important;
    padding: 16px 20px 40px 20px !important;
    background-color: transparent !important;
    overflow-x: hidden !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── HERO BANNER ─────────────────────────────────────────── */
.hero-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.4)) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 20px !important;
    padding: 22px 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35) !important;
    backdrop-filter: blur(16px) !important;
    text-align: center !important;
}

.hero-brand {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 14px !important;
    margin-bottom: 8px !important;
}

.brand-logo {
    width: 46px !important;
    height: 46px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 4px 16px rgba(56, 189, 248, 0.25) !important;
    object-fit: cover !important;
    display: inline-block !important;
    flex-shrink: 0 !important;
}

#hero-title {
    font-size: clamp(1.8rem, 3.5vw, 2.5rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px !important;
    line-height: 1.1 !important;
    margin: 0 !important;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

#hero-subtitle {
    font-size: clamp(0.95rem, 1.8vw, 1.1rem) !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    margin: 4px 0 2px 0 !important;
    line-height: 1.4 !important;
}

#hero-tagline {
    font-size: clamp(0.82rem, 1.4vw, 0.92rem) !important;
    color: #94a3b8 !important;
    font-weight: 400 !important;
    margin: 0 0 12px 0 !important;
    line-height: 1.4 !important;
}

.hero-badges-container {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    gap: 8px !important;
    margin-top: 4px !important;
}

.tech-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 4px 11px !important;
    border-radius: 9999px !important;
    letter-spacing: 0.2px !important;
}

.tech-pill.accent {
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.08) !important;
}

/* ── PRODUCTION CLASSES STRIP ────────────────────────────── */
.classes-bar {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    gap: 8px !important;
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    margin-bottom: 20px !important;
}

.class-chip {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 4px 10px !important;
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ── STUDIO CARD PANELS ──────────────────────────────────── */
.product-card {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.25s ease !important;
}

.product-card:hover {
    border-color: rgba(56, 189, 248, 0.25) !important;
}

.card-title-bar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    padding-bottom: 10px !important;
}

.card-title-text {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.2px !important;
    margin: 0 !important;
}

.card-tag {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    padding: 3px 8px !important;
    border-radius: 6px !important;
    white-space: nowrap !important;
}

/* Primary Action Button */
button.primary-analyze {
    background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    border: none !important;
    border-radius: 12px !important;
    height: 50px !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 18px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    margin-top: 10px !important;
}

button.primary-analyze:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.5) !important;
    background: linear-gradient(135deg, #0369a1 0%, #4f46e5 100%) !important;
}

/* Image Upload Element Overrides */
div[data-testid="image"] {
    background: rgba(10, 15, 26, 0.5) !important;
    border: 2px dashed rgba(56, 189, 248, 0.18) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

div[data-testid="image"]:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    background: rgba(10, 15, 26, 0.7) !important;
}

/* ── DIAGNOSTIC RESULTS STYLING ──────────────────────────── */
.result-card-inner {
    display: flex !important;
    flex-direction: column !important;
    gap: 14px !important;
}

.result-top-bar {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
}

.status-badge {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 3px 10px !important;
    border-radius: 9999px !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
}

.badge-success { background: rgba(16, 185, 129, 0.12) !important; color: #10b981 !important; border: 1px solid rgba(16, 185, 129, 0.3) !important; }
.badge-warning { background: rgba(245, 158, 11, 0.12) !important; color: #f59e0b !important; border: 1px solid rgba(245, 158, 11, 0.3) !important; }

.badge-dot { width: 6px !important; height: 6px !important; border-radius: 50% !important; }
.dot-success { background: #10b981 !important; box-shadow: 0 0 8px #10b981 !important; }
.dot-warning { background: #f59e0b !important; box-shadow: 0 0 8px #f59e0b !important; }

.latency-pill {
    font-size: 0.72rem !important;
    color: #64748b !important;
    font-weight: 600 !important;
}

.result-header {
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    padding-bottom: 12px !important;
}

.result-kicker {
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: #38bdf8 !important;
    margin-bottom: 4px !important;
}

.result-title-group {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 4px !important;
}

.result-emoji { font-size: 1.8rem !important; line-height: 1 !important; }
.result-title { font-size: 1.6rem !important; font-weight: 800 !important; color: #ffffff !important; margin: 0 !important; letter-spacing: -0.4px !important; }
.result-desc { font-size: 0.92rem !important; color: #94a3b8 !important; margin: 0 !important; line-height: 1.45 !important; }

/* Abstention Warning Banner */
.abstention-banner {
    display: flex !important;
    gap: 10px !important;
    align-items: flex-start !important;
    background: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    font-size: 0.85rem !important;
    color: #fde68a !important;
    line-height: 1.4 !important;
}

.abstention-icon { font-size: 1.2rem !important; line-height: 1 !important; flex-shrink: 0 !important; }

/* Gauge Card */
.metric-gauge-card {
    display: flex !important;
    align-items: center !important;
    gap: 20px !important;
    background: rgba(10, 15, 26, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
}

.gauge-ring {
    width: 90px !important;
    height: 90px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}

.gauge-center {
    width: 74px !important;
    height: 74px !important;
    border-radius: 50% !important;
    background: #0f172a !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}

.gauge-pct { font-size: 1.15rem !important; font-weight: 800 !important; line-height: 1 !important; }
.gauge-sub { font-size: 0.65rem !important; color: #64748b !important; text-transform: uppercase !important; font-weight: 700 !important; margin-top: 3px !important; }

.gauge-meta {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
    flex-grow: 1 !important;
}

.meta-item {
    display: flex !important;
    justify-content: space-between !important;
    font-size: 0.8rem !important;
}

.meta-label { color: #94a3b8 !important; font-weight: 500 !important; }
.meta-value { color: #e2e8f0 !important; font-weight: 700 !important; }

/* Softmax Probability Bars */
.probs-section {
    background: rgba(10, 15, 26, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
}

.probs-heading {
    margin: 0 0 10px 0 !important;
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    padding-bottom: 6px !important;
}

.probs-list { display: flex !important; flex-direction: column !important; gap: 7px !important; }
.prob-row { display: flex !important; flex-direction: column !important; gap: 3px !important; }
.prob-info { display: flex !important; justify-content: space-between !important; font-size: 0.82rem !important; }
.prob-name { font-weight: 600 !important; color: #cbd5e1 !important; }
.prob-val { font-weight: 700 !important; }
.prob-bar-track { width: 100% !important; height: 6px !important; background: #1e293b !important; border-radius: 3px !important; overflow: hidden !important; }
.prob-bar-fill { height: 100% !important; border-radius: 3px !important; transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1) !important; }

/* ── EMPTY / AWAITING STATE ──────────────────────────────── */
.awaiting-input-card {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 40px 16px !important;
    text-align: center !important;
    border: 2px dashed rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    background: rgba(10, 15, 26, 0.3) !important;
}

.awaiting-icon { font-size: 2.6rem !important; margin-bottom: 10px !important; opacity: 0.85 !important; }
.awaiting-title { color: #f1f5f9 !important; font-size: 1.1rem !important; font-weight: 700 !important; margin-bottom: 4px !important; }
.awaiting-subtitle { color: #64748b !important; font-size: 0.85rem !important; max-width: 280px !important; line-height: 1.4 !important; }

/* ── CALLOUT BANNERS ─────────────────────────────────────── */
.status-callout {
    display: flex !important;
    align-items: flex-start !important;
    gap: 12px !important;
    padding: 14px 16px !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    line-height: 1.45 !important;
}
.callout-warning { background: rgba(245, 158, 11, 0.1) !important; border: 1px solid rgba(245, 158, 11, 0.3) !important; color: #fde68a !important; }
.callout-error { background: rgba(239, 68, 68, 0.1) !important; border: 1px solid rgba(239, 68, 68, 0.3) !important; color: #fca5a5 !important; }
.callout-icon { font-size: 1.3rem !important; line-height: 1 !important; flex-shrink: 0 !important; }

/* ── EXAMPLES GALLERY ────────────────────────────────────── */
.gr-examples {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-top: 20px !important;
}
.gr-examples .gallery { gap: 10px !important; }
.gr-examples .gallery-item {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    transition: all 0.2s ease !important;
}
.gr-examples .gallery-item:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
}

/* ── HOW IT WORKS PIPELINE CARDS ─────────────────────────── */
.pipeline-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)) !important;
    gap: 14px !important;
    margin-top: 14px !important;
}

.step-card {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    transition: border-color 0.2s ease !important;
}

.step-card:hover { border-color: rgba(56, 189, 248, 0.3) !important; }

.step-num {
    display: inline-block !important;
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
}

.step-title {
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin: 0 0 6px 0 !important;
}

.step-desc {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    margin: 0 !important;
    line-height: 1.45 !important;
}

/* ── METRIC CALLOUT CARDS ────────────────────────────────── */
.metrics-row {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
    gap: 12px !important;
    margin: 16px 0 !important;
}

.metric-stat-card {
    background: rgba(10, 15, 26, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    text-align: center !important;
}

.metric-stat-card.featured {
    border-color: rgba(56, 189, 248, 0.35) !important;
    background: rgba(56, 189, 248, 0.05) !important;
}

.stat-val { font-size: 1.6rem !important; font-weight: 800 !important; color: #38bdf8 !important; line-height: 1.1 !important; }
.stat-val.emerald { color: #10b981 !important; }
.stat-lbl { font-size: 0.78rem !important; color: #94a3b8 !important; font-weight: 600 !important; margin-top: 4px !important; text-transform: uppercase !important; }
.stat-sub { font-size: 0.7rem !important; color: #64748b !important; margin-top: 2px !important; }

/* ── FOOTER STYLING ──────────────────────────────────────── */
.footer-wrap {
    text-align: center !important;
    padding: 30px 16px 10px 16px !important;
    margin-top: 20px !important;
    border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
}

.footer-author { font-size: 1.05rem !important; font-weight: 700 !important; color: #f1f5f9 !important; margin-bottom: 2px !important; }
.footer-role { font-size: 0.85rem !important; color: #64748b !important; margin-bottom: 14px !important; }
.footer-badge-links { display: flex !important; justify-content: center !important; gap: 8px !important; flex-wrap: wrap !important; margin-bottom: 14px !important; }
.footer-badge-links img { height: 24px !important; border-radius: 4px !important; transition: transform 0.15s ease !important; }
.footer-badge-links img:hover { transform: scale(1.05) !important; }
.footer-footnote { font-size: 0.78rem !important; color: #475569 !important; }

/* ── MOBILE-FIRST TAB NAVIGATION & TABLE OVERFLOW ──────────── */
/* Horizontal swipeable tab bar on mobile */
div[role="tablist"], .tab-nav {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
    gap: 6px !important;
    padding: 4px 2px 10px 2px !important;
    margin-bottom: 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}

div[role="tablist"]::-webkit-scrollbar, .tab-nav::-webkit-scrollbar {
    display: none !important;
}

div[role="tablist"] button, .tab-nav button {
    flex-shrink: 0 !important;
    font-size: 0.84rem !important;
    padding: 8px 14px !important;
    border-radius: 9px !important;
    white-space: nowrap !important;
    font-weight: 600 !important;
}

/* Prevent tables and markdown blocks from blowing out mobile width */
.gradio-container table {
    display: block !important;
    width: 100% !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    border-collapse: collapse !important;
    margin: 12px 0 !important;
}

.gradio-container .block,
.gradio-container .form {
    min-width: 0 !important;
}

/* Studio Row - Desktop Side-by-Side */
#studio-main-row,
.studio-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 20px !important;
    width: 100% !important;
    align-items: stretch !important;
}

#studio-main-row > div,
.studio-row > div,
.studio-row > .product-card {
    flex: 1 1 0% !important;
    min-width: 0 !important;
}

/* Examples gallery mobile optimization */
@media (max-width: 600px) {
    .gr-examples .gallery {
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 6px !important;
    }
}

@media (max-width: 400px) {
    .gr-examples .gallery {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 6px !important;
    }
}

/* ── RESPONSIVE MEDIA QUERIES ────────────────────────────── */
/* Large Tablet & Desktop (<= 1024px) */
@media (max-width: 1024px) {
    .gradio-container { max-width: 100% !important; padding: 14px 16px !important; }
    .hero-card { padding: 18px 20px !important; }
    .product-card { padding: 18px !important; }
    .metrics-row { grid-template-columns: repeat(2, 1fr) !important; }
}

/* Tablet Portrait & Handheld Devices (<= 920px: iPad, Surface, and all Mobiles) */
@media (max-width: 920px) {
    .gradio-container { padding: 10px 12px !important; }

    /* Force studio row and all multi-column rows into full-width vertical stack */
    #studio-main-row,
    .studio-row,
    .gradio-container .unequal-height,
    .gradio-container .stretch,
    .tabitem > div > div[class*="unequal-height"],
    .tabitem > div > div[class*="stretch"] {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        gap: 16px !important;
    }

    #studio-main-row > div,
    .studio-row > div,
    .product-card,
    .input-card,
    .results-card {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
        box-sizing: border-box !important;
        padding: 18px 16px !important;
        margin-bottom: 0 !important;
    }

    .hero-card { padding: 16px 14px !important; margin-bottom: 14px !important; }
    .hero-brand { display: flex !important; flex-direction: row !important; align-items: center !important; justify-content: center !important; gap: 10px !important; }
    .brand-logo { width: 40px !important; height: 40px !important; }
    #hero-title { font-size: 1.7rem !important; }
    #hero-subtitle { font-size: 0.92rem !important; }
    #hero-tagline { font-size: 0.82rem !important; margin-bottom: 8px !important; }

    .card-title-bar {
        gap: 8px !important;
        margin-bottom: 12px !important;
        padding-bottom: 8px !important;
    }
    .card-title-text {
        font-size: 1rem !important;
    }
    .card-tag {
        font-size: 0.68rem !important;
        padding: 2px 7px !important;
    }

    .metric-gauge-card {
        padding: 14px 14px !important;
        gap: 14px !important;
    }

    #image-input, #image-input .image-container,
    #output-image, #output-image .image-container {
        max-height: 280px !important;
        min-height: 180px !important;
    }
    #image-input img, #output-image img {
        max-height: 280px !important;
        object-fit: contain !important;
    }

    button.primary-analyze {
        height: 50px !important;
        font-size: 0.98rem !important;
        width: 100% !important;
        touch-action: manipulation !important;
    }

    .pipeline-grid { grid-template-columns: 1fr !important; }
    div[data-testid="gallery"] > div { grid-template-columns: 1fr !important; }
}

/* Standard Mobile Screens (<= 480px: iPhone 14/15 Pro Max, Galaxy S23, Pixel) */
@media (max-width: 480px) {
    .gradio-container { padding: 6px 8px !important; }
    .hero-card { padding: 14px 10px !important; border-radius: 14px !important; margin-bottom: 10px !important; }
    .brand-logo { width: 34px !important; height: 34px !important; border-radius: 9px !important; }
    #hero-title { font-size: 1.4rem !important; letter-spacing: -0.5px !important; }
    #hero-subtitle { font-size: 0.82rem !important; line-height: 1.3 !important; }
    #hero-tagline { font-size: 0.75rem !important; line-height: 1.3 !important; margin-bottom: 8px !important; }
    .hero-badges-container { gap: 5px !important; }
    .tech-pill { font-size: 0.68rem !important; padding: 2px 7px !important; }
    .classes-bar { gap: 4px !important; padding: 6px 8px !important; border-radius: 10px !important; margin-bottom: 12px !important; }
    .class-chip { font-size: 0.68rem !important; padding: 2px 6px !important; border-radius: 6px !important; }

    .product-card { padding: 14px 12px !important; border-radius: 14px !important; }
    .card-title-text { font-size: 0.95rem !important; }
    .card-tag { font-size: 0.65rem !important; padding: 2px 6px !important; }

    button.primary-analyze { height: 48px !important; font-size: 0.92rem !important; }

    .result-kicker { font-size: 0.68rem !important; }
    .result-title { font-size: 1.25rem !important; }
    .result-desc { font-size: 0.82rem !important; line-height: 1.35 !important; }
    .abstention-banner { font-size: 0.78rem !important; padding: 8px 10px !important; }
    .stat-val { font-size: 1.25rem !important; }
    .metrics-row { grid-template-columns: 1fr !important; gap: 8px !important; }
    .gauge-ring { width: 76px !important; height: 76px !important; }
    .gauge-center { width: 62px !important; height: 62px !important; }
    .gauge-pct { font-size: 1rem !important; }
    .meta-item { font-size: 0.75rem !important; }
    .prob-info { font-size: 0.78rem !important; }
    .prob-bar-track { height: 5px !important; }
    .footer-wrap { padding: 18px 8px !important; }
    .footer-author { font-size: 0.92rem !important; }
    .footer-role { font-size: 0.76rem !important; }
    .footer-badge-links img { height: 20px !important; }
    .footer-footnote { font-size: 0.68rem !important; line-height: 1.35 !important; }
}

/* Compact Mobile Screens (<= 360px: Small Android / Fold Cover) */
@media (max-width: 360px) {
    #hero-title { font-size: 1.22rem !important; }
    #hero-subtitle { font-size: 0.78rem !important; }
    #hero-tagline { font-size: 0.7rem !important; }
    .class-chip { font-size: 0.62rem !important; padding: 2px 4px !important; }
    .tech-pill { font-size: 0.64rem !important; padding: 2px 5px !important; }
    .card-title-text { font-size: 0.88rem !important; }
    .metric-gauge-card { flex-direction: column !important; text-align: center !important; }
    .gauge-meta { width: 100% !important; }
}
"""

# Theme definition
app_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    text_size="md",
    radius_size="lg",
).set(
    body_background_fill="#080c14",
    block_background_fill="#0f172a",
    block_border_width="1px",
    block_label_text_color="#94a3b8",
)


# --- GRADIO APPLICATION LAYOUT ---
with gr.Blocks(title=f"{PROJECT_NAME} — Intelligent Cricket Shot Classification", theme=app_theme, css=CSS) as demo:
    # Inject Google Font
    gr.HTML("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap'>", visible=False)

    # Force dark theme mode
    demo.load(None, None, None, js="""
        function() {
            document.querySelector('body').classList.add('dark');
        }
    """)

    # 1. HEADER / HERO SECTION
    with gr.Column(elem_classes=["hero-card"]):
        gr.HTML(
            f"""
            <div class="hero-brand">
                {LOGO_HTML}
                <h1 id="hero-title">{PROJECT_NAME}</h1>
            </div>
            """
        )

        gr.Markdown(
            "AI-powered cricket shot classification from a single image.",
            elem_id="hero-subtitle",
        )
        gr.Markdown(
            "Pose-based biomechanical features + XGBoost for lightweight shot recognition.",
            elem_id="hero-tagline",
        )

        gr.HTML("""
            <div class="hero-badges-container">
                <span class="tech-pill accent">⚡ 51 3D Biomechanical Features</span>
                <span class="tech-pill">🌲 Regularized XGBoost</span>
                <span class="tech-pill">🛡️ 65% Confidence Gate</span>
                <span class="tech-pill">⏱️ &lt; 65ms CPU Latency</span>
                <span class="tech-pill">🟢 System Online</span>
            </div>
        """)

    # 2. PRODUCTION CLASSES SHOWCASE STRIP
    gr.HTML("""
        <div class="classes-bar">
            <div class="class-chip">🏏 Cover Drive</div>
            <div class="class-chip">💪 Pull Shot</div>
            <div class="class-chip">✂️ Cut Shot</div>
            <div class="class-chip">🧹 Sweep Shot</div>
            <div class="class-chip">🥄 Scoop Shot</div>
            <div class="class-chip">⬅️ Leg Glance</div>
        </div>
    """)

    # 3. MAIN PRODUCT TABS
    with gr.Tabs():
        # ─────────────────────────────────────────────────────────────────
        # TAB 1: ANALYZE SHOT (Primary Studio)
        # ─────────────────────────────────────────────────────────────────
        with gr.Tab("⚡ Analyze Shot"):
            with gr.Row(equal_height=False, elem_id="studio-main-row", elem_classes=["studio-row"]):
                # LEFT COLUMN: Input Card
                with gr.Column(scale=1, elem_classes=["product-card", "input-card"]):
                    gr.HTML("""
                    <div class="card-title-bar">
                        <h3 class="card-title-text">📸 Biomechanical Frame Capture</h3>
                        <span class="card-tag">Upload / Drag & Drop</span>
                    </div>
                    """)

                    image_input = gr.Image(
                        label="Upload a cricket shot image (drag & drop or click)",
                        type="pil",
                        elem_id="image-input",
                        sources=["upload", "clipboard"],
                    )

                    submit_btn = gr.Button(
                        "⚡ ANALYZE BIOMECHANICS",
                        variant="primary",
                        elem_classes=["primary-analyze"],
                    )

                    with gr.Accordion("💡 Best Capture Guidelines", open=False):
                        gr.Markdown(
                            "- **Full Body Framing:** Include head to feet so knee and ankle angles can be measured.\n"
                            "- **Angle:** Side-on or 45° camera perspective provides the richest kinematic depth.\n"
                            "- **Lighting:** Clear contrast minimizes MediaPipe skeletal landmark jitter.\n"
                            "- **Safety Guardrail:** Predictions under 65% probability trigger safe abstention (`Uncertain Shot`)."
                        )

                # RIGHT COLUMN: Results Card
                with gr.Column(scale=1, elem_classes=["product-card", "results-card"]):
                    gr.HTML("""
                    <div class="card-title-bar">
                        <h3 class="card-title-text">📊 Diagnostic Result</h3>
                        <span class="card-tag">XGBoost &bull; 6 Classes</span>
                    </div>
                    """)

                    output_html = gr.HTML(
                        value="""
                        <div class="awaiting-input-card">
                            <div class="awaiting-icon">🏏</div>
                            <div class="awaiting-title">Awaiting Biomechanical Input</div>
                            <div class="awaiting-subtitle">
                                Select a sample image below or upload a batting photo and click <strong>Analyze Biomechanics</strong>.
                            </div>
                        </div>
                        """
                    )

                    output_img = gr.Image(
                        label="Annotated 3D Skeletal Graph",
                        elem_id="output-image",
                        interactive=False,
                    )

            # Core Execution Event with smooth auto-scroll on mobile
            submit_btn.click(
                fn=classify_shot,
                inputs=[image_input],
                outputs=[output_html, output_img],
                js="""() => {
                    setTimeout(() => {
                        const resultsEl = document.querySelector('.results-card');
                        if (resultsEl && window.innerWidth <= 920) {
                            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    }, 250);
                }"""
            )

            # Example Gallery
            if EXAMPLES_DIR.exists():
                example_files = sorted(
                    path for path in EXAMPLES_DIR.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if example_files:
                    example_labels = [
                        SHOT_META.get(path.stem, ('', path.stem.replace('_', ' ').title(), ''))[1]
                        for path in example_files[:6]
                    ]
                    gr.Examples(
                        examples=[[str(path)] for path in example_files[:6]],
                        inputs=image_input,
                        example_labels=example_labels,
                        label="⚡ Benchmark Test Examples (Click to Load)",
                        examples_per_page=6,
                    )

        # ─────────────────────────────────────────────────────────────────
        # TAB 2: HOW IT WORKS & ARCHITECTURE
        # ─────────────────────────────────────────────────────────────────
        with gr.Tab("🔬 How It Works & Architecture"):
            with gr.Column(elem_classes=["product-card"]):
                gr.HTML("""
                <div class="card-title-bar">
                    <h3 class="card-title-text">End-to-End Inference Pipeline</h3>
                    <span class="card-tag">6-Step Flow</span>
                </div>
                """)

                gr.HTML("""
                <div class="pipeline-grid">
                    <div class="step-card">
                        <span class="step-num">Step 1</span>
                        <h4 class="step-title">🖼️ Input Image</h4>
                        <p class="step-desc">Receives raw RGB frame (upload or video frame). Verified for dimension and format constraints (max 10MB).</p>
                    </div>
                    <div class="step-card">
                        <span class="step-num">Step 2</span>
                        <h4 class="step-title">🦴 MediaPipe Pose</h4>
                        <p class="step-desc">Extracts 33 skeletal anatomical landmarks with (x, y, z, visibility) coordinates. Discards background noise and player jerseys.</p>
                    </div>
                    <div class="step-card">
                        <span class="step-num">Step 3</span>
                        <h4 class="step-title">📐 51 Biomechanical Features</h4>
                        <p class="step-desc">Normalizes limb vectors by 3D torso scale; computes 8 3D joint angles, trunk lean, bat vectors, and lateral symmetry.</p>
                    </div>
                    <div class="step-card">
                        <span class="step-num">Step 4</span>
                        <h4 class="step-title">⚖️ StandardScaler</h4>
                        <p class="step-desc">Applies feature normalization fitted strictly on training data to ensure zero distribution drift.</p>
                    </div>
                    <div class="step-card">
                        <span class="step-num">Step 5</span>
                        <h4 class="step-title">🌲 XGBoost Classifier</h4>
                        <p class="step-desc">Gradient boosted decision trees compute multi-class probability distribution in &lt; 1 millisecond on CPU.</p>
                    </div>
                    <div class="step-card">
                        <span class="step-num">Step 6</span>
                        <h4 class="step-title">🛡️ Confidence Gate</h4>
                        <p class="step-desc">If top probability &ge; 65%, outputs certified shot label. If &lt; 65%, abstains safely as "Uncertain Shot".</p>
                    </div>
                </div>
                """)

                gr.Markdown("---")
                gr.Markdown("### 💡 Why Biomechanical Pose Instead of Raw Pixels?")
                gr.Markdown(
                    "- **Zero Background Bias:** Standard CNNs learn pitch grass, stadium advertising, and team jerseys rather than technique. MediaPipe strips out 100% of pixel backgrounds.\n"
                    "- **Scale Invariance:** All distance measurements are normalized by the 3D torso length (mid-shoulder to mid-hip), making inference invariant to batsman height and camera zoom.\n"
                    "- **Ultra-Lightweight CPU Inference:** Total model payload is only **2.82 MB**, executing end-to-end in **< 65 ms** on commodity CPUs without GPU acceleration."
                )

        # ─────────────────────────────────────────────────────────────────
        # TAB 3: MODEL PERFORMANCE & BENCHMARKS
        # ─────────────────────────────────────────────────────────────────
        with gr.Tab("📈 Model Performance & Evaluation"):
            with gr.Column(elem_classes=["product-card"]):
                gr.HTML("""
                <div class="card-title-bar">
                    <h3 class="card-title-text">Quantitative Validation & Metrics</h3>
                    <span class="card-tag">Dual-Metric Framing</span>
                </div>
                """)

                # Metric Cards Strip
                gr.HTML(f"""
                <div class="metrics-row">
                    <div class="metric-stat-card">
                        <div class="stat-val">{PROD_METRICS['overall_acc']}</div>
                        <div class="stat-lbl">Raw Holdout Accuracy</div>
                        <div class="stat-sub">1,304 unseen test frames</div>
                    </div>
                    <div class="metric-stat-card featured">
                        <div class="stat-val emerald">{PROD_METRICS['post_acc']}</div>
                        <div class="stat-lbl">Selective Accuracy</div>
                        <div class="stat-sub">882 accepted frames (@ 0.65)</div>
                    </div>
                    <div class="metric-stat-card">
                        <div class="stat-val">67.6%</div>
                        <div class="stat-lbl">Sample Coverage</div>
                        <div class="stat-sub">{PROD_METRICS['reject_rate']} rejected as uncertain</div>
                    </div>
                    <div class="metric-stat-card">
                        <div class="stat-val">{PROD_METRICS['cv_score'] or '78.13%'}</div>
                        <div class="stat-lbl">5-Fold GroupKFold CV</div>
                        <div class="stat-sub">Group-aware baseline audit</div>
                    </div>
                </div>
                """)

                gr.HTML("""
                <div class="status-callout callout-warning" style="margin-bottom: 16px;">
                    <span class="callout-icon">ℹ️</span>
                    <div>
                        <strong>Scientific Honesty Note:</strong> We explicitly distinguish between 
                        <strong>Raw Generalization Accuracy (77.53%)</strong> across all unseen frames and 
                        <strong>Selective Accuracy (92.52%)</strong> achieved when deploying the 0.65 confidence guardrail. 
                        In sports coaching, abstaining on ambiguous postures is critical to maintain user trust.
                    </div>
                </div>
                """)

                gr.Markdown("### 📊 Diagnostic Evaluation Gallery")
                gr.Gallery(
                    value=get_performance_gallery_items(),
                    label="Evaluation Diagnostics",
                    show_label=False,
                    columns=2,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                )

        # ─────────────────────────────────────────────────────────────────
        # TAB 4: SYSTEM SPECIFICATIONS & GUIDE
        # ─────────────────────────────────────────────────────────────────
        with gr.Tab("⚙️ System Specifications & Shot Guide"):
            with gr.Column(elem_classes=["product-card"]):
                gr.HTML("""
                <div class="card-title-bar">
                    <h3 class="card-title-text">System Specifications & Kinematic Guide</h3>
                    <span class="card-tag">Technical Specs</span>
                </div>
                """)

                gr.Markdown("""
                | Component | Specification | Technical Notes |
                | :--- | :--- | :--- |
                | **Model Architecture** | Extreme Gradient Boosting (`XGBClassifier`) | Multi-class softmax objective (`multi:softprob`) |
                | **Input Vector** | 51 continuous 3D numerical features | Angle degrees, torso-scaled distances, bat vectors |
                | **Active Classes (6)** | Cover Drive, Pull, Cut, Sweep, Scoop, Leg Glance | `straight_drive` excluded due to scarcity (345 samples) |
                | **Hardware Target** | 100% CPU edge-deployable | Zero GPU required; ~250MB container RAM footprint |
                | **Model Binary Size** | 2.82 MB (`models/xgboost_v1.pkl`) | Serialized via Joblib |
                | **Guardrail Threshold** | 0.65 softmax confidence | Configurable via `CRICKET_CONFIDENCE_THRESHOLD` |
                | **REST API** | FastAPI asynchronous endpoints | `/health` (GET) and `/predict` (POST) |
                """)

                gr.Markdown("---")
                gr.Markdown("### 🏏 Six Supported Shot Classes & Kinematics")
                gr.Markdown("""
                1. **Cover Drive (`cover_drive`):** Classical front-foot off-side stroke. Characterized by bent front knee, high forward elbow, and upright torso.
                2. **Pull Shot (`pull_shot`):** Powerful cross-bat hook against short pitch deliveries. Distinguished by horizontal arm extension and open chest facing square leg.
                3. **Cut Shot (`cut_shot`):** Back-foot stroke through the point region. Marked by backward weight transfer, raised wrists, and cross-bat chop.
                4. **Sweep Shot (`sweep_shot`):** Front-knee knelt paddle on the leg side. Marked by lowered center of gravity, bent lead knee, and horizontal bat trajectory.
                5. **Scoop Shot (`scoop_shot`):** Unorthodox ramp over the wicketkeeper. Characterized by deep crouch, low knee bend, and vertical bat lift behind the shoulders.
                6. **Leg Glance (`leg_glance_shot`):** Subtle wrist deflection off hip/pads toward fine leg. Distinguished by closed shoulders and subtle wrist roll at impact.
                """)

    # 4. FOOTER & BRANDING
    gr.HTML(
        f"""
        <div class="footer-wrap">
            <div class="footer-author">Created by Himanshu Jadhav</div>
            <div class="footer-role">AI & Data Science Engineering &bull; CricketVision AI</div>
            <div class="footer-badge-links">
                <a href='https://github.com/himanshu-jadhav108/CricketVision_AI' target='_blank'><img src='https://img.shields.io/badge/GitHub-Repository-100000?style=for-the-badge&logo=github&logoColor=white' alt='GitHub'></a>
                <a href='https://www.linkedin.com/in/himanshu-jadhav-328082339' target='_blank'><img src='https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='LinkedIn'></a>
                <a href='https://himanshu-jadhav-portfolio.vercel.app/' target='_blank'><img src='https://img.shields.io/badge/Portfolio-Visit-FFD700?style=for-the-badge&logo=google-chrome&logoColor=black' alt='Portfolio'></a>
            </div>
            <div class="footer-footnote">
                CricketVision AI &bull; MediaPipe Pose + 51 Biomechanical Features + XGBoost &bull; {PROD_METRICS['post_acc']} Selective Accuracy (@ 67.6% Coverage)
            </div>
        </div>
        """
    )


if __name__ == "__main__":
    if not models_exist():
        missing = ", ".join(path.name for path in missing_model_files())
        raise SystemExit(f"CRITICAL: models/ files missing: {missing}")

    display_host = "127.0.0.1" if GRADIO_HOST == "0.0.0.0" else GRADIO_HOST
    logger.info("Launching Gradio UI on http://%s:%s", display_host, GRADIO_PORT)
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )
