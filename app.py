from __future__ import annotations

import logging

import cv2
import gradio as gr
from PIL import Image

from config import (
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
    SCALER_PATH,
    missing_model_files,
)
from predictor import ShotPredictor

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Current production model paths reflecting the V2 architecture
def models_exist():
    return not missing_model_files()

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        if not models_exist():
            return None
        logger.info("Loading predictor from %s", MODEL_DIR)
        predictor = ShotPredictor(str(MODEL_PATH), str(SCALER_PATH), str(ENCODER_PATH))
    return predictor

# Metadata for rich UI display (Emojis, Human-readable names, Descriptions)
SHOT_META = {
    'cover_drive'      : ('🏏', 'Cover Drive',    'A classic front-foot off-side shot'),
    'pull_shot'        : ('💪', 'Pull Shot',       'An aggressive short-ball hook'),
    'leg_glance_shot'  : ('⬅️', 'Leg Glance',     'A deflection to the leg side'),
    'cut_shot'         : ('✂️', 'Cut Shot',        'A back-foot cut through point'),
    'scoop_shot'       : ('🥄', 'Scoop Shot',      'An unorthodox loft over the keeper'),
    'sweep_shot'       : ('🧹', 'Sweep Shot',      'A sweeping shot played on the leg side'),
    'Uncertain Shot'   : ('⚠️', 'Uncertain Shot',  'Confidence too low. Please provide a clearer image.'),
}

def format_results(label, confidence, all_probs):
    """Generates premium HTML/CSS for the diagnostic results panel."""
    emoji, name, desc = SHOT_META.get(label, ('🏏', label, ''))
    
    # Calculate gauge color based on confidence
    conf_pct = confidence * 100
    if label == 'Uncertain Shot':
        color = "#ef4444" # Red
    else:
        color = "#10b981" if conf_pct >= 80 else "#f59e0b" if conf_pct >= 50 else "#ef4444"
    
    html = f"""
    <div class="result-header">
        <h2 class="result-title">{emoji} {name}</h2>
        <p class="result-desc">{desc}</p>
        
        <div class="gauge-container" style="background: conic-gradient({color} {conf_pct}%, #334155 0); box-shadow: 0 0 30px {color}30;">
            <div class="gauge-inner">
                <span class="gauge-value" style="color: {color};">{conf_pct:.1f}%</span>
                <span class="gauge-label">Confidence</span>
            </div>
        </div>
    </div>
    
    <div class="probs-container">
        <h3 class="probs-title">Biomechanical Probability</h3>
    """
    
    # Add probability bars for each class
    for shot, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        sh_em, sh_nm, _ = SHOT_META.get(shot, ('🏏', shot, ''))
        bar_w = prob * 100
        is_winner = shot == label
        class_name = "prob-row winner" if is_winner else "prob-row"
        bar_style = f"width: {bar_w}%; background: { 'linear-gradient(90deg, #38bdf8, #818cf8)' if is_winner else '#475569' };"
        val_color_style = f"color: {color};" if is_winner else ""
        star = "⚡ " if is_winner else ""
        
        html += f"""
        <div class="{class_name}">
            <div class="prob-info">
                <span class="prob-name">{star}{sh_nm}</span>
                <span class="prob-value" style="{val_color_style}">{bar_w:.1f}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="{bar_style}"></div>
            </div>
        </div>
        """
        
    html += "</div>"
    return html

def classify_shot(image):
    """Main interface hook: Processes images and returns rich diagnostic UI."""
    if image is None:
        return "<p style='color: #ef4444; text-align: center; padding: 20px;'>⬆️ Please select or upload a cricket batting image.</p>", None
        
    pred_instance = get_predictor()
    if pred_instance is None:
        return "<p style='color: #ef4444; text-align: center; padding: 20px;'>❌ <b>Model missing!</b> Please ensure .pkl files exist in the models/ directory.</p>", None
        
    # Execute V2 Inference Pipeline
    result = pred_instance.predict(image)
    if not result['success']:
        return f"<div style='background: rgba(239, 68, 68, 0.1); padding: 20px; border-radius: 12px; border: 1px solid #ef4444; color: #ef4444; text-align: center;'>⚠️ <b>Diagnostic Warning:</b><br>{result['error']}</div>", None
        
    # Convert BGR (OpenCV) to RGB (PIL) for Gradio
    annotated_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    
    # Generate the premium HTML report
    output_html_content = format_results(result['label'], result['confidence'], result['all_probs'])
    return output_html_content, annotated_pil

# --- CUSTOM CSS & THEME ---
CSS = """
body { font-family: 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important; background-color: #0b0f19 !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; padding: 20px !important; }
#logo-img { display: block !important; margin: 0 auto 15px auto !important; background: transparent !important; border: none !important; box-shadow: none !important; }
#logo-img img { border-radius: 20px !important; box-shadow: 0 8px 32px rgba(56, 189, 248, 0.25) !important; border: 2px solid rgba(255,255,255,0.05) !important; }
#title { text-align: center; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 3.5rem; margin: 10px 0 10px 0 !important; letter-spacing: -1.5px; line-height: 1.2; }
#subtitle { text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 0px !important; font-weight: 500; letter-spacing: 0.2px; line-height: 1.6; }
.glass-panel { background: rgba(30, 41, 59, 0.45) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 24px !important; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3) !important; padding: 30px !important; transition: all 0.3s ease; }
button.primary { background: linear-gradient(135deg, #38bdf8, #818cf8) !important; border: none !important; color: white !important; font-weight: 800 !important; font-size: 1.15rem !important; box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3) !important; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; border-radius: 14px !important; height: 56px !important; letter-spacing: 0.5px !important; }
button.primary:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(56, 189, 248, 0.5) !important; }
button.primary:active { transform: translateY(0); }
.dark { background-color: #0b0f19 !important; }
label { color: #94a3b8 !important; text-transform: uppercase !important; font-weight: 700 !important; font-size: 0.75em !important; letter-spacing: 1px !important; }

/* Custom Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0b0f19; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; border: 2px solid #0b0f19; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* Cohesive Header Card */
.header-card {
    text-align: center;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6)) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    padding: 35px 20px !important;
    margin-bottom: 40px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* Accordion Custom Styling */
.accordion {
    background: rgba(15, 23, 42, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    margin-top: 20px !important;
    overflow: hidden;
}
.accordion > button {
    background: transparent !important;
    padding: 12px 16px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.95em !important;
    transition: background 0.3s ease !important;
}
.accordion > button:hover {
    background: rgba(255, 255, 255, 0.03) !important;
}

/* Examples Gallery Custom Styling */
.gr-examples {
    background: rgba(30, 41, 59, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin-top: 30px !important;
}
.gr-examples .gallery {
    gap: 12px !important;
}
.gr-examples .gallery-item {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 2px solid rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
}
.gr-examples .gallery-item:hover {
    transform: translateY(-4px) scale(1.03) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 10px 20px rgba(56, 189, 248, 0.15) !important;
}

/* Input File / Image upload overrides */
div[data-testid="image"] {
    background: rgba(15, 23, 42, 0.3) !important;
    border: 2px dashed rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}
div[data-testid="image"]:hover {
    border-color: rgba(56, 189, 248, 0.3) !important;
    background: rgba(15, 23, 42, 0.4) !important;
}

/* Diagnostic results styles */
.result-header { text-align: center; margin-bottom: 20px; }
.result-title { margin: 0; font-size: 2.2em; color: white; letter-spacing: -0.5px; }
.result-desc { color: #94a3b8; margin: 4px 0 20px 0; font-size: 1.1em; }
.gauge-container { width: 160px; height: 160px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; border: 4px solid rgba(255,255,255,0.05); animation: gauge-glow 4s infinite alternate ease-in-out; }
.gauge-inner { width: 140px; height: 140px; border-radius: 50%; background: #1e293b; display: flex; align-items: center; justify-content: center; flex-direction: column; }
.gauge-value { font-size: 2em; font-weight: 800; line-height: 1; }
.gauge-label { font-size: 0.85em; color: #94a3b8; text-transform: uppercase; margin-top: 5px; opacity: 0.7; }

.probs-container { background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(8px); }
.probs-title { margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; color: #e2e8f0; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; }

.prob-row { margin-bottom: 12px; transition: all 0.3s ease; }
.prob-row:not(.winner) { opacity: 0.6; }
.prob-info { display: flex; justify-content: space-between; font-size: 0.95em; margin-bottom: 5px; color: #cbd5e1; }
.prob-name { font-weight: 500; }
.prob-value { font-weight: bold; }
.prob-bar-bg { width: 100%; background: #0f172a; height: 10px; border-radius: 5px; overflow: hidden; border: 1px solid #334155; }
.prob-bar-fill { height: 100%; transition: width 1s cubic-bezier(0.17, 0.67, 0.83, 0.67); }

/* Premium Empty State */
.awaiting-input {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    text-align: center;
    border: 2px dashed rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.2);
}
.awaiting-icon {
    font-size: 3.5em;
    margin-bottom: 15px;
    animation: pulse-slow 3s infinite ease-in-out;
}
.awaiting-text {
    color: #e2e8f0;
    font-size: 1.25em;
    font-weight: 600;
    margin-bottom: 8px;
}
.awaiting-subtext {
    color: #64748b;
    font-size: 0.95em;
    max-width: 320px;
    line-height: 1.5;
}

@keyframes pulse-slow {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.08); opacity: 1; }
}
@keyframes gauge-glow {
    0% { filter: drop-shadow(0 0 5px rgba(56, 189, 248, 0.15)); }
    100% { filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.3)); }
}

/* Footer Styles */
.footer-section { text-align: center; padding: 40px 20px 20px 20px; }
.footer-divider { width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin-bottom: 30px; }
.footer-name { color: #e2e8f0; font-size: 1.5em; font-weight: 800; margin-bottom: 5px; }
.footer-title { color: #94a3b8; font-size: 1.05em; margin-bottom: 25px; font-weight: 500; }
.footer-links { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-bottom: 25px; }
.footer-links img { height: 28px; transition: transform 0.2s ease; }
.footer-links img:hover { transform: scale(1.05); }
.footer-architecture { color: #475569; font-size: 0.9em; max-width: 600px; margin: 0 auto; line-height: 1.5; }

/* Responsive Media Queries */
@media (max-width: 768px) {
    .gradio-container { padding: 10px !important; }
    .header-card { padding: 25px 15px !important; margin-bottom: 25px !important; border-radius: 16px !important; }
    #logo-img img { border-radius: 16px !important; }
    #title { font-size: 2.2rem !important; }
    #subtitle { font-size: 1rem !important; }
    .glass-panel { padding: 20px 15px !important; border-radius: 16px !important; }
    button.primary { height: 50px !important; font-size: 1.05rem !important; }
    .accordion { margin-top: 15px !important; }
    .gr-examples { padding: 15px !important; margin-top: 20px !important; border-radius: 16px !important; }
    
    /* Responsive image scaling */
    #image-input, #image-input img, #image-input .image-container {
        max-height: 300px !important;
    }
    #output-image, #output-image img, #output-image .image-container {
        max-height: 280px !important;
    }
    
    /* Result styling adaptations */
    .result-title { font-size: 1.8em; }
    .result-desc { font-size: 0.95em; margin-bottom: 15px; }
    .gauge-container { width: 140px; height: 140px; }
    .gauge-inner { width: 120px; height: 120px; }
    .gauge-value { font-size: 1.6em; }
    .probs-container { padding: 15px; }
    .prob-info { font-size: 0.85em; }
    .prob-bar-bg { height: 8px; }
    
    /* Footer responsiveness */
    .footer-name { font-size: 1.25em; }
    .footer-title { font-size: 0.9em; margin-bottom: 20px; }
    .footer-links { gap: 8px; }
    .footer-links img { height: 24px; }
    .footer-architecture { font-size: 0.8em; }
}
"""

premium_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    text_size="md",
    radius_size="lg"
).set(
    body_background_fill="#0b0f19",
    block_background_fill="#1e293b",
    block_border_width="1px",
    block_label_text_color="#94a3b8",
)

# --- APP LAYOUT ---
with gr.Blocks(title=PROJECT_NAME, theme=premium_theme, css=CSS) as demo:
    # Load Google Fonts stylesheet dynamically
    gr.HTML("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;800&display=swap'>", visible=False)
    
    # Initialization JS to force dark mode
    demo.load(None, None, None, js="""
        function() {
            document.querySelector('body').classList.add('dark');
        }
    """)
    
    with gr.Column(elem_classes=["header-card"]):
        if LOGO_PATH.exists():
            gr.Image(
                value=str(LOGO_PATH),
                show_label=False,
                show_download_button=False,
                interactive=False,
                container=False,
                elem_id="logo-img",
                width=100,
                height=100,
            )
        
        gr.Markdown(f"# {PROJECT_NAME}", elem_id="title")
        gr.Markdown("Transforming batting motion into data. Upload a cricket photo and see **XGBoost biomechanics** in action.", elem_id="subtitle")
    
    with gr.Row(equal_height=True):
        # LEFT COLUMN: Input & Tips
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### 📸 Biomechanical Capture")
            image_input = gr.Image(label="Input Image", type="pil", height=450, elem_id="image-input")
            submit_btn = gr.Button("🔍 ANALYZE BIOMECHANICS", variant="primary")
            
            with gr.Accordion("🛠️ Advanced Capture Tips", open=False, elem_classes=["accordion"]):
                gr.Markdown(
                    "- **Body Visibility:** Ensure head-to-toe visibility for optimal pose detection.\n"
                    "- **Angle:** Side-on or 45° angles provide the richest biomechanical data.\n"
                    "- **Resolution:** Motion blur affects MediaPipe quality. Use clear daylight images.\n"
                    "- **System Guardrail:** V2 identifies shots only with > 65% confidence for professional reliability."
                )
                
        # RIGHT COLUMN: Results & Diagnostic Output
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### ⚙️ Diagnostic AI Analysis")
            output_html = gr.HTML(
                value="""
                <div class="awaiting-input">
                    <div class="awaiting-icon">🏏</div>
                    <div class="awaiting-text">Awaiting Biomechanical Input</div>
                    <div class="awaiting-subtext">Upload a batting image and click analyze to extract skeletal angles and shot classification.</div>
                </div>
                """
            )
            output_img  = gr.Image(label="Pose Skeleton Extraction", height=380, elem_id="output-image")

    # Core Event Logic
    submit_btn.click(
        fn=classify_shot,
        inputs=[image_input],
        outputs=[output_html, output_img],
    )

    # Auto-load Examples if folder exists
    if EXAMPLES_DIR.exists():
        example_files = sorted(
            path for path in EXAMPLES_DIR.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if example_files:
            gr.Examples(
                examples=[[str(path)] for path in example_files[:7]],
                inputs=image_input,
                label="Sample Biomechanical Tests",
                examples_per_page=7,
            )

    # Footer / Personal Branding
    gr.HTML(
        f"""
        <div class="footer-section">
            <div class="footer-divider"></div>
            <div class="footer-name">Created by Himanshu Jadhav</div>
            <div class="footer-title">AI & Data Science Engineering Student</div>
            <div class="footer-links">
                <a href='https://github.com/himanshu-jadhav108' target='_blank'><img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white'></a>
                <a href='https://www.linkedin.com/in/himanshu-jadhav-328082339' target='_blank'><img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white'></a>
                <a href='https://www.instagram.com/himanshu_jadhav_108' target='_blank'><img src='https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white'></a>
                <a href='https://himanshu-jadhav-portfolio.vercel.app/' target='_blank'><img src='https://img.shields.io/badge/Portfolio-FFD700?style=for-the-badge&logo=google-chrome&logoColor=black'></a>
            </div>
            <div class="footer-architecture">
                {PROJECT_NAME} Model Architecture: MediaPipe Pose + XGBoost + Confidence Thresholding · 91.45% Safe Accuracy
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
        show_error=True
    )
