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
    'straight_drive'   : ('⬆️', 'Straight Drive', 'An elegant drive back past the bowler'),
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
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="margin: 0; font-size: 2.2em; color: white; letter-spacing: -0.5px;">{emoji} {name}</h2>
        <p style="color: #94a3b8; margin: 4px 0 20px 0; font-size: 1.1em;">{desc}</p>
        
        <div style="width: 160px; height: 160px; border-radius: 50%; background: conic-gradient({color} {conf_pct}%, #334155 0); display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 0 30px {color}30; border: 4px solid rgba(255,255,255,0.05);">
            <div style="width: 140px; height: 140px; border-radius: 50%; background: #1e293b; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                <span style="font-size: 2em; font-weight: 800; color: {color}; line-height: 1;">{conf_pct:.1f}%</span>
                <span style="font-size: 0.85em; color: #94a3b8; text-transform: uppercase; margin-top: 5px; opacity: 0.7;">Confidence</span>
            </div>
        </div>
    </div>
    
    <div style="background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(8px);">
        <h3 style="margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; color: #e2e8f0; font-size: 1em; text-transform: uppercase; letter-spacing: 1px;">Biomechanical Probability</h3>
    """
    
    # Add probability bars for each class
    for shot, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        sh_em, sh_nm, _ = SHOT_META.get(shot, ('🏏', shot, ''))
        bar_w = prob * 100
        highlight = "background: linear-gradient(90deg, #38bdf8, #818cf8);" if shot == label else "background: #475569;"
        opacity = "opacity: 1;" if shot == label else "opacity: 0.6;"
        star = "⚡ " if shot == label else ""
        
        html += f"""
        <div style="margin-bottom: 12px; {opacity}">
            <div style="display: flex; justify-content: space-between; font-size: 0.95em; margin-bottom: 5px; color: #cbd5e1;">
                <span style="font-weight: 500;">{star}{sh_nm}</span>
                <span style="font-weight: bold; color: {color if shot == label else '#94a3b8'};">{bar_w:.1f}%</span>
            </div>
            <div style="width: 100%; background: #0f172a; height: 10px; border-radius: 5px; overflow: hidden; border: 1px solid #334155;">
                <div style="height: 100%; width: {bar_w}%; {highlight} transition: width 1s cubic-bezier(0.17, 0.67, 0.83, 0.67);"></div>
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
.gradio-container { max-width: 1300px !important; margin: auto; }
#logo-img { display: block !important; margin: 40px auto 10px auto !important; border-radius: 24px !important; box-shadow: 0 8px 32px rgba(56, 189, 248, 0.25) !important; border: 2px solid rgba(255,255,255,0.05) !important; }
#title { text-align: center; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 3.5em; margin-bottom: 0px; padding-top: 15px; letter-spacing: -1px; }
#subtitle { text-align: center; color: #94a3b8; font-size: 1.2em; margin-bottom: 30px; font-weight: 500; letter-spacing: 0.2px; }
.glass-panel { background: rgba(30, 41, 59, 0.5) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 24px !important; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important; padding: 25px !important; }
button.primary { background: linear-gradient(135deg, #38bdf8, #818cf8) !important; border: none !important; color: white !important; font-weight: 800 !important; font-size: 1.2em !important; box-shadow: 0 4px 20px rgba(56, 189, 248, 0.4) !important; transition: all 0.4s cubic-bezier(0.17, 0.67, 0.83, 0.67) !important; border-radius: 14px !important; height: 60px !important; }
button.primary:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(56, 189, 248, 0.6) !important; }
.dark { background-color: #0b0f19 !important; }
label { color: #94a3b8 !important; text-transform: uppercase !important; font-weight: 700 !important; font-size: 0.75em !important; letter-spacing: 1px !important; }
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
    
    if LOGO_PATH.exists():
        gr.Image(
            value=str(LOGO_PATH),
            show_label=False,
            show_download_button=False,
            interactive=False,
            container=False,
            elem_id="logo-img",
            width=120,
            height=120,
        )
    
    gr.Markdown(f"# {PROJECT_NAME}", elem_id="title")
    gr.Markdown("Transforming batting motion into data. Upload a cricket photo and see **XGBoost biomechanics** in action.", elem_id="subtitle")
    
    with gr.Row(equal_height=True):
        # LEFT COLUMN: Input & Tips
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### 📸 Biomechanical Capture")
            image_input = gr.Image(label="Input Image", type="pil", height=450)
            submit_btn = gr.Button("🔍 ANALYZE BIOMECHANICS", variant="primary")
            
            with gr.Accordion("🛠️ Advanced Capture Tips", open=False):
                gr.Markdown(
                    "- **Body Visibility:** Ensure head-to-toe visibility for optimal pose detection.\n"
                    "- **Angle:** Side-on or 45° angles provide the richest biomechanical data.\n"
                    "- **Resolution:** Motion blur affects MediaPipe quality. Use clear daylight images.\n"
                    "- **System Guardrail:** V2 identifies shots only with > 65% confidence for professional reliability."
                )
                
        # RIGHT COLUMN: Results & Diagnostic Output
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### ⚙️ Diagnostic AI Analysis")
            output_html = gr.HTML(value="<div style='text-align: center; padding: 80px; color: #475569; font-size: 1.2em;'><em>Awaiting biomechanical input...</em></div>")
            output_img  = gr.Image(label="Pose Skeleton Extraction", height=380)

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
    gr.Markdown(
        "---\n"
        "<div style='text-align: center; padding: 30px;'>"
        "  <div style='color: #e2e8f0; font-size: 1.5em; font-weight: 800; margin-bottom: 5px;'>Created by Himanshu Jadhav</div>"
        "  <div style='color: #94a3b8; font-size: 1em; margin-bottom: 25px; font-weight: 500;'>AI & Data Science Engineering Student</div>"
        "  <div style='display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;'>"
        "    <a href='https://github.com/himanshu-jadhav108' target='_blank'><img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white'></a>"
        "    <a href='https://www.linkedin.com/in/himanshu-jadhav-328082339' target='_blank'><img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white'></a>"
        "    <a href='https://www.instagram.com/himanshu_jadhav_108' target='_blank'><img src='https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white'></a>"
        "    <a href='https://himanshu-jadhav-portfolio.vercel.app/' target='_blank'><img src='https://img.shields.io/badge/Portfolio-FFD700?style=for-the-badge&logo=google-chrome&logoColor=black'></a>"
        "  </div>"
        "  <div style='margin-top: 30px; color: #475569; font-size: 0.9em;'>"
        f"    {PROJECT_NAME} Model Architecture: MediaPipe Pose + XGBoost + Confidence Thresholding · 91.45% Safe Accuracy"
        "  </div>"
        "</div>"
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
