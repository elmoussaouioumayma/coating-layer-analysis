"""Small web app: upload a cross-section image and get the layer analysis.

    python app.py           # opens http://127.0.0.1:7860

In Jupyter:  from app import build_app; build_app().launch()
"""

import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from layer_analysis import Params, analyze  # noqa: E402
from plot import plot_analysis  # noqa: E402

DEMO = Path(__file__).parent / "demo" / "cross_section.png"


def run(image, scale_um, scale_px, layers, has_substrate):
    if image is None:
        raise gr.Error("Please upload an image first.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if not scale_um or not scale_px or scale_um <= 0 or scale_px <= 0:
        raise gr.Error("Enter the scale bar: its length in µm and in pixels.")
    p = Params(px_per_um=scale_px / scale_um, n_layers=None if layers == "Auto" else int(layers),
               detect_substrate=has_substrate)
    res = analyze(gray, p)
    fig = plot_analysis(res)
    tmp = Path(tempfile.mkdtemp())
    fig.savefig(tmp / "analysis.png", dpi=110)
    plt.close(fig)
    table = res.layer_table()
    table.to_csv(tmp / "layers.csv", index=False)
    res.boundary_table().to_csv(tmp / "boundaries.csv", index=False)
    res.gradient_table().to_csv(tmp / "gradient.csv", index=False)
    res.particles.round(4).to_csv(tmp / "particles.csv", index=False)
    zpath = tmp / "layer_analysis_results.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in ("analysis.png", "layers.csv", "boundaries.csv", "gradient.csv", "particles.csv"):
            z.write(tmp / f, f)
    show = table[["region", "mean_thickness_um", "thickness_std_um", "particles", "median_diameter_um"]].copy()
    show.columns = ["Region", "Thickness (µm)", "± (µm)", "Particles", "Median diameter (µm)"]
    summary = f"**{res.n_layers} coating layer(s)** found. Calibration: {p.px_per_um:.2f} px/µm."
    return str(tmp / "analysis.png"), show.round(3), summary, str(zpath)


def build_app():
    with gr.Blocks(title="Coating layer analysis") as app:
        gr.Markdown("# Coating layer analysis\nUpload a cross-section image of a coating. Layers are found from "
                    "changes in the **size and density of the particles** inside the coating.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(label="Cross-section image", type="numpy", image_mode="L",
                               value=str(DEMO) if DEMO.exists() else None)
                with gr.Row():
                    scale_um = gr.Number(label="Scale bar (µm)", value=10)
                    scale_px = gr.Number(label="Scale bar (px)", value=165)
                layers = gr.Dropdown(["Auto", "1", "2", "3", "4", "5"], value="Auto", label="Number of layers")
                sub = gr.Checkbox(value=True, label="Image shows the substrate at the bottom")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                fig = gr.Image(label="Result", type="filepath")
                table = gr.Dataframe(label="Layers")
                files = gr.File(label="Download results (figure + CSV)")
        btn.click(run, [img, scale_um, scale_px, layers, sub], [fig, table, summary, files])
    return app


if __name__ == "__main__":
    build_app().launch()
