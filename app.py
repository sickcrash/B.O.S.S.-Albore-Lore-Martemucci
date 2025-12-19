import os
import sys
import json
import time
import queue
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

import customtkinter as ctk

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_path(p: str) -> str:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p))


def resolve_yaml_variables(cfg: dict) -> dict:
    paths = cfg.get("paths", {})

    dataset_root_abs = resolve_path(paths.get("dataset_root", ""))
    paths["dataset_root"] = dataset_root_abs

    for k, v in list(paths.items()):
        if isinstance(v, str) and "${paths.dataset_root}" in v:
            paths[k] = v.replace("${paths.dataset_root}", dataset_root_abs)

    for k, v in list(paths.items()):
        if isinstance(v, str):
            paths[k] = resolve_path(v)

    cfg["paths"] = paths
    return cfg


# =========================
# LOAD CONFIG
# =========================
CONFIG_PATH = resolve_path("config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"config.yaml non trovato in: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

CFG = resolve_yaml_variables(CFG)
PATHS = CFG.get("paths", {})

# =========================
# PATHS DEL PROGETTO
# =========================
for sub in ["source_adapter", "pipeline_integration", "temporal_tracking"]:
    p = os.path.join(PROJECT_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from source_adapter.file_video_source import FileVideoSource
from model_builder import EnhancedTemporalSSD
from temporal_tracking.temporal_consistency_integration import TemporalConsistencyIntegration
from pipeline_integration.logging_system import LoggingSystem


# =========================
# ENGINE & CONFIG
# =========================
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        exports_dir = PATHS.get("exports_dir", resolve_path("./exports"))
        self.output_dir = os.path.join(exports_dir, "inference_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.confidence_threshold = 0.5

        self.class_names = {
            0: "background",
            1: "person",
            2: "vehicle",
            3: "bike",
            4: "motor",
        }
        self.class_colors = {
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (0, 165, 255),
            4: (128, 0, 128),
        }


class InferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.temporal = TemporalConsistencyIntegration(iou_threshold=0.5, max_lost_frames=5)
        self.logger = LoggingSystem(debug=False)

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.config.device)
        num_classes = ckpt.get("num_classes", 5)

        self.model = EnhancedTemporalSSD(
            num_classes=num_classes,
            device=self.config.device,
            use_adapters=True,
            use_temporal=True,
            use_weather_aug=False
        ).to(self.config.device)

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        if self.config.device == "cuda":
            self.model.half()

    def _parse_output_frame(self, frame_out):
        parsed = []

        if isinstance(frame_out, dict) and "boxes" in frame_out:
            boxes = frame_out["boxes"]
            scores = frame_out["scores"]
            labels = frame_out["labels"]

            if torch.is_tensor(boxes):  boxes = boxes.detach().cpu().numpy()
            if torch.is_tensor(scores): scores = scores.detach().cpu().numpy()
            if torch.is_tensor(labels): labels = labels.detach().cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                parsed.append({"bbox": b[:4].tolist(), "score": float(s), "label": int(l)})

        elif isinstance(frame_out, list):
            for d in frame_out:
                if not isinstance(d, dict):
                    continue
                bbox = d.get("bbox", d.get("boxes"))
                score = d.get("score", d.get("scores"))
                label = d.get("label", d.get("labels"))
                if bbox is None or score is None or label is None:
                    continue

                if torch.is_tensor(bbox):
                    bbox = bbox.detach().cpu().numpy().tolist()
                elif isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()

                if torch.is_tensor(score):
                    score = float(score.detach().cpu().item())
                else:
                    score = float(score)

                if torch.is_tensor(label):
                    label = int(label.detach().cpu().item())
                else:
                    label = int(label)

                parsed.append({"bbox": bbox[:4], "score": score, "label": label})

        return parsed

    def process_video(self, video_path: str, max_frames: int, batch_size: int = 8, progress_callback=None):
        if self.model is None:
            raise RuntimeError("Modello non caricato. Chiama load_model() prima.")

        # total frames best effort
        try:
            cap = cv2.VideoCapture(video_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_frames = min(max_frames, total_video_frames) if total_video_frames > 0 else max_frames
        except Exception:
            total_frames = max_frames

        src = FileVideoSource(video_path)
        src.open()

        frames = []
        tensors = []
        frame_indices = []
        drops = 0

        self.logger.start()

        t_read0 = time.perf_counter()
        preproc_time = 0.0

        for i, item in enumerate(src):
            if i >= max_frames:
                break

            if progress_callback:
                progress_callback(i + 1, total_frames, "Lettura video...")

            img = item["image"]
            meta = item.get("meta", {}) if isinstance(item, dict) else {}
            frame_index = meta.get("frame_index", i)

            if frame_indices:
                prev = frame_indices[-1]
                if frame_index > prev + 1:
                    drops += int(frame_index - prev - 1)

            frame_indices.append(int(frame_index))
            frames.append(img)

            t_p0 = time.perf_counter()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
            preproc_time += (time.perf_counter() - t_p0)

            tensors.append(t)
            self.logger.log({"frame_index": frame_index}, [])

        read_time = time.perf_counter() - t_read0
        src.close()

        if not tensors:
            return None, None, None

        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t_fwd0 = time.perf_counter()

        clip_dets = []
        num_batches = (len(tensors) + batch_size - 1) // batch_size

        for start in range(0, len(tensors), batch_size):
            bidx = start // batch_size + 1
            if progress_callback:
                progress_callback(bidx, num_batches, "Elaborazione modello...")

            batch = torch.stack(tensors[start:start + batch_size]).to(self.config.device)
            if self.config.device == "cuda":
                batch = batch.half()

            with torch.no_grad():
                raw = self.model.forward_video_clip(batch)

            for frame_out in raw:
                clip_dets.append(self._parse_output_frame(frame_out))

        if self.config.device == "cuda":
            torch.cuda.synchronize()
        fwd_time = time.perf_counter() - t_fwd0

        n_frames = len(frames)
        fwd_fps = (n_frames / fwd_time) if fwd_time > 0 else 0.0
        fwd_latency_ms = (fwd_time / n_frames) * 1000.0 if n_frames > 0 else 0.0

        stats = {
            "frames": n_frames,
            "read_time": read_time,
            "preproc_time": preproc_time,
            "fwd_time": fwd_time,
            "fwd_fps": fwd_fps,
            "fwd_latency_ms": fwd_latency_ms,
            "drops": drops,
        }
        return frames, clip_dets, stats


# =========================
# REPORT
# =========================
def format_report(total_frames, stats, metrics_dict):
    def fmt(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "NA"

    frames = int(stats.get("frames", total_frames) or 0)
    read_time = float(stats.get("read_time", 0.0))
    preproc_time = float(stats.get("preproc_time", 0.0))
    fwd_time = float(stats.get("fwd_time", 0.0))
    post_time = float(stats.get("post_time", 0.0))

    e2e_time = read_time + preproc_time + fwd_time + post_time
    e2e_fps = (frames / e2e_time) if e2e_time > 0 else 0.0
    e2e_latency_ms = (e2e_time / frames) * 1000.0 if frames > 0 else 0.0

    report = []
    report.append("=" * 60)
    report.append("B.O.S.S. VIDEO INFERENCE REPORT")
    report.append("=" * 60)
    report.append(f"Temporal metrics | Total frames: {int(total_frames)}")
    report.append("-" * 60)
    report.append("PERFORMANCE METRICS")
    report.append("-" * 60)
    report.append(f"✓ Frames elaborati: {frames}")
    report.append(f"✓ FPS forward (solo modello): {fmt(stats.get('fwd_fps', 0.0), 2)}")
    report.append(f"✓ Latenza forward (ms/frame): {fmt(stats.get('fwd_latency_ms', 0.0), 2)}")
    report.append(f"✓ FPS end-to-end: {fmt(e2e_fps, 2)}")
    report.append(f"✓ Latenza end-to-end (ms/frame): {fmt(e2e_latency_ms, 2)}")
    report.append("")
    report.append("TIMING DETAILS")
    report.append(f"  • Lettura video: {fmt(read_time, 3)}s")
    report.append(f"  • Pre-processing: {fmt(preproc_time, 3)}s")
    report.append(f"  • Inference: {fmt(fwd_time, 3)}s")
    report.append(f"  • Post-processing: {fmt(post_time, 3)}s")
    report.append(f"  • Totale: {fmt(e2e_time, 3)}s")
    report.append("")
    report.append("DROP FRAMES ANALYSIS")
    report.append(f"  • Frame persi: {int(stats.get('drops', 0))}")
    report.append("")
    report.append("TEMPORAL TRACKING ANALYSIS")
    report.append("-" * 60)

    for cat, m in (metrics_dict or {}).items():
        if not isinstance(m, dict):
            continue
        tracks = m.get("total_tracks", m.get("tracks", m.get("track_count", 0)))
        avg_len = m.get("avg_track_length", m.get("avg_len", None))
        avg_iou = m.get("avg_iou", None)
        completeness = m.get("completeness", None)

        report.append(
            f"├─ {cat.upper():<12} "
            f"| Tracciati: {int(tracks):3d} "
            f"| Len media: {fmt(avg_len, 1):>6} "
            f"| IoU: {fmt(avg_iou, 3):>6} "
            f"| Completezza: {fmt(completeness, 1):>5}"
        )

    report.append("=" * 60)
    report.append(f"Report generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    return "\n".join(report)


# =========================
# DRAW
# =========================
def draw_bbox_with_label(img_bgr, bbox, label_text, color, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(img_bgr, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_th = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_th)

    y_text_top = y1 - th - baseline - 6
    if y_text_top < 0:
        y_text_top = y1 + th + baseline + 6

    x_text_left = x1
    cv2.rectangle(
        img_bgr,
        (x_text_left, y_text_top - 2),
        (x_text_left + tw + 8, y_text_top + th + baseline + 2),
        color,
        -1
    )
    cv2.putText(
        img_bgr,
        label_text,
        (x_text_left + 4, y_text_top + th),
        font,
        font_scale,
        (255, 255, 255),
        font_th,
        cv2.LINE_AA
    )


# =========================
# MODERN UI (CustomTkinter)
# =========================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("B.O.S.S. — Video Inference")
        self.geometry("1480x920")
        self.minsize(1300, 820)

        # --- state ---
        self.cfg = Config()
        self.engine = InferenceEngine(self.cfg)

        self.queue = queue.Queue()

        self.model_loaded = False
        self.session_dir = ""
        self.stats = {}
        self.report = ""

        self.frames = []
        self.clip_dets = []
        self.display_frames = []
        self.current_frame = 0

        self.playing = False
        self.video_timer = None
        self._updating_slider = False

        self._queue_after_id = None
        self._timer_after_id = None
        self.timer_running = False
        self.start_time = None

        # --- layout grid ---
        self.grid_columnconfigure(0, weight=0)   # sidebar
        self.grid_columnconfigure(1, weight=1)   # main
        self.grid_rowconfigure(0, weight=1)

        self.build_sidebar()
        self.build_main()

        self.start_queue_loop()

    # ---------------------------
    # SIDEBAR
    # ---------------------------
    def build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, corner_radius=18)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=16, pady=16)
        self.sidebar.grid_rowconfigure(99, weight=1)

        title = ctk.CTkLabel(self.sidebar, text="B.O.S.S.", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 4))

        sub = ctk.CTkLabel(self.sidebar, text="Video Inference Dashboard", text_color=("gray70", "gray70"))
        sub.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 14))

        # Model path
        ctk.CTkLabel(self.sidebar, text="Model path", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, sticky="w", padx=16, pady=(8, 6)
        )
        self.model_entry = ctk.CTkEntry(self.sidebar, width=360)
        self.model_entry.grid(row=3, column=0, sticky="w", padx=16)
        self.model_entry.insert(0, PATHS.get("model_save_path", ""))

        btn_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_row.grid(row=4, column=0, sticky="w", padx=16, pady=(8, 16))
        ctk.CTkButton(btn_row, text="Sfoglia", width=110, command=self.browse_model).pack(side="left", padx=(0, 10))
        self.btn_load = ctk.CTkButton(btn_row, text="Carica", width=110, fg_color="#22c55e",
                                      hover_color="#16a34a", text_color="black",
                                      command=self.load_model)
        self.btn_load.pack(side="left")

        # Video path
        ctk.CTkLabel(self.sidebar, text="Video path", font=ctk.CTkFont(weight="bold")).grid(
            row=5, column=0, sticky="w", padx=16, pady=(6, 6)
        )
        self.video_entry = ctk.CTkEntry(self.sidebar, width=360)
        self.video_entry.grid(row=6, column=0, sticky="w", padx=16)
        self.video_entry.insert(0, PATHS.get("test_video_path", ""))

        vrow = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        vrow.grid(row=7, column=0, sticky="w", padx=16, pady=(8, 16))
        ctk.CTkButton(vrow, text="Sfoglia", width=110, command=self.browse_video).pack(side="left", padx=(0, 10))

        # Params
        params = ctk.CTkFrame(self.sidebar, corner_radius=16)
        params.grid(row=8, column=0, sticky="we", padx=16, pady=(0, 16))
        params.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(params, text="Parametri", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        # max frames
        row = ctk.CTkFrame(params, fg_color="transparent")
        row.grid(row=1, column=0, sticky="we", padx=14, pady=(6, 0))
        ctk.CTkLabel(row, text="Max frames", text_color=("gray70", "gray70")).pack(side="left")
        self.max_frames = ctk.CTkEntry(row, width=110)
        self.max_frames.pack(side="right")
        self.max_frames.insert(0, "150")

        # batch size
        row2 = ctk.CTkFrame(params, fg_color="transparent")
        row2.grid(row=2, column=0, sticky="we", padx=14, pady=(10, 0))
        ctk.CTkLabel(row2, text="Batch size", text_color=("gray70", "gray70")).pack(side="left")
        self.batch_size = ctk.CTkEntry(row2, width=110)
        self.batch_size.pack(side="right")
        self.batch_size.insert(0, "8")

        # confidence slider
        ctk.CTkLabel(params, text="Confidence", text_color=("gray70", "gray70")).grid(row=3, column=0, sticky="w", padx=14, pady=(12, 0))
        self.conf_slider = ctk.CTkSlider(params, from_=0.0, to=1.0, number_of_steps=100, command=self.on_conf_change)
        self.conf_slider.grid(row=4, column=0, sticky="we", padx=14, pady=(6, 0))
        self.conf_slider.set(0.5)
        self.conf_value = ctk.CTkLabel(params, text="0.50", font=ctk.CTkFont(weight="bold"))
        self.conf_value.grid(row=5, column=0, sticky="e", padx=14, pady=(6, 12))

        # Actions
        self.btn_process = ctk.CTkButton(
            self.sidebar, text="🚀 Processa Video", height=44,
            command=self.process_video, state="disabled"
        )
        self.btn_process.grid(row=9, column=0, sticky="we", padx=16, pady=(0, 10))

        actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        actions.grid(row=10, column=0, sticky="we", padx=16, pady=(0, 0))
        self.btn_save = ctk.CTkButton(actions, text="💾 Salva risultati", command=self.save_results, state="disabled")
        self.btn_save.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.btn_export = ctk.CTkButton(actions, text="🎬 Esporta video", command=self.export_video, state="disabled")
        self.btn_export.pack(side="left", expand=True, fill="x")

        # bottom (status badge)
        self.badge = ctk.CTkLabel(self.sidebar, text="● Pronto", text_color="#22c55e")
        self.badge.grid(row=99, column=0, sticky="w", padx=16, pady=(0, 16))

    # ---------------------------
    # MAIN AREA
    # ---------------------------
    def build_main(self):
        self.main = ctk.CTkFrame(self, corner_radius=18)
        self.main.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # topbar
        top = ctk.CTkFrame(self.main, corner_radius=16)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        top.grid_columnconfigure(1, weight=1)

        self.top_title = ctk.CTkLabel(top, text="Dashboard", font=ctk.CTkFont(size=18, weight="bold"))
        self.top_title.grid(row=0, column=0, sticky="w", padx=14, pady=12)

        self.kpi = ctk.CTkLabel(top, text="FPS: --   |   Tempo: --", text_color=("gray70", "gray70"))
        self.kpi.grid(row=0, column=1, sticky="e", padx=14, pady=12)

        # content split
        content = ctk.CTkFrame(self.main, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)

        # Preview card
        self.preview_card = ctk.CTkFrame(content, corner_radius=18)
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.preview_card.grid_rowconfigure(1, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.preview_card, text="Anteprima", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 8)
        )

        self.preview_label = ctk.CTkLabel(self.preview_card, text="")
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))

        # playback controls
        pb = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        pb.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        pb.grid_columnconfigure(2, weight=1)

        self.btn_play = ctk.CTkButton(pb, text="▶️", width=44, command=self.play_video, state="disabled")
        self.btn_play.grid(row=0, column=0, padx=(0, 8))
        self.btn_pause = ctk.CTkButton(pb, text="⏸️", width=44, command=self.pause_video, state="disabled")
        self.btn_pause.grid(row=0, column=1, padx=(0, 12))

        self.slider = ctk.CTkSlider(pb, from_=0, to=0, number_of_steps=1, command=self.on_slider_move)
        self.slider.grid(row=0, column=2, sticky="ew")

        self.frame_lbl = ctk.CTkLabel(pb, text="Frame: 0/0", text_color=("gray70", "gray70"))
        self.frame_lbl.grid(row=0, column=3, padx=(12, 0))

        # Report card
        self.report_card = ctk.CTkFrame(content, corner_radius=18)
        self.report_card.grid(row=0, column=1, sticky="nsew")
        self.report_card.grid_rowconfigure(1, weight=1)
        self.report_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.report_card, text="Report", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 8)
        )

        self.report_box = ctk.CTkTextbox(self.report_card, corner_radius=14)
        self.report_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.report_box.insert("1.0",
                               "B.O.S.S. VIDEO INFERENCE SYSTEM\n\n"
                               "1) Carica un modello (.pth)\n"
                               "2) Seleziona un video\n"
                               "3) Premi 'Processa Video'\n")
        self.report_box.configure(state="disabled")

        # bottom status + progress
        bottom = ctk.CTkFrame(self.main, corner_radius=16)
        bottom.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        bottom.grid_columnconfigure(1, weight=1)

        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.grid(row=0, column=0, sticky="w", padx=14, pady=14)
        self.progress.set(0)

        self.status_text = ctk.CTkLabel(bottom, text="Sistema pronto", text_color=("gray70", "gray70"))
        self.status_text.grid(row=0, column=1, sticky="w", padx=10)

    # ---------------------------
    # UI helpers
    # ---------------------------
    def set_badge(self, text, kind="ok"):
        color = {"ok": "#22c55e", "warn": "#f97316", "bad": "#ef4444", "info": "#60a5fa"}.get(kind, "#60a5fa")
        self.badge.configure(text=f"● {text}", text_color=color)

    def set_status(self, text):
        self.status_text.configure(text=text)

    def set_kpi(self, fps=None, elapsed=None):
        fps_s = f"{fps:.1f}" if isinstance(fps, (int, float)) else "--"
        if elapsed is None:
            t_s = "--"
        else:
            mins, secs = divmod(int(elapsed), 60)
            t_s = f"{mins:02d}:{secs:02d}"
        self.kpi.configure(text=f"FPS: {fps_s}   |   Tempo: {t_s}")

    def write_report(self, text: str):
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", text)
        self.report_box.configure(state="disabled")

    def append_report(self, text: str):
        self.report_box.configure(state="normal")
        self.report_box.insert("end", text)
        self.report_box.see("end")
        self.report_box.configure(state="disabled")

    def on_conf_change(self, value):
        self.conf_value.configure(text=f"{float(value):.2f}")

    def _get_int(self, entry: ctk.CTkEntry, default: int):
        try:
            v = int(entry.get().strip())
            return v if v > 0 else default
        except Exception:
            return default

    # ---------------------------
    # File dialogs
    # ---------------------------
    def browse_model(self):
        path = ctk.filedialog.askopenfilename(
            title="Seleziona modello",
            filetypes=[("PyTorch model", "*.pth *.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, path)

    def browse_video(self):
        path = ctk.filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_entry.delete(0, "end")
            self.video_entry.insert(0, path)

    # ---------------------------
    # Model
    # ---------------------------
    def load_model(self):
        mp = self.model_entry.get().strip()
        if not mp:
            ctk.CTkMessagebox(title="Errore", message="model_save_path vuoto (config.yaml) o non selezionato.", icon="cancel")
            return
        if not os.path.exists(mp):
            ctk.CTkMessagebox(title="Errore", message=f"Modello non trovato:\n{mp}", icon="cancel")
            return

        self.set_badge("Caricamento modello", "warn")
        self.set_status("Caricamento modello in corso...")
        self.btn_load.configure(state="disabled")
        self.progress.set(0.15)

        try:
            self.cfg.confidence_threshold = float(self.conf_slider.get())
            self.engine.load_model(mp)
            self.model_loaded = True
            self.btn_process.configure(state="normal")
            self.set_badge("Modello caricato", "ok")
            self.set_status(f"Modello OK — device: {self.cfg.device}")

            self.write_report(
                f"✓ Modello caricato: {os.path.basename(mp)}\n"
                f"✓ Dispositivo: {self.cfg.device}\n"
                f"✓ Confidence: {float(self.conf_slider.get()):.2f}\n\n"
                f"Pronto per l'elaborazione video.\n"
            )
            self.progress.set(0)

        except Exception as e:
            self.set_badge("Errore caricamento", "bad")
            self.set_status("Errore nel caricamento del modello")
            self.progress.set(0)
            ctk.CTkMessagebox(title="Errore", message=str(e), icon="cancel")

        finally:
            self.btn_load.configure(state="normal")

    # ---------------------------
    # Process
    # ---------------------------
    def process_video(self):
        if not self.model_loaded:
            ctk.CTkMessagebox(title="Errore", message="Carica prima un modello!", icon="cancel")
            return

        vp = self.video_entry.get().strip()
        if not vp:
            ctk.CTkMessagebox(title="Errore", message="test_video_path vuoto (config.yaml) o non selezionato.", icon="cancel")
            return
        if not os.path.exists(vp):
            ctk.CTkMessagebox(title="Errore", message=f"Video non trovato:\n{vp}", icon="cancel")
            return

        # stop playback
        self.playing = False
        if self.video_timer is not None:
            try:
                self.after_cancel(self.video_timer)
            except Exception:
                pass
            self.video_timer = None

        # stop old timer callback
        if self._timer_after_id is not None:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
            self._timer_after_id = None

        self.btn_process.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.btn_export.configure(state="disabled")
        self.btn_play.configure(state="disabled")
        self.btn_pause.configure(state="disabled")

        self.progress.set(0.05)
        self.set_badge("Elaborazione", "warn")
        self.set_status("Avvio elaborazione...")
        self.start_time = time.time()
        self.timer_running = True
        self.update_timer_tick()

        max_frames = self._get_int(self.max_frames, 150)
        batch_size = self._get_int(self.batch_size, 8)
        thr = float(self.conf_slider.get())

        self.write_report(
            "ELABORAZIONE VIDEO IN CORSO\n\n"
            f"Video: {os.path.basename(vp)}\n"
            f"Max frames: {max_frames}\n"
            f"Batch size: {batch_size}\n"
            f"Confidence: {thr:.2f}\n\n"
        )

        threading.Thread(target=self._process_thread, daemon=True).start()

    def update_timer_tick(self):
        if self.timer_running and self.start_time:
            elapsed = time.time() - self.start_time
            # fps in tempo reale lo settiamo a fine inferenza
            self.set_kpi(fps=None, elapsed=elapsed)
            self._timer_after_id = self.after(1000, self.update_timer_tick)

    def _process_thread(self):
        try:
            def progress_cb(current, total, msg):
                pct = (current / total) if total else 0.0
                self.queue.put(("progress", (pct, msg, current, total)))

            max_frames = self._get_int(self.max_frames, 150)
            batch_size = self._get_int(self.batch_size, 8)
            thr = float(self.conf_slider.get())

            frames, clip_dets, stats = self.engine.process_video(
                self.video_entry.get().strip(),
                max_frames,
                batch_size,
                progress_cb
            )
            if frames is None:
                self.queue.put(("error", "Nessun frame elaborato"))
                return

            # temporal analysis
            v_name = os.path.basename(self.video_entry.get().strip())
            v_results = {v_name: []}
            for fid, dets in enumerate(clip_dets):
                for d in dets:
                    cat = self.cfg.class_names.get(int(d["label"]), str(int(d["label"])))
                    v_results[v_name].append({
                        "frame_id": fid,
                        "bbox": d["bbox"],
                        "score": d["score"],
                        "label": cat
                    })

            analysis = self.engine.temporal.analyze_temporal_consistency(v_results, thr)
            metrics = analysis[v_name]["metrics"]
            total_frames = analysis[v_name]["total_frames"]

            # session dir
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.cfg.output_dir, f"gui_{ts}")
            os.makedirs(session_dir, exist_ok=True)

            # draw bboxes -> display_frames
            t_post0 = time.perf_counter()
            display_frames = []
            for i, frame in enumerate(frames):
                disp = frame.copy()
                for d in clip_dets[i]:
                    if float(d["score"]) < thr:
                        continue
                    label_id = int(d["label"])
                    cls_name = self.cfg.class_names.get(label_id, f"class_{label_id}")
                    conf = float(d["score"])
                    label_text = f"{cls_name} {conf:.2f}"
                    color = self.cfg.class_colors.get(label_id, (0, 255, 0))
                    draw_bbox_with_label(disp, d["bbox"], label_text, color, thickness=2)
                display_frames.append(disp)

            stats["post_time"] = time.perf_counter() - t_post0

            # save processed video in session
            processed_video_path = os.path.join(session_dir, "processed_video.mp4")
            fps = 30.0
            try:
                cap = cv2.VideoCapture(self.video_entry.get().strip())
                fps_read = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                if fps_read and fps_read > 0:
                    fps = fps_read
            except Exception:
                pass

            self._save_video(processed_video_path, display_frames, fps=fps)
            stats["processed_video_path"] = processed_video_path

            # report
            report_txt = format_report(total_frames=total_frames, stats=stats, metrics_dict=metrics)

            # save files
            with open(os.path.join(session_dir, "detections.json"), "w", encoding="utf-8") as f:
                json.dump(clip_dets, f, indent=2)
            with open(os.path.join(session_dir, "temporal_report.txt"), "w", encoding="utf-8") as rf:
                rf.write(report_txt)

            self.queue.put(("success", (report_txt, session_dir, stats, display_frames)))

        except Exception as e:
            self.queue.put(("error", str(e)))

    # ---------------------------
    # Queue loop
    # ---------------------------
    def start_queue_loop(self):
        self._queue_after_id = self.after(100, self.check_queue)

    def check_queue(self):
        try:
            while True:
                msg, data = self.queue.get_nowait()

                if msg == "progress":
                    pct, txt, cur, tot = data
                    self.progress.set(max(0.05, min(0.95, pct)))
                    self.set_status(f"{txt} ({cur}/{tot})")

                elif msg == "success":
                    report_txt, session_dir, stats, display_frames = data

                    self.timer_running = False
                    if self._timer_after_id is not None:
                        try:
                            self.after_cancel(self._timer_after_id)
                        except Exception:
                            pass
                        self._timer_after_id = None

                    self.session_dir = session_dir
                    self.stats = stats
                    self.report = report_txt
                    self.display_frames = display_frames
                    self.current_frame = 0

                    self.progress.set(1.0)
                    self.set_badge("Completato", "ok")
                    self.set_status(f"Risultati salvati in: {session_dir}")

                    fps_val = float(stats.get("fwd_fps", 0.0) or 0.0)
                    elapsed = time.time() - self.start_time if self.start_time else None
                    self.set_kpi(fps=fps_val, elapsed=elapsed)

                    self.write_report(report_txt)

                    # slider setup + first frame
                    n = len(self.display_frames)
                    if n > 0:
                        self._updating_slider = True
                        self.slider.configure(from_=0, to=max(0, n - 1), number_of_steps=max(1, n - 1))
                        self.slider.set(0)
                        self._updating_slider = False
                        self.frame_lbl.configure(text=f"Frame: 0/{n - 1}")
                        self.show_frame(0)
                        self.btn_play.configure(state="normal")
                        self.btn_pause.configure(state="disabled")

                    self.btn_process.configure(state="normal")
                    self.btn_save.configure(state="normal")
                    self.btn_export.configure(state="normal")
                    self.progress.set(0.0)

                elif msg == "error":
                    err = data

                    self.timer_running = False
                    if self._timer_after_id is not None:
                        try:
                            self.after_cancel(self._timer_after_id)
                        except Exception:
                            pass
                        self._timer_after_id = None

                    self.progress.set(0.0)
                    self.set_badge("Errore", "bad")
                    self.set_status("Elaborazione fallita")
                    self.append_report(f"\n\n✗ ERRORE: {err}\n")
                    self.btn_process.configure(state="normal")

        except queue.Empty:
            pass

        self._queue_after_id = self.after(100, self.check_queue)

    # ---------------------------
    # Preview / playback
    # ---------------------------
    def show_frame(self, idx: int):
        if not self.display_frames:
            return
        idx = max(0, min(idx, len(self.display_frames) - 1))
        self.current_frame = idx

        frame = self.display_frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # fit into label size
        w = max(1, self.preview_label.winfo_width())
        h = max(1, self.preview_label.winfo_height())
        iw, ih = img.size
        ratio = min(w / iw, h / ih)
        new_size = (max(1, int(iw * ratio)), max(1, int(ih * ratio)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img  # keep ref

        self.frame_lbl.configure(text=f"Frame: {idx}/{len(self.display_frames) - 1}")

    def play_video(self):
        if not self.display_frames:
            return
        self.playing = True
        self.btn_play.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.play_next()

    def play_next(self):
        if not self.playing or self.current_frame >= len(self.display_frames) - 1:
            self.playing = False
            self.btn_play.configure(state="normal")
            self.btn_pause.configure(state="disabled")
            return

        self._updating_slider = True
        self.slider.set(self.current_frame + 1)
        self._updating_slider = False
        self.show_frame(self.current_frame + 1)

        delay = 33
        fps = float(self.stats.get("fwd_fps", 0.0) or 0.0)
        if fps > 0:
            delay = int(1000 / min(fps, 60.0))

        self.video_timer = self.after(delay, self.play_next)

    def pause_video(self):
        self.playing = False
        self.btn_play.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        if self.video_timer is not None:
            try:
                self.after_cancel(self.video_timer)
            except Exception:
                pass
            self.video_timer = None

    def on_slider_move(self, value):
        if self._updating_slider:
            return
        try:
            idx = int(float(value))
            self.show_frame(idx)
        except Exception:
            pass

    # ---------------------------
    # Save/export
    # ---------------------------
    def save_results(self):
        if not self.session_dir or not os.path.exists(self.session_dir):
            ctk.CTkMessagebox(title="Errore", message="Nessun risultato da salvare!", icon="cancel")
            return

        dest = ctk.filedialog.askdirectory(title="Seleziona cartella destinazione")
        if not dest:
            return

        try:
            out_dir = os.path.join(dest, os.path.basename(self.session_dir))
            if os.path.exists(out_dir):
                import shutil
                shutil.rmtree(out_dir)
            import shutil
            shutil.copytree(self.session_dir, out_dir)
            ctk.CTkMessagebox(title="OK", message=f"Salvato in:\n{out_dir}", icon="check")
        except Exception as e:
            ctk.CTkMessagebox(title="Errore", message=str(e), icon="cancel")

    def export_video(self):
        if not self.display_frames:
            ctk.CTkMessagebox(title="Errore", message="Nessun video da esportare!", icon="cancel")
            return

        out = ctk.filedialog.asksaveasfilename(
            title="Esporta video elaborato",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")]
        )
        if not out:
            return

        try:
            self._save_video(out, self.display_frames, fps=30.0)
            ctk.CTkMessagebox(title="OK", message=f"Video esportato:\n{out}", icon="check")
        except Exception as e:
            ctk.CTkMessagebox(title="Errore", message=str(e), icon="cancel")

    @staticmethod
    def _save_video(out_path: str, frames_bgr, fps: float = 30.0):
        if not frames_bgr:
            return
        h, w = frames_bgr[0].shape[:2]
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps) if fps > 0 else 30.0, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"Impossibile aprire VideoWriter: {out_path}")
        for fr in frames_bgr:
            vw.write(fr)
        vw.release()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
