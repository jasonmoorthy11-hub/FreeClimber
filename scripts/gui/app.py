"""FreeClimber v2.0 — customtkinter GUI.

Professional desktop app replacing the legacy wxPython interface.
Designed for non-technical lab members: sliders, tooltips, drag-and-drop,
dark mode, progressive disclosure.
"""

import logging
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

# Bootstrap sys.path so bare imports work from any launch location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import customtkinter as ctk
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from gui.controller import AnalysisController
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C = {
    "bg":           "#0f0f1a",
    "bg_panel":     "#161625",
    "bg_card":      "#1c1c30",
    "bg_hover":     "#252540",
    "bg_input":     "#12121f",
    "border":       "#2a2a45",
    "text":         "#e4e4ed",
    "text_dim":     "#7a7a95",
    "text_disabled":"#44445a",
    "accent":       "#53a8b6",
    "accent_hover": "#6bc4d0",
    "accent_muted": "#2a5a64",
    "danger":       "#e94560",
    "danger_hover": "#ff5a75",
    "success":      "#48bb78",
    "warning":      "#ecc94b",
}

S = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32, "xxl": 48}

FONT_FAMILY = "Helvetica"

plt.rcParams.update({
    'figure.facecolor': C["bg"],
    'axes.facecolor':   C["bg_card"],
    'axes.edgecolor':   C["border"],
    'axes.labelcolor':  C["text"],
    'text.color':       C["text"],
    'xtick.color':      C["text_dim"],
    'ytick.color':      C["text_dim"],
    'grid.color':       C["border"],
    'grid.alpha':       0.4,
    'axes.grid':        True,
    'grid.linestyle':   '--',
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'legend.facecolor': C["bg_card"],
    'legend.edgecolor': C["border"],
    'legend.fontsize':  8,
})

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'theme.json'))

# ---------------------------------------------------------------------------
# Tooltip text for parameters
# ---------------------------------------------------------------------------
TOOLTIPS = {
    'diameter': "Expected diameter of fly spots in pixels (must be odd). Start with 7; increase if flies are large blobs.",
    'minmass': "Minimum brightness for a spot to count as a fly. Lower = more detections (but more noise). Start with 100.",
    'maxsize': "Maximum allowed spot diameter. Spots larger than this are rejected (e.g. merged flies).",
    'threshold': "Brightness threshold for spot detection. 'auto' uses Otsu's method. Or enter a number.",
    'ecc_low': "Minimum eccentricity (0 = perfect circle). Rejects too-round artifacts.",
    'ecc_high': "Maximum eccentricity (1 = very elongated). Rejects non-circular objects.",
    'vials': "Number of vials visible in the video. FreeClimber divides the ROI into this many vertical strips.",
    'window': "Number of frames for the local linear regression window. ~2 seconds of video is typical.",
    'pixel_to_cm': "Pixels per centimeter. Use the calibration wizard or measure manually.",
    'frame_rate': "Video frame rate (fps). Auto-detected from video if possible.",
    'blank_0': "First frame for background computation (usually 0).",
    'blank_n': "Last frame for background computation.",
    'crop_0': "First frame to analyze (skip earlier frames).",
    'crop_n': "Last frame to analyze.",
    'check_frame': "Frame to display for parameter testing preview.",
    'outlier_TB': "Top/bottom edge trim (% of vial height). Removes detections near edges.",
    'outlier_LR': "Left/right edge trim (% of vial width). Removes detections near edges.",
    'vial_id_vars': "Number of underscore-separated fields in filename that identify the vial.",
    'background_method': "Background subtraction method. 'temporal_median' is the published default.",
    'individual_tracking': "Enable per-fly tracking with trackpy. Slower but gives individual fly metrics.",
}

# Section definitions: (key, icon, title)
SECTIONS = [
    ("video",      "\u25B6",  "Video"),
    ("roi",        "\u2B21",  "Region of Interest"),
    ("detection",  "\u25C9",  "Detection"),
    ("experiment", "\u2699",  "Experiment"),
    ("frames",     "\u25A4",  "Frames"),
    ("naming",     "\u270E",  "Naming"),
    ("profiles",   "\u2606",  "Profiles"),
]


# ---------------------------------------------------------------------------
# ToolTip
# ---------------------------------------------------------------------------
class ToolTip:
    """Hover tooltip for any widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background=C["bg_card"], foreground=C["text"],
            relief=tk.SOLID, borderwidth=1,
            font=(FONT_FAMILY, 11), wraplength=300, padx=8, pady=6,
        )
        label.pack()

    def hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ---------------------------------------------------------------------------
# CollapsibleCard
# ---------------------------------------------------------------------------
class CollapsibleCard(ctk.CTkFrame):
    """A card with a clickable header that collapses/expands its body."""

    def __init__(self, parent, icon="", title="", initially_open=True, **kwargs):
        super().__init__(parent, fg_color=C["bg_card"], corner_radius=12, **kwargs)

        self._open = initially_open

        # Header
        self.header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        self.header.pack(fill="x", padx=S["sm"], pady=(S["sm"], 0))

        self.chevron_label = ctk.CTkLabel(
            self.header, text="\u25BE" if self._open else "\u25B8",
            font=(FONT_FAMILY, 14), text_color=C["accent"], width=18, anchor="w",
        )
        self.chevron_label.pack(side="left")

        icon_label = ctk.CTkLabel(
            self.header, text=icon, font=(FONT_FAMILY, 14),
            text_color=C["accent"], width=22, anchor="w",
        )
        icon_label.pack(side="left", padx=(0, S["xs"]))

        title_label = ctk.CTkLabel(
            self.header, text=title.upper(),
            font=(FONT_FAMILY, 12, "bold"), text_color=C["text"], anchor="w",
        )
        title_label.pack(side="left", fill="x", expand=True)

        # Make entire header clickable
        for w in (self.header, self.chevron_label, icon_label, title_label):
            w.bind("<Button-1>", self._toggle)

        # Body container
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        if self._open:
            self.body.pack(fill="x", padx=S["sm"], pady=(S["xs"], S["sm"]))

    def _toggle(self, event=None):
        self._open = not self._open
        self.chevron_label.configure(text="\u25BE" if self._open else "\u25B8")
        if self._open:
            self.body.pack(fill="x", padx=S["sm"], pady=(S["xs"], S["sm"]))
        else:
            self.body.pack_forget()


# ---------------------------------------------------------------------------
# ParameterSlider
# ---------------------------------------------------------------------------
class ParameterSlider(ctk.CTkFrame):
    """Labeled slider with manual entry box and optional tooltip."""

    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, dtype=int, tooltip: str = "", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.dtype = dtype
        self.grid_columnconfigure(1, weight=1)

        self.label = ctk.CTkLabel(
            self, text=label, width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        )
        self.label.grid(row=0, column=0, padx=(0, S["sm"]), sticky="w")

        self.var = ctk.DoubleVar(value=default)
        self.slider = ctk.CTkSlider(
            self, from_=from_, to=to, variable=self.var,
            command=self._on_slide,
            fg_color=C["border"], progress_color=C["accent"],
            button_color=C["accent"], button_hover_color=C["accent_hover"],
        )
        self.slider.grid(row=0, column=1, padx=S["xs"], sticky="ew")

        self.entry = ctk.CTkEntry(
            self, width=55, justify="center",
            fg_color=C["bg_input"], border_color=C["border"],
            font=(FONT_FAMILY, 12),
        )
        self.entry.grid(row=0, column=2, padx=(S["xs"], 0))
        self.entry.insert(0, str(dtype(default)))
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

        if tooltip:
            info_btn = ctk.CTkLabel(
                self, text="\u24D8", width=20, cursor="hand2",
                text_color=C["text_dim"], font=(FONT_FAMILY, 13),
            )
            info_btn.grid(row=0, column=3, padx=(S["xs"], 0))
            ToolTip(info_btn, tooltip)

    def _on_slide(self, value):
        v = self.dtype(float(value))
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(v))

    def _on_entry(self, event=None):
        try:
            v = self.dtype(float(self.entry.get()))
            self.var.set(v)
        except ValueError:
            pass

    def get(self):
        try:
            return self.dtype(float(self.entry.get()))
        except ValueError:
            return self.dtype(self.var.get())

    def set(self, value):
        self.var.set(value)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.dtype(value)))


# ---------------------------------------------------------------------------
# FreeClimberApp
# ---------------------------------------------------------------------------
class FreeClimberApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("FreeClimber v2.0")
        self.geometry("1340x860")
        self.minsize(1060, 700)
        self.configure(fg_color=C["bg"])

        self.controller = AnalysisController()
        self.video_meta = None
        self._roi_rect = None
        self._roi_press = False
        self._roi_x0 = self._roi_y0 = 0
        self._roi_x1 = self._roi_y1 = 0
        self.recent_files: list[str] = []
        self._log_visible = False

        self._build_menu_bar()
        self._build_layout()
        self._bind_shortcuts()

        # Status bar
        sep = ctk.CTkFrame(self, height=1, fg_color=C["border"])
        sep.pack(side="bottom", fill="x")
        self.status_var = ctk.StringVar(value="Ready")
        self.status_bar = ctk.CTkLabel(
            self, textvariable=self.status_var, anchor="w",
            height=26, font=(FONT_FAMILY, 11), text_color=C["text_dim"],
            fg_color=C["bg"],
        )
        self.status_bar.pack(side="bottom", fill="x", padx=S["md"], pady=(S["xs"], S["xs"]))

    # ------------------------------------------------------------------
    # Menu / shortcuts
    # ------------------------------------------------------------------
    def _build_menu_bar(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video\u2026  \u2318O", command=self._browse_video)
        file_menu.add_command(label="Load Config\u2026", command=self._load_config)
        file_menu.add_command(label="Save Config\u2026  \u2318S", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results\u2026  \u2318E", command=self._export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Quit  \u2318Q", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Toggle Log Viewer", command=self._toggle_log_viewer)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About FreeClimber", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menubar)

    def _bind_shortcuts(self):
        self.bind("<Command-o>", lambda e: self._browse_video())
        self.bind("<Command-r>", lambda e: self._run_analysis())
        self.bind("<Command-t>", lambda e: self._test_parameters())
        self.bind("<Command-s>", lambda e: self._save_config())
        self.bind("<Command-e>", lambda e: self._export_dialog())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.main_frame = ctk.CTkFrame(self, fg_color=C["bg"])
        self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content()

    def _build_sidebar(self):
        sidebar_outer = ctk.CTkFrame(
            self.main_frame, width=340, fg_color=C["bg_panel"], corner_radius=0,
        )
        sidebar_outer.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)
        sidebar_outer.grid_propagate(False)

        # App title area
        title_frame = ctk.CTkFrame(sidebar_outer, fg_color="transparent")
        title_frame.pack(fill="x", padx=S["md"], pady=(S["lg"], S["sm"]))

        ctk.CTkLabel(
            title_frame, text="FreeClimber",
            font=(FONT_FAMILY, 20, "bold"), text_color=C["text"], anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            title_frame, text="v2.0",
            font=(FONT_FAMILY, 12), text_color=C["accent"], anchor="w",
        ).pack(side="left", padx=(S["sm"], 0), pady=(4, 0))

        ctk.CTkLabel(
            sidebar_outer, text="Drosophila RING Assay Analysis",
            font=(FONT_FAMILY, 11), text_color=C["text_dim"], anchor="w",
        ).pack(fill="x", padx=S["md"], pady=(0, S["md"]))

        # Thin accent line
        ctk.CTkFrame(sidebar_outer, height=2, fg_color=C["accent_muted"]).pack(
            fill="x", padx=S["md"], pady=(0, S["sm"]),
        )

        # Scrollable area for cards
        sidebar = ctk.CTkScrollableFrame(
            sidebar_outer, fg_color=C["bg_panel"], corner_radius=0,
            scrollbar_button_color=C["text_dim"],
            scrollbar_button_hover_color=C["accent"],
        )
        sidebar.pack(fill="both", expand=True, padx=0, pady=0)
        self.sidebar = sidebar
        # Ensure scroll starts at top
        sidebar.after(100, lambda: sidebar._parent_canvas.yview_moveto(0))

        # --- VIDEO card ---
        video_card = CollapsibleCard(sidebar, icon="\u25B6", title="Video")
        video_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        btn_frame = ctk.CTkFrame(video_card.body, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, S["xs"]))
        ctk.CTkButton(
            btn_frame, text="Open Video\u2026", width=140,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=(FONT_FAMILY, 13, "bold"),
            corner_radius=8, command=self._browse_video,
        ).pack(side="left", padx=(0, S["sm"]))
        ctk.CTkButton(
            btn_frame, text="Reload", width=80,
            fg_color=C["bg_hover"], hover_color=C["border"],
            text_color=C["text"], corner_radius=8,
            command=self._reload_video,
        ).pack(side="left")

        self.video_label = ctk.CTkLabel(
            video_card.body, text="No video selected",
            text_color=C["text_dim"], font=(FONT_FAMILY, 11),
            wraplength=280, anchor="w",
        )
        self.video_label.pack(anchor="w", pady=(S["xs"], 0))

        # --- ROI card ---
        roi_card = CollapsibleCard(sidebar, icon="\u2B21", title="Region of Interest")
        roi_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        roi_grid = ctk.CTkFrame(roi_card.body, fg_color="transparent")
        roi_grid.pack(fill="x", pady=(0, S["xs"]))
        self.roi_x = self._labeled_entry(roi_grid, "X", "0", row=0, col=0)
        self.roi_y = self._labeled_entry(roi_grid, "Y", "0", row=0, col=2)
        self.roi_w = self._labeled_entry(roi_grid, "W", "0", row=1, col=0)
        self.roi_h = self._labeled_entry(roi_grid, "H", "0", row=1, col=2)

        self.fixed_roi_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            roi_card.body, text="Fixed ROI dimensions", variable=self.fixed_roi_var,
            font=(FONT_FAMILY, 12), text_color=C["text"],
            fg_color=C["accent"], hover_color=C["accent_hover"],
            border_color=C["border"],
        ).pack(anchor="w", pady=(S["xs"], 0))

        # --- DETECTION card ---
        det_card = CollapsibleCard(sidebar, icon="\u25C9", title="Detection", initially_open=True)
        det_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        self.sl_diameter = ParameterSlider(det_card.body, "Diameter", 3, 31, 7, int, TOOLTIPS['diameter'])
        self.sl_diameter.pack(fill="x", pady=S["xs"])
        self.sl_minmass = ParameterSlider(det_card.body, "Min Mass", 0, 5000, 100, int, TOOLTIPS['minmass'])
        self.sl_minmass.pack(fill="x", pady=S["xs"])
        self.sl_maxsize = ParameterSlider(det_card.body, "Max Size", 1, 50, 11, int, TOOLTIPS['maxsize'])
        self.sl_maxsize.pack(fill="x", pady=S["xs"])

        th_frame = ctk.CTkFrame(det_card.body, fg_color="transparent")
        th_frame.pack(fill="x", pady=S["xs"])
        th_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            th_frame, text="Threshold", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).grid(row=0, column=0, padx=(0, S["sm"]), sticky="w")
        self.threshold_entry = ctk.CTkEntry(
            th_frame, width=80, justify="center",
            fg_color=C["bg_input"], border_color=C["border"],
            font=(FONT_FAMILY, 12),
        )
        self.threshold_entry.grid(row=0, column=1, sticky="w")
        self.threshold_entry.insert(0, "auto")
        info = ctk.CTkLabel(
            th_frame, text="\u24D8", width=20, cursor="hand2",
            text_color=C["text_dim"], font=(FONT_FAMILY, 13),
        )
        info.grid(row=0, column=2, padx=(S["xs"], 0))
        ToolTip(info, TOOLTIPS['threshold'])

        self.sl_ecc_low = ParameterSlider(det_card.body, "Ecc Low", 0, 1, 0, float, TOOLTIPS['ecc_low'])
        self.sl_ecc_low.pack(fill="x", pady=S["xs"])
        self.sl_ecc_high = ParameterSlider(det_card.body, "Ecc High", 0, 1, 1, float, TOOLTIPS['ecc_high'])
        self.sl_ecc_high.pack(fill="x", pady=S["xs"])

        # --- EXPERIMENT card ---
        exp_card = CollapsibleCard(sidebar, icon="\u2699", title="Experiment", initially_open=False)
        exp_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        bg_frame = ctk.CTkFrame(exp_card.body, fg_color="transparent")
        bg_frame.pack(fill="x", pady=S["xs"])
        ctk.CTkLabel(
            bg_frame, text="Background", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).pack(side="left")
        self.bg_method_var = ctk.StringVar(value="temporal_median")
        ctk.CTkOptionMenu(
            bg_frame, variable=self.bg_method_var, width=150,
            values=["temporal_median", "mog2", "running_avg"],
            fg_color=C["bg_input"], button_color=C["accent"],
            button_hover_color=C["accent_hover"],
            font=(FONT_FAMILY, 12),
        ).pack(side="left", padx=(S["sm"], 0))

        self.sl_vials = ParameterSlider(exp_card.body, "Vials", 1, 20, 3, int, TOOLTIPS['vials'])
        self.sl_vials.pack(fill="x", pady=S["xs"])
        self.sl_window = ParameterSlider(exp_card.body, "Window", 5, 500, 50, int, TOOLTIPS['window'])
        self.sl_window.pack(fill="x", pady=S["xs"])

        fr_frame = ctk.CTkFrame(exp_card.body, fg_color="transparent")
        fr_frame.pack(fill="x", pady=S["xs"])
        ctk.CTkLabel(
            fr_frame, text="Frame Rate", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).pack(side="left")
        self.frame_rate_entry = ctk.CTkEntry(
            fr_frame, width=60, justify="center",
            fg_color=C["bg_input"], border_color=C["border"],
            font=(FONT_FAMILY, 12),
        )
        self.frame_rate_entry.pack(side="left", padx=(S["sm"], S["xs"]))
        self.frame_rate_entry.insert(0, "30")
        ctk.CTkLabel(
            fr_frame, text="fps", text_color=C["text_dim"],
            font=(FONT_FAMILY, 11),
        ).pack(side="left")

        px_frame = ctk.CTkFrame(exp_card.body, fg_color="transparent")
        px_frame.pack(fill="x", pady=S["xs"])
        ctk.CTkLabel(
            px_frame, text="Pixel/cm", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).pack(side="left")
        self.pixel_cm_entry = ctk.CTkEntry(
            px_frame, width=60, justify="center",
            fg_color=C["bg_input"], border_color=C["border"],
            font=(FONT_FAMILY, 12),
        )
        self.pixel_cm_entry.pack(side="left", padx=(S["sm"], 0))
        self.pixel_cm_entry.insert(0, "1")

        check_frame = ctk.CTkFrame(exp_card.body, fg_color="transparent")
        check_frame.pack(fill="x", pady=(S["sm"], 0))
        self.convert_cm_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            check_frame, text="Convert to cm/sec", variable=self.convert_cm_var,
            font=(FONT_FAMILY, 12), text_color=C["text"],
            fg_color=C["accent"], hover_color=C["accent_hover"],
            border_color=C["border"],
        ).pack(anchor="w", pady=2)
        self.trim_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            check_frame, text="Trim outliers", variable=self.trim_var,
            font=(FONT_FAMILY, 12), text_color=C["text"],
            fg_color=C["accent"], hover_color=C["accent_hover"],
            border_color=C["border"],
        ).pack(anchor="w", pady=2)
        self.tracking_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            check_frame, text="Individual fly tracking", variable=self.tracking_var,
            font=(FONT_FAMILY, 12), text_color=C["text"],
            fg_color=C["accent"], hover_color=C["accent_hover"],
            border_color=C["border"],
        ).pack(anchor="w", pady=2)

        # --- FRAMES card ---
        frames_card = CollapsibleCard(sidebar, icon="\u25A4", title="Frames", initially_open=False)
        frames_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        self.sl_blank_0 = ParameterSlider(frames_card.body, "Blank Start", 0, 10000, 0, int, TOOLTIPS['blank_0'])
        self.sl_blank_0.pack(fill="x", pady=S["xs"])
        self.sl_blank_n = ParameterSlider(frames_card.body, "Blank End", 0, 10000, 145, int, TOOLTIPS['blank_n'])
        self.sl_blank_n.pack(fill="x", pady=S["xs"])
        self.sl_crop_0 = ParameterSlider(frames_card.body, "Crop Start", 0, 10000, 0, int, TOOLTIPS['crop_0'])
        self.sl_crop_0.pack(fill="x", pady=S["xs"])
        self.sl_crop_n = ParameterSlider(frames_card.body, "Crop End", 0, 10000, 145, int, TOOLTIPS['crop_n'])
        self.sl_crop_n.pack(fill="x", pady=S["xs"])
        self.sl_check = ParameterSlider(frames_card.body, "Check Frame", 0, 10000, 0, int, TOOLTIPS['check_frame'])
        self.sl_check.pack(fill="x", pady=S["xs"])

        self.sl_outlier_tb = ParameterSlider(frames_card.body, "Outlier TB", 0, 20, 1, float, TOOLTIPS['outlier_TB'])
        self.sl_outlier_tb.pack(fill="x", pady=S["xs"])
        self.sl_outlier_lr = ParameterSlider(frames_card.body, "Outlier LR", 0, 20, 3, float, TOOLTIPS['outlier_LR'])
        self.sl_outlier_lr.pack(fill="x", pady=S["xs"])

        # --- NAMING card ---
        naming_card = CollapsibleCard(sidebar, icon="\u270E", title="Naming", initially_open=False)
        naming_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        nc_frame = ctk.CTkFrame(naming_card.body, fg_color="transparent")
        nc_frame.pack(fill="x", pady=S["xs"])
        ctk.CTkLabel(
            nc_frame, text="Convention", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).pack(side="left")
        self.naming_entry = ctk.CTkEntry(
            nc_frame, width=160, fg_color=C["bg_input"],
            border_color=C["border"], font=(FONT_FAMILY, 12),
        )
        self.naming_entry.pack(side="left", padx=(S["sm"], 0))

        vid_frame = ctk.CTkFrame(naming_card.body, fg_color="transparent")
        vid_frame.pack(fill="x", pady=S["xs"])
        ctk.CTkLabel(
            vid_frame, text="Vial ID vars", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).pack(side="left")
        self.vial_id_entry = ctk.CTkEntry(
            vid_frame, width=60, justify="center",
            fg_color=C["bg_input"], border_color=C["border"],
            font=(FONT_FAMILY, 12),
        )
        self.vial_id_entry.pack(side="left", padx=(S["sm"], 0))
        self.vial_id_entry.insert(0, "2")

        proj_frame = ctk.CTkFrame(naming_card.body, fg_color="transparent")
        proj_frame.pack(fill="x", pady=S["xs"])
        ctk.CTkLabel(
            proj_frame, text="Project Path", width=100, anchor="w",
            font=(FONT_FAMILY, 13), text_color=C["text"],
        ).pack(side="left")
        self.project_entry = ctk.CTkEntry(
            proj_frame, width=160, fg_color=C["bg_input"],
            border_color=C["border"], font=(FONT_FAMILY, 12),
        )
        self.project_entry.pack(side="left", padx=(S["sm"], 0))

        # --- PROFILES card ---
        prof_card = CollapsibleCard(sidebar, icon="\u2606", title="Profiles", initially_open=False)
        prof_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        prof_frame = ctk.CTkFrame(prof_card.body, fg_color="transparent")
        prof_frame.pack(fill="x")
        self.profile_var = ctk.StringVar(value="")
        self.profile_menu = ctk.CTkOptionMenu(
            prof_frame, variable=self.profile_var, width=150,
            values=self.controller.list_profiles() or ["(none)"],
            command=self._on_profile_selected,
            fg_color=C["bg_input"], button_color=C["accent"],
            button_hover_color=C["accent_hover"], font=(FONT_FAMILY, 12),
        )
        self.profile_menu.pack(side="left", padx=(0, S["sm"]))
        ctk.CTkButton(
            prof_frame, text="Save", width=55,
            fg_color=C["bg_hover"], hover_color=C["border"],
            text_color=C["text"], corner_radius=6,
            font=(FONT_FAMILY, 12), command=self._save_profile,
        ).pack(side="left", padx=(0, S["xs"]))
        ctk.CTkButton(
            prof_frame, text="Del", width=40,
            fg_color=C["bg_hover"], hover_color=C["danger"],
            text_color=C["text"], corner_radius=6,
            font=(FONT_FAMILY, 12), command=self._delete_profile,
        ).pack(side="left")

        # --- ACTION BUTTONS (fixed at bottom of sidebar) ---
        action_area = ctk.CTkFrame(sidebar_outer, fg_color=C["bg_panel"])
        action_area.pack(side="bottom", fill="x", padx=0, pady=0)

        ctk.CTkFrame(action_area, height=1, fg_color=C["border"]).pack(fill="x")

        action_pad = ctk.CTkFrame(action_area, fg_color="transparent")
        action_pad.pack(fill="x", padx=S["md"], pady=S["md"])

        btn_row1 = ctk.CTkFrame(action_pad, fg_color="transparent")
        btn_row1.pack(fill="x", pady=(0, S["sm"]))
        ctk.CTkButton(
            btn_row1, text="Preview", width=140,
            fg_color=C["bg_hover"], hover_color=C["accent_muted"],
            text_color=C["text"], corner_radius=8,
            font=(FONT_FAMILY, 13), command=self._test_parameters,
        ).pack(side="left", padx=(0, S["sm"]))
        ctk.CTkButton(
            btn_row1, text="Save Config", width=100,
            fg_color=C["bg_hover"], hover_color=C["border"],
            text_color=C["text"], corner_radius=8,
            font=(FONT_FAMILY, 13), command=self._save_config,
        ).pack(side="left")

        btn_row2 = ctk.CTkFrame(action_pad, fg_color="transparent")
        btn_row2.pack(fill="x", pady=(0, S["md"]))
        ctk.CTkButton(
            btn_row2, text="Batch Mode", width=140,
            fg_color=C["bg_hover"], hover_color=C["accent_muted"],
            text_color=C["text"], corner_radius=8,
            font=(FONT_FAMILY, 13), command=self._batch_mode,
        ).pack(side="left", padx=(0, S["sm"]))
        ctk.CTkButton(
            btn_row2, text="Copy Methods", width=100,
            fg_color=C["bg_hover"], hover_color=C["border"],
            text_color=C["text"], corner_radius=8,
            font=(FONT_FAMILY, 13), command=self._copy_methods,
        ).pack(side="left")

        # Run button with highlight effect
        run_wrapper = ctk.CTkFrame(action_pad, fg_color=C["danger"], corner_radius=10)
        run_wrapper.pack(fill="x")
        # Top highlight bar for faux gradient
        ctk.CTkFrame(
            run_wrapper, height=2, fg_color=C["danger_hover"], corner_radius=0,
        ).pack(fill="x")
        self.run_btn = ctk.CTkButton(
            run_wrapper, text="RUN ANALYSIS", height=44,
            font=(FONT_FAMILY, 15, "bold"),
            fg_color=C["danger"], hover_color=C["danger_hover"],
            text_color="#ffffff", corner_radius=8,
            command=self._run_analysis,
        )
        self.run_btn.pack(fill="x", padx=2, pady=(0, 2))

        self.progress_label = ctk.CTkLabel(
            action_pad, text="", text_color=C["text_dim"],
            font=(FONT_FAMILY, 11),
        )
        self.progress_label.pack(anchor="w", pady=(S["sm"], 0))
        self.progress_bar = ctk.CTkProgressBar(
            action_pad, height=6, corner_radius=3,
            fg_color=C["border"], progress_color=C["accent"],
        )
        self.progress_bar.pack(fill="x", pady=(S["xs"], 0))
        self.progress_bar.set(0)

    def _build_content(self):
        content = ctk.CTkFrame(self.main_frame, corner_radius=0, fg_color=C["bg"])
        content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        self.content_frame = content

        # Empty state overlay
        self.empty_state = ctk.CTkFrame(content, fg_color=C["bg"])
        self.empty_state.grid(row=0, column=0, sticky="nsew")

        empty_inner = ctk.CTkFrame(self.empty_state, fg_color="transparent")
        empty_inner.place(relx=0.5, rely=0.45, anchor="center")

        ctk.CTkLabel(
            empty_inner, text="\U0001F3AC",
            font=(FONT_FAMILY, 64), text_color=C["text_dim"],
        ).pack(pady=(0, S["md"]))
        ctk.CTkLabel(
            empty_inner, text="Drop a video here",
            font=(FONT_FAMILY, 22, "bold"), text_color=C["text"],
        ).pack()
        ctk.CTkLabel(
            empty_inner, text="or",
            font=(FONT_FAMILY, 13), text_color=C["text_dim"],
        ).pack(pady=S["sm"])
        ctk.CTkButton(
            empty_inner, text="Open Video", width=160, height=40,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=(FONT_FAMILY, 14, "bold"),
            corner_radius=10, command=self._browse_video,
        ).pack()
        ctk.CTkLabel(
            empty_inner, text="\u2318O to open  \u00b7  \u2318R to run  \u00b7  \u2318T to preview",
            font=(FONT_FAMILY, 11), text_color=C["text_disabled"],
        ).pack(pady=(S["lg"], 0))

        # Tab view (hidden until video loaded)
        self.tabview = ctk.CTkTabview(
            content, fg_color=C["bg"], segmented_button_fg_color=C["bg_card"],
            segmented_button_selected_color=C["accent"],
            segmented_button_selected_hover_color=C["accent_hover"],
            segmented_button_unselected_color=C["bg_card"],
            segmented_button_unselected_hover_color=C["bg_hover"],
            corner_radius=8,
        )

        self._build_setup_tab()
        self._build_diagnostics_tab()
        self._build_results_tab()
        self._build_statistics_tab()

    def _show_tabs(self):
        self.empty_state.grid_forget()
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=S["sm"], pady=S["sm"])

    def _build_setup_tab(self):
        tab = self.tabview.add("Setup")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.setup_fig = Figure(figsize=(8, 4), dpi=100)
        self.setup_canvas = FigureCanvasTkAgg(self.setup_fig, master=tab)
        self.setup_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(tab, fg_color="transparent", height=30)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.setup_toolbar = NavigationToolbar2Tk(self.setup_canvas, toolbar_frame)
        self.setup_toolbar.update()

        self.setup_canvas.mpl_connect('button_press_event', self._roi_press_event)
        self.setup_canvas.mpl_connect('button_release_event', self._roi_release_event)
        self.setup_canvas.mpl_connect('motion_notify_event', self._roi_motion_event)

    def _build_diagnostics_tab(self):
        tab = self.tabview.add("Diagnostics")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.diag_fig = Figure(figsize=(10, 6), dpi=100)
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=tab)
        self.diag_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(tab, fg_color="transparent", height=30)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.diag_canvas, toolbar_frame).update()

    def _build_results_tab(self):
        tab = self.tabview.add("Results")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=2)
        tab.grid_columnconfigure(0, weight=1)

        # --- Summary card at top ---
        self.summary_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8, height=60)
        self.summary_frame.grid(row=0, column=0, sticky="ew", padx=S["xs"], pady=(S["xs"], S["xs"]))
        self.summary_frame.grid_propagate(False)
        self.summary_label = ctk.CTkLabel(
            self.summary_frame, text="Run an analysis to see results here.",
            font=(FONT_FAMILY, 12), text_color=C["text_dim"], anchor="w",
        )
        self.summary_label.pack(fill="x", padx=S["md"], pady=S["sm"])

        # --- Data table section with sub-tabs ---
        table_outer = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8)
        table_outer.grid(row=1, column=0, sticky="nsew", padx=S["xs"], pady=S["xs"])
        table_outer.grid_rowconfigure(0, weight=1)
        table_outer.grid_columnconfigure(0, weight=1)

        self.data_tabview = ctk.CTkTabview(
            table_outer, height=180,
            fg_color=C["bg_card"],
            segmented_button_fg_color=C["bg"],
            segmented_button_selected_color=C["accent"],
            segmented_button_selected_hover_color=C["accent_hover"],
            segmented_button_unselected_color=C["bg"],
            segmented_button_unselected_hover_color=C["bg_hover"],
        )
        self.data_tabview.grid(row=0, column=0, sticky="nsew")

        slopes_tab = self.data_tabview.add("Slopes")
        slopes_tab.grid_rowconfigure(0, weight=1)
        slopes_tab.grid_columnconfigure(0, weight=1)

        self._configure_treeview_style()
        self.slopes_tree = tk.ttk.Treeview(slopes_tab, show="headings", style="Dark.Treeview")
        self.slopes_tree.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        scrollbar = tk.ttk.Scrollbar(slopes_tab, orient="vertical", command=self.slopes_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.slopes_tree.configure(yscrollcommand=scrollbar.set)

        # Per-fly metrics table (populated when individual tracking is on)
        perfly_tab = self.data_tabview.add("Per-Fly Metrics")
        perfly_tab.grid_rowconfigure(0, weight=1)
        perfly_tab.grid_columnconfigure(0, weight=1)
        self.perfly_tree = tk.ttk.Treeview(perfly_tab, show="headings", style="Dark.Treeview")
        self.perfly_tree.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        perfly_scroll = tk.ttk.Scrollbar(perfly_tab, orient="vertical", command=self.perfly_tree.yview)
        perfly_scroll.grid(row=0, column=1, sticky="ns")
        self.perfly_tree.configure(yscrollcommand=perfly_scroll.set)

        # Population stats table
        pop_tab = self.data_tabview.add("Population Stats")
        pop_tab.grid_rowconfigure(0, weight=1)
        pop_tab.grid_columnconfigure(0, weight=1)
        self.pop_stats_text = ctk.CTkTextbox(
            pop_tab, font=("Menlo", 11), state="disabled", wrap="word",
            fg_color=C["bg_card"], text_color=C["text"],
        )
        self.pop_stats_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # --- Plot section with richer tabs ---
        plot_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8)
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=S["xs"], pady=(S["xs"], S["xs"]))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.results_plot_tabview = ctk.CTkTabview(
            plot_frame, height=300,
            fg_color=C["bg_card"],
            segmented_button_fg_color=C["bg"],
            segmented_button_selected_color=C["accent"],
            segmented_button_selected_hover_color=C["accent_hover"],
            segmented_button_unselected_color=C["bg"],
            segmented_button_unselected_hover_color=C["bg_hover"],
        )
        self.results_plot_tabview.grid(row=0, column=0, sticky="nsew")

        speed_tab = self.results_plot_tabview.add("Speed")
        speed_tab.grid_rowconfigure(0, weight=1)
        speed_tab.grid_columnconfigure(0, weight=1)
        self.speed_fig = Figure(figsize=(6, 3), dpi=100)
        self.speed_canvas = FigureCanvasTkAgg(self.speed_fig, master=speed_tab)
        self.speed_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        traj_tab = self.results_plot_tabview.add("Trajectories")
        traj_tab.grid_rowconfigure(0, weight=1)
        traj_tab.grid_columnconfigure(0, weight=1)
        self.traj_fig = Figure(figsize=(6, 3), dpi=100)
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, master=traj_tab)
        self.traj_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        dist_tab = self.results_plot_tabview.add("Distribution")
        dist_tab.grid_rowconfigure(0, weight=1)
        dist_tab.grid_columnconfigure(0, weight=1)
        self.dist_fig = Figure(figsize=(6, 3), dpi=100)
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=dist_tab)
        self.dist_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        perfly_plot_tab = self.results_plot_tabview.add("Per-Fly")
        perfly_plot_tab.grid_rowconfigure(0, weight=1)
        perfly_plot_tab.grid_columnconfigure(0, weight=1)
        self.perfly_fig = Figure(figsize=(6, 3), dpi=100)
        self.perfly_canvas = FigureCanvasTkAgg(self.perfly_fig, master=perfly_plot_tab)
        self.perfly_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_statistics_tab(self):
        tab = self.tabview.add("Statistics")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Box + swarm plot at top
        stats_plot_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8)
        stats_plot_frame.grid(row=0, column=0, sticky="nsew", padx=S["xs"], pady=(S["xs"], S["xs"]))
        stats_plot_frame.grid_rowconfigure(0, weight=1)
        stats_plot_frame.grid_columnconfigure(0, weight=1)
        self.stats_fig = Figure(figsize=(6, 3), dpi=100)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=stats_plot_frame)
        self.stats_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Formatted text at bottom
        self.stats_text = ctk.CTkTextbox(
            tab, font=("Menlo", 11), state="disabled", wrap="word",
            fg_color=C["bg_card"], text_color=C["text"],
        )
        self.stats_text.grid(row=1, column=0, sticky="nsew", padx=S["xs"], pady=S["xs"])
        self._set_stats_text("Run an analysis to see statistics here.")

    # ------------------------------------------------------------------
    # Treeview dark style
    # ------------------------------------------------------------------
    def _configure_treeview_style(self):
        style = tk.ttk.Style()
        style.theme_use("default")
        style.configure("Dark.Treeview",
                         background=C["bg_card"],
                         foreground=C["text"],
                         fieldbackground=C["bg_card"],
                         rowheight=28,
                         font=("Menlo", 11))
        style.configure("Dark.Treeview.Heading",
                         background=C["bg_hover"],
                         foreground=C["text"],
                         font=(FONT_FAMILY, 11, "bold"))
        style.map("Dark.Treeview",
                   background=[("selected", C["accent"])],
                   foreground=[("selected", "#ffffff")])
        return style

    # ------------------------------------------------------------------
    # Helper: labeled entry in a grid
    # ------------------------------------------------------------------
    def _labeled_entry(self, parent, label, default, row, col):
        ctk.CTkLabel(
            parent, text=label, width=20,
            font=(FONT_FAMILY, 12), text_color=C["text_dim"],
        ).grid(row=row, column=col, padx=2, sticky="e")
        entry = ctk.CTkEntry(
            parent, width=70, justify="center",
            fg_color=C["bg_input"], border_color=C["border"],
            font=(FONT_FAMILY, 12),
        )
        entry.grid(row=row, column=col + 1, padx=2, pady=3)
        entry.insert(0, default)
        return entry

    # ------------------------------------------------------------------
    # Collect parameters from GUI into dict
    # ------------------------------------------------------------------
    def _collect_params(self) -> dict:
        def _int(entry):
            try:
                return int(float(entry.get()))
            except (ValueError, TypeError):
                return 0

        def _float(entry):
            try:
                return float(entry.get())
            except (ValueError, TypeError):
                return 0.0

        threshold = self.threshold_entry.get().strip()
        if threshold.lower() != 'auto':
            try:
                threshold = int(float(threshold))
            except ValueError:
                threshold = 'auto'

        return {
            'x': _int(self.roi_x), 'y': _int(self.roi_y),
            'w': _int(self.roi_w), 'h': _int(self.roi_h),
            'check_frame': self.sl_check.get(),
            'blank_0': self.sl_blank_0.get(), 'blank_n': self.sl_blank_n.get(),
            'crop_0': self.sl_crop_0.get(), 'crop_n': self.sl_crop_n.get(),
            'threshold': threshold,
            'diameter': self.sl_diameter.get(),
            'minmass': self.sl_minmass.get(),
            'maxsize': self.sl_maxsize.get(),
            'ecc_low': self.sl_ecc_low.get(),
            'ecc_high': self.sl_ecc_high.get(),
            'vials': self.sl_vials.get(),
            'window': self.sl_window.get(),
            'pixel_to_cm': _float(self.pixel_cm_entry),
            'frame_rate': _int(self.frame_rate_entry),
            'vial_id_vars': _int(self.vial_id_entry),
            'outlier_TB': self.sl_outlier_tb.get(),
            'outlier_LR': self.sl_outlier_lr.get(),
            'naming_convention': self.naming_entry.get() or '',
            'path_project': self.project_entry.get() or '',
            'file_suffix': getattr(self, '_file_suffix', 'h264'),
            'convert_to_cm_sec': self.convert_cm_var.get(),
            'trim_outliers': self.trim_var.get(),
            'background_method': self.bg_method_var.get(),
            'individual_tracking': self.tracking_var.get(),
        }

    def _apply_params_to_gui(self, params: dict):
        """Set GUI controls from a params dict."""
        def _set_entry(entry, val):
            entry.delete(0, tk.END)
            entry.insert(0, str(val))

        _set_entry(self.roi_x, params.get('x', 0))
        _set_entry(self.roi_y, params.get('y', 0))
        _set_entry(self.roi_w, params.get('w', 0))
        _set_entry(self.roi_h, params.get('h', 0))

        self.sl_check.set(params.get('check_frame', 0))
        self.sl_blank_0.set(params.get('blank_0', 0))
        self.sl_blank_n.set(params.get('blank_n', 145))
        self.sl_crop_0.set(params.get('crop_0', 0))
        self.sl_crop_n.set(params.get('crop_n', 145))
        _set_entry(self.threshold_entry, params.get('threshold', 'auto'))
        self.sl_diameter.set(params.get('diameter', 7))
        self.sl_minmass.set(params.get('minmass', 100))
        self.sl_maxsize.set(params.get('maxsize', 11))
        self.sl_ecc_low.set(params.get('ecc_low', 0))
        self.sl_ecc_high.set(params.get('ecc_high', 1))
        self.sl_vials.set(params.get('vials', 3))
        self.sl_window.set(params.get('window', 50))
        _set_entry(self.pixel_cm_entry, params.get('pixel_to_cm', 1))
        _set_entry(self.frame_rate_entry, params.get('frame_rate', 30))
        _set_entry(self.vial_id_entry, params.get('vial_id_vars', 2))
        self.sl_outlier_tb.set(params.get('outlier_TB', 1))
        self.sl_outlier_lr.set(params.get('outlier_LR', 3))
        _set_entry(self.naming_entry, params.get('naming_convention', ''))
        _set_entry(self.project_entry, params.get('path_project', ''))
        self.convert_cm_var.set(params.get('convert_to_cm_sec', False))
        self.trim_var.set(params.get('trim_outliers', False))
        self.bg_method_var.set(params.get('background_method', 'temporal_median'))
        self.tracking_var.set(params.get('individual_tracking', False))

    # ------------------------------------------------------------------
    # ROI drawing on Setup tab
    # ------------------------------------------------------------------
    def _roi_press_event(self, event):
        if event.inaxes is None:
            return
        self._roi_press = True
        self._roi_x0 = int(event.xdata)
        self._roi_y0 = int(event.ydata)

    def _roi_release_event(self, event):
        if not self._roi_press:
            return
        self._roi_press = False
        if event.xdata is not None:
            self._roi_x1 = int(event.xdata)
            self._roi_y1 = int(event.ydata)
        self._update_roi_entries()
        self._draw_roi_rect()

    def _roi_motion_event(self, event):
        if not self._roi_press or event.xdata is None:
            return
        self._roi_x1 = int(event.xdata)
        self._roi_y1 = int(event.ydata)
        self._draw_roi_rect()

    def _draw_roi_rect(self):
        ax = self.setup_fig.axes[0] if self.setup_fig.axes else None
        if ax is None:
            return
        if self._roi_rect:
            self._roi_rect.remove()
        x0, y0 = min(self._roi_x0, self._roi_x1), min(self._roi_y0, self._roi_y1)
        w = abs(self._roi_x1 - self._roi_x0)
        h = abs(self._roi_y1 - self._roi_y0)
        self._roi_rect = Rectangle((x0, y0), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(self._roi_rect)
        self.setup_canvas.draw_idle()

    def _update_roi_entries(self):
        x0, y0 = min(self._roi_x0, self._roi_x1), min(self._roi_y0, self._roi_y1)
        w = abs(self._roi_x1 - self._roi_x0)
        h = abs(self._roi_y1 - self._roi_y0)
        for entry, val in [(self.roi_x, x0), (self.roi_y, y0), (self.roi_w, w), (self.roi_h, h)]:
            entry.delete(0, tk.END)
            entry.insert(0, str(val))

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.h264 *.mkv *.wmv"),
                        ("All files", "*.*")]
        )
        if path:
            self._open_video(path)

    def _reload_video(self):
        if self.controller.video_path:
            self._open_video(self.controller.video_path)

    def _open_video(self, path: str):
        self.status_var.set(f"Loading: {os.path.basename(path)}...")
        self.update_idletasks()

        try:
            params = self._collect_params()
            self.video_meta = self.controller.load_video(path, params)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load video:\n\n{e}")
            self.status_var.set("Error loading video")
            return

        # Switch from empty state to tabs
        self._show_tabs()

        folder, name_ext = os.path.split(path)
        name, ext = os.path.splitext(name_ext)
        self._file_suffix = ext.lstrip('.')

        self.video_label.configure(
            text=f"{name_ext}  ({self.video_meta['n_frames']} frames, "
                 f"{self.video_meta['fps']:.0f} fps)",
            text_color=C["text"],
        )

        n = self.video_meta['n_frames']
        fps = int(self.video_meta['fps'])
        self.sl_blank_n.set(n)
        self.sl_crop_n.set(n)
        self.frame_rate_entry.delete(0, tk.END)
        self.frame_rate_entry.insert(0, str(fps))
        self.sl_window.set(min(fps * 2, n))
        self.naming_entry.delete(0, tk.END)
        self.naming_entry.insert(0, name)
        self.project_entry.delete(0, tk.END)
        self.project_entry.insert(0, folder)
        self.vial_id_entry.delete(0, tk.END)
        self.vial_id_entry.insert(0, str(len(name.split('_'))))

        self.setup_fig.clear()
        if self.video_meta.get('first_frame') is not None:
            ax1 = self.setup_fig.add_subplot(121)
            ax1.imshow(self.video_meta['first_frame'])
            ax1.set_title("Frame 0 (draw ROI)", fontsize=10)
            ax1.axis("off")

            h, w = self.video_meta['first_frame'].shape[:2]
            self._roi_x0, self._roi_y0 = 0, 0
            self._roi_x1, self._roi_y1 = w, h
            self._update_roi_entries()

        if self.video_meta.get('last_frame') is not None:
            ax2 = self.setup_fig.add_subplot(122)
            ax2.imshow(self.video_meta['last_frame'])
            ax2.set_title("Final frame (reference)", fontsize=10)
            ax2.axis("off")

        self.setup_fig.tight_layout()
        self.setup_canvas.draw()
        self.tabview.set("Setup")

        if path not in self.recent_files:
            self.recent_files.insert(0, path)
            self.recent_files = self.recent_files[:10]

        self.status_var.set(f"Loaded: {name_ext}")

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------
    def _test_parameters(self):
        if self.controller.detector is None:
            messagebox.showinfo("FreeClimber", "Load a video first.")
            return

        self.status_var.set("Testing parameters...")
        self.update_idletasks()

        params = self._collect_params()

        self.diag_fig.clear()
        axes = [self.diag_fig.add_subplot(2, 3, i + 1) for i in range(6)]

        try:
            result = self.controller.test_parameters(params, axes)
        except Exception as e:
            messagebox.showerror("Error", f"Parameter testing failed:\n\n{e}")
            self.status_var.set("Error during testing")
            return

        self.diag_fig.tight_layout()
        self.diag_canvas.draw()
        self.tabview.set("Diagnostics")

        if 'slopes_df' in result:
            self._populate_results(result)
            self._populate_statistics(result)

        self.status_var.set("Parameter testing complete")

    # ------------------------------------------------------------------
    # Full analysis (threaded pipeline, main-thread plotting)
    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self.controller.detector is None:
            messagebox.showinfo("FreeClimber", "Load a video first.")
            return

        # Auto-save config before running
        if self.controller.video_path:
            try:
                cfg_path = self.controller.video_path.rsplit('.', 1)[0] + '.cfg'
                self.controller.save_config(cfg_path, self._collect_params())
            except Exception:
                pass

        self.status_var.set("Running analysis...")
        self.run_btn.configure(state="disabled", text="Running...")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Processing...")
        self.update_idletasks()

        params = self._collect_params()
        self.diag_fig.clear()
        axes = [self.diag_fig.add_subplot(2, 3, i + 1) for i in range(6)]

        try:
            self.update_idletasks()
            result = self.controller.test_parameters(params, axes)
        except Exception as e:
            result = e
        self._on_analysis_done(result)

    def _on_analysis_done(self, result):
        self.run_btn.configure(state="normal", text="RUN ANALYSIS")
        self.progress_bar.set(1.0)

        if isinstance(result, Exception):
            messagebox.showerror("Error", f"Analysis failed:\n\n{result}")
            self.status_var.set("Analysis failed")
            self.progress_label.configure(text="Failed")
            return

        self.diag_fig.tight_layout()
        self.diag_canvas.draw()

        if 'slopes_df' in result:
            self._populate_results(result)
            self._populate_statistics(result)
            self.tabview.set("Results")

        self.progress_label.configure(text="Complete")
        self.status_var.set("Analysis complete \u2014 results saved")
        messagebox.showinfo("FreeClimber", "Analysis complete. Results saved to project folder.")

    # ------------------------------------------------------------------
    # Populate Results tab
    # ------------------------------------------------------------------
    def _populate_results(self, result: dict):
        df = result.get('slopes_df') if isinstance(result, dict) else result
        if df is None:
            return

        # --- Summary card ---
        summary_parts = []
        if self.controller.video_path:
            summary_parts.append(os.path.basename(self.controller.video_path))

        quality = result.get('quality') if isinstance(result, dict) else None
        if quality:
            score = quality.get('overall_score', 0)
            dots = self._quality_dots(score)
            level = quality.get('overall_level', '?')
            summary_parts.append(f"Quality: {dots} {level.title()} ({score:.2f})")

        n_vials = len(df)
        summary_parts.append(f"Vials: {n_vials}")

        pop = result.get('population_metrics') if isinstance(result, dict) else None
        if pop:
            if 'mean_speed' in pop:
                summary_parts.append(f"Mean speed: {pop['mean_speed']:.3f}")
            if 'fly_count_per_vial' in pop:
                total_flies = sum(pop['fly_count_per_vial'].values())
                summary_parts.append(f"Flies tracked: ~{total_flies:.0f}")

        self.summary_label.configure(text="  |  ".join(summary_parts), text_color=C["text"])

        # --- Slopes table with quality dots ---
        self.slopes_tree.delete(*self.slopes_tree.get_children())
        cols = list(df.columns) + (['quality'] if quality else [])
        self.slopes_tree["columns"] = cols
        for col in cols:
            self.slopes_tree.heading(col, text=col)
            max_width = max(len(col) * 10, 70)
            self.slopes_tree.column(col, width=min(max_width, 150), minwidth=50)

        per_vial_quality = quality.get('per_vial', {}) if quality else {}
        for idx, row in df.iterrows():
            values = [str(round(v, 4)) if isinstance(v, float) else str(v) for v in row]
            if quality:
                vial_id = str(row.get('vial_ID', idx))
                vq = per_vial_quality.get(vial_id, {})
                values.append(self._quality_dots(vq.get('score', 0)))
            self.slopes_tree.insert("", "end", values=values)

        # --- Per-fly metrics table ---
        per_fly = result.get('per_fly_metrics') if isinstance(result, dict) else None
        self.perfly_tree.delete(*self.perfly_tree.get_children())
        if per_fly is not None and len(per_fly) > 0:
            pcols = list(per_fly.columns)
            self.perfly_tree["columns"] = pcols
            for col in pcols:
                self.perfly_tree.heading(col, text=col)
                self.perfly_tree.column(col, width=min(max(len(col) * 9, 60), 130), minwidth=40)
            for _, row in per_fly.iterrows():
                values = [str(round(v, 3)) if isinstance(v, float) else str(v) for v in row]
                self.perfly_tree.insert("", "end", values=values)

        # --- Population stats text ---
        self._update_pop_stats_text(pop)

        # --- Speed chart ---
        self._plot_speed_chart(df)

        # --- Trajectory plot (using figures.py) ---
        positions = self.controller.get_positions()
        self._plot_trajectory(positions, result)

        # --- Distribution plot ---
        self._plot_distribution(df)

        # --- Per-fly overlay plot ---
        self._plot_perfly_overlay(result)

    def _quality_dots(self, score: float) -> str:
        filled = round(score * 4)
        return '\u25cf' * filled + '\u25cb' * (4 - filled)

    def _plot_speed_chart(self, df: pd.DataFrame):
        self.speed_fig.clear()
        ax = self.speed_fig.add_subplot(111)

        ycol = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                ycol = c
                break
        if ycol is None and len(df.columns) >= 2:
            ycol = df.columns[1]

        if ycol:
            y = pd.to_numeric(df[ycol], errors="coerce").values
            try:
                from output.figures import get_color_palette
                colors = get_color_palette(len(y))
            except ImportError:
                colors = [C["accent"]] * len(y)

            x = list(range(1, len(df) + 1))
            bars = ax.bar(x, y, color=colors[:len(y)], edgecolor='black',
                         linewidth=0.5, alpha=0.85)
            # Value labels on bars
            for bar, val in zip(bars, y):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                            color=C["text_dim"])
            ax.set_xlabel("Vial")
            ax.set_ylabel(ycol)
            ax.set_title("Climbing Speed by Vial", fontsize=11)

        self.speed_canvas.draw()

    def _plot_trajectory(self, positions, result):
        self.traj_fig.clear()
        if positions is None:
            self.traj_canvas.draw()
            return

        ax = self.traj_fig.add_subplot(111)
        params = self.controller.config
        try:
            from output.figures import trajectory_plot
            n_vials = params.get('vials', 1)
            trajectory_plot(
                positions, vials=n_vials,
                convert_to_cm_sec=params.get('convert_to_cm_sec', False),
                frame_rate=params.get('frame_rate', 30),
                pixel_to_cm=params.get('pixel_to_cm', 1.0),
                ax=ax,
            )
        except Exception as e:
            ax.set_title(f"Trajectory plot error: {e}", fontsize=9)
        self.traj_canvas.draw()

    def _plot_distribution(self, df: pd.DataFrame):
        self.dist_fig.clear()
        ax = self.dist_fig.add_subplot(111)

        ycol = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                ycol = c
                break
        if ycol is None:
            self.dist_canvas.draw()
            return

        # Build groups by vial or genotype
        group_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['geno', 'genotype', 'group', 'condition']):
                group_col = c
                break
        if group_col is None:
            for c in df.columns:
                if 'vial' in c.lower():
                    group_col = c
                    break

        if group_col and df[group_col].nunique() >= 2:
            groups = {}
            for name, sub in df.groupby(group_col):
                vals = pd.to_numeric(sub[ycol], errors='coerce').dropna().values
                if len(vals) >= 2:
                    groups[str(name)] = vals

            if len(groups) >= 1:
                try:
                    from output.figures import speed_distribution
                    speed_distribution(groups, ax=ax)
                except Exception:
                    ax.set_title("Distribution: insufficient data")
        else:
            vals = pd.to_numeric(df[ycol], errors='coerce').dropna().values
            if len(vals) >= 2:
                try:
                    from output.figures import speed_distribution
                    speed_distribution({'All': vals}, ax=ax)
                except Exception:
                    pass

        self.dist_canvas.draw()

    def _plot_perfly_overlay(self, result):
        self.perfly_fig.clear()
        ax = self.perfly_fig.add_subplot(111)

        has_tracking = result.get('has_individual_tracking', False) if isinstance(result, dict) else False
        raw_df = result.get('raw_tracking_df') if isinstance(result, dict) else None
        first_frame = result.get('first_frame') if isinstance(result, dict) else None

        if has_tracking and raw_df is not None and 'particle' in raw_df.columns:
            try:
                from output.figures import per_fly_trajectory_overlay
                n_vials = self.controller.config.get('vials', 3)
                per_fly_trajectory_overlay(raw_df, first_frame=first_frame,
                                           vials=n_vials, ax=ax)
            except Exception as e:
                ax.set_title(f"Per-fly overlay error: {e}", fontsize=9)
        else:
            ax.set_title("Per-fly tracking not available", fontsize=10, color=C["text_dim"])
            ax.set_facecolor(C["bg_card"])

        self.perfly_canvas.draw()

    def _update_pop_stats_text(self, pop):
        self.pop_stats_text.configure(state="normal")
        self.pop_stats_text.delete("0.0", "end")
        if pop:
            lines = []
            for key, val in pop.items():
                if isinstance(val, dict):
                    lines.append(f"{key}:")
                    for k, v in val.items():
                        lines.append(f"  Vial {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {val}")
            self.pop_stats_text.insert("0.0", "\n".join(lines))
        else:
            self.pop_stats_text.insert("0.0", "Run analysis to see population metrics.")
        self.pop_stats_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Populate Statistics tab
    # ------------------------------------------------------------------
    def _populate_statistics(self, result: dict):
        df = result.get('slopes_df') if isinstance(result, dict) else result
        if df is None:
            return

        try:
            from analysis.stats import (
                check_normality,
                cohens_d,
                compare_multiple_groups,
                compare_two_groups,
                confidence_interval,
                publication_stats_table,
            )
        except ImportError:
            self._set_stats_text("Statistics module not available.")
            return

        slope_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                slope_col = c
                break
        if slope_col is None:
            self._set_stats_text("No slope/velocity column found in results.")
            return

        group_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['geno', 'genotype', 'group', 'condition', 'treatment']):
                group_col = c
                break

        # Build groups dict for stats and plotting
        groups = {}
        if group_col and df[group_col].nunique() >= 2:
            for name, sub in df.groupby(group_col):
                vals = pd.to_numeric(sub[slope_col], errors='coerce').dropna().values
                if len(vals) > 0:
                    groups[str(name)] = vals
        else:
            # Use vial as grouping for the plot
            vial_col = None
            for c in df.columns:
                if 'vial' in c.lower():
                    vial_col = c
                    break
            if vial_col and df[vial_col].nunique() >= 2:
                for name, sub in df.groupby(vial_col):
                    vals = pd.to_numeric(sub[slope_col], errors='coerce').dropna().values
                    if len(vals) > 0:
                        groups[f"Vial {name}"] = vals

        # --- Box + swarm plot ---
        self.stats_fig.clear()
        ax = self.stats_fig.add_subplot(111)
        if len(groups) >= 2:
            try:
                from output.figures import box_swarm_plot
                box_swarm_plot(groups, ylabel=slope_col, title="Distribution by Group", ax=ax)
            except Exception:
                # Fallback: basic boxplot
                ax.boxplot(list(groups.values()), tick_labels=list(groups.keys()))
                ax.set_ylabel(slope_col)
        elif len(groups) == 1:
            name, vals = next(iter(groups.items()))
            ax.hist(vals, bins='auto', color=C["accent"], alpha=0.7, edgecolor='black')
            ax.set_xlabel(slope_col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution: {name}")
        self.stats_canvas.draw()

        # --- Formatted text ---
        lines = ["\u2550" * 50, "  STATISTICAL ANALYSIS", "\u2550" * 50, ""]

        if group_col and df[group_col].nunique() >= 2 and len(groups) >= 2:
            norm = check_normality(groups)
            lines.append("\u25b6 NORMALITY (Shapiro-Wilk)")
            lines.append("\u2500" * 40)
            for gname, res in norm['results'].items():
                check = "\u2713" if res['normal'] else "\u2717"
                lines.append(f"  {check} {gname}: W={res['statistic']:.4f}, p={res['p_value']:.4f}")
            lines.append(f"  All normal: {'Yes' if norm['all_normal'] else 'No'}")
            lines.append("")

            if len(groups) == 2:
                keys = list(groups.keys())
                stat_result = compare_two_groups(groups[keys[0]], groups[keys[1]])
                lines.append("\u25b6 TWO-GROUP COMPARISON")
                lines.append("\u2500" * 40)
                lines.append(f"  Test: {stat_result['test']}")
                lines.append(f"  Statistic: {stat_result['statistic']:.4f}")
                sig = stat_result['significance']
                lines.append(f"  p-value: {stat_result['p_value']:.6f}  {sig}")
                lines.append(f"  Cohen's d: {stat_result['effect_size_d']:.3f}")
                lines.append("")
            elif len(groups) >= 3:
                stat_result = compare_multiple_groups(groups)
                lines.append(f"\u25b6 MULTI-GROUP COMPARISON ({len(groups)} groups)")
                lines.append("\u2500" * 40)
                lines.append(f"  Test: {stat_result['test']}")
                lines.append(f"  Statistic: {stat_result['statistic']:.4f}")
                lines.append(f"  p-value: {stat_result['p_value']:.6f}")
                lines.append(f"  Effect size ({stat_result['effect_size_name']}): {stat_result['effect_size']:.4f}")
                lines.append(f"  Significant: {'Yes' if stat_result['significant'] else 'No'}")
                lines.append("")

                if stat_result.get('post_hoc'):
                    lines.append(f"\u25b6 POST-HOC: {stat_result['post_hoc_method']}")
                    lines.append("\u2500" * 40)
                    # Aligned table
                    lines.append(f"  {'Group 1':<12} {'Group 2':<12} {'p-value':<10} {'Sig':<5} {'d':<8}")
                    sep = "\u2500"
                    lines.append(f"  {sep*12} {sep*12} {sep*10} {sep*5} {sep*8}")
                    for comp in stat_result['post_hoc']:
                        lines.append(
                            f"  {comp['group1']:<12} {comp['group2']:<12} "
                            f"{comp['p_value']:<10.6f} {comp['significance']:<5} "
                            f"{comp['effect_size_d']:<8.3f}"
                        )
                    lines.append("")

            lines.append("\u25b6 GROUP SUMMARIES")
            lines.append("\u2500" * 40)
            for gname, vals in groups.items():
                ci = confidence_interval(vals)
                sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                lines.append(
                    f"  {gname}: n={len(vals)}, mean={np.mean(vals):.4f} "
                    f"\u00b1 {sem:.4f} (SEM), 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                )
        else:
            vals = pd.to_numeric(df[slope_col], errors='coerce').dropna().values
            if len(vals) > 0:
                ci = confidence_interval(vals)
                lines.append("\u25b6 DESCRIPTIVE STATISTICS")
                lines.append("\u2500" * 40)
                lines.append(f"  n = {len(vals)}")
                lines.append(f"  mean = {np.mean(vals):.4f}")
                lines.append(f"  std = {np.std(vals, ddof=1):.4f}")
                lines.append(f"  SEM = {np.std(vals, ddof=1) / np.sqrt(len(vals)):.4f}")
                lines.append(f"  95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
                lines.append(f"  min = {np.min(vals):.4f}")
                lines.append(f"  max = {np.max(vals):.4f}")

        lines.append("")
        lines.append("\u2550" * 50)
        self._set_stats_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _set_results_text(self, text: str):
        self.summary_label.configure(text=text)

    def _set_stats_text(self, text: str):
        self.stats_text.configure(state="normal")
        self.stats_text.delete("0.0", "end")
        self.stats_text.insert("0.0", text)
        self.stats_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Config save/load
    # ------------------------------------------------------------------
    def _save_config(self):
        if not self.controller.video_path:
            messagebox.showinfo("FreeClimber", "Load a video first to save its configuration.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".cfg",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")],
            initialdir=self.project_entry.get() or None,
        )
        if path:
            params = self._collect_params()
            self.controller.save_config(path, params)
            self.status_var.set(f"Config saved: {os.path.basename(path)}")

    def _load_config(self):
        path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
        )
        if path:
            try:
                params = self.controller.load_config(path)
                self._apply_params_to_gui(params)
                self.status_var.set(f"Config loaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load config:\n\n{e}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_dialog(self):
        if self.controller.get_slopes() is None:
            messagebox.showinfo("FreeClimber", "Run analysis first to have results to export.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Export Results")
        win.geometry("420x320")
        win.transient(self)
        win.configure(fg_color=C["bg"])

        ctk.CTkLabel(
            win, text="Export Results",
            font=(FONT_FAMILY, 18, "bold"), text_color=C["text"],
        ).pack(pady=(S["lg"], S["md"]))

        # Check Excel availability
        try:
            import openpyxl  # noqa: F401
            has_excel = True
        except ImportError:
            has_excel = False

        formats = [
            ("Slopes CSV (backward compatible)", "csv"),
            ("Tidy CSV (R-ready)", "tidy"),
            ("GraphPad Prism CSV", "prism"),
            ("Excel Workbook (.xlsx)" if has_excel else "Excel (install openpyxl)", "excel"),
            ("Per-fly Tracks CSV", "tracks"),
        ]

        fmt_var = ctk.StringVar(value="csv")
        for label, val in formats:
            ctk.CTkRadioButton(
                win, text=label, variable=fmt_var, value=val,
                font=(FONT_FAMILY, 13), text_color=C["text"],
                fg_color=C["accent"], hover_color=C["accent_hover"],
                border_color=C["border"],
            ).pack(anchor="w", padx=S["xl"], pady=3)

        def do_export():
            fmt = fmt_var.get()
            if fmt == 'excel' and not has_excel:
                messagebox.showinfo(
                    "Missing Dependency",
                    "Install openpyxl for Excel export:\n\npip install openpyxl",
                    parent=win,
                )
                return
            if fmt == 'excel':
                ext = '.xlsx'
                ftypes = [("Excel files", "*.xlsx"), ("All files", "*.*")]
            else:
                ext = '.csv'
                ftypes = [("CSV files", "*.csv"), ("All files", "*.*")]
            path = filedialog.asksaveasfilename(
                parent=win, title="Save As",
                defaultextension=ext, filetypes=ftypes,
            )
            if path:
                try:
                    self.controller.export_results(fmt, path)
                    messagebox.showinfo("Exported", f"Results saved to:\n{path}", parent=win)
                    win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Export failed:\n\n{e}", parent=win)

        ctk.CTkButton(
            win, text="Export", command=do_export,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=(FONT_FAMILY, 14, "bold"),
            corner_radius=8, width=120, height=36,
        ).pack(pady=S["lg"])

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------
    def _show_about(self):
        messagebox.showinfo("About FreeClimber",
                            "FreeClimber v2.0\n\n"
                            "Drosophila RING Assay Analysis\n"
                            "Climbing velocity from video\n\n"
                            "Based on Spierer et al. (2021) J Exp Biol\n"
                            "doi: 10.1242/jeb.229377\n\n"
                            "Upgraded by Jason Moorthy, Wayne State University")

    # ------------------------------------------------------------------
    # Config profiles
    # ------------------------------------------------------------------
    def _on_profile_selected(self, name):
        if name == "(none)":
            return
        try:
            params = self.controller.load_profile(name)
            self._apply_params_to_gui(params)
            self.status_var.set(f"Profile loaded: {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load profile:\n\n{e}")

    def _save_profile(self):
        name = ctk.CTkInputDialog(text="Profile name:", title="Save Profile").get_input()
        if not name:
            return
        params = self._collect_params()
        self.controller.save_profile(name, params)
        self._refresh_profiles()
        self.status_var.set(f"Profile saved: {name}")

    def _delete_profile(self):
        name = self.profile_var.get()
        if name and name != "(none)":
            self.controller.delete_profile(name)
            self._refresh_profiles()
            self.status_var.set(f"Profile deleted: {name}")

    def _refresh_profiles(self):
        profiles = self.controller.list_profiles() or ["(none)"]
        self.profile_menu.configure(values=profiles)
        if profiles:
            self.profile_var.set(profiles[0])

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    def _batch_mode(self):
        paths = filedialog.askopenfilenames(
            title="Select Videos for Batch Processing",
            filetypes=[
                ("Video files", "*.h264 *.mp4 *.avi *.mov"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        params = self._collect_params()
        self.run_btn.configure(state="disabled")
        self.status_var.set(f"Batch: 0/{len(paths)} videos...")
        self.progress_bar.set(0)

        def progress(i, total, path, status):
            self.after(0, lambda: self._batch_progress(i, total, path, status))

        def worker():
            try:
                combined = self.controller.run_batch(list(paths), params, progress_callback=progress)
                self.after(0, lambda c=combined: self._batch_done(c))
            except Exception as ex:
                self.after(0, lambda msg=str(ex): self._batch_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _batch_progress(self, i, total, path, status):
        self.progress_bar.set(i / total)
        self.progress_label.configure(text=f"Video {i}/{total} \u2014 {os.path.basename(path)}")
        self.status_var.set(f"Batch: {i}/{total} \u2014 {status}")

    def _batch_done(self, combined):
        self.run_btn.configure(state="normal")
        self.progress_bar.set(1)
        if combined is not None and len(combined) > 0:
            result = {'slopes_df': combined}
            self._populate_results(result)
            self._populate_statistics(result)
            self.tabview.set("Results")
            self.status_var.set(f"Batch complete: {len(combined)} rows")
        else:
            self.status_var.set("Batch complete: no results")

    def _batch_error(self, msg):
        self.run_btn.configure(state="normal")
        messagebox.showerror("Batch Error", msg)

    # ------------------------------------------------------------------
    # Methods paragraph
    # ------------------------------------------------------------------
    def _copy_methods(self):
        from output.reports import generate_methods_paragraph
        params = self._collect_params()
        text = generate_methods_paragraph(params)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Methods paragraph copied to clipboard")

    # ------------------------------------------------------------------
    # Log viewer (hidden by default, toggled from View menu)
    # ------------------------------------------------------------------
    def _build_log_viewer(self):
        self.log_frame = ctk.CTkFrame(self, height=120, fg_color=C["bg_card"], corner_radius=8)
        self.log_text = ctk.CTkTextbox(
            self.log_frame, height=100, font=("Courier", 10),
            state="disabled", wrap="word",
            fg_color=C["bg_card"], text_color=C["text_dim"],
        )
        self.log_text.pack(fill="both", expand=True, padx=S["xs"], pady=S["xs"])

        handler = _TextboxHandler(self.log_text, self)
        logging.getLogger().addHandler(handler)
        handler.setLevel(logging.INFO)

    def _toggle_log_viewer(self):
        if self._log_visible:
            self.log_frame.pack_forget()
            self._log_visible = False
        else:
            self.log_frame.pack(side="bottom", fill="x", padx=S["sm"], pady=(0, S["xs"]),
                                before=self.status_bar)
            self._log_visible = True

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------
    def _setup_drag_drop(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self._on_drop)
        except ImportError:
            pass

    def _on_drop(self, event):
        path = event.data.strip('{}')
        if os.path.isfile(path):
            self._open_video(path)

    # ------------------------------------------------------------------
    # First-run welcome overlay (single overlay instead of 3 modals)
    # ------------------------------------------------------------------
    def _check_first_run(self):
        marker = os.path.expanduser('~/.freeclimber/first_run')
        if os.path.exists(marker):
            return
        os.makedirs(os.path.dirname(marker), exist_ok=True)

        overlay = ctk.CTkFrame(self, fg_color=C["bg"])
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()

        inner = ctk.CTkFrame(overlay, fg_color=C["bg_card"], corner_radius=16, width=520, height=400)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        inner.pack_propagate(False)

        ctk.CTkLabel(
            inner, text="Welcome to FreeClimber",
            font=(FONT_FAMILY, 22, "bold"), text_color=C["text"],
        ).pack(pady=(S["xl"], S["md"]))

        ctk.CTkLabel(
            inner, text="Drosophila RING Assay Analysis",
            font=(FONT_FAMILY, 13), text_color=C["accent"],
        ).pack(pady=(0, S["lg"]))

        steps = [
            ("\u2460", "Load a Video", "Open or drag a video file onto the window"),
            ("\u2461", "Adjust Parameters", "Tweak detection settings, then click Preview"),
            ("\u2462", "Run Analysis", "Hit RUN ANALYSIS to process and view results"),
        ]

        for num, title, desc in steps:
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", padx=S["xl"], pady=S["sm"])
            ctk.CTkLabel(
                row, text=num, font=(FONT_FAMILY, 20),
                text_color=C["accent"], width=36,
            ).pack(side="left")
            text_frame = ctk.CTkFrame(row, fg_color="transparent")
            text_frame.pack(side="left", fill="x", padx=(S["sm"], 0))
            ctk.CTkLabel(
                text_frame, text=title,
                font=(FONT_FAMILY, 14, "bold"), text_color=C["text"], anchor="w",
            ).pack(anchor="w")
            ctk.CTkLabel(
                text_frame, text=desc,
                font=(FONT_FAMILY, 12), text_color=C["text_dim"], anchor="w",
            ).pack(anchor="w")

        def dismiss():
            overlay.destroy()
            with open(marker, 'w') as f:
                f.write('done')

        ctk.CTkButton(
            inner, text="Get Started", width=160, height=40,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=(FONT_FAMILY, 14, "bold"),
            corner_radius=10, command=dismiss,
        ).pack(pady=(S["lg"], S["xl"]))


class _TextboxHandler(logging.Handler):
    """Logging handler that writes to a CTkTextbox."""
    def __init__(self, textbox, app):
        super().__init__()
        self.textbox = textbox
        self.app = app

    def emit(self, record):
        msg = self.format(record) + '\n'
        self.app.after(0, self._append, msg)

    def _append(self, msg):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", msg)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")


def main():
    app = FreeClimberApp()
    app._build_log_viewer()
    app._setup_drag_drop()
    app._check_first_run()
    app.mainloop()


if __name__ == "__main__":
    main()
