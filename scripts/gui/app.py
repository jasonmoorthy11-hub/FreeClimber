"""FreeClimber v4.0 — customtkinter GUI.

Professional desktop app replacing the legacy wxPython interface.
Designed for non-technical lab members: sliders, tooltips, drag-and-drop,
dark mode, progressive disclosure.
"""

import logging
import os
import re
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

from PIL import Image

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('trackpy').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Bootstrap sys.path so bare imports work from any launch location
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _asset_path(filename):
    base = getattr(sys, '_MEIPASS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    return os.path.join(base, 'scripts', 'gui', 'assets', filename)

import customtkinter as ctk
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gui.controller import AnalysisController
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C = {
    # 5-tier dark theme elevation system
    "bg":           "#0a0a14",   # Level 0: Base (window background)
    "bg_panel":     "#0f0f1a",   # Level 1: Surface (main content)
    "bg_card":      "#161625",   # Level 2: Raised (cards, sidebar, panels)
    "bg_hover":     "#1e1e30",   # Level 3: Overlay (dropdowns, tooltips, modals)
    "bg_elevated":  "#28283d",   # Level 4: Elevated (hover states, active items)
    "bg_input":     "#12121f",
    "border":       "#ffffff12",
    "border_subtle":"#ffffff08",
    # Text hierarchy
    "text":         "#f0f0f5",   # Primary
    "text_dim":     "#a0a0b8",   # Secondary
    "text_disabled":"#6b6b85",   # Disabled
    # Accents
    "accent":       "#53a8b6",
    "accent_hover": "#6bc4d0",
    "accent_muted": "#2a5a64",
    "danger":       "#e94560",
    "danger_hover": "#ff5a75",
    "success":      "#48bb78",
    "warning":      "#ecc94b",
    "run":          "#4CAF50",
    "run_hover":    "#5CC462",
    "text_on_accent": "#ffffff",
}

S = {"xxs": 2, "xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32, "xxxl": 48}

FONT_FAMILY = "Helvetica"
MONO_FAMILY = "SF Mono"

F = {
    "h1":      (FONT_FAMILY, 24, "bold"),   # Page titles
    "h2":      (FONT_FAMILY, 18, "bold"),   # Section headings
    "h3":      (FONT_FAMILY, 15, "bold"),   # Subsection
    "body":    (FONT_FAMILY, 13),           # Data-dense body
    "body_b":  (FONT_FAMILY, 13, "bold"),
    "caption": (FONT_FAMILY, 11),           # Labels
    "mono":    (MONO_FAMILY, 12),           # Data/code
}

R = {"btn": 6, "card": 8, "modal": 12, "pill": 9999}

PLOT_STYLE = {
    'figure.facecolor': C["bg_panel"],
    'axes.facecolor':   C["bg_card"],
    'axes.edgecolor':   '#2a2a3e',
    'axes.labelcolor':  '#E0E0E0',
    'text.color':       '#E0E0E0',
    'xtick.color':      '#8a8a9a',
    'ytick.color':      '#8a8a9a',
    'grid.color':       C["bg_hover"],
    'grid.alpha':       0.4,
    'axes.grid':        True,
    'grid.linestyle':   '--',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.facecolor': C["bg_card"],
    'legend.edgecolor': '#2a2a3e',
    'legend.fontsize':  10,
}
plt.rcParams.update(PLOT_STYLE)

EXPORT_STYLE = {
    'figure.facecolor': '#ffffff',
    'axes.facecolor':   '#ffffff',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#000000',
    'text.color':       '#000000',
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'grid.color':       '#cccccc',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'font.family':      'sans-serif',
    'font.size':        7,
}

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
    'individual_tracking': "Per-fly tracking with trackpy (always enabled).",
}

# Workflow step definitions
STEPS = [
    ("load",       "\u2460", "LOAD VIDEO"),
    ("roi",        "\u2461", "REGION OF INTEREST"),
    ("detection",  "\u2462", "DETECTION"),
    ("analyze",    "\u2463", "ANALYZE"),
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
        tw.configure(bg=C["bg_card"])
        frame = ctk.CTkFrame(tw, fg_color=C["bg_card"],
            border_color=C["accent_muted"], border_width=1, corner_radius=6)
        frame.pack(fill="both", expand=True, padx=1, pady=1)
        ctk.CTkLabel(frame, text=self.text, justify="left",
            text_color=C["text"], font=F["caption"],
            wraplength=300).pack(padx=S["sm"], pady=S["xs"])

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
        super().__init__(parent, fg_color=C["bg_card"], corner_radius=10, **kwargs)

        self._open = initially_open

        # Header
        self.header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        self.header.pack(fill="x", padx=S["sm"], pady=(S["sm"], 0))

        self.chevron_label = ctk.CTkLabel(
            self.header, text="\u25BC" if self._open else "\u25B6",
            font=(FONT_FAMILY, 14), text_color=C["accent"], width=18, anchor="w",
        )
        self.chevron_label.pack(side="left")

        icon_badge = ctk.CTkFrame(self.header, width=22, height=22,
            fg_color=C["accent_muted"], corner_radius=6)
        icon_badge.pack(side="left", padx=(0, S["sm"]))
        icon_badge.pack_propagate(False)
        ctk.CTkLabel(icon_badge, text=icon, font=(FONT_FAMILY, 10, "bold"),
            text_color=C["accent"], anchor="center").pack(expand=True)

        title_label = ctk.CTkLabel(
            self.header, text=title,
            font=(FONT_FAMILY, 12, "bold"), text_color=C["text"], anchor="w",
        )
        title_label.pack(side="left", fill="x", expand=True)

        # Make entire header clickable
        for w in (self.header, self.chevron_label, icon_badge, title_label):
            w.bind("<Button-1>", self._toggle)

        # Body container
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        if self._open:
            self.body.pack(fill="x", padx=S["sm"], pady=(S["xs"], S["sm"]))

    def _toggle(self, event=None):
        if hasattr(self, '_enabled') and not self._enabled:
            return
        self._open = not self._open
        self.chevron_label.configure(text="\u25BC" if self._open else "\u25B6")
        if self._open:
            self.body.pack(fill="x", padx=S["sm"], pady=(S["xs"], S["sm"]))
        else:
            self.body.pack_forget()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        color = C["text"] if enabled else C["text_disabled"]
        opacity_fg = C["bg_card"] if enabled else C["bg_hover"]
        self.configure(fg_color=opacity_fg)
        for w in self.header.winfo_children():
            if hasattr(w, 'configure'):
                try:
                    w.configure(text_color=color)
                except Exception:
                    pass
        if not enabled and self._open:
            self._toggle()


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

    def set_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.slider.configure(state=state)
        self.entry.configure(state=state)
        color = C["text"] if enabled else C["text_disabled"]
        self.label.configure(text_color=color)


# ---------------------------------------------------------------------------
# PlotToolbar
# ---------------------------------------------------------------------------
class PlotToolbar(ctk.CTkFrame):
    """Minimal dark-theme plot toolbar replacing NavigationToolbar2Tk."""

    def __init__(self, parent, canvas, fig, save_callback=None):
        super().__init__(parent, fg_color=C["bg_card"], height=32, corner_radius=0)
        self.pack_propagate(False)
        self.canvas = canvas

        self._nav = NavigationToolbar2Tk(canvas, self)
        self._nav.pack_forget()

        ctk.CTkFrame(self, height=1, fg_color=C["border"]).pack(fill="x", side="top")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=S["sm"], pady=S["xs"])

        self._zoom_btn = self._make_btn(btn_frame, "\u2922", "Zoom", self._toggle_zoom)
        self._pan_btn = self._make_btn(btn_frame, "\u2725", "Pan", self._toggle_pan)
        self._make_btn(btn_frame, "\u21BA", "Reset view", lambda: (self._nav.home(), canvas.draw()))
        self._make_btn(btn_frame, "\u2913", "Save figure", save_callback or (lambda: None))
        self._active_mode = None

    def _make_btn(self, parent, symbol, tip, command):
        btn = ctk.CTkButton(parent, text=symbol, width=32, height=28,
            fg_color="transparent", hover_color=C["bg_hover"],
            text_color=C["text_dim"], font=("Menlo", 14),
            corner_radius=6, command=command)
        btn.pack(side="left", padx=2)
        ToolTip(btn, tip)
        return btn

    def _toggle_zoom(self):
        if self._active_mode == 'zoom':
            self._nav.zoom()
            self._zoom_btn.configure(fg_color="transparent")
            self._active_mode = None
        else:
            if self._active_mode == 'pan':
                self._nav.pan()
                self._pan_btn.configure(fg_color="transparent")
            self._nav.zoom()
            self._zoom_btn.configure(fg_color=C["accent_muted"])
            self._active_mode = 'zoom'

    def _toggle_pan(self):
        if self._active_mode == 'pan':
            self._nav.pan()
            self._pan_btn.configure(fg_color="transparent")
            self._active_mode = None
        else:
            if self._active_mode == 'zoom':
                self._nav.zoom()
                self._zoom_btn.configure(fg_color="transparent")
            self._nav.pan()
            self._pan_btn.configure(fg_color=C["accent_muted"])
            self._active_mode = 'pan'


# ---------------------------------------------------------------------------
# FreeClimberApp
# ---------------------------------------------------------------------------
class FreeClimberApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("FreeClimber v4.0")
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
        self._analysis_lock = threading.Lock()

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

        self._toast_stack: list[ctk.CTkFrame] = []

    # ------------------------------------------------------------------
    # Toast notifications
    # ------------------------------------------------------------------
    def _toast(self, message: str, level: str = "info", duration: int = 4000):
        """Show a brief toast notification in the bottom-right corner.

        level: 'success', 'error', 'info'
        """
        border_colors = {"success": C["success"], "error": C["danger"], "info": C["accent"]}
        border_color = border_colors.get(level, C["accent"])

        toast = ctk.CTkFrame(self, fg_color=C["bg_hover"], corner_radius=8,
                              border_width=0, width=320, height=40)
        toast.pack_propagate(False)

        inner = ctk.CTkFrame(toast, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=0, pady=0)

        accent_bar = ctk.CTkFrame(inner, fg_color=border_color, width=3,
                                   corner_radius=0)
        accent_bar.pack(side="left", fill="y")

        ctk.CTkLabel(inner, text=message, font=(FONT_FAMILY, 11),
                      text_color=C["text"], anchor="w", wraplength=280,
                      ).pack(side="left", padx=S["sm"], pady=S["xs"], fill="x", expand=True)

        close_btn = ctk.CTkLabel(inner, text="\u2715", font=(FONT_FAMILY, 11),
                                  text_color=C["text_dim"], cursor="hand2", width=20)
        close_btn.pack(side="right", padx=S["xs"])
        close_btn.bind("<Button-1>", lambda e, t=toast: self._dismiss_toast(t))

        self._toast_stack.append(toast)
        self._reposition_toasts()

        if level != "error":
            toast.after(duration, lambda: self._dismiss_toast(toast))

    def _dismiss_toast(self, toast):
        if toast in self._toast_stack:
            self._toast_stack.remove(toast)
            toast.place_forget()
            toast.destroy()
            self._reposition_toasts()

    def _reposition_toasts(self):
        y_offset = 50
        for t in reversed(self._toast_stack):
            t.place(relx=1.0, rely=1.0, x=-16, y=-y_offset, anchor="se")
            y_offset += 48

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
        file_menu.add_command(label="Save Current Figure\u2026  \u21e7\u2318S", command=self._save_current_figure)
        file_menu.add_separator()
        file_menu.add_command(label="Load Results\u2026  \u2318L", command=self._load_results)
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
        self.bind("<Command-Shift-S>", lambda e: self._save_current_figure())
        self.bind("<Command-l>", lambda e: self._load_results())
        self.bind("<Escape>", lambda e: self._cancel_analysis())

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
        self._preview_done = False

        sidebar_outer = ctk.CTkFrame(
            self.main_frame, width=340, fg_color=C["bg_panel"], corner_radius=0,
        )
        sidebar_outer.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)
        sidebar_outer.grid_propagate(False)

        # Logo + App title area
        try:
            _logo_img = Image.open(_asset_path('logo_64.png'))
            self._sidebar_logo = ctk.CTkImage(light_image=_logo_img, dark_image=_logo_img, size=(36, 36))
        except Exception:
            self._sidebar_logo = None

        title_frame = ctk.CTkFrame(sidebar_outer, fg_color="transparent")
        title_frame.pack(fill="x", padx=S["md"], pady=(S["lg"], S["sm"]))

        if self._sidebar_logo:
            ctk.CTkLabel(title_frame, image=self._sidebar_logo, text="").pack(side="left", padx=(0, S["sm"]))
        ctk.CTkLabel(
            title_frame, text="FreeClimber",
            font=F["h1"], text_color=C["text"], anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            title_frame, text="v3.0",
            font=F["caption"], text_color=C["accent"], anchor="w",
        ).pack(side="left", padx=(S["sm"], 0), pady=(4, 0))

        ctk.CTkLabel(
            sidebar_outer, text="Drosophila RING Assay Analysis",
            font=F["caption"], text_color=C["text_dim"], anchor="w",
        ).pack(fill="x", padx=S["md"], pady=(0, S["md"]))

        # Thin accent line
        ctk.CTkFrame(sidebar_outer, height=1, fg_color=C["accent"]).pack(
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
        sidebar._parent_canvas.configure(yscrollincrement=8)
        sidebar.after(200, lambda: sidebar._parent_canvas.yview_moveto(0))

        # ============================================================
        # STEP 1: LOAD VIDEO (always active)
        # ============================================================
        self.step1_card = CollapsibleCard(sidebar, icon="\u2460", title="LOAD VIDEO")
        self.step1_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        btn_frame = ctk.CTkFrame(self.step1_card.body, fg_color="transparent")
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
            self.step1_card.body, text="No video selected",
            text_color=C["text_dim"], font=(FONT_FAMILY, 11),
            wraplength=280, anchor="w",
        )
        self.video_label.pack(anchor="w", pady=(S["xs"], 0))

        # ============================================================
        # STEP 2: REGION OF INTEREST (enabled after video)
        # ============================================================
        self.step2_card = CollapsibleCard(sidebar, icon="\u2461", title="REGION OF INTEREST")
        self.step2_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))
        self.step2_card.set_enabled(False)

        roi_grid = ctk.CTkFrame(self.step2_card.body, fg_color="transparent")
        roi_grid.pack(fill="x", pady=(0, S["xs"]))
        self.roi_x = self._labeled_entry(roi_grid, "X", "0", row=0, col=0)
        self.roi_y = self._labeled_entry(roi_grid, "Y", "0", row=0, col=2)
        self.roi_w = self._labeled_entry(roi_grid, "W", "0", row=1, col=0)
        self.roi_h = self._labeled_entry(roi_grid, "H", "0", row=1, col=2)

        self.fixed_roi_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.step2_card.body, text="Fixed ROI dimensions", variable=self.fixed_roi_var,
            font=(FONT_FAMILY, 12), text_color=C["text"],
            fg_color=C["accent"], hover_color=C["accent_hover"],
            border_color=C["border"],
        ).pack(anchor="w", pady=(S["xs"], 0))

        # ============================================================
        # STEP 3: DETECTION (enabled after ROI)
        # ============================================================
        self.step3_card = CollapsibleCard(sidebar, icon="\u2462", title="DETECTION")
        self.step3_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))
        self.step3_card.set_enabled(False)

        self.sl_diameter = ParameterSlider(self.step3_card.body, "Diameter", 3, 31, 7, int, TOOLTIPS['diameter'])
        self.sl_diameter.pack(fill="x", pady=S["xs"])
        self.sl_minmass = ParameterSlider(self.step3_card.body, "Min Mass", 0, 5000, 100, int, TOOLTIPS['minmass'])
        self.sl_minmass.pack(fill="x", pady=S["xs"])
        self.sl_maxsize = ParameterSlider(self.step3_card.body, "Max Size", 1, 50, 11, int, TOOLTIPS['maxsize'])
        self.sl_maxsize.pack(fill="x", pady=S["xs"])

        th_frame = ctk.CTkFrame(self.step3_card.body, fg_color="transparent")
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

        self.sl_ecc_low = ParameterSlider(self.step3_card.body, "Ecc Low", 0, 1, 0, float, TOOLTIPS['ecc_low'])
        self.sl_ecc_low.pack(fill="x", pady=S["xs"])
        self.sl_ecc_high = ParameterSlider(self.step3_card.body, "Ecc High", 0, 1, 1, float, TOOLTIPS['ecc_high'])
        self.sl_ecc_high.pack(fill="x", pady=S["xs"])

        # Preview button inside Detection step
        ctk.CTkButton(
            self.step3_card.body, text="PREVIEW", width=280,
            fg_color="transparent", hover_color=C["accent_muted"],
            text_color=C["accent"], border_color=C["accent"], border_width=1,
            corner_radius=8, font=F["body_b"], command=self._test_parameters,
        ).pack(fill="x", pady=(S["sm"], S["xs"]))

        # ============================================================
        # STEP 4: ANALYZE (enabled after preview)
        # ============================================================
        self.step4_card = CollapsibleCard(sidebar, icon="\u2463", title="ANALYZE")
        self.step4_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))
        self.step4_card.set_enabled(False)

        # Vials + Groups
        vial_frame = ctk.CTkFrame(self.step4_card.body, fg_color="transparent")
        vial_frame.pack(fill="x", pady=S["xs"])
        self.sl_vials = ParameterSlider(vial_frame, "Vials", 1, 20, 3, int, TOOLTIPS['vials'])
        self.sl_vials.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            vial_frame, text="Groups", width=60,
            fg_color=C["bg_hover"], hover_color=C["accent_muted"],
            text_color=C["accent"], corner_radius=6,
            font=(FONT_FAMILY, 11), command=self._show_vial_grouping_dialog,
        ).pack(side="right", padx=(S["xs"], 0))

        self.sl_window = ParameterSlider(self.step4_card.body, "Window", 5, 500, 50, int, TOOLTIPS['window'])
        self.sl_window.pack(fill="x", pady=S["xs"])

        # Frame rate
        fr_frame = ctk.CTkFrame(self.step4_card.body, fg_color="transparent")
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

        # Crop range (essential for every analysis)
        self.sl_crop_0 = ParameterSlider(self.step4_card.body, "Crop Start", 0, 10000, 0, int, TOOLTIPS['crop_0'])
        self.sl_crop_0.pack(fill="x", pady=S["xs"])
        self.sl_crop_n = ParameterSlider(self.step4_card.body, "Crop End", 0, 10000, 145, int, TOOLTIPS['crop_n'])
        self.sl_crop_n.pack(fill="x", pady=S["xs"])

        # RUN button — green (semantic: go/execute)
        run_wrapper = ctk.CTkFrame(self.step4_card.body, fg_color=C["run"], corner_radius=10)
        run_wrapper.pack(fill="x", pady=(S["sm"], S["xs"]))
        ctk.CTkFrame(
            run_wrapper, height=2, fg_color=C["run_hover"], corner_radius=0,
        ).pack(fill="x")
        self.run_btn = ctk.CTkButton(
            run_wrapper, text="RUN ANALYSIS", height=44,
            font=F["h2"],
            fg_color=C["run"], hover_color=C["run_hover"],
            text_color=C["text_on_accent"], corner_radius=8,
            command=self._run_analysis,
        )
        self.run_btn.pack(fill="x", padx=2, pady=(0, 2))

        # Secondary action buttons
        btn_row = ctk.CTkFrame(self.step4_card.body, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, S["xs"]))
        ctk.CTkButton(
            btn_row, text="Batch Mode", width=90,
            fg_color="transparent", hover_color=C["accent_muted"],
            text_color=C["accent"], border_color=C["accent"], border_width=1,
            corner_radius=8, font=(FONT_FAMILY, 11), command=self._batch_mode,
        ).pack(side="left", padx=(0, S["xs"]))
        ctk.CTkButton(
            btn_row, text="Save Config", width=85,
            fg_color=C["bg_hover"], hover_color=C["border"],
            text_color=C["text"], corner_radius=8,
            font=(FONT_FAMILY, 11), command=self._save_config,
        ).pack(side="left", padx=(0, S["xs"]))
        ctk.CTkButton(
            btn_row, text="Copy Methods", width=90,
            fg_color=C["bg_hover"], hover_color=C["border"],
            text_color=C["text"], corner_radius=8,
            font=(FONT_FAMILY, 11), command=self._copy_methods,
        ).pack(side="left")

        # ============================================================
        # ADVANCED SETTINGS (collapsed)
        # ============================================================
        adv_card = CollapsibleCard(sidebar, icon="\u2699", title="Advanced Settings", initially_open=False)
        adv_card.pack(fill="x", padx=S["sm"], pady=(0, S["sm"]))

        # Background method
        bg_frame = ctk.CTkFrame(adv_card.body, fg_color="transparent")
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

        # Pixel/cm
        px_frame = ctk.CTkFrame(adv_card.body, fg_color="transparent")
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

        # Checkboxes
        check_frame = ctk.CTkFrame(adv_card.body, fg_color="transparent")
        check_frame.pack(fill="x", pady=(S["xs"], 0))
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
        self.tracking_var = ctk.BooleanVar(value=True)

        # Blank frames
        self.sl_blank_0 = ParameterSlider(adv_card.body, "Blank Start", 0, 10000, 0, int, TOOLTIPS['blank_0'])
        self.sl_blank_0.pack(fill="x", pady=S["xs"])
        self.sl_blank_n = ParameterSlider(adv_card.body, "Blank End", 0, 10000, 145, int, TOOLTIPS['blank_n'])
        self.sl_blank_n.pack(fill="x", pady=S["xs"])
        self.sl_check = ParameterSlider(adv_card.body, "Check Frame", 0, 10000, 0, int, TOOLTIPS['check_frame'])
        self.sl_check.pack(fill="x", pady=S["xs"])

        # Outlier trim
        self.sl_outlier_tb = ParameterSlider(adv_card.body, "Outlier TB", 0, 20, 1, float, TOOLTIPS['outlier_TB'])
        self.sl_outlier_tb.pack(fill="x", pady=S["xs"])
        self.sl_outlier_lr = ParameterSlider(adv_card.body, "Outlier LR", 0, 20, 3, float, TOOLTIPS['outlier_LR'])
        self.sl_outlier_lr.pack(fill="x", pady=S["xs"])

        # Naming
        nc_frame = ctk.CTkFrame(adv_card.body, fg_color="transparent")
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

        vid_frame = ctk.CTkFrame(adv_card.body, fg_color="transparent")
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

        proj_frame = ctk.CTkFrame(adv_card.body, fg_color="transparent")
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

        # ============================================================
        # PROFILES (collapsed)
        # ============================================================
        prof_card = CollapsibleCard(sidebar, icon="P", title="Profiles", initially_open=False)
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

        # ============================================================
        # PROGRESS (bottom of sidebar)
        # ============================================================
        action_area = ctk.CTkFrame(sidebar_outer, fg_color=C["bg_panel"])
        action_area.pack(side="bottom", fill="x", padx=0, pady=0)
        ctk.CTkFrame(action_area, height=1, fg_color=C["border"]).pack(fill="x")

        progress_pad = ctk.CTkFrame(action_area, fg_color="transparent")
        progress_pad.pack(fill="x", padx=S["md"], pady=S["sm"])

        self.progress_label = ctk.CTkLabel(
            progress_pad, text="", text_color=C["text_dim"],
            font=(FONT_FAMILY, 11),
        )
        self.progress_label.pack(anchor="w")
        self.progress_bar = ctk.CTkProgressBar(
            progress_pad, height=6, corner_radius=3,
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

        # Logo
        try:
            _logo_lg = Image.open(_asset_path('logo_256.png'))
            self._empty_logo = ctk.CTkImage(light_image=_logo_lg, dark_image=_logo_lg, size=(80, 80))
            ctk.CTkLabel(empty_inner, image=self._empty_logo, text="").pack(pady=(0, S["lg"]))
        except Exception:
            pass

        ctk.CTkLabel(empty_inner, text="FreeClimber", font=F["h1"], text_color=C["text"]).pack()
        ctk.CTkLabel(empty_inner, text="Drosophila RING Assay Analysis",
            font=F["caption"], text_color=C["text_dim"]).pack(pady=(S["xs"], S["xl"]))

        # Drop zone card
        drop_zone = ctk.CTkFrame(empty_inner, fg_color=C["bg_card"], corner_radius=12,
            border_width=2, border_color=C["border"], width=320, height=110)
        drop_zone.pack(pady=(0, S["md"]))
        drop_zone.pack_propagate(False)
        dz_inner = ctk.CTkFrame(drop_zone, fg_color="transparent")
        dz_inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(dz_inner, text="Drop a video here or", font=F["body"],
            text_color=C["text_dim"]).pack()
        ctk.CTkButton(dz_inner, text="Open Video", width=140, height=36,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=F["body_b"], corner_radius=8,
            command=self._browse_video).pack(pady=(S["sm"], 0))

        ctk.CTkLabel(empty_inner, text="Cmd+O open  \u00b7  Cmd+R run  \u00b7  Cmd+T preview",
            font=F["caption"], text_color=C["text_disabled"]).pack(pady=(S["lg"], 0))

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

        self.setup_toolbar = PlotToolbar(tab, self.setup_canvas, self.setup_fig,
            save_callback=lambda: self._save_figure_dialog(self.setup_fig))
        self.setup_toolbar.grid(row=1, column=0, sticky="ew")

        self.setup_canvas.mpl_connect('button_press_event', self._roi_press_event)
        self.setup_canvas.mpl_connect('button_release_event', self._roi_release_event)
        self.setup_canvas.mpl_connect('motion_notify_event', self._roi_motion_event)
        self._plot_empty_placeholder(self.setup_fig, self.setup_canvas)

    def _build_diagnostics_tab(self):
        tab = self.tabview.add("Diagnostics")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.diag_fig = Figure(figsize=(10, 6), dpi=100)
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=tab)
        self.diag_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        PlotToolbar(tab, self.diag_canvas, self.diag_fig,
            save_callback=lambda: self._save_figure_dialog(self.diag_fig)).grid(row=1, column=0, sticky="ew")
        self._plot_empty_placeholder(self.diag_fig, self.diag_canvas)

    def _build_results_tab(self):
        tab = self.tabview.add("Results")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=2)
        tab.grid_columnconfigure(0, weight=1)

        # --- KPI summary cards at top ---
        self.summary_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.summary_frame.grid(row=0, column=0, sticky="ew", padx=S["xs"], pady=(S["xs"], S["xs"]))
        self.summary_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self._kpi_cards = {}
        kpi_defs = [
            ("velocity", "Mean Velocity", "--", C["accent"]),
            ("flies", "Flies Tracked", "--", C["success"]),
            ("r_squared", "R-squared", "--", C["warning"]),
            ("quality", "Quality", "--", C["accent"]),
            ("climbing_idx", "Climbing Idx", "--", C["success"]),
        ]
        for col_i, (key, title, default, accent_color) in enumerate(kpi_defs):
            card = ctk.CTkFrame(self.summary_frame, fg_color=C["bg_card"], corner_radius=8)
            card.grid(row=0, column=col_i, sticky="nsew", padx=S["xxs"], pady=S["xxs"])
            ctk.CTkLabel(card, text=title, font=(FONT_FAMILY, 10),
                         text_color=C["text_dim"]).pack(padx=S["sm"], pady=(S["sm"], 0))
            val_label = ctk.CTkLabel(card, text=default,
                                      font=(FONT_FAMILY, 20, "bold"), text_color=accent_color)
            val_label.pack(padx=S["sm"], pady=(0, S["xxs"]))
            sub_label = ctk.CTkLabel(card, text="", font=(FONT_FAMILY, 9),
                                      text_color=C["text_dim"])
            sub_label.pack(padx=S["sm"], pady=(0, S["sm"]))
            self._kpi_cards[key] = (val_label, sub_label)

        # Export button row
        export_row = ctk.CTkFrame(self.summary_frame, fg_color="transparent")
        export_row.grid(row=1, column=0, columnspan=5, sticky="e", padx=S["xs"], pady=(S["xxs"], 0))
        ctk.CTkButton(
            export_row, text="Export\u2026", width=80,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=(FONT_FAMILY, 12, "bold"),
            corner_radius=6, command=self._export_dialog,
        ).pack(side="right")

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
        self.slopes_tree.bind("<Command-c>", lambda e: self._copy_tree_to_clipboard(self.slopes_tree))

        # Per-fly metrics table (populated when individual tracking is on)
        perfly_tab = self.data_tabview.add("Per-Fly Metrics")
        perfly_tab.grid_rowconfigure(0, weight=1)
        perfly_tab.grid_columnconfigure(0, weight=1)
        self.perfly_tree = tk.ttk.Treeview(perfly_tab, show="headings", style="Dark.Treeview")
        self.perfly_tree.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        perfly_scroll = tk.ttk.Scrollbar(perfly_tab, orient="vertical", command=self.perfly_tree.yview)
        perfly_scroll.grid(row=0, column=1, sticky="ns")
        self.perfly_tree.configure(yscrollcommand=perfly_scroll.set)
        self.perfly_tree.bind("<Command-c>", lambda e: self._copy_tree_to_clipboard(self.perfly_tree))

        # Population stats table
        pop_tab = self.data_tabview.add("Population Stats")
        pop_tab.grid_rowconfigure(0, weight=1)
        pop_tab.grid_columnconfigure(0, weight=1)
        self.pop_stats_text = ctk.CTkTextbox(
            pop_tab, font=F["mono"], state="disabled", wrap="word",
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

        # --- Lazy figure creation for plot tabs ---
        self._result_figs = {}
        self._result_tab_widgets = {}
        self._pending_draws = {}

        for name in ["Overview", "Trajectories", "Individual Flies", "Raincloud"]:
            tab = self.results_plot_tabview.add(name)
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(tab, text="Run analysis to see results",
                         text_color=C["text_dim"], font=F["caption"]).grid(row=0, column=0)
            self._result_tab_widgets[name] = tab

        # Individual Flies sub-tabs are created lazily too
        self.individual_subtabs = None
        self._individual_sub_widgets = {}

        # Initialize figure attributes as None (created on first use)
        self.overview_fig = self.overview_canvas = None
        self.traj_fig = self.traj_canvas = None
        self.overlay_fig = self.overlay_canvas = None
        self.heatmap_fig = self.heatmap_canvas = None
        self.speed_curves_fig = self.speed_curves_canvas = None
        self.raincloud_fig = self.raincloud_canvas = None

        # Tab-change callback for deferred drawing
        self.results_plot_tabview.configure(command=self._on_result_tab_changed)

    def _build_statistics_tab(self):
        tab = self.tabview.add("Statistics")
        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=4)
        tab.grid_rowconfigure(2, weight=3)
        tab.grid_rowconfigure(3, weight=3)
        tab.grid_columnconfigure(0, weight=1)

        # Stats control bar at top
        self._build_stats_controls(tab)

        # Raincloud plot (lazy)
        self._stats_plot_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8)
        self._stats_plot_frame.grid(row=1, column=0, sticky="nsew", padx=S["xs"], pady=(S["xs"], S["xs"]))
        self._stats_plot_frame.grid_rowconfigure(0, weight=1)
        self._stats_plot_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self._stats_plot_frame, text="Run analysis to see statistics",
                     text_color=C["text_dim"], font=F["caption"]).grid(row=0, column=0)
        self.stats_fig = self.stats_canvas = None

        # CDF plot (lazy)
        self._cdf_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8)
        self._cdf_frame.grid(row=2, column=0, sticky="nsew", padx=S["xs"], pady=(0, S["xs"]))
        self._cdf_frame.grid_rowconfigure(0, weight=1)
        self._cdf_frame.grid_rowconfigure(1, weight=0)
        self._cdf_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self._cdf_frame, text="Run analysis to see CDF",
                     text_color=C["text_dim"], font=F["caption"]).grid(row=0, column=0)
        self.cdf_fig = self.cdf_canvas = None

        # Formatted text at bottom
        self.stats_text = ctk.CTkTextbox(
            tab, font=F["mono"], state="disabled", wrap="word",
            fg_color=C["bg_card"], text_color=C["text"],
        )
        self.stats_text.grid(row=3, column=0, sticky="nsew", padx=S["xs"], pady=S["xs"])
        self._set_stats_text("Run an analysis to see statistics here.")

    # ------------------------------------------------------------------
    # Plot empty placeholder
    # ------------------------------------------------------------------
    def _plot_empty_placeholder(self, fig, canvas):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Run analysis to see results", transform=ax.transAxes,
            ha='center', va='center', fontsize=11, color=C["text_disabled"])
        ax.axis('off')
        fig.patch.set_facecolor(C["bg"])
        ax.set_facecolor(C["bg"])
        canvas.draw()

    # ------------------------------------------------------------------
    # Lazy figure creation
    # ------------------------------------------------------------------
    def _ensure_fig(self, attr_name, parent_widget, figsize=(8, 4)):
        """Create figure/canvas/toolbar on first use. Returns (fig, canvas)."""
        fig = getattr(self, attr_name + '_fig', None)
        if fig is not None:
            return fig, getattr(self, attr_name + '_canvas')

        for child in parent_widget.winfo_children():
            child.destroy()

        parent_widget.grid_rowconfigure(0, weight=1)
        parent_widget.grid_rowconfigure(1, weight=0)
        parent_widget.grid_columnconfigure(0, weight=1)

        fig = Figure(figsize=figsize, dpi=100)
        fig.set_facecolor(C["bg"])
        canvas = FigureCanvasTkAgg(fig, master=parent_widget)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        PlotToolbar(parent_widget, canvas, fig,
            save_callback=lambda f=fig: self._save_figure_dialog(f)).grid(row=1, column=0, sticky="ew")

        self._bind_plot_context_menu(canvas, fig)

        setattr(self, attr_name + '_fig', fig)
        setattr(self, attr_name + '_canvas', canvas)
        return fig, canvas

    def _ensure_individual_subtabs(self):
        """Create Individual Flies sub-tabs on first use."""
        if self.individual_subtabs is not None:
            return

        tab = self._result_tab_widgets["Individual Flies"]
        for child in tab.winfo_children():
            child.destroy()

        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.individual_subtabs = ctk.CTkTabview(
            tab, fg_color=C["bg_card"],
            segmented_button_fg_color=C["bg"],
            segmented_button_selected_color=C["accent"],
            segmented_button_selected_hover_color=C["accent_hover"],
            segmented_button_unselected_color=C["bg"],
            segmented_button_unselected_hover_color=C["bg_hover"],
        )
        self.individual_subtabs.grid(row=0, column=0, sticky="nsew")

        for name in ["Trajectory Overlay", "Metrics Heatmap", "Speed Curves"]:
            sub = self.individual_subtabs.add(name)
            sub.grid_rowconfigure(0, weight=1)
            sub.grid_columnconfigure(0, weight=1)
            self._individual_sub_widgets[name] = sub

    def _on_result_tab_changed(self):
        """Draw deferred plot when user switches to a new results tab."""
        try:
            tab = self.results_plot_tabview.get()
        except Exception:
            return
        if tab in self._pending_draws:
            items = self._pending_draws.pop(tab)
            for canvas in items:
                canvas.draw()

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
                         rowheight=30,
                         font=F["mono"])
        style.configure("Dark.Treeview.Heading",
                         background=C["bg_hover"],
                         foreground=C["text"],
                         font=(FONT_FAMILY, 11, "bold"), relief="flat")
        style.map("Dark.Treeview.Heading",
                   background=[("active", C["accent_muted"])])
        style.map("Dark.Treeview",
                   background=[("selected", C["accent"])],
                   foreground=[("selected", C["text_on_accent"])])
        self._tree_row_colors = (C["bg_card"], C["bg_hover"])
        return style

    def _sort_treeview(self, tree, col):
        """Sort treeview by column, toggling asc/desc."""
        data = [(tree.set(k, col), k) for k in tree.get_children('')]
        try:
            data.sort(key=lambda t: float(t[0]))
        except ValueError:
            data.sort(key=lambda t: t[0])
        if getattr(tree, '_sort_reverse', False):
            data.reverse()
        tree._sort_reverse = not getattr(tree, '_sort_reverse', False)
        for i, (_, k) in enumerate(data):
            tree.move(k, '', i)
            tree.item(k, tags=('even' if i % 2 == 0 else 'odd',))

    def _copy_tree_to_clipboard(self, tree):
        """Copy selected rows (or all if none selected) as TSV."""
        selected = tree.selection()
        items = selected if selected else tree.get_children()
        rows = [tree.item(k)['values'] for k in items]
        if not rows:
            return
        header = '\t'.join(tree['columns'])
        body = '\n'.join('\t'.join(str(v) for v in r) for r in rows)
        self.clipboard_clear()
        self.clipboard_append(header + '\n' + body)
        self.status_var.set(f"Copied {len(rows)} rows to clipboard")

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
        self.tracking_var.set(params.get('individual_tracking', True))
        self._update_sidebar_state()

    # ------------------------------------------------------------------
    # Progressive sidebar state
    # ------------------------------------------------------------------
    def _update_sidebar_state(self):
        has_video = self.controller.detector is not None
        has_roi = has_video and (int(self.roi_w.get() or 0) > 0)
        has_preview = has_video and self._preview_done

        self.step2_card.set_enabled(has_video)
        self.step3_card.set_enabled(has_roi)
        self.step4_card.set_enabled(has_preview)

        # If loading a config with valid params, skip to step 4
        if has_video and has_roi and self.controller.slopes_df is not None:
            self._preview_done = True
            self.step3_card.set_enabled(True)
            self.step4_card.set_enabled(True)

    # ------------------------------------------------------------------
    # ROI drawing on Setup tab
    # ------------------------------------------------------------------
    def _roi_press_event(self, event):
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._roi_press = True
        self._roi_x0 = int(event.xdata)
        self._roi_y0 = int(event.ydata)

    def _roi_release_event(self, event):
        if not self._roi_press:
            return
        self._roi_press = False
        if event.xdata is not None and event.ydata is not None:
            self._roi_x1 = int(event.xdata)
            self._roi_y1 = int(event.ydata)
        self._update_roi_entries()
        self._draw_roi_rect()
        self._update_sidebar_state()

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
        self._roi_rect = Rectangle((x0, y0), w, h, fill=False, edgecolor=C["accent"], linewidth=2)
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

        def _load_worker():
            try:
                params = self._collect_params()
                meta = self.controller.load_video(path, params)
                self.after(0, lambda: self._on_video_loaded(path, meta))
            except Exception as e:
                logger.exception("Failed to load video")
                self.after(0, lambda ex=e: self._on_video_load_error(ex))

        threading.Thread(target=_load_worker, daemon=True).start()

    def _on_video_load_error(self, e):
        messagebox.showerror("Error", f"Could not load video:\n\n{e}")
        self.status_var.set("Error loading video")

    def _on_video_loaded(self, path: str, meta: dict):
        self.video_meta = meta

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
        self._preview_done = False
        self._update_sidebar_state()

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
            logger.exception("Parameter testing failed")
            messagebox.showerror("Error", f"Parameter testing failed:\n\n{e}")
            self.status_var.set("Error during testing")
            return

        self.diag_fig.tight_layout()
        self.diag_canvas.draw()
        self.tabview.set("Diagnostics")

        if 'slopes_df' in result:
            self._populate_results(result)
            self._populate_statistics(result)

        self._preview_done = True
        self._update_sidebar_state()
        self.status_var.set("Parameter testing complete")

    # ------------------------------------------------------------------
    # Full analysis (threaded pipeline, main-thread plotting)
    # ------------------------------------------------------------------
    def _validate_params(self) -> list[str]:
        """Validate parameters before running analysis. Returns list of error messages."""
        errors = []
        params = self._collect_params()
        if params.get('crop_n', 0) <= params.get('crop_0', 0):
            errors.append("crop_n must be greater than crop_0")
        if params.get('blank_n', 0) < params.get('blank_0', 0):
            errors.append("blank_n must be >= blank_0")
        d = params.get('diameter', 7)
        if isinstance(d, (int, float)) and d % 2 == 0:
            errors.append(f"Diameter ({d}) must be odd — will auto-correct to {d+1}")
        ecc_lo = params.get('ecc_low', 0)
        ecc_hi = params.get('ecc_high', 1)
        if ecc_lo >= ecc_hi:
            errors.append(f"ecc_low ({ecc_lo}) must be less than ecc_high ({ecc_hi})")
        fr = params.get('frame_rate', 30)
        if isinstance(fr, (int, float)) and fr <= 0:
            errors.append("frame_rate must be positive")
        return errors

    def _run_analysis(self):
        if self.controller.detector is None:
            messagebox.showinfo("FreeClimber", "Load a video first.")
            return

        if not self._analysis_lock.acquire(blocking=False):
            return

        validation_errors = self._validate_params()
        if validation_errors:
            self._analysis_lock.release()
            messagebox.showwarning("Parameter Issues",
                "Fix before running:\n\n" + "\n".join(f"• {e}" for e in validation_errors))
            return

        # Auto-save config before running
        if self.controller.video_path:
            try:
                cfg_path = self.controller.video_path.rsplit('.', 1)[0] + '.cfg'
                self.controller.save_config(cfg_path, self._collect_params())
            except Exception:
                logger.exception("Failed to auto-save config")

        self.controller._cancel.clear()
        self.status_var.set("Running analysis...")
        self.run_btn.configure(
            state="normal", text="CANCEL",
            fg_color=C["danger"], hover_color="#ff3050",
            command=self._cancel_analysis,
        )
        self.progress_bar.set(0)
        self.progress_label.configure(text="Processing...")
        self.update_idletasks()

        params = self._collect_params()

        def progress(step, total, msg):
            self.after(0, lambda: self._analysis_progress(step, total, msg))

        def worker():
            try:
                result = self.controller.run_pipeline_only(params, progress_callback=progress)
                self.after(0, lambda r=result: self._on_analysis_done(r))
            except Exception as e:
                logger.exception("Analysis worker failed")
                self.after(0, lambda ex=e: self._on_analysis_done(ex))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_analysis(self):
        self.controller._cancel.set()
        self.status_var.set("Cancelling...")
        self.run_btn.configure(state="disabled", text="Cancelling...")

    def _analysis_progress(self, step, total, msg):
        self.progress_bar.set(step / max(total, 1))
        self.progress_label.configure(text=msg)
        self.status_var.set(msg)

    def _draw_diagnostics(self):
        """Draw diagnostic subplots using data stored on the detector object."""
        try:
            self._draw_diagnostics_inner()
        except Exception:
            logger.exception("Failed to draw diagnostics")

    def _draw_diagnostics_inner(self):
        import matplotlib.cm as cm
        from matplotlib.lines import Line2D

        det = self.controller.detector
        if det is None:
            return

        self.diag_fig.clear()
        axes = [self.diag_fig.add_subplot(2, 3, i + 1) for i in range(6)]

        # Subplot 0: Background image
        if hasattr(det, 'background') and det.background is not None:
            axes[0].set_title("Background Image")
            axes[0].imshow(det.background, cmap=cm.Greys_r)
            axes[0].set_xlim(0, det.w)
            axes[0].set_ylim(det.h, 0)

        # Subplot 1: Test frame with spot overlay
        if hasattr(det, 'df_big') and det.df_big is not None:
            spots_false = det.df_big[~det.df_big['True_particle']]
            spots_true = det.df_big[det.df_big['True_particle']].copy()

            spots_true['vial'] = np.repeat(0, spots_true.shape[0])
            vial_assignments = det.bin_vials(spots_true, vials=det.vials, bin_lines=det.bin_lines)[1]
            spots_true.loc[(spots_true.x >= det.bin_lines[0]) & (spots_true.x <= det.bin_lines[-1]), 'vial'] = vial_assignments

            axes[1].set_title(f'Frame: {det.check_frame}')
            if hasattr(det, 'clean_stack') and det.clean_stack is not None:
                axes[1].imshow(det.clean_stack[det.check_frame], cmap=cm.Greys_r)
            axes[1].scatter(
                spots_false[spots_false.frame == det.check_frame].x,
                spots_false[spots_false.frame == det.check_frame].y,
                color='b', marker='+', alpha=.5)
            a = axes[1].scatter(
                spots_true[spots_true.frame == det.check_frame].x,
                spots_true[spots_true.frame == det.check_frame].y,
                c=spots_true[spots_true.frame == det.check_frame].vial,
                cmap=det.vial_color_map, marker='o', alpha=.8)
            a.set_facecolor('none')
            axes[1].vlines(det.bin_lines, 0, det.df_big.y.max(), color='w')
            axes[1].set_xlim(0, det.w)
            axes[1].set_ylim(det.h, 0)

        # Subplot 2: Mean vertical position (local linear regression)
        if hasattr(det, 'df_filtered') and det.df_filtered is not None:
            df = det.df_filtered.sort_values(by='frame')
            convert_x, convert_y = 1, 1
            if det.convert_to_cm_sec:
                convert_x, convert_y = det.frame_rate, det.pixel_to_cm
            for V in range(1, det.vials + 1):
                label = f'Vial {V}'
                color = det.color_list[V - 1]
                _df = df[df.vial == V]
                begin = det.local_linear_regression(_df).iloc[0].first_frame.astype(int)
                end = begin + det.window
                axes[2].plot(_df.groupby('frame').frame.mean() / convert_x,
                             _df.groupby('frame').y.mean() / convert_y, alpha=.35, color=color, label='')
                _df = _df[(_df.frame >= begin) & (_df.frame <= end)]
                axes[2].plot(_df.groupby('frame').frame.mean() / convert_x,
                             _df.groupby('frame').y.mean() / convert_y, color=color, label=label)
            label_y, label_x = '(pixels)', 'Frames'
            if det.convert_to_cm_sec:
                label_x, label_y = 'Seconds', '(cm)'
            axes[2].set(title='Mean vertical position over time',
                        ylabel=f'Mean y-position {label_y}', xlabel=label_x)
            ncol = 3 if det.vials > 10 else (2 if det.vials > 5 else 1)
            axes[2].legend(frameon=False, fontsize='x-small', ncol=ncol)

        # Subplot 3: Mass distribution
        bins = 40
        if hasattr(det, 'df_big') and det.df_big is not None:
            axes[3].set_title('Mass Distribution')
            axes[3].hist(det.df_big.mass, bins=bins)
            y_max = np.histogram(det.df_big.mass, bins=bins)[0].max()
            axes[3].vlines(det.minmass, 0, y_max)

        # Subplot 4: Signal distribution
        if hasattr(det, 'df_big') and det.df_big is not None:
            axes[4].set_title('Signal Distribution')
            axes[4].hist(det.df_big.signal, bins=bins)
            y_max = np.histogram(det.df_big.signal, bins=bins)[0].max()
            if isinstance(det.threshold, (int, float)):
                axes[4].vlines(det.threshold, 0, y_max)

        # Subplot 5: Flies detected per frame
        if hasattr(det, 'df_filtered') and det.df_filtered is not None:
            df = det.df_filtered.sort_values(by='frame')
            for V in range(1, det.vials + 1):
                color = det.color_list[V - 1]
                _df = df[df.vial == V]
                begin = det.local_linear_regression(_df).iloc[0].first_frame.astype(int)
                end = begin + det.window
                axes[5].plot(_df.groupby('frame').frame.unique(),
                             _df.groupby('frame').y.count(), alpha=.3, color=color, label='')
                _df_lin = _df[(_df.frame >= begin) & (_df.frame <= end)]
                axes[5].plot(_df_lin.groupby('frame').frame.unique(),
                             _df_lin.groupby('frame').frame.count(), color=color, alpha=.5)
                axes[5].hlines(np.median(_df_lin.groupby('frame').frame.count()),
                               df.frame.min(), df.frame.max(), linestyle='--', alpha=.7, color=color)
            axes[5].set(title='Flies detected per frame', ylabel='Flies detected', xlabel='Frame')
            axes[5].set_ylim(ymin=0)
            custom_lines = [Line2D([0], [0], color='k', linestyle='--', alpha=.9),
                            Line2D([0], [0], color='k', linestyle='-', alpha=.5)]
            axes[5].legend(custom_lines, ['Median', 'All frames'], frameon=False, fontsize='x-small')

        self.diag_fig.tight_layout()
        self.diag_canvas.draw()

    def _on_analysis_done(self, result):
        try:
            self._analysis_lock.release()
        except RuntimeError:
            pass
        self.run_btn.configure(
            state="normal", text="RUN ANALYSIS",
            fg_color=C["run"], hover_color="#66BB6A",
            command=self._run_analysis,
        )
        self.progress_bar.set(1.0)

        if isinstance(result, Exception):
            logger.exception("Analysis failed: %s", result)
            self._toast(f"Analysis failed: {result}", level="error")
            self.status_var.set("Analysis failed")
            self.progress_label.configure(text="Failed")
            return

        self._draw_diagnostics()

        if 'slopes_df' in result:
            self._populate_results(result)
            self._populate_statistics(result)
            self.tabview.set("Results")
            self._on_result_tab_changed()

        self.progress_label.configure(text="Complete")
        self.status_var.set("Analysis complete \u2014 results saved")
        self._toast("Analysis complete — results saved", level="success")

    # ------------------------------------------------------------------
    # Populate Results tab
    # ------------------------------------------------------------------
    def _populate_results(self, result: dict):
        df = result.get('slopes_df') if isinstance(result, dict) else result
        if df is None:
            return

        # --- KPI cards ---
        quality = result.get('quality') if isinstance(result, dict) else None
        pop = result.get('population_metrics') if isinstance(result, dict) else None
        ci_data = result.get('climbing_index') if isinstance(result, dict) else None
        per_fly = result.get('per_fly_metrics') if isinstance(result, dict) else None

        # Mean Velocity
        val_l, sub_l = self._kpi_cards["velocity"]
        if pop and 'mean_speed' in pop:
            val_l.configure(text=f"{pop['mean_speed']:.3f}")
            sub_l.configure(text="px/frame" if not self.convert_cm_var.get() else "cm/s")
        else:
            val_l.configure(text="--")

        # Flies Tracked
        val_l, sub_l = self._kpi_cards["flies"]
        fly_count = 0
        if pop and 'fly_count_per_vial' in pop:
            fly_count = int(sum(pop['fly_count_per_vial'].values()))
        indiv = len(per_fly) if per_fly is not None and len(per_fly) > 0 else 0
        val_l.configure(text=str(fly_count) if fly_count else str(indiv) if indiv else "--")
        sub_l.configure(text=f"{indiv} individual" if indiv else "population avg")

        # R-squared
        val_l, sub_l = self._kpi_cards["r_squared"]
        if 'r_value' in df.columns:
            r_vals = df[~df.get('vial_ID', df.index).astype(str).str.endswith('_all')]['r_value']
            mean_r = r_vals.abs().mean()
            val_l.configure(text=f"{mean_r:.3f}")
            color = C["success"] if mean_r > 0.9 else C["warning"] if mean_r > 0.7 else C["danger"]
            val_l.configure(text_color=color)
            sub_l.configure(text="mean |r|")
        else:
            val_l.configure(text="--")

        # Quality
        val_l, sub_l = self._kpi_cards["quality"]
        if quality:
            score = quality.get('overall_score', 0)
            level = quality.get('overall_level', '?').title()
            val_l.configure(text=f"{self._quality_dots(score)} {level}")
            sub_l.configure(text=f"score: {score:.2f}")
        else:
            val_l.configure(text="--")

        # Climbing Index
        val_l, sub_l = self._kpi_cards["climbing_idx"]
        if ci_data:
            mean_ci = np.mean(list(ci_data.values()))
            val_l.configure(text=f"{mean_ci:.1f}%")
            sub_l.configure(text=f"{len(ci_data)} vials")
        else:
            val_l.configure(text="--")

        # --- Slopes table with quality dots ---
        self.slopes_tree.delete(*self.slopes_tree.get_children())
        cols = list(df.columns) + (['q_score'] if quality else [])
        self.slopes_tree["columns"] = cols
        for col in cols:
            display_name = col.replace('_', ' ').title()
            self.slopes_tree.heading(col, text=display_name,
                command=lambda c=col: self._sort_treeview(self.slopes_tree, c))
            data_width = max((len(str(v)) for v in df[col]), default=0) * 9 if col in df.columns else 0
            max_width = max(len(display_name) * 10, data_width, 70)
            self.slopes_tree.column(col, width=min(max_width, 200), minwidth=50)

        per_vial_quality = quality.get('per_vial', {}) if quality else {}
        for row_i, (idx, row) in enumerate(df.iterrows()):
            values = []
            for v in row:
                if isinstance(v, float) and v == int(v):
                    values.append(str(int(v)))
                elif isinstance(v, float):
                    values.append(str(round(v, 4)))
                else:
                    values.append(str(v))
            if quality:
                vial_id = str(row.get('vial_ID', idx))
                vq = per_vial_quality.get(vial_id, {})
                values.append(self._quality_dots(vq.get('score', 0)))
            tag = 'even' if row_i % 2 == 0 else 'odd'
            self.slopes_tree.insert("", "end", values=values, tags=(tag,))
        self.slopes_tree.tag_configure('even', background=self._tree_row_colors[0])
        self.slopes_tree.tag_configure('odd', background=self._tree_row_colors[1])

        # --- Per-fly metrics table ---
        per_fly = result.get('per_fly_metrics') if isinstance(result, dict) else None
        self.perfly_tree.delete(*self.perfly_tree.get_children())
        if per_fly is not None and len(per_fly) > 0:
            pcols = list(per_fly.columns)
            self.perfly_tree["columns"] = pcols
            for col in pcols:
                display_name = col.replace('_', ' ').title()
                self.perfly_tree.heading(col, text=display_name,
                    command=lambda c=col: self._sort_treeview(self.perfly_tree, c))
                self.perfly_tree.column(col, width=min(max(len(display_name) * 9, 60), 160), minwidth=40)
            for row_i, (_, row) in enumerate(per_fly.iterrows()):
                values = []
                for col, v in zip(pcols, row):
                    if col == 'particle' and isinstance(v, float):
                        values.append(str(int(v)))
                    elif isinstance(v, float):
                        values.append(str(round(v, 3)))
                    else:
                        values.append(str(v))
                tag = 'even' if row_i % 2 == 0 else 'odd'
                self.perfly_tree.insert("", "end", values=values, tags=(tag,))
            self.perfly_tree.tag_configure('even', background=self._tree_row_colors[0])
            self.perfly_tree.tag_configure('odd', background=self._tree_row_colors[1])

        # --- Population stats text ---
        self._update_pop_stats_text(pop)

        # --- Overview (Speed + Distribution side by side) ---
        self._plot_overview(df, result)

        # --- Raincloud plot ---
        self._plot_raincloud(df, result)

        # --- Trajectory plot ---
        positions = self.controller.get_positions()
        self._plot_trajectory(positions, result)

        # --- Individual Flies (Per-Fly + Heatmap + Speed Curves) ---
        self._plot_individual_flies(result)

        # Draw only the currently visible tab immediately; defer others
        try:
            active = self.results_plot_tabview.get()
            if active in self._pending_draws:
                for c in self._pending_draws.pop(active):
                    c.draw()
        except Exception:
            pass

    def _quality_dots(self, score: float) -> str:
        filled = round(score * 4)
        return '\u25cf' * filled + '\u25cb' * (4 - filled)

    def _plot_overview(self, df: pd.DataFrame, result=None):
        fig, canvas = self._ensure_fig('overview', self._result_tab_widgets["Overview"], figsize=(10, 4))
        fig.clear()

        ci_data = result.get('climbing_index') if isinstance(result, dict) else None
        n_panels = 3 if ci_data else 2

        ax_speed = fig.add_subplot(1, n_panels, 1)
        ax_dist = fig.add_subplot(1, n_panels, 2)

        groups = self._build_speed_groups(df)

        # Speed chart (left) — per-vial bar chart
        if groups:
            try:
                from output.figures import bar_chart_with_points
                bar_chart_with_points(groups, ylabel='Climbing Speed',
                                      title='Climbing Speed by Vial', ax=ax_speed)
            except Exception as e:
                logger.error("Speed chart failed: %s", e, exc_info=True)
                ax_speed.text(0.5, 0.5, f"Error: {e}", transform=ax_speed.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)

        # Distribution (center) — needs >=2 values per group for KDE
        dist_groups = self._build_distribution_groups(df, result)
        if dist_groups:
            try:
                from output.figures import speed_distribution
                speed_distribution(dist_groups, ax=ax_dist)
            except Exception as e:
                logger.error("Distribution plot failed: %s", e, exc_info=True)
                ax_dist.text(0.5, 0.5, f"Error: {e}", transform=ax_dist.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)

        # Climbing Index (right) — if available
        if ci_data:
            ax_ci = fig.add_subplot(1, n_panels, 3)
            try:
                from output.figures import climbing_index_chart
                climbing_index_chart(ci_data, ax=ax_ci)
            except Exception as e:
                logger.error("Climbing index chart failed: %s", e, exc_info=True)
                ax_ci.text(0.5, 0.5, f"Error: {e}", transform=ax_ci.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)

        fig.tight_layout()
        self._pending_draws.setdefault("Overview", []).append(canvas)

    def _build_speed_groups(self, df: pd.DataFrame) -> dict:
        ycol = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                ycol = c
                break
        if ycol is None:
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    ycol = c
                    break
        if ycol is None:
            return {}

        group_col = None
        for c in df.columns:
            if 'vial' in c.lower():
                group_col = c
                break

        if group_col:
            df = df[~df[group_col].astype(str).str.endswith('_all')].copy()
            df = df[df[group_col] != 0].copy()

        groups = {}
        if group_col and df[group_col].nunique() >= 2:
            for name, sub in df.groupby(group_col):
                vals = pd.to_numeric(sub[ycol], errors='coerce').dropna().values
                if len(vals) > 0:
                    m = re.search(r'(\d+)$', str(name))
                    label = f"Vial {m.group(1)}" if m else f"Vial {name}"
                    groups[label] = vals
        else:
            vals = pd.to_numeric(df[ycol], errors='coerce').dropna().values
            if len(vals) > 0:
                groups['All'] = vals
        return groups

    def _build_distribution_groups(self, df: pd.DataFrame, result=None) -> dict:
        """Build groups with >=2 values for KDE/violin plots.

        Tries per-fly metrics first (rich per-fly data), falls back to
        pooling all slopes into a single group.
        """
        per_fly = result.get('per_fly_metrics') if result and isinstance(result, dict) else None
        if per_fly is not None and len(per_fly) > 0:
            speed_col = None
            for c in per_fly.columns:
                if any(kw in c.lower() for kw in ['speed', 'velocity', 'slope']):
                    speed_col = c
                    break
            if speed_col:
                vial_col = None
                for c in per_fly.columns:
                    if 'vial' in c.lower():
                        vial_col = c
                        break
                if vial_col:
                    per_fly = per_fly[~per_fly[vial_col].astype(str).str.endswith('_all')].copy()
                    per_fly = per_fly[per_fly[vial_col] != 0].copy()
                groups = {}
                if vial_col and per_fly[vial_col].nunique() >= 2:
                    for name, sub in per_fly.groupby(vial_col):
                        vals = pd.to_numeric(sub[speed_col], errors='coerce').dropna().values
                        if len(vals) >= 2:
                            m = re.search(r'(\d+)$', str(name))
                            label = f"Vial {m.group(1)}" if m else f"Vial {name}"
                            groups[label] = vals
                if not groups:
                    vals = pd.to_numeric(per_fly[speed_col], errors='coerce').dropna().values
                    if len(vals) >= 2:
                        groups['All Flies'] = vals
                if groups:
                    return groups

        ycol = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                ycol = c
                break
        if ycol is None:
            return {}
        vals = pd.to_numeric(df[ycol], errors='coerce').dropna().values
        if len(vals) >= 2:
            return {'All Vials': vals}
        return {}

    def _plot_raincloud(self, df: pd.DataFrame, result=None):
        fig, canvas = self._ensure_fig('raincloud', self._result_tab_widgets["Raincloud"])
        fig.clear()
        ax = fig.add_subplot(111)

        groups = self._build_distribution_groups(df, result)
        if len(groups) >= 1:
            try:
                from output.figures import raincloud_plot
                raincloud_plot(groups, ylabel='Climbing Speed',
                               title='Speed Distribution (Raincloud)', ax=ax)
            except Exception as e:
                logger.error("Raincloud plot failed: %s", e, exc_info=True)
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)
        fig.tight_layout()
        self._pending_draws.setdefault("Raincloud", []).append(canvas)

    def _plot_trajectory(self, positions, result):
        fig, canvas = self._ensure_fig('traj', self._result_tab_widgets["Trajectories"])
        fig.clear()
        if positions is None:
            self._pending_draws.setdefault("Trajectories", []).append(canvas)
            return

        ax = fig.add_subplot(111)
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
        fig.tight_layout()
        self._pending_draws.setdefault("Trajectories", []).append(canvas)

    def _plot_individual_flies(self, result):
        self._ensure_individual_subtabs()
        has_tracking = result.get('has_individual_tracking', False) if isinstance(result, dict) else False
        raw_df = result.get('raw_tracking_df') if isinstance(result, dict) else None
        per_fly = result.get('per_fly_metrics') if isinstance(result, dict) else None
        first_frame = result.get('first_frame') if isinstance(result, dict) else None

        def _unavailable_msg(fig, canvas, label):
            fig.clear()
            ax = fig.add_subplot(111)
            reasons = []
            if not has_tracking:
                reasons.append("tracking disabled or failed")
            if raw_df is None:
                reasons.append("no raw tracking data")
            elif 'particle' not in raw_df.columns:
                reasons.append("no 'particle' column in data")
            msg = f"{label} not available"
            if reasons:
                msg += f"\n({', '.join(reasons)})"
            ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color=C["text_dim"], wrap=True)
            ax.set_facecolor(C["bg_card"])
            ax.axis('off')

        # Per-fly overlay
        overlay_fig, overlay_canvas = self._ensure_fig('overlay', self._individual_sub_widgets["Trajectory Overlay"])
        overlay_fig.clear()
        if has_tracking and raw_df is not None and 'particle' in raw_df.columns:
            try:
                from output.figures import per_fly_trajectory_overlay
                ax_overlay = overlay_fig.add_subplot(111)
                n_vials = self.controller.config.get('vials', 3)
                per_fly_trajectory_overlay(raw_df, first_frame=first_frame,
                                           vials=n_vials, ax=ax_overlay)
                overlay_fig.tight_layout()
            except Exception as e:
                logger.error("Per-fly overlay failed: %s", e, exc_info=True)
                overlay_fig.clear()
                ax = overlay_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)
        else:
            _unavailable_msg(overlay_fig, overlay_canvas, "Per-fly tracking")

        # Heatmap
        heatmap_fig, heatmap_canvas = self._ensure_fig('heatmap', self._individual_sub_widgets["Metrics Heatmap"])
        heatmap_fig.clear()
        if per_fly is not None and len(per_fly) > 0:
            try:
                from output.figures import per_fly_metrics_heatmap
                ax_heatmap = heatmap_fig.add_subplot(111)
                per_fly_metrics_heatmap(per_fly, ax=ax_heatmap)
                heatmap_fig.tight_layout()
            except Exception as e:
                logger.error("Heatmap failed: %s", e, exc_info=True)
                heatmap_fig.clear()
                ax = heatmap_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)
        else:
            _unavailable_msg(heatmap_fig, heatmap_canvas, "Per-fly metrics")

        # Speed curves
        speed_fig, speed_canvas = self._ensure_fig('speed_curves', self._individual_sub_widgets["Speed Curves"])
        speed_fig.clear()
        if has_tracking and raw_df is not None and 'particle' in raw_df.columns:
            try:
                from output.figures import per_fly_speed_timeseries
                ax_curves = speed_fig.add_subplot(111)
                per_fly_speed_timeseries(raw_df, ax=ax_curves)
                speed_fig.tight_layout()
            except Exception as e:
                logger.error("Speed curves failed: %s", e, exc_info=True)
                speed_fig.clear()
                ax = speed_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)
        else:
            _unavailable_msg(speed_fig, speed_canvas, "Speed curves")

        # Defer all individual draws to tab switch
        self._pending_draws.setdefault("Individual Flies", []).extend(
            [overlay_canvas, heatmap_canvas, speed_canvas])

    def _update_pop_stats_text(self, pop):
        self.pop_stats_text.configure(state="normal")
        self.pop_stats_text.delete("0.0", "end")
        if pop:
            lines = []
            lines.append("\u2501" * 34)
            lines.append("  POPULATION SUMMARY")
            lines.append("\u2501" * 34)

            for key in ['mean_speed', 'median_speed', 'std_speed', 'iqr']:
                if key in pop:
                    label = key.replace('_', ' ').title()
                    val = pop[key]
                    if isinstance(val, (list, tuple)):
                        lines.append(f"  {label + ':':<18} [{val[0]:.3f}, {val[1]:.3f}]")
                    elif isinstance(val, float):
                        lines.append(f"  {label + ':':<18} {val:.3f}")
                    else:
                        lines.append(f"  {label + ':':<18} {val}")

            fly_counts = pop.get('fly_count_per_vial')
            if fly_counts and isinstance(fly_counts, dict):
                lines.append("")
                lines.append("  FLY COUNTS PER VIAL")
                lines.append("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
                lines.append("  \u2502 Vial \u2502 Flies \u2502")
                lines.append("  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
                total = 0
                for k, v in sorted(fly_counts.items(), key=lambda x: str(x[0])):
                    count = int(v) if isinstance(v, (int, float)) else v
                    lines.append(f"  \u2502 {str(k):>4} \u2502 {str(count):>5} \u2502")
                    if isinstance(v, (int, float)):
                        total += int(v)
                lines.append("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
                lines.append(f"  Total: ~{total} flies")

            # Any remaining keys
            shown = {'mean_speed', 'median_speed', 'std_speed', 'iqr', 'fly_count_per_vial'}
            remaining = {k: v for k, v in pop.items() if k not in shown}
            if remaining:
                lines.append("")
                for key, val in remaining.items():
                    label = key.replace('_', ' ').title()
                    if isinstance(val, float):
                        lines.append(f"  {label + ':':<18} {val:.3f}")
                    elif isinstance(val, dict):
                        lines.append(f"  {label}:")
                        for k2, v2 in val.items():
                            lines.append(f"    {k2}: {v2:.1f}" if isinstance(v2, float) else f"    {k2}: {v2}")
                    else:
                        lines.append(f"  {label + ':':<18} {val}")

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

        # Filter out _all rows and vial 0
        vial_filter_col = next((c for c in df.columns if 'vial' in c.lower()), None)
        if vial_filter_col:
            df = df[~df[vial_filter_col].astype(str).str.endswith('_all')].copy()
            df = df[df[vial_filter_col] != 0].copy()

        slope_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                slope_col = c
                break
        if slope_col is None:
            self._set_stats_text("No slope/velocity column found in results.")
            return

        # Apply normalization if selected
        norm_choice = getattr(self, 'stats_norm_var', None)
        norm_choice = norm_choice.get() if norm_choice else "None"
        if norm_choice == "% of Control" and 'vial_ID' in df.columns:
            try:
                from analysis.normalization import normalize_to_control
                control_vials = [df['vial_ID'].iloc[0]]
                df = normalize_to_control(df, control_vials, metric_col=slope_col)
                norm_col = f'normalized_{slope_col}'
                if norm_col in df.columns:
                    slope_col = norm_col
            except Exception as e:
                logger.warning("Normalization to control failed: %s", e)
        elif norm_choice == "Z-score":
            try:
                from analysis.normalization import batch_zscore
                df = batch_zscore(df, metric_col=slope_col)
                zscore_col = f'zscore_{slope_col}'
                if zscore_col in df.columns:
                    slope_col = zscore_col
            except Exception as e:
                logger.warning("Z-score normalization failed: %s", e)

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

        # For distribution plots, use distribution groups (per-fly or pooled)
        dist_groups = groups if all(len(v) >= 2 for v in groups.values()) and groups else \
            self._build_distribution_groups(df, result)

        # --- Raincloud plot ---
        stats_fig, stats_canvas = self._ensure_fig('stats', self._stats_plot_frame)
        stats_fig.clear()
        ax = stats_fig.add_subplot(111)
        if len(dist_groups) >= 2:
            try:
                from output.figures import raincloud_plot
                raincloud_plot(dist_groups, ylabel=slope_col, title="Distribution by Group", ax=ax)
            except Exception as e:
                logger.error("Stats raincloud failed: %s", e, exc_info=True)
                try:
                    from output.figures import box_swarm_plot
                    box_swarm_plot(dist_groups, ylabel=slope_col, title="Distribution by Group", ax=ax)
                except Exception as e2:
                    logger.error("Stats box_swarm fallback failed: %s", e2, exc_info=True)
                    ax.text(0.5, 0.5, f"Error: {e2}", transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)
        elif len(dist_groups) == 1:
            name, vals = next(iter(dist_groups.items()))
            ax.hist(vals, bins='auto', color=C["accent"], alpha=0.7, edgecolor='black')
            ax.set_xlabel(slope_col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution: {name}")
        stats_fig.tight_layout()
        stats_canvas.draw()

        # --- CDF plot ---
        cdf_fig, cdf_canvas = self._ensure_fig('cdf', self._cdf_frame, figsize=(8, 3))
        cdf_fig.clear()
        cdf_ax = cdf_fig.add_subplot(111)
        if len(dist_groups) >= 1:
            try:
                from output.figures import cdf_comparison
                cdf_comparison(dist_groups, xlabel=slope_col, ax=cdf_ax)
            except Exception as e:
                logger.error("CDF plot failed: %s", e, exc_info=True)
                cdf_ax.text(0.5, 0.5, f"Error: {e}", transform=cdf_ax.transAxes,
                    ha='center', va='center', fontsize=9, color=C["danger"], wrap=True)
        cdf_fig.tight_layout()
        cdf_canvas.draw()

        # --- Formatted text ---
        lines = ["\u2550" * 50, "  STATISTICAL ANALYSIS", "\u2550" * 50, ""]

        # Read dropdown selections
        test_choice = getattr(self, 'stats_test_var', None)
        test_choice = test_choice.get() if test_choice else "Auto-detect"
        correction_choice = getattr(self, 'stats_correction_var', None)
        correction_choice = correction_choice.get() if correction_choice else "Holm-Bonferroni"

        correction_map = {
            "Holm-Bonferroni": "holm",
            "Benjamini-Hochberg": "bh",
            "Bonferroni": "holm",
            "None": None,
        }
        correction_method = correction_map.get(correction_choice, "holm")

        if group_col and df[group_col].nunique() >= 2 and len(groups) >= 2:
            norm = check_normality(groups)
            lines.append("\u25b6 NORMALITY (Shapiro-Wilk)")
            lines.append("\u2500" * 40)
            for gname, res in norm['results'].items():
                check = "\u2713" if res['normal'] else "\u2717"
                lines.append(f"  {check} {gname}: W={res['statistic']:.4f}, p={res['p_value']:.4f}")
            lines.append(f"  All normal: {'Yes' if norm['all_normal'] else 'No'}")
            lines.append("")

            # Determine forced normality from test choice
            force_normal = None
            if test_choice in ("ANOVA",):
                force_normal = True
            elif test_choice in ("Kruskal-Wallis",):
                force_normal = False

            if test_choice == "Two-group" or (test_choice == "Auto-detect" and len(groups) == 2):
                keys = list(groups.keys())
                stat_result = compare_two_groups(groups[keys[0]], groups[keys[1]], normal=force_normal)
                lines.append("\u25b6 TWO-GROUP COMPARISON")
                lines.append("\u2500" * 40)
                lines.append(f"  Test: {stat_result['test']}")
                lines.append(f"  Statistic: {stat_result['statistic']:.4f}")
                sig = stat_result['significance']
                lines.append(f"  p-value: {stat_result['p_value']:.6f}  {sig}")
                lines.append(f"  Cohen's d: {stat_result['effect_size_d']:.3f}")
                lines.append("")
            elif test_choice == "Dunnett's" and len(groups) >= 2:
                from analysis.stats import dunnett_vs_control
                control = list(groups.keys())[0]
                comparisons = dunnett_vs_control(groups, control)
                lines.append(f"\u25b6 DUNNETT'S TEST (control: {control})")
                lines.append("\u2500" * 40)
                for comp in comparisons:
                    lines.append(
                        f"  {comp['group']}: p={comp['p_value']:.6f} {comp['significance']} "
                        f"(d={comp['effect_size_d']:.3f})"
                    )
                lines.append("")
            elif len(groups) >= 3 or test_choice in ("Pairwise", "ANOVA", "Kruskal-Wallis"):
                stat_result = compare_multiple_groups(groups, normal=force_normal)
                lines.append(f"\u25b6 MULTI-GROUP COMPARISON ({len(groups)} groups)")
                lines.append("\u2500" * 40)
                lines.append(f"  Test: {stat_result['test']}")
                lines.append(f"  Statistic: {stat_result['statistic']:.4f}")
                lines.append(f"  p-value: {stat_result['p_value']:.6f}")
                lines.append(f"  Effect size ({stat_result['effect_size_name']}): {stat_result['effect_size']:.4f}")
                lines.append(f"  Significant: {'Yes' if stat_result['significant'] else 'No'}")
                lines.append("")

                if stat_result.get('post_hoc'):
                    post_hoc = stat_result['post_hoc']
                    if correction_method and len(post_hoc) > 1:
                        from analysis.stats import correct_pvalues
                        raw_ps = [c['p_value'] for c in post_hoc]
                        adj_ps = correct_pvalues(raw_ps, method=correction_method)
                        for c, ap in zip(post_hoc, adj_ps):
                            c['p_value'] = ap
                            c['significance'] = '***' if ap < 0.001 else '**' if ap < 0.01 else '*' if ap < 0.05 else 'ns'
                    correction_label = f" ({correction_choice})" if correction_method else ""
                    lines.append(f"\u25b6 POST-HOC: {stat_result['post_hoc_method']}{correction_label}")
                    lines.append("\u2500" * 40)
                    lines.append(f"  {'Group 1':<12} {'Group 2':<12} {'p-value':<10} {'Sig':<5} {'d':<8}")
                    sep = "\u2500"
                    lines.append(f"  {sep*12} {sep*12} {sep*10} {sep*5} {sep*8}")
                    for comp in post_hoc:
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
                logger.exception("Config load failed")
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

        # Check optional dependency availability
        try:
            import openpyxl  # noqa: F401
            has_excel = True
        except ImportError:
            has_excel = False
        try:
            import jinja2  # noqa: F401
            import weasyprint  # noqa: F401
            has_pdf = True
        except ImportError:
            has_pdf = False
        try:
            import plotly  # noqa: F401
            has_html = True
        except ImportError:
            has_html = False

        formats = [
            ("Slopes CSV (backward compatible)", "csv"),
            ("Tidy CSV (R-ready)", "tidy"),
            ("GraphPad Prism CSV", "prism"),
            ("Excel Workbook (.xlsx)" if has_excel else "Excel (install openpyxl)", "excel"),
            ("Per-fly Tracks CSV", "tracks"),
            ("PDF Report" if has_pdf else "PDF Report (install jinja2 + weasyprint)", "pdf"),
            ("HTML Report" if has_html else "HTML Report (install plotly)", "html"),
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
                messagebox.showinfo("Missing Dependency",
                    "Install openpyxl for Excel export:\n\npip install openpyxl", parent=win)
                return
            if fmt == 'pdf' and not has_pdf:
                messagebox.showinfo("Missing Dependency",
                    "Install jinja2 + weasyprint for PDF export:\n\npip install jinja2 weasyprint", parent=win)
                return
            if fmt == 'html' and not has_html:
                messagebox.showinfo("Missing Dependency",
                    "Install plotly for HTML export:\n\npip install plotly", parent=win)
                return
            ext_map = {'excel': '.xlsx', 'pdf': '.pdf', 'html': '.html'}
            ext = ext_map.get(fmt, '.csv')
            ftype_map = {
                '.xlsx': [("Excel files", "*.xlsx"), ("All files", "*.*")],
                '.pdf': [("PDF files", "*.pdf"), ("All files", "*.*")],
                '.html': [("HTML files", "*.html"), ("All files", "*.*")],
            }
            ftypes = ftype_map.get(ext, [("CSV files", "*.csv"), ("All files", "*.*")])
            path = filedialog.asksaveasfilename(
                parent=win, title="Save As",
                defaultextension=ext, filetypes=ftypes,
            )
            if path:
                try:
                    self.controller.export_results(fmt, path)
                    win.destroy()
                    self._toast(f"Exported: {os.path.basename(path)}", level="success")
                except Exception as e:
                    logger.exception("Export failed")
                    self._toast(f"Export failed: {e}", level="error")

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
                            "FreeClimber v4.0\n\n"
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
            logger.exception("Profile load failed")
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
            if not messagebox.askyesno("Delete Profile", f"Delete profile '{name}'?"):
                return
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
            self._on_result_tab_changed()
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
    # Right-click context menu on plots
    # ------------------------------------------------------------------
    def _bind_plot_context_menu(self, canvas, fig):
        def _on_right_click(event):
            menu = tk.Menu(self, tearoff=0, bg=C["bg_card"], fg=C["text"],
                activebackground=C["accent"], activeforeground=C["bg"],
                font=F["body"], relief="flat", borderwidth=1)
            menu.add_command(label="Save Figure As\u2026", command=lambda: self._save_figure_dialog(fig))
            menu.add_command(label="Copy to Clipboard", command=lambda: self._copy_figure_to_clipboard(fig))
            menu.add_separator()
            menu.add_command(label="Reset View", command=lambda: self._reset_figure_view(fig, canvas))
            menu.post(event.x_root, event.y_root)
        canvas.get_tk_widget().bind("<Button-2>", _on_right_click)
        canvas.get_tk_widget().bind("<Control-Button-1>", _on_right_click)

    def _save_figure_dialog(self, fig):
        path = filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("EPS", "*.eps")],
        )
        if path:
            fig.savefig(path, dpi=300, bbox_inches='tight')
            self.status_var.set(f"Figure saved: {os.path.basename(path)}")

    def _copy_figure_to_clipboard(self, fig):
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        try:
            import subprocess
            proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            proc.communicate(buf.read())
            self.status_var.set("Figure copied to clipboard")
        except Exception:
            self.status_var.set("Clipboard copy not available")

    def _reset_figure_view(self, fig, canvas):
        for ax in fig.axes:
            ax.autoscale()
        canvas.draw()

    def _save_current_figure(self):
        fig = self._get_active_figure()
        if fig:
            self._save_figure_dialog(fig)
        else:
            self.status_var.set("No figure to save")

    def _get_active_figure(self):
        try:
            active_tab = self.tabview.get()
        except Exception:
            return None

        if active_tab == "Setup":
            return self.setup_fig
        elif active_tab == "Diagnostics":
            return self.diag_fig
        elif active_tab == "Results":
            try:
                plot_tab = self.results_plot_tabview.get()
            except Exception:
                return self.overview_fig
            tab_map = {
                "Overview": self.overview_fig,
                "Trajectories": self.traj_fig,
                "Individual Flies": None,
                "Raincloud": self.raincloud_fig,
            }
            if plot_tab == "Individual Flies":
                try:
                    if self.individual_subtabs is None:
                        return None
                    sub = self.individual_subtabs.get()
                    sub_map = {
                        "Trajectory Overlay": self.overlay_fig,
                        "Metrics Heatmap": self.heatmap_fig,
                        "Speed Curves": self.speed_curves_fig,
                    }
                    return sub_map.get(sub, self.overlay_fig)
                except Exception:
                    return self.overlay_fig
            return tab_map.get(plot_tab, self.overview_fig)
        elif active_tab == "Statistics":
            return self.stats_fig
        return None

    # ------------------------------------------------------------------
    # Load existing results
    # ------------------------------------------------------------------
    def _load_results(self):
        path = filedialog.askopenfilename(
            title="Load Results",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            if path.endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load results:\n\n{e}")
            return

        self._show_tabs()
        self.controller.slopes_df = df
        result = {'slopes_df': df}
        self._populate_results(result)
        self._populate_statistics(result)
        self.tabview.set("Results")
        self._on_result_tab_changed()
        self.status_var.set(f"Loaded results: {os.path.basename(path)} ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Vial Grouping Dialog (Phase 3.1)
    # ------------------------------------------------------------------
    def _show_vial_grouping_dialog(self):
        n_vials = self.sl_vials.get()
        win = ctk.CTkToplevel(self)
        win.title("Assign Vial Groups")
        win.geometry("400x500")
        win.transient(self)
        win.configure(fg_color=C["bg"])

        ctk.CTkLabel(
            win, text="Assign Groups",
            font=(FONT_FAMILY, 18, "bold"), text_color=C["text"],
        ).pack(pady=(S["lg"], S["sm"]))
        ctk.CTkLabel(
            win, text="Assign each vial to a named group (e.g., control, mutant A)",
            font=F["caption"], text_color=C["text_dim"],
        ).pack(pady=(0, S["md"]))

        group_entries = {}
        scroll = ctk.CTkScrollableFrame(win, fg_color=C["bg_card"], corner_radius=8)
        scroll.pack(fill="both", expand=True, padx=S["md"], pady=S["sm"])

        existing = self.controller.config.get('vial_groups', {})
        for v in range(1, n_vials + 1):
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=S["xs"])
            ctk.CTkLabel(row, text=f"Vial {v}:", width=60, font=F["body"],
                         text_color=C["text"]).pack(side="left", padx=(0, S["sm"]))
            entry = ctk.CTkEntry(row, width=200, fg_color=C["bg_input"],
                                 border_color=C["border"], font=F["body"])
            entry.pack(side="left", fill="x", expand=True)
            default = existing.get(str(v), existing.get(v, ''))
            if default:
                entry.insert(0, str(default))
            group_entries[v] = entry

        # Control group selection
        ctrl_frame = ctk.CTkFrame(win, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=S["md"], pady=S["sm"])
        ctk.CTkLabel(ctrl_frame, text="Control group:", font=F["body"],
                     text_color=C["text"]).pack(side="left", padx=(0, S["sm"]))
        ctrl_entry = ctk.CTkEntry(ctrl_frame, width=200, fg_color=C["bg_input"],
                                  border_color=C["border"], font=F["body"])
        ctrl_entry.pack(side="left", fill="x", expand=True)
        ctrl_entry.insert(0, self.controller.config.get('control_group', ''))

        def save_groups():
            groups = {}
            for v, entry in group_entries.items():
                val = entry.get().strip()
                if val:
                    groups[v] = val
            self.controller.config['vial_groups'] = groups
            self.controller.config['control_group'] = ctrl_entry.get().strip()
            win.destroy()
            self.status_var.set(f"Groups assigned: {len(groups)} vials mapped")

        ctk.CTkButton(
            win, text="Save Groups", command=save_groups,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=(FONT_FAMILY, 14, "bold"),
            corner_radius=8, width=120, height=36,
        ).pack(pady=S["md"])

    # ------------------------------------------------------------------
    # Enhanced Statistics Controls (Phase 3.2)
    # ------------------------------------------------------------------
    def _build_stats_controls(self, tab):
        ctrl_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"], corner_radius=8)
        ctrl_frame.grid(row=0, column=0, sticky="ew", padx=S["xs"], pady=(S["xs"], 0))

        inner = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        inner.pack(fill="x", padx=S["sm"], pady=S["xs"])

        ctk.CTkLabel(inner, text="Test:", font=F["caption"], text_color=C["text_dim"]).pack(side="left", padx=(0, 2))
        self.stats_test_var = ctk.StringVar(value="Auto-detect")
        ctk.CTkOptionMenu(
            inner, variable=self.stats_test_var, width=130,
            values=["Auto-detect", "Dunnett's", "Pairwise", "Two-group", "ANOVA", "Kruskal-Wallis"],
            fg_color=C["bg_input"], button_color=C["accent"],
            button_hover_color=C["accent_hover"], font=(FONT_FAMILY, 11),
            command=lambda _: self._rerun_statistics(),
        ).pack(side="left", padx=(0, S["sm"]))

        ctk.CTkLabel(inner, text="Norm:", font=F["caption"], text_color=C["text_dim"]).pack(side="left", padx=(0, 2))
        self.stats_norm_var = ctk.StringVar(value="None")
        ctk.CTkOptionMenu(
            inner, variable=self.stats_norm_var, width=120,
            values=["None", "% of Control", "Z-score"],
            fg_color=C["bg_input"], button_color=C["accent"],
            button_hover_color=C["accent_hover"], font=(FONT_FAMILY, 11),
            command=lambda _: self._rerun_statistics(),
        ).pack(side="left", padx=(0, S["sm"]))

        ctk.CTkLabel(inner, text="Correction:", font=F["caption"], text_color=C["text_dim"]).pack(side="left", padx=(0, 2))
        self.stats_correction_var = ctk.StringVar(value="Holm-Bonferroni")
        ctk.CTkOptionMenu(
            inner, variable=self.stats_correction_var, width=140,
            values=["Holm-Bonferroni", "Benjamini-Hochberg", "Bonferroni", "None"],
            fg_color=C["bg_input"], button_color=C["accent"],
            button_hover_color=C["accent_hover"], font=(FONT_FAMILY, 11),
            command=lambda _: self._rerun_statistics(),
        ).pack(side="left", padx=(0, S["sm"]))

        ctk.CTkButton(
            inner, text="Assign Groups", width=100,
            fg_color=C["bg_hover"], hover_color=C["accent_muted"],
            text_color=C["accent"], border_color=C["accent"], border_width=1,
            corner_radius=6, font=(FONT_FAMILY, 11),
            command=self._show_vial_grouping_dialog,
        ).pack(side="right")

    def _rerun_statistics(self):
        if self.controller.slopes_df is not None:
            result = {'slopes_df': self.controller.slopes_df}
            if hasattr(self.controller, 'per_fly_metrics') and self.controller.per_fly_metrics is not None:
                result['per_fly_metrics'] = self.controller.per_fly_metrics
            if hasattr(self.controller, 'climbing_index') and self.controller.climbing_index is not None:
                result['climbing_index'] = self.controller.climbing_index
            result['has_individual_tracking'] = getattr(self.controller.detector, 'has_individual_tracking', False) if self.controller.detector else False
            self._populate_statistics(result)

    # ------------------------------------------------------------------
    # Log viewer (hidden by default, toggled from View menu)
    # ------------------------------------------------------------------
    def _build_log_viewer(self):
        self.log_frame = ctk.CTkFrame(self, height=120, fg_color=C["bg_card"], corner_radius=8)
        self.log_text = ctk.CTkTextbox(
            self.log_frame, height=100, font=F["mono"],
            state="disabled", wrap="word",
            fg_color=C["bg_card"], text_color=C["text_dim"],
        )
        self.log_text.pack(fill="both", expand=True, padx=S["xs"], pady=S["xs"])

        handler = _TextboxHandler(self.log_text, self)
        logging.getLogger().addHandler(handler)
        handler.setLevel(logging.INFO)

        import os
        log_path = os.path.expanduser('~/FreeClimber_debug.log')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(file_handler)

    def _toggle_log_viewer(self):
        if not hasattr(self, 'log_frame'):
            return
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

        inner = ctk.CTkFrame(overlay, fg_color=C["bg_card"], corner_radius=16, width=520, height=440)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        inner.pack_propagate(False)

        if self._sidebar_logo:
            ctk.CTkLabel(inner, image=self._sidebar_logo, text="").pack(pady=(S["xl"], S["sm"]))

        ctk.CTkLabel(
            inner, text="Welcome to FreeClimber",
            font=F["h1"], text_color=C["text"],
        ).pack(pady=(S["sm"], S["sm"]))

        ctk.CTkLabel(
            inner, text="Drosophila RING Assay Analysis",
            font=F["body"], text_color=C["accent"],
        ).pack(pady=(0, S["lg"]))

        steps = [
            ("1", "Load a Video", "Open or drag-drop a video file"),
            ("2", "Adjust Parameters", "Tweak detection settings, then Preview"),
            ("3", "Run Analysis", "Press RUN to process and view results"),
        ]

        for num, title, desc in steps:
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", padx=S["xl"], pady=S["sm"])
            badge = ctk.CTkFrame(row, width=28, height=28, fg_color=C["accent"], corner_radius=14)
            badge.pack(side="left")
            badge.pack_propagate(False)
            ctk.CTkLabel(badge, text=num, font=F["body_b"],
                text_color=C["text_on_accent"], anchor="center").pack(expand=True)
            text_frame = ctk.CTkFrame(row, fg_color="transparent")
            text_frame.pack(side="left", fill="x", padx=(S["md"], 0))
            ctk.CTkLabel(
                text_frame, text=title,
                font=F["body_b"], text_color=C["text"], anchor="w",
            ).pack(anchor="w")
            ctk.CTkLabel(
                text_frame, text=desc,
                font=F["caption"], text_color=C["text_dim"], anchor="w",
            ).pack(anchor="w")

        def dismiss():
            overlay.destroy()
            with open(marker, 'w') as f:
                f.write('done')

        ctk.CTkButton(
            inner, text="Get Started", width=160, height=40,
            fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["bg"], font=F["body_b"],
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
