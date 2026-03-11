"""FreeClimber v2.0 — customtkinter GUI.

Professional desktop app replacing the legacy wxPython interface.
Designed for non-technical lab members: sliders, tooltips, drag-and-drop,
dark mode, progressive disclosure.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Local imports — scripts/ is on sys.path
from gui.controller import AnalysisController

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Tooltip text for parameters
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
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#333333", foreground="#eeeeee",
                         relief=tk.SOLID, borderwidth=1,
                         font=("Helvetica", 11), wraplength=300, padx=6, pady=4)
        label.pack()

    def hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class ParameterSlider(ctk.CTkFrame):
    """Labeled slider with manual entry box and optional tooltip."""

    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, dtype=int, tooltip: str = "", **kwargs):
        super().__init__(parent, **kwargs)
        self.dtype = dtype

        self.label = ctk.CTkLabel(self, text=label, width=110, anchor="w")
        self.label.grid(row=0, column=0, padx=(4, 2), sticky="w")

        self.var = ctk.DoubleVar(value=default)
        self.slider = ctk.CTkSlider(self, from_=from_, to=to, variable=self.var,
                                     width=140, command=self._on_slide)
        self.slider.grid(row=0, column=1, padx=2)

        self.entry = ctk.CTkEntry(self, width=60, justify="center")
        self.entry.grid(row=0, column=2, padx=(2, 4))
        self.entry.insert(0, str(dtype(default)))
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

        if tooltip:
            info_btn = ctk.CTkLabel(self, text="ⓘ", width=20, cursor="hand2",
                                     text_color="#888888")
            info_btn.grid(row=0, column=3, padx=(0, 4))
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


class FreeClimberApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("FreeClimber v2.0")
        self.geometry("1280x800")
        self.minsize(1000, 650)

        self.controller = AnalysisController()
        self.video_meta = None
        self._roi_rect = None
        self._roi_press = False
        self._roi_x0 = self._roi_y0 = 0
        self._roi_x1 = self._roi_y1 = 0
        self.recent_files: list[str] = []

        self._build_menu_bar()
        self._build_layout()
        self._bind_shortcuts()

        self.status_var = ctk.StringVar(value="Ready")
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w",
                                        height=24, font=("Helvetica", 11))
        self.status_bar.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

    # ------------------------------------------------------------------
    # Menu / shortcuts
    # ------------------------------------------------------------------
    def _build_menu_bar(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video…  ⌘O", command=self._browse_video)
        file_menu.add_command(label="Load Config…", command=self._load_config)
        file_menu.add_command(label="Save Config…  ⌘S", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results…  ⌘E", command=self._export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Quit  ⌘Q", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

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
        # Main horizontal split: sidebar (left) + content (right)
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content()

    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self.main_frame, width=320, corner_radius=8)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 4), pady=0)
        self.sidebar = sidebar

        # --- VIDEO section ---
        ctk.CTkLabel(sidebar, text="VIDEO", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=8)
        ctk.CTkButton(btn_frame, text="Open Video…", width=140, command=self._browse_video).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_frame, text="Reload", width=80, fg_color="#555555", command=self._reload_video).pack(side="left")

        self.video_label = ctk.CTkLabel(sidebar, text="No video selected", text_color="#999999",
                                         font=("Helvetica", 11), wraplength=290, anchor="w")
        self.video_label.pack(anchor="w", padx=8, pady=(2, 4))

        # --- ROI section ---
        ctk.CTkLabel(sidebar, text="ROI", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        roi_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        roi_frame.pack(fill="x", padx=8)

        self.roi_x = self._labeled_entry(roi_frame, "X", "0", row=0, col=0)
        self.roi_y = self._labeled_entry(roi_frame, "Y", "0", row=0, col=2)
        self.roi_w = self._labeled_entry(roi_frame, "W", "0", row=1, col=0)
        self.roi_h = self._labeled_entry(roi_frame, "H", "0", row=1, col=2)

        self.fixed_roi_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Fixed ROI dimensions", variable=self.fixed_roi_var).pack(anchor="w", padx=8)

        # --- DETECTION section ---
        ctk.CTkLabel(sidebar, text="DETECTION", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=8, pady=(12, 2))

        self.sl_diameter = ParameterSlider(sidebar, "Diameter", 3, 31, 7, int, TOOLTIPS['diameter'])
        self.sl_diameter.pack(fill="x", padx=8, pady=1)
        self.sl_minmass = ParameterSlider(sidebar, "Min Mass", 0, 5000, 100, int, TOOLTIPS['minmass'])
        self.sl_minmass.pack(fill="x", padx=8, pady=1)
        self.sl_maxsize = ParameterSlider(sidebar, "Max Size", 1, 50, 11, int, TOOLTIPS['maxsize'])
        self.sl_maxsize.pack(fill="x", padx=8, pady=1)

        # Threshold: entry (can be "auto" or a number)
        th_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        th_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(th_frame, text="Threshold", width=110, anchor="w").pack(side="left")
        self.threshold_entry = ctk.CTkEntry(th_frame, width=80, justify="center")
        self.threshold_entry.pack(side="left", padx=4)
        self.threshold_entry.insert(0, "auto")
        info = ctk.CTkLabel(th_frame, text="ⓘ", width=20, cursor="hand2", text_color="#888888")
        info.pack(side="left")
        ToolTip(info, TOOLTIPS['threshold'])

        self.sl_ecc_low = ParameterSlider(sidebar, "Ecc Low", 0, 1, 0, float, TOOLTIPS['ecc_low'])
        self.sl_ecc_low.pack(fill="x", padx=8, pady=1)
        self.sl_ecc_high = ParameterSlider(sidebar, "Ecc High", 0, 1, 1, float, TOOLTIPS['ecc_high'])
        self.sl_ecc_high.pack(fill="x", padx=8, pady=1)

        # --- EXPERIMENT section ---
        ctk.CTkLabel(sidebar, text="EXPERIMENT", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=8, pady=(12, 2))

        # Background method
        bg_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        bg_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(bg_frame, text="Background", width=110, anchor="w").pack(side="left")
        self.bg_method_var = ctk.StringVar(value="temporal_median")
        ctk.CTkOptionMenu(bg_frame, variable=self.bg_method_var, width=150,
                           values=["temporal_median", "mog2", "running_avg"]).pack(side="left")

        self.sl_vials = ParameterSlider(sidebar, "Vials", 1, 20, 3, int, TOOLTIPS['vials'])
        self.sl_vials.pack(fill="x", padx=8, pady=1)
        self.sl_window = ParameterSlider(sidebar, "Window", 5, 500, 50, int, TOOLTIPS['window'])
        self.sl_window.pack(fill="x", padx=8, pady=1)

        # Frame rate + pixel_to_cm
        fr_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        fr_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(fr_frame, text="Frame Rate", width=110, anchor="w").pack(side="left")
        self.frame_rate_entry = ctk.CTkEntry(fr_frame, width=60, justify="center")
        self.frame_rate_entry.pack(side="left", padx=4)
        self.frame_rate_entry.insert(0, "30")
        ctk.CTkLabel(fr_frame, text="fps", text_color="#999999").pack(side="left")

        px_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        px_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(px_frame, text="Pixel/cm", width=110, anchor="w").pack(side="left")
        self.pixel_cm_entry = ctk.CTkEntry(px_frame, width=60, justify="center")
        self.pixel_cm_entry.pack(side="left", padx=4)
        self.pixel_cm_entry.insert(0, "1")

        # Checkboxes
        self.convert_cm_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Convert to cm/sec", variable=self.convert_cm_var).pack(anchor="w", padx=8, pady=1)
        self.trim_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Trim outliers", variable=self.trim_var).pack(anchor="w", padx=8, pady=1)
        self.tracking_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sidebar, text="Individual fly tracking", variable=self.tracking_var).pack(anchor="w", padx=8, pady=1)

        # Frame controls
        ctk.CTkLabel(sidebar, text="FRAMES", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=8, pady=(12, 2))
        self.sl_blank_0 = ParameterSlider(sidebar, "Blank Start", 0, 10000, 0, int, TOOLTIPS['blank_0'])
        self.sl_blank_0.pack(fill="x", padx=8, pady=1)
        self.sl_blank_n = ParameterSlider(sidebar, "Blank End", 0, 10000, 145, int, TOOLTIPS['blank_n'])
        self.sl_blank_n.pack(fill="x", padx=8, pady=1)
        self.sl_crop_0 = ParameterSlider(sidebar, "Crop Start", 0, 10000, 0, int, TOOLTIPS['crop_0'])
        self.sl_crop_0.pack(fill="x", padx=8, pady=1)
        self.sl_crop_n = ParameterSlider(sidebar, "Crop End", 0, 10000, 145, int, TOOLTIPS['crop_n'])
        self.sl_crop_n.pack(fill="x", padx=8, pady=1)
        self.sl_check = ParameterSlider(sidebar, "Check Frame", 0, 10000, 0, int, TOOLTIPS['check_frame'])
        self.sl_check.pack(fill="x", padx=8, pady=1)

        # Outlier trim
        self.sl_outlier_tb = ParameterSlider(sidebar, "Outlier TB", 0, 20, 1, float, TOOLTIPS['outlier_TB'])
        self.sl_outlier_tb.pack(fill="x", padx=8, pady=1)
        self.sl_outlier_lr = ParameterSlider(sidebar, "Outlier LR", 0, 20, 3, float, TOOLTIPS['outlier_LR'])
        self.sl_outlier_lr.pack(fill="x", padx=8, pady=1)

        # --- NAMING section ---
        ctk.CTkLabel(sidebar, text="NAMING", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=8, pady=(12, 2))
        nc_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        nc_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(nc_frame, text="Convention", width=110, anchor="w").pack(side="left")
        self.naming_entry = ctk.CTkEntry(nc_frame, width=160)
        self.naming_entry.pack(side="left", padx=4)

        vid_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        vid_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(vid_frame, text="Vial ID vars", width=110, anchor="w").pack(side="left")
        self.vial_id_entry = ctk.CTkEntry(vid_frame, width=60, justify="center")
        self.vial_id_entry.pack(side="left", padx=4)
        self.vial_id_entry.insert(0, "2")

        proj_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        proj_frame.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(proj_frame, text="Project Path", width=110, anchor="w").pack(side="left")
        self.project_entry = ctk.CTkEntry(proj_frame, width=160)
        self.project_entry.pack(side="left", padx=4)

        # --- ACTION BUTTONS ---
        ctk.CTkFrame(sidebar, height=2, fg_color="#444444").pack(fill="x", padx=8, pady=8)
        btn_actions = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_actions.pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(btn_actions, text="Test Parameters", width=140, fg_color="#555555",
                       command=self._test_parameters).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_actions, text="Save Config", width=100, fg_color="#555555",
                       command=self._save_config).pack(side="left")

        self.run_btn = ctk.CTkButton(sidebar, text="RUN ANALYSIS", height=40,
                                      font=("Helvetica", 15, "bold"),
                                      command=self._run_analysis)
        self.run_btn.pack(fill="x", padx=8, pady=(8, 4))

        self.progress_label = ctk.CTkLabel(sidebar, text="", text_color="#999999",
                                            font=("Helvetica", 11))
        self.progress_label.pack(anchor="w", padx=8)
        self.progress_bar = ctk.CTkProgressBar(sidebar, width=280)
        self.progress_bar.pack(padx=8, pady=(0, 8))
        self.progress_bar.set(0)

    def _build_content(self):
        content = ctk.CTkFrame(self.main_frame, corner_radius=8)
        content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(content)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._build_setup_tab()
        self._build_diagnostics_tab()
        self._build_results_tab()
        self._build_statistics_tab()

    def _build_setup_tab(self):
        tab = self.tabview.add("Setup")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.setup_fig = Figure(figsize=(8, 4), dpi=100)
        self.setup_fig.patch.set_facecolor('#2b2b2b')
        self.setup_canvas = FigureCanvasTkAgg(self.setup_fig, master=tab)
        self.setup_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(tab, fg_color="transparent", height=30)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.setup_toolbar = NavigationToolbar2Tk(self.setup_canvas, toolbar_frame)
        self.setup_toolbar.update()

        # Connect ROI drawing events
        self.setup_canvas.mpl_connect('button_press_event', self._roi_press_event)
        self.setup_canvas.mpl_connect('button_release_event', self._roi_release_event)
        self.setup_canvas.mpl_connect('motion_notify_event', self._roi_motion_event)

    def _build_diagnostics_tab(self):
        tab = self.tabview.add("Diagnostics")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.diag_fig = Figure(figsize=(10, 6), dpi=100)
        self.diag_fig.patch.set_facecolor('#2b2b2b')
        self.diag_canvas = FigureCanvasTkAgg(self.diag_fig, master=tab)
        self.diag_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(tab, fg_color="transparent", height=30)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.diag_canvas, toolbar_frame).update()

    def _build_results_tab(self):
        tab = self.tabview.add("Results")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Explanation text
        self.results_text = ctk.CTkTextbox(tab, height=100, font=("Helvetica", 11),
                                            state="disabled", wrap="word")
        self.results_text.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        self._set_results_text("Run an analysis to see results here.")

        # Slopes table (Treeview wrapped in a frame)
        table_frame = ctk.CTkFrame(tab)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Use tkinter Treeview for the data table
        style = self._configure_treeview_style()
        self.slopes_tree = tk.ttk.Treeview(table_frame, show="headings", style="Dark.Treeview")
        self.slopes_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.ttk.Scrollbar(table_frame, orient="vertical", command=self.slopes_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.slopes_tree.configure(yscrollcommand=scrollbar.set)

        # Results plot area with sub-tabs
        plot_frame = ctk.CTkFrame(tab)
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 4))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.results_plot_tabview = ctk.CTkTabview(plot_frame, height=280)
        self.results_plot_tabview.grid(row=0, column=0, sticky="nsew")

        # Speed plot
        speed_tab = self.results_plot_tabview.add("Speed")
        speed_tab.grid_rowconfigure(0, weight=1)
        speed_tab.grid_columnconfigure(0, weight=1)
        self.speed_fig = Figure(figsize=(6, 3), dpi=100)
        self.speed_fig.patch.set_facecolor('#2b2b2b')
        self.speed_canvas = FigureCanvasTkAgg(self.speed_fig, master=speed_tab)
        self.speed_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # X plot
        x_tab = self.results_plot_tabview.add("Horizontal (X)")
        x_tab.grid_rowconfigure(0, weight=1)
        x_tab.grid_columnconfigure(0, weight=1)
        self.x_fig = Figure(figsize=(6, 3), dpi=100)
        self.x_fig.patch.set_facecolor('#2b2b2b')
        self.x_canvas = FigureCanvasTkAgg(self.x_fig, master=x_tab)
        self.x_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Y plot
        y_tab = self.results_plot_tabview.add("Vertical (Y)")
        y_tab.grid_rowconfigure(0, weight=1)
        y_tab.grid_columnconfigure(0, weight=1)
        self.y_fig = Figure(figsize=(6, 3), dpi=100)
        self.y_fig.patch.set_facecolor('#2b2b2b')
        self.y_canvas = FigureCanvasTkAgg(self.y_fig, master=y_tab)
        self.y_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_statistics_tab(self):
        tab = self.tabview.add("Statistics")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self.stats_text = ctk.CTkTextbox(tab, font=("Courier", 12), state="disabled", wrap="word")
        self.stats_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._set_stats_text("Run an analysis to see statistics here.")

    # ------------------------------------------------------------------
    # Treeview dark style
    # ------------------------------------------------------------------
    def _configure_treeview_style(self):
        style = tk.ttk.Style()
        style.theme_use("default")
        style.configure("Dark.Treeview",
                         background="#333333",
                         foreground="#eeeeee",
                         fieldbackground="#333333",
                         rowheight=24,
                         font=("Helvetica", 11))
        style.configure("Dark.Treeview.Heading",
                         background="#444444",
                         foreground="#eeeeee",
                         font=("Helvetica", 11, "bold"))
        style.map("Dark.Treeview",
                   background=[("selected", "#1f6aa5")],
                   foreground=[("selected", "#ffffff")])
        return style

    # ------------------------------------------------------------------
    # Helper: labeled entry in a grid
    # ------------------------------------------------------------------
    def _labeled_entry(self, parent, label, default, row, col):
        ctk.CTkLabel(parent, text=label, width=20).grid(row=row, column=col, padx=2, sticky="e")
        entry = ctk.CTkEntry(parent, width=70, justify="center")
        entry.grid(row=row, column=col + 1, padx=2, pady=2)
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

        # Update GUI state
        folder, name_ext = os.path.split(path)
        name, ext = os.path.splitext(name_ext)
        self._file_suffix = ext.lstrip('.')

        self.video_label.configure(text=f"{name_ext}  ({self.video_meta['n_frames']} frames, "
                                        f"{self.video_meta['fps']:.0f} fps)")

        # Auto-set params from video
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

        # Show first frame in Setup tab
        self.setup_fig.clear()
        if self.video_meta.get('first_frame') is not None:
            ax1 = self.setup_fig.add_subplot(121)
            ax1.imshow(self.video_meta['first_frame'])
            ax1.set_title("Frame 0 (draw ROI)", color='white', fontsize=10)
            ax1.axis("off")

            # Set ROI to full frame by default
            h, w = self.video_meta['first_frame'].shape[:2]
            self._roi_x0, self._roi_y0 = 0, 0
            self._roi_x1, self._roi_y1 = w, h
            self._update_roi_entries()

        if self.video_meta.get('last_frame') is not None:
            ax2 = self.setup_fig.add_subplot(122)
            ax2.imshow(self.video_meta['last_frame'])
            ax2.set_title("Final frame (reference)", color='white', fontsize=10)
            ax2.axis("off")

        self.setup_fig.tight_layout()
        self.setup_canvas.draw()
        self.tabview.set("Setup")

        # Add to recent files
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

        # Prepare 6-panel diagnostic axes
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

        # Populate results if available
        if 'slopes_df' in result:
            self._populate_results(result['slopes_df'])
            self._populate_statistics(result['slopes_df'])

        self.status_var.set("Parameter testing complete")

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self.controller.detector is None:
            messagebox.showinfo("FreeClimber", "Load a video first.")
            return

        self.status_var.set("Running analysis...")
        self.run_btn.configure(state="disabled", text="Running...")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Processing...")
        self.update_idletasks()

        params = self._collect_params()
        self.diag_fig.clear()
        axes = [self.diag_fig.add_subplot(2, 3, i + 1) for i in range(6)]

        def on_done(result):
            self.after(0, lambda: self._on_analysis_done(result))

        def _worker():
            try:
                result = self.controller.test_parameters(params, axes)
                on_done(result)
            except Exception as e:
                on_done(e)

        threading.Thread(target=_worker, daemon=True).start()

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
            self._populate_results(result['slopes_df'])
            self._populate_statistics(result['slopes_df'])
            self.tabview.set("Results")

        self.progress_label.configure(text="Complete")
        self.status_var.set("Analysis complete — results saved")
        messagebox.showinfo("FreeClimber", "Analysis complete. Results saved to project folder.")

    # ------------------------------------------------------------------
    # Populate Results tab
    # ------------------------------------------------------------------
    def _populate_results(self, df: pd.DataFrame):
        # Update explanation
        self._set_results_text(
            "Results loaded.\n\n"
            "slope = climbing speed (pixels/frame, or cm/sec if conversion enabled)\n"
            "r_value = goodness-of-fit (closer to 1 = more linear climb)\n"
            "p_value = significance of linear trend\n"
            "std_err = uncertainty on slope"
        )

        # Clear and populate treeview
        self.slopes_tree.delete(*self.slopes_tree.get_children())
        cols = list(df.columns)
        self.slopes_tree["columns"] = cols
        for col in cols:
            self.slopes_tree.heading(col, text=col)
            max_width = max(len(col) * 10, 70)
            self.slopes_tree.column(col, width=min(max_width, 150), minwidth=50)

        for _, row in df.iterrows():
            values = [str(round(v, 4)) if isinstance(v, float) else str(v) for v in row]
            self.slopes_tree.insert("", "end", values=values)

        # Speed plot
        self.speed_fig.clear()
        ax = self.speed_fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        ycol = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                ycol = c
                break
        if ycol is None and len(df.columns) >= 2:
            ycol = df.columns[1]

        if ycol:
            y = pd.to_numeric(df[ycol], errors="coerce").values
            x = list(range(1, len(df) + 1))
            ax.bar(x, y, color="#1f6aa5", edgecolor="#3a8fd4", alpha=0.85)
            ax.set_xlabel("Vial", color='white')
            ax.set_ylabel(ycol, color='white')
            ax.set_title("Climbing Speed by Vial", color='white', fontsize=11)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.15)

        self.speed_canvas.draw()

        # X and Y position plots
        positions = self.controller.get_positions()
        if positions is not None:
            self._plot_position_timeseries(positions)

    def _plot_position_timeseries(self, pos_df: pd.DataFrame):
        cols = {c.lower(): c for c in pos_df.columns}

        def _pick(*cands):
            for c in cands:
                if c in cols:
                    return cols[c]
            return None

        frame_col = _pick('frame', 'frames', 't')
        vial_col = _pick('vial', 'vial_id')
        x_col = _pick('x', 'xpos')
        y_col = _pick('y', 'ypos')

        if frame_col is None:
            return

        # X plot
        self.x_fig.clear()
        ax = self.x_fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        if x_col and vial_col:
            grp = pos_df.groupby([frame_col, vial_col])[x_col].mean().reset_index()
            for v in sorted(grp[vial_col].unique()):
                sub = grp[grp[vial_col] == v]
                ax.plot(sub[frame_col].values, sub[x_col].values, label=str(v), alpha=0.7)
            ax.set_title("Mean X Position Over Time", color='white', fontsize=10)
            ax.set_xlabel("Frame", color='white')
            ax.set_ylabel("X", color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.15)
            if len(grp[vial_col].unique()) <= 12:
                ax.legend(fontsize=8)
        self.x_canvas.draw()

        # Y plot
        self.y_fig.clear()
        ax = self.y_fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        if y_col and vial_col:
            grp = pos_df.groupby([frame_col, vial_col])[y_col].mean().reset_index()
            for v in sorted(grp[vial_col].unique()):
                sub = grp[grp[vial_col] == v]
                ax.plot(sub[frame_col].values, sub[y_col].values, label=str(v), alpha=0.7)
            ax.set_title("Mean Y Position Over Time", color='white', fontsize=10)
            ax.set_xlabel("Frame", color='white')
            ax.set_ylabel("Y", color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.15)
            if len(grp[vial_col].unique()) <= 12:
                ax.legend(fontsize=8)
        self.y_canvas.draw()

    # ------------------------------------------------------------------
    # Populate Statistics tab
    # ------------------------------------------------------------------
    def _populate_statistics(self, df: pd.DataFrame):
        try:
            from analysis.stats import (
                check_normality, compare_two_groups, compare_multiple_groups,
                cohens_d, confidence_interval, publication_stats_table,
            )
        except ImportError:
            self._set_stats_text("Statistics module not available.")
            return

        # Find slope column and group column
        slope_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['slope', 'velocity', 'speed']):
                slope_col = c
                break
        if slope_col is None:
            self._set_stats_text("No slope/velocity column found in results.")
            return

        # Try to find a grouping column
        group_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in ['geno', 'genotype', 'group', 'condition', 'treatment']):
                group_col = c
                break

        lines = ["=" * 60, "  STATISTICAL ANALYSIS", "=" * 60, ""]

        if group_col and df[group_col].nunique() >= 2:
            # Build groups
            groups = {}
            for name, sub in df.groupby(group_col):
                vals = pd.to_numeric(sub[slope_col], errors='coerce').dropna().values
                if len(vals) > 0:
                    groups[str(name)] = vals

            if len(groups) >= 2:
                # Normality
                norm = check_normality(groups)
                lines.append("NORMALITY TEST (Shapiro-Wilk)")
                lines.append("-" * 40)
                for gname, res in norm['results'].items():
                    status = "normal" if res['normal'] else "NOT normal"
                    lines.append(f"  {gname}: W={res['statistic']}, p={res['p_value']} ({status})")
                lines.append(f"  All normal: {norm['all_normal']}")
                lines.append("")

                # Group comparison
                if len(groups) == 2:
                    keys = list(groups.keys())
                    result = compare_two_groups(groups[keys[0]], groups[keys[1]])
                    lines.append("TWO-GROUP COMPARISON")
                    lines.append("-" * 40)
                    lines.append(f"  Test: {result['test']}")
                    lines.append(f"  Statistic: {result['statistic']}")
                    lines.append(f"  p-value: {result['p_value']} {result['significance']}")
                    lines.append(f"  Cohen's d: {result['effect_size_d']}")
                    lines.append("")
                elif len(groups) >= 3:
                    result = compare_multiple_groups(groups)
                    lines.append(f"MULTI-GROUP COMPARISON ({len(groups)} groups)")
                    lines.append("-" * 40)
                    lines.append(f"  Test: {result['test']}")
                    lines.append(f"  Statistic: {result['statistic']}")
                    lines.append(f"  p-value: {result['p_value']}")
                    lines.append(f"  Effect size ({result['effect_size_name']}): {result['effect_size']}")
                    lines.append(f"  Significant: {result['significant']}")
                    lines.append("")

                    if result.get('post_hoc'):
                        lines.append(f"POST-HOC: {result['post_hoc_method']}")
                        lines.append("-" * 40)
                        for comp in result['post_hoc']:
                            lines.append(f"  {comp['group1']} vs {comp['group2']}: "
                                        f"p={comp['p_value']} {comp['significance']}, "
                                        f"d={comp['effect_size_d']}")
                        lines.append("")

                # Confidence intervals
                lines.append("GROUP SUMMARIES")
                lines.append("-" * 40)
                for gname, vals in groups.items():
                    ci = confidence_interval(vals)
                    lines.append(f"  {gname}: n={len(vals)}, mean={np.mean(vals):.4f} "
                                f"+/- {np.std(vals, ddof=1):.4f}, "
                                f"95% CI [{ci[0]}, {ci[1]}]")
        else:
            # No grouping — just descriptive stats
            vals = pd.to_numeric(df[slope_col], errors='coerce').dropna().values
            if len(vals) > 0:
                ci = confidence_interval(vals)
                lines.append("DESCRIPTIVE STATISTICS")
                lines.append("-" * 40)
                lines.append(f"  n = {len(vals)}")
                lines.append(f"  mean = {np.mean(vals):.4f}")
                lines.append(f"  std = {np.std(vals, ddof=1):.4f}")
                lines.append(f"  95% CI = [{ci[0]}, {ci[1]}]")
                lines.append(f"  min = {np.min(vals):.4f}")
                lines.append(f"  max = {np.max(vals):.4f}")

        lines.append("")
        lines.append("=" * 60)
        self._set_stats_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _set_results_text(self, text: str):
        self.results_text.configure(state="normal")
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", text)
        self.results_text.configure(state="disabled")

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
        win.geometry("400x300")
        win.transient(self)

        ctk.CTkLabel(win, text="Export Results", font=("Helvetica", 16, "bold")).pack(pady=(16, 8))

        formats = [
            ("Slopes CSV (backward compatible)", "csv"),
            ("Tidy CSV (R-ready)", "tidy"),
            ("GraphPad Prism CSV", "prism"),
            ("Per-fly Tracks CSV", "tracks"),
        ]

        fmt_var = ctk.StringVar(value="csv")
        for label, val in formats:
            ctk.CTkRadioButton(win, text=label, variable=fmt_var, value=val).pack(anchor="w", padx=24, pady=2)

        def do_export():
            fmt = fmt_var.get()
            ext = '.csv'
            path = filedialog.asksaveasfilename(
                parent=win,
                title="Save As",
                defaultextension=ext,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if path:
                try:
                    self.controller.export_results(fmt, path)
                    messagebox.showinfo("Exported", f"Results saved to:\n{path}", parent=win)
                    win.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Export failed:\n\n{e}", parent=win)

        ctk.CTkButton(win, text="Export", command=do_export).pack(pady=16)

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


def main():
    app = FreeClimberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
