#!/usr/bin/env python
# -*- coding: utf-8 -*- 

## File name : FreeClimber_main.py
## Created by: Adam N. Spierer, Modified by Jason Moorthy
## Date      : May 2020
## Purpose   : Graphical User Interface wrapper for FreeClimber

## Version number
version = '0.4.0'
doi =  'https://doi.org/10.1242/jeb.229377' ## Link to published paper

## More universal modules
import wx
import os
import glob
import sys
import time
import argparse
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime

## Matplotlib configurations
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
matplotlib.use('WXAgg')

## Local imports
from detector import detector


## Set up wxPython application window
class App(wx.App):
    '''Create the main window and insert the custom frame'''
    def OnInit(self):
        frame = create(None, file_name)
        self.SetTopWindow(frame)
        frame.Show(True)
        return True

## Main GUI frame
class main_gui(wx.Frame):    
    '''Setting up the placement and '''
    def __init__(self, parent, file_path):
        if args.debug: print('main_gui.__init__')
        self.args = args
        self.initialize_controls(parent)
        self.box_sizer.Add(self.panel1, 0, wx.EXPAND)
        ## Main content area: notebook with Diagnostics + Results tabs
        self.notebook = wx.Notebook(self)
        self.box_sizer.Add(self.notebook, 1, wx.EXPAND)

        ## Diagnostics tab (existing matplotlib canvas)
        self.panel_diag = wx.Panel(self.notebook)
        self.notebook.AddPage(self.panel_diag, "Diagnostics")
        diag_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel_diag.SetSizer(diag_sizer)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.panel_diag, -1, self.figure)
        diag_sizer.Add(self.canvas, 1, wx.EXPAND)

        ## Results tab (persistent summary + table + plots)
        self.panel_results = wx.Panel(self.notebook)
        self.notebook.AddPage(self.panel_results, "Results")
        results_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel_results.SetSizer(results_sizer)

        # Results tab (persistent, non-overlapping): splitter = top (explain + table) / bottom (plot)
        self.results_splitter = wx.SplitterWindow(self.panel_results, style=wx.SP_LIVE_UPDATE | wx.SP_3D)
        results_sizer.Add(self.results_splitter, 1, wx.EXPAND | wx.ALL, 8)

        self.results_top = wx.Panel(self.results_splitter)
        self.results_bottom = wx.Panel(self.results_splitter)

        # --- Top pane: summary + explanations + slopes table ---
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        self.results_top.SetSizer(top_sizer)

        self.results_header = wx.StaticText(self.results_top, label='Results (run an analysis to populate)')
        top_sizer.Add(self.results_header, 0, wx.BOTTOM, 6)

        self.results_explain = wx.TextCtrl(self.results_top, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_NONE)
        self.results_explain.SetMinSize((-1, 110))
        self.results_explain.SetValue(
            """How to read these results:

• slopes.csv is the main output. Each row is one vial (plus an optional summary row).
• slope: estimated climbing speed (pixels/frame, or cm/sec if you enabled Convert to cm/sec).
• intercept: fitted y value at frame 0 for the regression window.
• r_value: goodness-of-fit of the straight-line model (closer to 1.0 = more linear).
• p_value: significance of the linear trend (smaller = more confident slope).
• std_err: uncertainty on slope (smaller = more precise).

QC tips:
• If slope is ~0 or negative, flies may not be climbing or tracking failed.
• Check Diagnostics: detections should sit on flies; vials should stay separated horizontally.
"""
        )
        top_sizer.Add(self.results_explain, 0, wx.EXPAND | wx.BOTTOM, 6)

        self.slopes_table = wx.ListCtrl(self.results_top, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.slopes_table.SetMinSize((-1, 240))
        top_sizer.Add(self.slopes_table, 1, wx.EXPAND)

        # --- Bottom pane: plots ---
        bottom_sizer = wx.BoxSizer(wx.VERTICAL)
        self.results_bottom.SetSizer(bottom_sizer)

        # Notebook so we can show multiple result plots clearly.
        self.results_plotbook = wx.Notebook(self.results_bottom)
        bottom_sizer.Add(self.results_plotbook, 1, wx.EXPAND)

        # Plot 1: slopes / climbing speed
        self.results_plot_panel_speed = wx.Panel(self.results_plotbook)
        sp_sizer = wx.BoxSizer(wx.VERTICAL)
        self.results_plot_panel_speed.SetSizer(sp_sizer)
        self.results_figure_speed = Figure(figsize=(5, 3))
        self.results_canvas_speed = FigureCanvas(self.results_plot_panel_speed, -1, self.results_figure_speed)
        sp_sizer.Add(self.results_canvas_speed, 1, wx.EXPAND)
        self.results_plotbook.AddPage(self.results_plot_panel_speed, "Speed")

        # Plot 2: mean horizontal position over time (QC for vial separation/drift)
        self.results_plot_panel_x = wx.Panel(self.results_plotbook)
        xp_sizer = wx.BoxSizer(wx.VERTICAL)
        self.results_plot_panel_x.SetSizer(xp_sizer)
        self.results_figure_x = Figure(figsize=(5, 3))
        self.results_canvas_x = FigureCanvas(self.results_plot_panel_x, -1, self.results_figure_x)
        xp_sizer.Add(self.results_canvas_x, 1, wx.EXPAND)
        self.results_plotbook.AddPage(self.results_plot_panel_x, "Horizontal (X)")

        # Plot 3: mean vertical position over time (mirrors Diagnostics plot, but zoomable here)
        self.results_plot_panel_y = wx.Panel(self.results_plotbook)
        yp_sizer = wx.BoxSizer(wx.VERTICAL)
        self.results_plot_panel_y.SetSizer(yp_sizer)
        self.results_figure_y = Figure(figsize=(5, 3))
        self.results_canvas_y = FigureCanvas(self.results_plot_panel_y, -1, self.results_figure_y)
        yp_sizer.Add(self.results_canvas_y, 1, wx.EXPAND)
        self.results_plotbook.AddPage(self.results_plot_panel_y, "Vertical (Y)")

        self.results_splitter.SplitHorizontally(self.results_top, self.results_bottom, sashPosition=360)
        self.results_splitter.SetMinimumPaneSize(150)

        # Auto-refresh Results when the user clicks the Results tab
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnNotebookPageChanged)

        ## Default rectangle variable
        self.pressed=False
        
        ## Initialize bottom text bar
        self.box_sizer.Add(self.status_bar, 0, border=0, flag=0)
        self.status_bar.SetStatusText('Ready', 0)
        rect = self.status_bar.GetFieldRect(0)
    
        ## Set up names
        self.video_file = file_path

        ## Create a dialog box at the beginning if the video path is not a real file
        if self.args.video_file == None:
            openFileDialog = wx.FileDialog(self, "Open Video file", "", "",
                                       "Video files (*.*)|*.*", 
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

            ## Closing program if Browse box is closed
            if openFileDialog.ShowModal() == wx.ID_CANCEL:
                print('\nExiting Program. Must select a video from box or enter path into command line\n')
                raise SystemExit

            ## Setting video name from dialog box selection
            self.video_file = openFileDialog.GetPath()        

        ## Passing individual inputs from GUI to a list
        self.update_names()
        self.input_names = ['x','y',
                            'w','h',
                            'check_frame','blank_0',
                            'blank_n','crop_0',
                            'crop_n','threshold',
                            'diameter','minmass',
                            'maxsize','ecc_low',
                            'ecc_high','vials',
                            'window','pixel_to_cm',
                            'frame_rate','vial_id_vars',
                            'outlier_TB','outlier_LR',
                            
                            'naming_convention','path_project','file_suffix',
                            
                            'convert_to_cm_sec','trim_outliers']
        self.parameter_names = ['self.input_'+item for item in self.input_names]
        self.input_values = [## int
                             self.input_x,self.input_y,
                             self.input_w,self.input_h,
                             self.input_check_frame,self.input_blank_0,
                             self.input_blank_n,self.input_crop_0,
                             self.input_crop_n,self.input_threshold,
                             self.input_diameter,self.input_minmass,
                             self.input_maxsize,self.input_ecc_low,
                             self.input_ecc_high,self.input_vials,
                             self.input_window,self.input_pixel_to_cm,
                             self.input_frame_rate,self.input_vial_id_vars,
                             self.input_outlier_TB,self.input_outlier_LR,
                             ## str
                             self.input_naming_convention,self.input_path_project,self.input_file_suffix,
                             ## bool
                             self.input_convert_to_cm_sec,self.input_checkBox_trim_outliers]

        ## Enable all buttons
        button_list = ['browse_video','reload_video','test_parameters','store_parameters']
        for button in button_list:
            exec('self.button_' + button + '.Enable(True)')

        ## Load video
        self.load_video()
        self.status_bar.SetStatusText("Ready...",0)
        if args.debug: print('End of init')
        return

    def update_variables(self):
        '''Updates the detection variables'''
        if args.debug: print('main_gui.update_variables')
        variables = []
        
        
        ## Including integers
        for item,jtem in zip(self.input_names[:22],self.input_values[:22]):
#             print('int',item,jtem)
            phrase = str(item + '='+ jtem.GetValue())
            if args.debug: print('    '+phrase)
            variables.append(phrase)

        ## Including strings - type I
        for item,jtem in zip(self.input_names[22:24],self.input_values[22:24]):
#             print('str-I',item,jtem)
            phrase = str(item + '="'+ jtem.GetValue()+'"')
            if args.debug: print('    '+phrase)
            variables.append(phrase)

        ## Including strings - type II
        for item,jtem in zip(self.input_names[24:25],self.input_values[24:25]):
#             print('str-II',item,jtem)
            phrase = str(item + '="'+ str(jtem)+'"')
            if args.debug: print(phrase)
            variables.append(phrase)
        
        ## Including booleans
        for item,jtem in zip(self.input_names[25:],self.input_values[25:]):
#             print('bool',item,jtem)
            phrase = str(item + '=%s' % str(jtem.GetValue()))
            if args.debug: print('    '+phrase)
            variables.append(phrase)

        return variables

    def load_video(self):
        '''Function for loading the video when the respective button is pressed'''
        if args.debug: print('main_gui.load_video')
        
        ## Set up
        self.figure.clear()
        self.axes = [
            self.figure.add_subplot(121),  # first frame (interactive ROI)
            self.figure.add_subplot(122)   # last frame (reference)
]

        
        ## Confirm file is a path, or folder has specified suffix
        status = self.check_specified_video()
        if status:
            print("Loading:", self.video_file)
            self.update_names()
            for item in self.input_values[:4]:
                item.SetEditable(True)
                item.Enable(True)
            self.checkBox_fixed_ROI.Enable(True)
            self.input_convert_to_cm_sec.Enable(True)

            ## Busy cursor while the detector object is called and initialized
            wx.BeginBusyCursor()
            try:
                vars = self.update_variables()
                self.detector = detector(self.video_file,
                                        gui=True,
                                        variables = vars)
            
                self.axes[0].imshow(self.detector.image_stack[0])
                self.axes[0].set_title("Frame 0 (ROI selection)")
                self.axes[0].axis("off")

                # Show last frame as reference
                try:
                    self.axes[1].imshow(self.detector.image_stack[-1])
                    self.axes[1].set_title("Final frame (reference)")
                    self.axes[1].axis("off")
                except:
                    self.axes[1].set_title("Final frame unavailable")
                    self.axes[1].axis("off")

                self.figure.tight_layout()
                self.figure.canvas.draw()

            finally:
                wx.EndBusyCursor()
            
            ## Setting mechanism for drawing the ROI rectangle
            self.canvas.Bind(wx.EVT_ENTER_WINDOW, self.ChangeCursor)
            self.canvas.mpl_connect('button_press_event', self.draw_rectangle)
            self.canvas.mpl_connect('button_release_event', self.on_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.rect = Rectangle((0,0), 1, 1, fill=False, ec='r')
            self.axes[0].add_patch(self.rect)

            ## Auto-set GUI parameters from the video
            self.input_blank_0.SetValue('0')
            self.input_blank_n.SetValue(str(self.detector.n_frames))
            self.input_crop_0.SetValue('0')
            self.input_crop_n.SetValue(str(self.detector.n_frames))
            self.input_check_frame.SetValue('0')
            self.input_ecc_low.SetValue('0')
            self.input_ecc_high.SetValue('1')
            self.input_ecc_high.SetValue('1')
            self.input_path_project.SetValue(self.folder)
            self.input_naming_convention.SetValue(self.name)
            self.input_vial_id_vars.SetValue(str(len(self.input_naming_convention.GetValue().split('_'))))
            
            ## Display the 0th and frame corresponding with (most likely) t = 2 seconds
            try:
                self.input_frame_rate = int(self.input_frame_rate.GetValue())
            except:
                pass
            if self.detector.n_frames < self.input_frame_rate*2:            
                self.input_check_frame.SetValue(str(self.detector.n_frames))
            
            ## Try to make the local linear regression window size 2 seconds, but if not then 35% of the frames in the video
            if self.detector.n_frames < self.input_frame_rate*2:
                self.input_window.SetValue(str(int(len(self.detector.image_stack) * .35)))                
            else:
                self.input_window.SetValue(str(int(self.input_frame_rate)*2))

            ## Enable Test parameter button if disabled from prior testing
            self.button_test_parameters.Enable(True)
            try:
                self.button_run_full_analysis.Enable(True)
            except:
                pass
            self.x0,self.y0 = 0, 0
            self.x1,self.y1 = self.detector.width,self.detector.height
        
            ## Display the first frame of the video in the GUI
            self.update_ROIdisp()
            self.canvas.draw()
        else:
            return

    def update_names(self):
        '''Updates the names of variables within the program. Generally variables set for naming files.'''
        if args.debug: print('main_gui.update_names')
        self.status_bar.SetStatusText("Updating file names...",0)
        # Display selected video path.
        # Earlier versions used a dedicated TextCtrl (a bright white bar) for this,
        # but we removed it to reclaim UI space. Keep the status bar as the single
        # source of truth for the currently selected file.
        self.folder, self.name        = os.path.split(self.video_file)
        self.name,   self.input_file_suffix = self.name.split('.')

        ## Naming files to be generated
        self.name_noext  = os.path.join(self.folder,self.name)
        self.path_data   = self.name_noext + '.raw.csv'
        self.path_filter = self.name_noext + '.filter.csv'
        self.path_plot   = self.name_noext + '.diag.png'
        self.path_slope  = self.name_noext + '.slopes.csv'
        if args.debug: print('name:',self.name_noext,"+ file suffixes")
                
        ## Set path_project default to the folder of the selected video file
        if self.input_path_project == '': self.input_path_project = self.folder

        # Update bottom-bar label (filename only; full path as tooltip).
        try:
            if hasattr(self, "video_path_label"):
                if self.video_file and os.path.isfile(self.video_file):
                    base = os.path.basename(self.video_file)
                    self.video_path_label.SetLabel(f"Video: {base}")
                    self.video_path_label.SetToolTip(self.video_file)
                else:
                    self.video_path_label.SetLabel("No video selected")
                    self.video_path_label.SetToolTip("")
        except Exception:
            pass
        return
        
    def check_specified_video(self):
        if args.debug: print('main_gui.check_specified_video')
        self.status_bar.SetStatusText("Checking specified video...",0)
    
        ## Check file path and update names
        if os.path.isfile(self.video_file):
            self.button_reload_video.Enable(True)
            self.button_test_parameters.Enable(True)
            self.update_names()
            self.input_file_suffix = '.' + self.video_file.split('/')[-1].split('.')[-1]
            return True
        
        else:
            self.video_file = "No or invalid file entered. Please change the file path"
            self.button_browse_video.Enable(True)
            return False

    ## Commands for drawing the ROI rectangle
    def ChangeCursor(self, event):
        '''Change cursor into crosshair type when enter the plot area'''
        self.canvas.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        return

    def draw_rectangle(self, event):
        '''Draw ROI rectangle'''
        self.status_bar.SetStatusText("Draw rectangle from upper-left to lower-right",0)
        self.pressed = True
        if self.checkBox_fixed_ROI.Enabled:
            try:
                self.x0 = int(event.xdata)
                self.y0 = int(event.ydata)
                
                ## If the fixed_ROI box is checked, handle values differently
                if self.checkBox_fixed_ROI.GetValue():
                    self.x1 = self.x0 + int(eval(self.input_w.GetValue()))
                    self.y1 = self.y0 + int(eval(self.input_h.GetValue()))
                    self.rect.set_width(self.x1 - self.x0)
                    self.rect.set_height(self.y1 - self.y0)
                    self.rect.set_xy((self.x0, self.y0))
                    self.canvas.draw()
                
                ## Set the values in the GUI and program to drawn rectangle
                self.input_x.SetValue(str(self.x0))
                self.input_y.SetValue(str(self.y0))
                self.input_h.SetValue(str(self.rect.get_height()))
                self.input_w.SetValue(str(self.rect.get_width()))
            except:
                pass
        return

    def on_release(self, event):
        '''When mouse is on plot and button is released, redraw ROI rectangle, update ROI values'''
        self.status_bar.SetStatusText("Specify the detector parameters...",0)
        self.pressed = False
        if self.checkBox_fixed_ROI.Enabled:
            if self.checkBox_fixed_ROI.GetValue():
                pass
            else:
                self.redraw_rect(event)
            self.update_ROIdisp()
        return

    def on_motion(self, event):
        '''If the mouse is on plot and if the mouse button is pressed, redraw ROI rectangle'''
        if self.pressed & self.checkBox_fixed_ROI.Enabled & (not self.checkBox_fixed_ROI.GetValue()):
            # Redraw the rectangle
            self.redraw_rect(event)
            self.update_ROIdisp()
        return

    def redraw_rect(self, event):
        '''Draw the ROI rectangle overlay'''
        try:
            x1 = int(event.xdata)
            y1 = int(event.ydata)
            if any([self.x1!=x1, self.y1!=y1]):                
                self.x1 = x1
                self.y1 = y1
                self.rect.set_xy((self.x0, self.y0))
                self.rect.set_width(self.x1 - self.x0)
                self.rect.set_height(self.y1 - self.y0)
                
                self.canvas.draw()
            else:
                pass
        except:
            pass
        return

    def update_ROIdisp(self):
        '''Updates the ROI coordinates as the rectangle is drawn.'''
        self.input_x.SetValue(str(self.x0))
        self.input_y.SetValue(str(self.y0))
        self.input_h.SetValue(str(int(self.y1) - int(self.y0)))
        self.input_w.SetValue(str(int(self.x1) - int(self.x0)))
        return

    def OnButton_testParButton(self, event):
        '''Tests the entered parameters when the `Test parameters` button is pressed'''
        if args.debug: print('main_gui.OnButton_testParButton')
        self.status_bar.SetStatusText("Testing parameters...",0)

        #Prep the parameters
        variables = self.update_variables()
        self.checkBox_fixed_ROI.Enable(False)

        ## Set up figure for plots
        self.figure.clear()
        self.axes = [self.figure.add_subplot(231),
                     self.figure.add_subplot(232),
                     self.figure.add_subplot(233),
                     self.figure.add_subplot(234),
                     self.figure.add_subplot(235),
                     self.figure.add_subplot(236)]

        ## Busy cursor while the main function runs
        wx.BeginBusyCursor()
        try:
            variables = variables + ['debug='+str(args.debug)]
            self.detector.parameter_testing(variables, self.axes)
        finally:
            wx.EndBusyCursor()

        ## Renders plots in the GUI
        self.figure.tight_layout()
        self.figure.canvas.draw()
        
        # Enable buttons and print statements once parameter testing is complete
        self.button_reload_video.Enable(True)
        self.button_store_parameters.Enable(True)
        if args.debug: print('Parameter testing complete')
        self.status_bar.SetStatusText("Refine detector parameters by reloading the video, or finish optimization by pressing 'Save configuration'",0)

        # Update Results tab if outputs were written
        try:
            self.populate_results_tab()
        except Exception:
            pass
        return

    def OnButton_strParButton(self, event):
        '''Runs the 'save_parameter' function for creating the configuration file'''
        if args.debug: print('main_gui.OnButton_strParButton')
        try:
            self.status_bar.SetStatusText("Saving configuration...", 0)
            self.save_parameter()
            # On macOS the pressed highlight can 'stick' unless we refresh + return focus
            wx.CallAfter(self.panel1.SetFocus)
            wx.CallAfter(self.button_store_parameters.SetFocus)
            wx.CallAfter(self.panel1.Refresh)
            wx.CallAfter(self.panel1.Update)
            self.status_bar.SetStatusText("Configuration saved.", 0)
            wx.MessageBox("Configuration saved (.cfg).", "FreeClimber", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.status_bar.SetStatusText("Error saving configuration (see terminal).", 0)
            wx.MessageBox(f"Error saving configuration:\n\n{e}", "FreeClimber", wx.OK | wx.ICON_ERROR)
        finally:
            try:
                event.Skip()
            except Exception:
                pass

    def _set_video_path_label(self, path):
        """Show a short filename in the bottom bar; keep the full path in a tooltip."""
        try:
            import os
            base = os.path.basename(path) if path else ""
            if not base:
                self.video_path_label.SetLabel("")
                self.video_path_label.SetToolTip("")
                return
            # truncate long names but preserve extension
            if len(base) > 60:
                root, ext = os.path.splitext(base)
                base = (root[:45] + "…" + root[-10:] + ext) if len(root) > 58 else (root[:57] + "…" + ext)
            self.video_path_label.SetLabel(base)
            self.video_path_label.SetToolTip(path)
        except Exception:
            pass


    def OnButton_runFullButton(self, event):
        '''Runs full processing (same pipeline as Test parameters) and writes output files.'''
        # The safest approach is to reuse the already-working GUI pipeline:
        # detector.parameter_testing(...) runs steps 1–7 and saves the outputs.
        try:
            debug = False
            try:
                debug = bool(getattr(globals().get('args', None), 'debug', False))
            except Exception:
                debug = False
            if debug:
                print('main_gui.OnButton_runFullButton')

            # Guardrail: ensure a video is selected
            video_file = getattr(self, 'video_file', None)
            if (video_file is None) or (str(video_file).strip() == ''):
                wx.MessageBox("Please select a video first (Step 1a: Browse...).",
                              "FreeClimber", wx.OK | wx.ICON_INFORMATION)
                return

            # Run the same workflow as the Test button (this already saves outputs)
            self.OnButton_testParButton(event)

            # Populate Results tab
            try:
                self.populate_results_tab()
                self.notebook.ChangeSelection(1)  # Results tab
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=20)
                # Make the failure visible (previous versions silently swallowed it)
                try:
                    self.results_explain.SetValue(
                        "Error rendering results. The analysis may have completed, but the GUI failed to load the output files.\n\n"
                        + str(e)
                        + "\n\n"
                        + tb
                    )
                except Exception:
                    pass
                wx.MessageBox(
                    "Analysis finished, but the Results view couldn't be rendered.\n\n" + str(e),
                    "FreeClimber",
                    wx.OK | wx.ICON_WARNING,
                )

            # If we reached here without throwing, analysis completed
            try:
                proj_path = getattr(getattr(self, 'detector', None), 'path_project', None)
                if proj_path is None:
                    proj_path = self.input_project_path.GetValue()
            except Exception:
                proj_path = None

            if proj_path:
                wx.MessageBox("Full analysis complete.\n\nOutputs saved in:\n%s\n\nOpen the Results tab to view slopes + plot inside the app." % str(proj_path),
                              "FreeClimber", wx.OK | wx.ICON_INFORMATION)
            else:
                wx.MessageBox("Full analysis complete. Outputs saved to the project folder.",
                              "FreeClimber", wx.OK | wx.ICON_INFORMATION)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            wx.MessageBox("Run failed.\n\n%s\n\n(See terminal for full traceback.)" % str(e),
                          "FreeClimber", wx.OK | wx.ICON_ERROR)

    def _find_output_paths(self, det):
        """Return the most likely outputs (slopes + key diagnostic images).

        The upstream pipeline has a few naming variations (e.g. base.slopes.csv vs slopes.csv
        sitting in the project folder). This resolver first checks the expected "base.*" files,
        then falls back to scanning the project folder for the newest matching outputs.
        """
        paths = {}

        # 1) Try the canonical base prefix used by the pipeline
        base = getattr(det, 'path_noext', None)
        if base is None:
            proj = getattr(det, 'path_project', None) or getattr(det, 'project_path', None)
            name = getattr(det, 'name', None)
            if proj and name:
                base = os.path.join(str(proj), str(name))

        if base:
            candidates = {
                "slopes": [base + ".slopes.csv", base + "_slopes.csv", base + "-slopes.csv", os.path.join(os.path.dirname(base), "slopes.csv")],
                "raw": [base + ".raw.csv", base + "_raw.csv", base + "-raw.csv"],
                "filter": [base + ".filter.csv", base + "_filter.csv", base + "-filter.csv", base + ".filtered.csv", base + "_filtered.csv"],
                "diagnostic": [base + ".diagnostic.png", base + ".diag.png", os.path.join(os.path.dirname(base), "diagnostic.png")],
                "frame0": [base + ".processed_frame0.png", base + "_processed_frame0.png"],
                "finalframe": [base + ".processed_finalframe.png", base + "_processed_finalframe.png"],
            }
            for k, lst in candidates.items():
                for p in lst:
                    if p and os.path.exists(p):
                        paths[k] = p
                        break

        # 2) Fallback: scan the project folder for the newest relevant outputs
        proj_dir = None
        try:
            proj_dir = getattr(det, 'path_project', None) or getattr(det, 'project_path', None)
        except Exception:
            proj_dir = None
        try:
            if hasattr(self, 'input_project_path') and self.input_project_path is not None:
                gui_proj = self.input_project_path.GetValue()
                if gui_proj:
                    proj_dir = gui_proj
        except Exception:
            pass

        if proj_dir and os.path.isdir(proj_dir):
            # newest slopes-like csv
            if "slopes" not in paths:
                slope_glob = sorted(glob.glob(os.path.join(proj_dir, "**", "*slopes*.csv"), recursive=True),
                                   key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                                   reverse=True)
                if slope_glob:
                    paths["slopes"] = slope_glob[0]

            # newest raw/filter per-frame CSVs (if exported)
            if "raw" not in paths:
                raw_glob = sorted(glob.glob(os.path.join(proj_dir, "**", "*raw*.csv"), recursive=True),
                                  key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                                  reverse=True)
                # avoid picking slopes/config as raw
                raw_glob = [p for p in raw_glob if "slopes" not in os.path.basename(p).lower() and "config" not in os.path.basename(p).lower()]
                if raw_glob:
                    paths["raw"] = raw_glob[0]

            if "filter" not in paths:
                filt_glob = sorted(glob.glob(os.path.join(proj_dir, "**", "*filter*.csv"), recursive=True) +
                                   glob.glob(os.path.join(proj_dir, "**", "*filtered*.csv"), recursive=True),
                                   key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                                   reverse=True)
                filt_glob = [p for p in filt_glob if "slopes" not in os.path.basename(p).lower() and "config" not in os.path.basename(p).lower()]
                if filt_glob:
                    paths["filter"] = filt_glob[0]

            # newest diagnostic png
            if "diagnostic" not in paths:
                diag_glob = sorted(glob.glob(os.path.join(proj_dir, "**", "*diagnostic*.png"), recursive=True),
                                  key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                                  reverse=True)
                if diag_glob:
                    paths["diagnostic"] = diag_glob[0]

            # reference frames if present
            if "frame0" not in paths:
                f0_glob = sorted(glob.glob(os.path.join(proj_dir, "**", "*frame0*.png"), recursive=True),
                                key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                                reverse=True)
                if f0_glob:
                    paths["frame0"] = f0_glob[0]
            if "finalframe" not in paths:
                ff_glob = sorted(glob.glob(os.path.join(proj_dir, "**", "*finalframe*.png"), recursive=True) +
                                 glob.glob(os.path.join(proj_dir, "*final_frame*.png")),
                                 key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                                 reverse=True)
                if ff_glob:
                    paths["finalframe"] = ff_glob[0]

        return paths

    def _load_positions_df(self, paths):
        """Load per-frame fly positions from the filter/raw CSV if available.

        Returns (df, frame_col, vial_col, x_col, y_col) or (None, None, None, None, None)
        if the file/columns can't be inferred.
        """
        try:
            import pandas as pd
        except Exception:
            return None, None, None, None, None

        csv_path = paths.get("filter") or paths.get("raw")
        if not csv_path or not os.path.exists(csv_path):
            return None, None, None, None, None

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None, None, None, None, None

        # Infer likely columns
        cols = {c.lower(): c for c in df.columns}

        def pick(*cands):
            for c in cands:
                if c in cols:
                    return cols[c]
            return None

        frame_col = pick("frame", "frames", "t", "time", "frame_idx")
        vial_col = pick("vial", "vial_id", "vialid", "tube", "roi", "roi_id", "channel")

        # x/y columns: prefer centroid naming, then generic x/y.
        x_col = pick("x", "xpos", "x_pos", "x_position", "centroid_x", "cx")
        y_col = pick("y", "ypos", "y_pos", "y_position", "centroid_y", "cy")

        # If still missing, try any column that looks like x/y but avoid obvious non-position fields.
        if x_col is None:
            for c in df.columns:
                cl = c.lower()
                if "x" in cl and not any(bad in cl for bad in ["max", "min", "std", "err", "index"]):
                    x_col = c
                    break
        if y_col is None:
            for c in df.columns:
                cl = c.lower()
                if "y" in cl and not any(bad in cl for bad in ["max", "min", "std", "err", "index"]):
                    y_col = c
                    break

        if frame_col is None or x_col is None or y_col is None:
            return None, None, None, None, None

        return df, frame_col, vial_col, x_col, y_col

    def populate_results_tab(self):
        """Load outputs (slopes + plots) into the Results tab."""
        det = getattr(self, 'detector', None)
        if det is None:
            self.results_header.SetLabel("Results (no detector object yet)")
            return

        paths = self._find_output_paths(det)
        import time

        def _wait_for_file(p, tries=10, delay=0.2):
            """Wait briefly for a file to appear/non-empty (helps right after analysis)."""
            if p is None:
                return None
            for _ in range(tries):
                try:
                    if os.path.exists(p) and os.path.getsize(p) > 0:
                        return p
                except Exception:
                    pass
                time.sleep(delay)
            return p

        slopes_path = _wait_for_file(paths.get("slopes", None))
        positions_path = _wait_for_file(paths.get("positions", None))
        diag_img_path = _wait_for_file(paths.get("diagnostic", None))

        # Header
        proj_path = getattr(det, 'path_project', None)
        if proj_path:
            self.results_header.SetLabel(f"Results saved in: {proj_path}")
        else:
            self.results_header.SetLabel("Results")

        # Clear table
        self.slopes_table.ClearAll()

        if slopes_path is None:
            self.results_explain.SetValue(
                "No slopes.csv found yet. Run analysis, then check the output folder."
            )
            for fig, canvas in [
                (self.results_figure_speed, self.results_canvas_speed),
                (self.results_figure_x, self.results_canvas_x),
                (self.results_figure_y, self.results_canvas_y),
            ]:
                try:
                    fig.clear()
                    canvas.draw()
                except Exception:
                    pass
            return

        # Load slopes
        try:
            import pandas as pd  # local import so the app can still open if pandas isn't installed globally
        except Exception as e:
            self.results_explain.SetValue(
                "Could not import pandas, which is required to display results.\n\n"
                f"Error: {e}"
            )
            return

        try:
            df = pd.read_csv(slopes_path)
        except Exception as e:
            self.results_explain.SetValue(
                "Found slopes.csv but could not read it.\n\n"
                f"File: {slopes_path}\n"
                f"Error: {e}"
            )
            return


        explain_text = """How to read these results:

• slopes.csv: one row per vial (and sometimes a summary row).
  - slope: estimated climbing speed. Units depend on your settings (pixels/frame, or cm/sec if you enabled conversion).
  - intercept: y-value at frame 0 for the fit (mainly QC).
  - r_value: goodness-of-fit (closer to 1 = more linear climb).
  - p_value / std_err: regression diagnostics.
  - first_frame / last_frame: frame window used for the linear fit.

Quick QC tips:
• Near-zero slope can mean no climbing OR detection failed.
• Low r_value often means noisy tracking (threshold too low/high, glare, crowded vials).
"""
        self.results_explain.SetValue(explain_text)

        # Table columns
        for i, col in enumerate(df.columns):
            self.slopes_table.InsertColumn(i, col)

        # Table rows
        for row_i in range(len(df)):
            self.slopes_table.InsertItem(row_i, str(df.iloc[row_i, 0]))
            for col_i in range(1, len(df.columns)):
                self.slopes_table.SetItem(row_i, col_i, str(df.iloc[row_i, col_i]))

        # Auto-size columns (cap width)
        for i in range(len(df.columns)):
            self.slopes_table.SetColumnWidth(i, wx.LIST_AUTOSIZE_USEHEADER)
            w = self.slopes_table.GetColumnWidth(i)
            self.slopes_table.SetColumnWidth(i, min(max(w, 70), 220))

        # Plot: try to find a slope/velocity column
        # --- Plot 1: speed per vial (from slopes.csv)
        self.results_figure_speed.clear()
        ax = self.results_figure_speed.add_subplot(111)

        # Pick a y column
        ycol = None
        for c in df.columns:
            cl = c.lower()
            if "slope" in cl or "velocity" in cl or "speed" in cl:
                ycol = c
                break
        if ycol is None and len(df.columns) >= 2:
            ycol = df.columns[1]

        x = list(range(1, len(df) + 1))
        try:
            y = pd.to_numeric(df[ycol], errors="coerce").values
        except Exception:
            y = df[ycol].values

        ax.plot(x, y, marker="o")
        ax.set_title("Climbing speed by vial")
        ax.set_xlabel("Vial")
        ax.set_ylabel(str(ycol))
        ax.grid(True, alpha=0.2)

        self.results_canvas_speed.draw()

        # --- Plot 2/3: mean horizontal/vertical position over time (from filter/raw.csv)
        pos_df, frame_col, vial_col, x_col, y_col = self._load_positions_df(paths)
        if pos_df is not None and frame_col and (x_col or y_col):
            try:
                import pandas as pd
            except Exception:
                pd = None

            # Mean X over time (per vial)
            self.results_figure_x.clear()
            axx = self.results_figure_x.add_subplot(111)
            if x_col:
                grp = pos_df.groupby([frame_col, vial_col])[x_col].mean().reset_index()
                for v in sorted(grp[vial_col].unique()):
                    sub = grp[grp[vial_col] == v]
                    axx.plot(sub[frame_col].values, sub[x_col].values, label=str(v))
                axx.set_title("Mean horizontal position (x) over time")
                axx.set_xlabel("Frame")
                axx.set_ylabel(x_col)
                axx.grid(True, alpha=0.2)
                # Only show legend if manageable
                if len(grp[vial_col].unique()) <= 12:
                    axx.legend(loc="best", fontsize=7)
            else:
                axx.text(0.5, 0.5, "No x-position column found in filter/raw CSV", ha="center", va="center")
                axx.set_axis_off()
            self.results_canvas_x.draw()

            # Mean Y over time (per vial)
            self.results_figure_y.clear()
            axy = self.results_figure_y.add_subplot(111)
            if y_col:
                grp = pos_df.groupby([frame_col, vial_col])[y_col].mean().reset_index()
                for v in sorted(grp[vial_col].unique()):
                    sub = grp[grp[vial_col] == v]
                    axy.plot(sub[frame_col].values, sub[y_col].values, label=str(v))
                axy.set_title("Mean vertical position (y) over time")
                axy.set_xlabel("Frame")
                axy.set_ylabel(y_col)
                axy.grid(True, alpha=0.2)
                if len(grp[vial_col].unique()) <= 12:
                    axy.legend(loc="best", fontsize=7)
            else:
                axy.text(0.5, 0.5, "No y-position column found in filter/raw CSV", ha="center", va="center")
                axy.set_axis_off()
            self.results_canvas_y.draw()
        else:
            # If positions aren't available, clear the X/Y plots so the UI isn't confusing.
            for fig, canvas in [
                (self.results_figure_x, self.results_canvas_x),
                (self.results_figure_y, self.results_canvas_y),
            ]:
                try:
                    fig.clear()
                    canvas.draw()
                except Exception:
                    pass

        # Update explanation with plain-English labeling
        self.results_explain.SetValue(
            "How to read these outputs:\n"
            "\n"
            "Main result (slopes.csv)\n"
            "• Each row = one vial (or one group, depending on naming).\n"
            "• slope = climbing speed (in pixels/frame, or cm/sec if you enable Convert to cm/sec).\n"
            "• r_value/p_value/std_err describe how well a straight line fit explains the climbing trajectory.\n"
            "\n"
            "Quality control (Diagnostics tab + X/Y plots)\n"
            "• Mean Y vs time should generally increase if flies are climbing.\n"
            "• Mean X vs time should stay roughly flat (large drifts suggest vial mixing or camera/ROI issues).\n"
            "• If speed is near 0 and Y is flat, flies didn’t climb or detections failed."
        )

        # Force a relayout/refresh so the wx + matplotlib canvases reliably render
        # on macOS (especially when switching tabs right after analysis).
        try:
            self.results_panel.Layout()
            self.results_panel.SendSizeEvent()
            self.results_panel.Refresh()
        except Exception:
            pass

    def OnNotebookPageChanged(self, event):
        """Refresh Results when the Results tab is selected."""
        try:
            idx = event.GetSelection()
            try:
                label = self.notebook.GetPageText(idx)
            except Exception:
                label = ""
            if label.strip().lower() == "results":
                self.populate_results_tab()
        finally:
            event.Skip()

    def set_config_file(self):
        '''Set path for the project folder'''
        if args.debug: print('main_gui.OnButton_strParButton')
        ## Figure out where to save configuration file
        if os.path.isdir(self.input_path_project):
            if not self.input_path_project.endswith('/'):
                self.input_path_project = self.input_path_project + '/'
            self.path_parameters = self.input_path_project + self.name + '.cfg'
        else:
            self.path_parameters = self.path_noext+'.cfg'
        return self.path_parameters

    def save_parameter(self):
        if args.debug: print('main_gui.save_parameter')
        '''
        Save parameters as python list.
        New parameter sets appended to the config file.
        Each parameter sets come with a comment line, contain the datetime of analysis
        '''

        variables = self.update_variables()
        try: self.input_path_project = self.input_path_project.GetValue()
        except: pass

        self.path_parameters = self.set_config_file()

        ## Printing output to configuration file
        print('Saving parameters to:', self.path_parameters)
        with open(self.path_parameters, 'w') as f:
            print('## FreeClimber ##', file=f)
        f.close()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path_parameters,'a') as f:
            print('## Generated from file: '+self.video_file,file = f)
            print('##     @ ' + now, file = f)
            print('##',file = f)
            print('## Analysis parameters:',file = f)
            for item in variables:
                print(item,file = f)
        f.close()
        print("Configuration settings saved")
        return

    def OnButton_Browse(self, event):
        if args.debug: print('main_gui.OnButton_Browse')
        openFileDialog = wx.FileDialog(self, "Open Video file", "", "",
                                       "Video files (*.*)|*.*", 
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            pass
        else:
            self.video_file = openFileDialog.GetPath()
            self.update_names()
            self.load_video()
        self.figure.clear()
        return

    def OnButton_LoadVideo(self,event):
        '''Calls function to load the video when the `reload` button is pressed'''
        if args.debug: print('main_gui.OnButton_LoadVideo')
        self.load_video()
        return
        
    def initialize_sizers(self):
        '''Initializes the GUI window orientation'''
        if args.debug: print('main_gui.initialize_sizers')
        # Generated method, do not edit
        self.box_sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self.SetSizer(self.box_sizer)
        return

    def initialize_controls(self, prnt):
        '''Initializes the GUI controls, for which there are many'''
        if args.debug: print('main_gui.initialize_controls')
        # Generated method, do not edit
        wx.Frame.__init__(self, id=wxID_text_title, name='', parent=prnt,
              pos=wx.Point(100, 30), size=wx.Size(950, 759),
              style=wx.DEFAULT_FRAME_STYLE,
              title='FreeClimber')
        self.SetClientSize(wx.Size(950, 737))

        ######
        ## Inputs for ROI Rectangle
        self.panel1 = wx.Panel(id=wxID_panel_1, name='panel1',
              parent=self, pos=wx.Point(0, 0), size=wx.Size(950, 260),
              style=wx.TAB_TRAVERSAL)
       
        ## Step 1 boxes
        self.text_step_1a = wx.StaticText(id=wxID_text_step_1a,
              label=u'Step 1a: Specify a video', name='text_step_1a', parent=self.panel1, 
              pos=wx.Point(col1,10),
              size=wx.Size(box_dimensions),style = wx.ALIGN_CENTER)
                      
        ## Browse
        self.button_browse_video = wx.Button(id=wxID_browse_video,
              label=u'Browse...', name=u'button_browse_video', parent=self.panel1,
              pos=wx.Point(col1, 30), size=wx.Size(box_dimensions), style=0)
        self.button_browse_video.Bind(wx.EVT_BUTTON, self.OnButton_Browse,
              id=wxID_browse_video)
        
        self.text_step_1b = wx.StaticText(id=wxID_text_step_1b,
              label=u'Step 1b: Define options', name='text_step_1b', parent=self.panel1, 
              pos=wx.Point(col1,65),
              size=wx.Size(box_dimensions),style = wx.ALIGN_CENTER)

        ## Pixel to cm
        self.text_pixel_to_cm = wx.StaticText(id=wxID_text_pixel_to_cm,
              label=u"Pixels / cm:", name='text_pixel_to_cm',
              parent=self.panel1, pos=wx.Point(col1, 85), size=wx.Size(medium_box_dimensions),
              style=0)
        self.input_pixel_to_cm = wx.TextCtrl(id=wxID_input_pixel_to_cm,
              name=u'input_pixel_to_cm', parent=self.panel1, pos=wx.Point(col1 + 95, 85), 
              size=wx.Size(medium_box_dimensions), style=0, value=u"1")
        
        ## Frame Rate    
        self.text_frame_rate = wx.StaticText(id=wxID_frame_rate,
              label=u'Frames / sec:', name='text_frame_rate',
              parent=self.panel1, pos=wx.Point(col1,115), size=wx.Size(box_dimensions),
              style=0)
        self.input_frame_rate = wx.TextCtrl(id=wxID_frame_rate,
              name=u'input_frame_rate', parent=self.panel1, pos=wx.Point(col1 + 95,115),
               size=wx.Size(medium_box_dimensions), style=0, value='25')
        
        ## Check box to convert final slope to cm
        self.input_convert_to_cm_sec = wx.CheckBox(id=wxID_input_convert_to_cm_sec,
              label=u'Convert to cm / sec', name=u'input_convert_to_cm_sec',
              parent=self.panel1, pos=wx.Point(col1, 145), size=wx.Size(250,22),
              style=0)

        ## Step 2 boxes
        self.text_step_2 = wx.StaticText(id=wxID_text_step_2,
              label=u'Step 2: Select ROI', name='text_step_2', parent=self.panel1, 
              pos=wx.Point(col2,10),
              size=wx.Size(box_dimensions),style = wx.ALIGN_LEFT)
        ## X
        self.text_x = wx.StaticText(id=wxID_text_x,
              label=u'x-pos.', name='text_x', parent=self.panel1, 
              pos=wx.Point(col2, 30),
              size=wx.Size(medium_box_dimensions),style = wx.ALIGN_LEFT)
        self.input_x = wx.TextCtrl(id=wxID_input_x,
              name=u'input_x', parent=self.panel1, 
              style=0, value=u'0',
              pos=wx.Point(col2 + 55, 30),size=wx.Size(medium_box_dimensions))
        
        ## Y      
        self.text_y = wx.StaticText(id=wxID_text_y,
              label=u'y-pos.', name='text_y', parent=self.panel1,
              pos=wx.Point(col2, 55), size=wx.Size(medium_box_dimensions), style=wx.ALIGN_LEFT)
        self.input_y = wx.TextCtrl(id=wxID_input_y,
              name=u'input_y', parent=self.panel1, pos=wx.Point(col2 + 55,55),
              size=wx.Size(medium_box_dimensions), style=0, value=u'0')
        
        ## Width
        self.text_w = wx.StaticText(id=wxID_text_w,
              label=u'Width:', name='text_w', parent=self.panel1,
              pos=wx.Point(col2, 80), size=wx.Size(medium_box_dimensions), style=wx.ALIGN_LEFT)
        self.input_w = wx.TextCtrl(id=wxID_input_w,
              name=u'input_w', parent=self.panel1, pos=wx.Point(col2 + 55, 80),
              size=wx.Size(medium_box_dimensions), style=0, value=u'0')
        
        ## Height
        self.text_h = wx.StaticText(id=wxID_text_h,
              label=u'Height:', name='text_h', parent=self.panel1,
              pos=wx.Point(col2,105), size=wx.Size(medium_box_dimensions), style=wx.ALIGN_LEFT)
        self.input_h = wx.TextCtrl(id=wxID_input_h,
              name=u'input_h', parent=self.panel1, pos=wx.Point(col2 + 55, 105),
               size=wx.Size(medium_box_dimensions), style=0, value=u'0')
        
        ## ROI rectangle stays same dimensions but can be redrawn. Not critical to keep
        self.checkBox_fixed_ROI = wx.CheckBox(id=wxID_check_box_ROI,
              label=u'Fixed ROI Size?', name=u'checkBox_fixed_ROI',
              parent=self.panel1, pos=wx.Point(col2, 145), size=wx.Size(250,22),
              style=0)
        self.checkBox_fixed_ROI.SetValue(False)
        ######
        
       ## Detection parameters
        self.text_step_3 = wx.StaticText(id=wxID_text_step_3,
              label=u'Step 3: Specify spot parameters', name='text_step_3', parent=self.panel1, 
              pos=wx.Point(col3,10),
              size=wx.Size(100,22),style = wx.ALIGN_LEFT)
                      
        ## Expected spot diameter
        self.text_diameter = wx.StaticText(id=wxID_text_diameter,
              label=u'Diameter:', name='text_diameter', parent=self.panel1,
              pos=wx.Point(col3 ,30), size=wx.Size(medium_box_dimensions), style=0)
        self.input_diameter = wx.TextCtrl(id=wxID_input_diameter,
              name=u'input_diameter', parent=self.panel1, pos=wx.Point(col3 + 100,30), 
              size=wx.Size(medium_box_dimensions), style=0, value=u'7')
        
        ## Maximum spot diameter
        self.text_maxsize = wx.StaticText(id=wxID_text_maxsize,
              label=u'MaxDiameter:', name='text_maxsize', parent=self.panel1,
              pos=wx.Point(col3,55), size=wx.Size(medium_box_dimensions), style=0)
        self.input_maxsize = wx.TextCtrl(id=wxID_input_maxsize,
              name=u'input_maxsize', parent=self.panel1, pos=wx.Point(col3 + 100,55), 
              size=wx.Size(medium_box_dimensions), style=0, value=u'11')

        ## Minimum spot 'mass'
        self.text_minmass = wx.StaticText(id=wxID_text_minmass,
              label=u'MinMass:', name='text_minmass', parent=self.panel1,
              pos=wx.Point(col3,80), size=wx.Size(medium_box_dimensions), style=0)
        self.input_minmass = wx.TextCtrl(id=wxID_input_minmass,
              name=u'input_minmass', parent=self.panel1, pos=wx.Point(col3 + 100,80), 
              size=wx.Size(medium_box_dimensions), style=0, value=u'100')

        ## Spot threshold
        self.text_threshold = wx.StaticText(id=wxID_text_threshold,
              label=u'Threshold:', name='text_threshold', parent=self.panel1,
              pos=wx.Point(col3, 105), size=wx.Size(medium_box_dimensions), style=0)
        self.input_threshold = wx.TextCtrl(id=wxID_input_threshold,
              name=u'input_threshold', parent=self.panel1, pos=wx.Point(col3 + 100,105),
              size=wx.Size(medium_box_dimensions), style=0, value=u'"auto"')
        
        ## Eccentricity range
        self.text_ecc = wx.StaticText(id=wxID_text_ecc,
              label=u'Ecc/circularity:', name='text_ecc',
              parent=self.panel1, pos=wx.Point(col3, 130), size=wx.Size(medium_box_dimensions),
              style=0)
        self.input_ecc_low = wx.TextCtrl(id=wxID_input_ecc_low,
              name=u'input_ecc_low', parent=self.panel1, pos=wx.Point(col3+ 100, 130),
            size=wx.Size(small_box_dimensions), style=0, value=u'0')
        self.input_ecc_high = wx.TextCtrl(id=wxID_input_ecc_high,
              name=u'input_ecc_high', parent=self.panel1, pos=wx.Point(col3 + 140,130),
              size=wx.Size(small_box_dimensions), style=0, value=u'0')        
        
        #### Step 4 arguments
        ## Check frames
        self.text_step_4 = wx.StaticText(id=wxID_text_step_4,
              label=u'Step 4: Additional parameters', name='text_step_4', parent=self.panel1, 
              pos=wx.Point(col4,10),
              size=wx.Size(100,22),style = wx.ALIGN_LEFT)
        
        ## Background frames
        self.text_background_frames = wx.StaticText(id=wxID_text_background_frames,
              label=u'Background frames:', name='text_background_frames',
              parent=self.panel1, pos=wx.Point(col4, 30), size=wx.Size(medium_box_dimensions),
              style=0)
        self.input_blank_0 = wx.TextCtrl(id=wxID_input_blank_0,
              name=u'input_blank_0', parent=self.panel1, pos=wx.Point(col4+ 130, 30),
            size=wx.Size(small_box_dimensions), style=0, value=u'0')
        self.input_blank_n = wx.TextCtrl(id=wxID_input_blank_n,
              name=u'input_blank_n', parent=self.panel1, pos=wx.Point(col4 + 170,30),
              size=wx.Size(small_box_dimensions), style=0, value=u'0')

        ## crop frames
        self.text_crop_frames = wx.StaticText(id=wxID_text_crop_frames,
              label=u'Crop frames:', name='text_crop_frames',
              parent=self.panel1, pos=wx.Point(col4, 55), size=wx.Size(medium_box_dimensions),
              style=0)
        self.input_crop_0 = wx.TextCtrl(id=wxID_input_crop_0,
              name=u'input_crop_0', parent=self.panel1, pos=wx.Point(col4+ 130, 55),
            size=wx.Size(small_box_dimensions), style=0, value=u'0')
        self.input_crop_n = wx.TextCtrl(id=wxID_input_crop_n,
              name=u'input_crop_n', parent=self.panel1, pos=wx.Point(col4 + 170,55),
              size=wx.Size(small_box_dimensions), style=0, value=u'0')

        ## Check frames
        self.text_check_frames = wx.StaticText(id=wxID_text_check_frames,
              label=u'Check frame:', name='text_check_frames', parent=self.panel1,
              pos=wx.Point(col4, 80), size=wx.Size(115, 17), style=0)

        self.input_check_frame = wx.TextCtrl(id=wxID_input_check_frame,
              name=u'input_check_frame', parent=self.panel1, pos=wx.Point(col4 + 130, 80),
               size=wx.Size(small_box_dimensions), style=0, value=u'0')
        
        ## Vials
        self.text_vials = wx.StaticText(id=wxID_text_vials,
              label=u'Number of vials:', name='text_vials',
              parent=self.panel1, pos=wx.Point(col4, 105), size=wx.Size(133,22),
              style=0)
        self.input_vials = wx.TextCtrl(id=wxID_input_vials,
              name=u'input_vials', parent=self.panel1, pos=wx.Point(col4+130,105), 
              size=wx.Size(small_box_dimensions), style=0, value=u'1')
        
        ## Window size
        self.text_window = wx.StaticText(id=wxID_text_window,
              label=u'Window size:', name='text_window',
              parent=self.panel1, pos=wx.Point(col4, 130), size=wx.Size(133,22),
              style=0)
        self.input_window = wx.TextCtrl(id=wxID_input_window,
              name=u'input_window', parent=self.panel1, pos=wx.Point(col4+130,130), 
              size=wx.Size(small_box_dimensions), style=0, value='1')    
          

        ## Edge trim
        self.input_checkBox_trim_outliers = wx.CheckBox(id=wxID_check_box_outlier,
              label=u'Trim outliers? (TB                      LR)', name=u'checkBox_outlier',
              parent=self.panel1, pos=wx.Point(col4, 155), size=wx.Size(250,22),
              style=0)
        self.input_checkBox_trim_outliers.SetValue(False)
        self.input_outlier_TB = wx.TextCtrl(id=wxID_outlier_TB,
              name=u'input_outlier_TB', parent=self.panel1, pos=wx.Point(col4+130,155),
            size=wx.Size(small_box_dimensions), style=0, value=u'1')
        self.input_outlier_LR = wx.TextCtrl(id=wxID_outlier_LR,
              name=u'input_outlier_LR', parent=self.panel1, pos=wx.Point(col4+170,155),
              size=wx.Size(small_box_dimensions), style=0, value=u'3')


        self.text_step_5 = wx.StaticText(id=wxID_text_step_5,
              label=u'Step 5: Naming parameters', name='text_step_5', parent=self.panel1, 
              pos=wx.Point(col5,10),
              size=wx.Size(100,22),style = wx.ALIGN_LEFT)

        ## Naming convention
        self.text_naming_convention = wx.StaticText(id=wxID_text_naming_convention,
              label=u"Naming pattern:", name='text_naming_convention',
              parent=self.panel1, pos=wx.Point(col5, 30), size=wx.Size(medium_box_dimensions),
              style=0)
        self.input_naming_convention = wx.TextCtrl(id=wxID_input_naming_convention,
              name=u'input_naming_convention', parent=self.panel1, pos=wx.Point(col5,50), 
              size=wx.Size(large_box_dimensions), style=0, value='') 
        
        ## Variables
        self.text_vial_id_vars = wx.StaticText(id=wxID_text_vial_id_vars,
              label=u'Vial_ID variables:', name='text_vial_id_vars',
              parent=self.panel1, pos=wx.Point(col5, 80), size=wx.Size(133, 22),
              style=0)
        self.input_vial_id_vars = wx.TextCtrl(id=wxID_input_vial_id_vars,
              name=u'input_vial_id_vars', parent=self.panel1, pos=wx.Point(col5 + 130, 80), 
              size=wx.Size(small_box_dimensions), style=0, value=u'2')  
        
        self.text_path_project = wx.StaticText(id=wxID_text_path_project,
              label=u"Project path:", name='text_path_project',
              parent=self.panel1, pos=wx.Point(col4-45, 180), size=wx.Size(medium_box_dimensions),
              style=0)
        self.input_path_project = wx.TextCtrl(id=wxID_input_path_project,
              name=u'input_path_project', parent=self.panel1, pos=wx.Point(col4 + 40, 180), 
              size=wx.Size(350,22), style=0, value='') 



        ## Bottom bar: selected video + Run analysis (kept visible; never overlaps)
        # NOTE: panel1 is now taller; place this bar lower so it can't collide with the button row.
        # Give this bar enough height so the button and text never collide on macOS.
        self.bottom_bar = wx.Panel(self.panel1, pos=wx.Point(10, 228), size=wx.Size(1480, 52))
        # Match macOS dark appearance by inheriting the parent background
        try:
            self.bottom_bar.SetBackgroundColour(self.panel1.GetBackgroundColour())
        except Exception:
            pass

        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left: prominent Run analysis button (replaces the old white path bar)
        self.btn_run_full = wx.Button(self.bottom_bar, wx.ID_ANY, "RUN ANALYSIS")
        self.btn_run_full.SetMinSize(wx.Size(190, 34))
        try:
            f = self.btn_run_full.GetFont()
            f.SetWeight(wx.FONTWEIGHT_BOLD)
            self.btn_run_full.SetFont(f)
        except Exception:
            pass
        self.btn_run_full.SetToolTip("Run full analysis and save outputs into the Project path folder")

        # Right: short video filename (full path in tooltip)
        self.video_path_label = wx.StaticText(self.bottom_bar, label="No video selected")
        self.video_path_label.SetForegroundColour(wx.Colour(230, 230, 230))
        self.video_path_label.SetToolTip("")

        bottom_sizer.Add(self.btn_run_full, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 8)
        bottom_sizer.AddStretchSpacer(1)
        bottom_sizer.Add(self.video_path_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 8)

        # Force layout so sizers take effect even with absolute-position parent panel.
        self.bottom_bar.SetSizer(bottom_sizer)
        self.bottom_bar.Layout()

        # Bind directly to the button (DO NOT bind by id=wx.ID_ANY; that can fire from other buttons)
        self.btn_run_full.Bind(wx.EVT_BUTTON, self.OnButton_runFullButton)
        self.button_test_parameters = wx.Button(id=wxID_test_parameters,
              label=u'Test parameters', name=u'button_test_parameters',
              parent=self.panel1, pos=wx.Point(col1,180), size=wx.Size(140, 22),
              style=wx.ALIGN_CENTER)
        self.button_test_parameters.Bind(wx.EVT_BUTTON, self.OnButton_testParButton,
              id=wxID_test_parameters)

        self.button_reload_video = wx.Button(id=wxID_reload_video,
              label=u'Reload video', name=u'button_reload_video', parent=self.panel1,
              pos=wx.Point(col1 + 160*1, 180), size=wx.Size(140, 22), style=wx.ALIGN_CENTER)
        self.button_reload_video.Bind(wx.EVT_BUTTON, self.OnButton_LoadVideo,
              id=wxID_reload_video)
              
        self.button_store_parameters = wx.Button(id=wxID_store_parameters,
              label=u'Save configuration', name=u'button_store_parameters',
              parent=self.panel1, pos=wx.Point(col1 + 160*2, 180), size=wx.Size(140, 22),
              style=wx.ALIGN_CENTER)
        self.button_store_parameters.Bind(wx.EVT_BUTTON, self.OnButton_strParButton,
              id=wxID_store_parameters)

        # Run button moved into the bottom bar (see "Bottom bar" block above)


        ## Text box at the bottom
        self.status_bar = wx.StatusBar(id=wxID_status_bar,
              name='status_bar', parent=self, style=0)
        self.initialize_sizers()
        return   

## Create the GUI
def create(parent, file_path):
    '''Begins running the main GUI object
    ----
    Inputs:
      file_path (str): path to the video file
    ----
    Returns:
      main_gui (object): begins running the main_gui object
    '''
    if args.debug: print('main_gui.create')
    if file_path==None:
        file_path='Select a video to begin'
    return main_gui(parent, file_path)

## Print startup message and setup argument parser
def startup():
    '''Function prints top line and runs argument parsing function.
    ----
    Inputs:
      None
    ----
    Returns:
      args (list): list of arguments passed to program
    '''
    def print_line(line,line_length):
        '''Formats line to print'''
        if len(line) <= line_length: string = line + '#'*(line_length-len(line))
        else: string = line
        print(string)
        return

    line_length = 72
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line0 = '#'*line_length
    line1 = '## FreeClimber v.%s ' % str(version)
    line2 = '## Please cite: %s ' % doi
    line3 = "## Beginning program @ %s " % str(now)
    line4 = line0

    ## Printing formated lines
    print('\n')
    for item in range(5):
        print_line(eval('line'+str(item)),line_length)

    args = define_argument_parser()
    return args

def define_argument_parser():
    '''Parses out arguments (which are admittedly few for the GUI. Use the help flag '-h' for more information when running the program.
    ----
    Inputs:
      None
    ----
    Returns:
      args (list): list of arguments passed to program
    '''
    parser = argparse.ArgumentParser(prog='FreeClimber',
                                    description='Calculates the climbing velocity of flies in a negative geotaxis assay',
                                    usage='%(prog)s [options] path',
                                    epilog='For documentation and a tutorial, see https://github.com/adamspierer/FreeClimber',
                                    allow_abbrev=False)

    ## Specifying file (video file)
    parser.add_argument('--video_file', 
                        type=str, 
                        required=False, 
                        help="Path to file")

    ## Debug printing
    parser.add_argument('--debug', 
                        required=False, 
                        default=False, 
                        action='store_true',
                        help="Prints for every function and output being run. Not very pretty")

    ## Final parser specifications
    parser.version = version ## Program version
    parser.add_argument('-v','--version', 
                        action='version')
    args = parser.parse_args()
    return args

## Check video argument arguments
def check_args(args):
    '''Confirms video path is valid, and if not hecks arguments passed to parser and kills program if invalid path.
    ----
    Inputs:
      args (list): Arguments passed to parsed
    ----
    Returns:
      args.video_file (str): Video file path if valid, or None if not.
    '''
    if args.debug: print('main_gui.check_args')
    try:
        if os.path.isfile(args.video_file):
            print('--> Video file: ',args.video_file)
        else:
            print('--> No video file specified, select from Browser...')
            args.video_file = None
    except:
        args.video_file = None
    return args.video_file


## Initializing wx text boxes and input fields
[wxID_text_step_1a,wxID_text_step_1b,wxID_text_step_2,wxID_text_step_3,wxID_text_step_4,wxID_text_step_5,
wxID_panel_1, wxID_text_title, 
wxID_browse_video, wxID_reload_video, wxID_store_parameters,
wxID_run_full_analysis, wxID_test_parameters, wxID_display_parameters,
wxID_video_path, 
wxID_frame_rate,
wxID_status_bar, 
wxID_text_minmass,wxID_input_minmass, 
wxID_text_threshold, wxID_input_threshold,
wxID_text_maxsize, wxID_input_maxsize,
wxID_text_diameter, wxID_input_diameter,
wxID_text_ecc, wxID_input_ecc_low, wxID_input_ecc_high,
wxID_text_x, wxID_input_x,
wxID_text_y, wxID_input_y,
wxID_text_h, wxID_input_h,
wxID_text_w, wxID_input_w,
wxID_text_check_frames, wxID_input_check_frame, 
wxID_text_background_frames,wxID_input_blank_0, wxID_input_blank_n,
wxID_text_crop_frames,wxID_input_crop_0,wxID_input_crop_n,
wxID_text_vials, wxID_input_vials,
wxID_text_window, wxID_input_window,
wxID_text_pixel_to_cm,wxID_input_pixel_to_cm,
wxID_check_box_outlier,wxID_outlier_TB,wxID_outlier_LR,
wxID_text_naming_convention,wxID_input_naming_convention,
wxID_text_vial_id_vars,wxID_input_vial_id_vars,
wxID_text_path_project,wxID_input_path_project,
wxID_input_convert_to_cm_sec,wxID_check_box_ROI] = [wx.ID_ANY for item in range(61)] 


## Basic GUI sizes and spacers
col1,col2,col3,col4,col5 = 10,180,320,525,750
small_box_dimensions = 35,22
medium_box_dimensions = 50,22
box_dimensions = 100,22
large_box_dimensions = 165,22

## Datetime
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__=='__main__':
    args = startup()
    file_name = check_args(args)
    app = App(0)
    app.MainLoop()
