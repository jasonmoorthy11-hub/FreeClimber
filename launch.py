"""Entry point for PyInstaller-packaged FreeClimber app."""
import multiprocessing
import os
import sys

# CRITICAL: Must be called before anything else in a PyInstaller bundle.
# Without this, every fork/spawn re-executes this script, opening new windows.
multiprocessing.freeze_support()

# When running from a PyInstaller bundle, resources are in _MEIPASS
if getattr(sys, '_MEIPASS', None):
    base = sys._MEIPASS
    sys.path.insert(0, base)
    os.chdir(base)
else:
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

sys.path.insert(0, os.path.join(base, 'scripts'))

from scripts.gui.app import main
main()
