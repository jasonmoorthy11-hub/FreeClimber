# -*- mode: python ; coding: utf-8 -*-
import sys


a = Analysis(
    ['launch.py'],
    pathex=[],
    binaries=[],
    datas=[('scripts', 'scripts')],
    hiddenimports=['customtkinter', 'PIL', 'cv2', 'trackpy', 'scipy', 'scipy.signal', 'scipy.stats', 'scipy.ndimage', 'scipy.optimize', 'scipy.interpolate', 'pandas', 'matplotlib', 'numpy', 'scripts.analysis.metrics', 'scripts.analysis.stats', 'scripts.analysis.quality', 'scripts.analysis.normalization', 'scripts.output.export', 'scripts.output.figures', 'scripts.output.database', 'scripts.output.reports', 'scripts.output.video', 'scripts.gui.controller'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FreeClimber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['FreeClimber.icns'] if sys.platform == 'darwin' else ['FreeClimber.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FreeClimber',
)
app = BUNDLE(
    coll,
    name='FreeClimber.app',
    icon='FreeClimber.icns',
    bundle_identifier='com.jasonmoorthy.freeclimber',
)
