"""Visual verification of FreeClimber GUI.

Launches the app, takes screenshots at key stages, and saves them
for inspection. Uses macOS screencapture (no browser needed).

Usage:
    python3 scripts/gui/visual_test.py [--output-dir /path/to/screenshots]
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def take_screenshot(output_dir, name):
    path = os.path.join(output_dir, f"freeclimber_{name}.png")
    subprocess.run(["screencapture", "-x", path], check=True)
    print(f"  Screenshot saved: {path}")
    return path


def run_visual_test(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Screenshots will be saved to: {output_dir}")

    import customtkinter as ctk
    from gui.app import FreeClimberApp

    ctk.set_appearance_mode("dark")

    app = FreeClimberApp()
    app._build_log_viewer()
    app._setup_drag_drop()
    screenshots = []

    def stage_launch():
        print("\n[Stage 1] App launched — taking screenshot...")
        app.update_idletasks()
        time.sleep(1)
        screenshots.append(take_screenshot(output_dir, "01_launch"))
        app.after(500, stage_resize)

    def stage_resize():
        print("[Stage 2] Resizing window...")
        app.geometry("1200x800")
        app.update_idletasks()
        time.sleep(0.5)
        screenshots.append(take_screenshot(output_dir, "02_resized"))
        app.after(500, stage_done)

    def stage_done():
        print(f"\nVisual test complete. {len(screenshots)} screenshots saved.")
        for s in screenshots:
            print(f"  {s}")
        app.destroy()

    app.after(2000, stage_launch)
    app.mainloop()


def main():
    parser = argparse.ArgumentParser(description="FreeClimber GUI visual test")
    parser.add_argument("--output-dir", default="/tmp/freeclimber_visual",
                        help="Directory for screenshots (default: /tmp/freeclimber_visual)")
    args = parser.parse_args()
    run_visual_test(args.output_dir)


if __name__ == "__main__":
    main()
