"""Annotated video export for FreeClimber.

Overlay detected particles, ROI boundaries, vial lines, and trajectory trails
onto the original video.  Output as H.264 .mp4.
"""

import logging
import os

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Okabe-Ito in BGR for OpenCV
VIAL_COLORS_BGR = [
    (0, 159, 230),   # orange  #E69F00
    (233, 180, 86),  # sky blue #56B4E9
    (115, 158, 0),   # green   #009E73
    (66, 228, 240),  # yellow  #F0E442
    (178, 114, 0),   # blue    #0072B2
    (0, 94, 213),    # vermillion #D55E00
    (167, 121, 204), # pink    #CC79A7
    (255, 255, 255), # white fallback
]


def export_annotated_video(
    video_path: str,
    output_path: str,
    positions_df: pd.DataFrame | None = None,
    roi: tuple | None = None,
    vials: int = 1,
    fps: float | None = None,
    trail_length: int = 10,
    show_particles: bool = True,
    show_roi: bool = True,
    show_vial_lines: bool = True,
    show_frame_counter: bool = True,
    show_trails: bool = True,
):
    """Export annotated video with detection overlays.

    Args:
        video_path: path to original video
        output_path: output .mp4 path
        positions_df: DataFrame with columns [frame, x, y, vial] and optionally [particle]
        roi: (x, y, w, h) tuple for ROI rectangle
        vials: number of vials
        fps: output frame rate (auto-detect from source if None)
        trail_length: number of previous frames to show as trajectory trail
        show_particles: overlay circles on detected spots
        show_roi: draw ROI rectangle
        show_vial_lines: draw vial boundary lines
        show_frame_counter: overlay frame number
        show_trails: draw trajectory trails for tracked particles
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        fps = src_fps if src_fps > 0 else 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise OSError(f"Cannot create output video: {output_path}")

    # Pre-index positions by frame for fast lookup
    frame_data = {}
    has_particle = False
    if positions_df is not None:
        cols = {c.lower(): c for c in positions_df.columns}
        frame_col = _pick_col(cols, 'frame', 'frames', 't')
        x_col = _pick_col(cols, 'x', 'xpos')
        y_col = _pick_col(cols, 'y', 'ypos')
        vial_col = _pick_col(cols, 'vial', 'vial_id')
        particle_col = _pick_col(cols, 'particle', 'fly_id')
        has_particle = particle_col is not None

        if frame_col and x_col and y_col:
            for f, grp in positions_df.groupby(frame_col):
                pts = []
                for _, row in grp.iterrows():
                    pt = {
                        'x': int(row[x_col]),
                        'y': int(row[y_col]),
                        'vial': int(row[vial_col]) if vial_col else 1,
                    }
                    if has_particle:
                        pt['particle'] = int(row[particle_col])
                    pts.append(pt)
                frame_data[int(f)] = pts

    # Build particle trail history
    trail_history = {}  # particle_id -> [(x, y), ...]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI rectangle
        if show_roi and roi:
            rx, ry, rw, rh = roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        # Vial lines
        if show_vial_lines and roi and vials > 1:
            rx, ry, rw, rh = roi
            for v in range(1, vials):
                vx = rx + int(v * rw / vials)
                cv2.line(frame, (vx, ry), (vx, ry + rh), (255, 255, 255), 1, cv2.LINE_AA)

        # Particles
        if show_particles and frame_idx in frame_data:
            for pt in frame_data[frame_idx]:
                vial_idx = min(pt['vial'] - 1, len(VIAL_COLORS_BGR) - 1)
                color = VIAL_COLORS_BGR[max(0, vial_idx)]
                cv2.circle(frame, (pt['x'], pt['y']), 5, color, 2, cv2.LINE_AA)

                # Build trail
                if show_trails and has_particle and 'particle' in pt:
                    pid = pt['particle']
                    if pid not in trail_history:
                        trail_history[pid] = []
                    trail_history[pid].append((pt['x'], pt['y']))
                    if len(trail_history[pid]) > trail_length:
                        trail_history[pid] = trail_history[pid][-trail_length:]

        # Draw trails
        if show_trails and has_particle:
            for _pid, trail in trail_history.items():
                if len(trail) < 2:
                    continue
                for i in range(1, len(trail)):
                    alpha = int(255 * i / len(trail))
                    cv2.line(frame, trail[i - 1], trail[i], (alpha, alpha, 255), 1, cv2.LINE_AA)

        # Frame counter
        if show_frame_counter:
            text = f"Frame {frame_idx}/{total_frames}"
            time_sec = frame_idx / fps if fps > 0 else 0
            text += f"  ({time_sec:.1f}s)"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    logger.info(f"Annotated video saved: {output_path} ({frame_idx} frames)")


def export_annotated_stills(
    video_path: str,
    output_dir: str,
    frames: list[int] | None = None,
    positions_df: pd.DataFrame | None = None,
    roi: tuple | None = None,
    vials: int = 1,
):
    """Export specific frames as annotated PNG stills.

    Args:
        frames: list of frame indices to export (default: first, middle, last)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames is None:
        frames = [0, total // 2, max(0, total - 1)]

    os.makedirs(output_dir, exist_ok=True)

    for target_frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        # ROI
        if roi:
            rx, ry, rw, rh = roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        # Vial lines
        if roi and vials > 1:
            rx, ry, rw, rh = roi
            for v in range(1, vials):
                vx = rx + int(v * rw / vials)
                cv2.line(frame, (vx, ry), (vx, ry + rh), (255, 255, 255), 1)

        out_path = os.path.join(output_dir, f"frame_{target_frame:06d}.png")
        cv2.imwrite(out_path, frame)
        logger.info(f"Still saved: {out_path}")

    cap.release()


def _pick_col(cols_lower: dict, *candidates) -> str | None:
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None
