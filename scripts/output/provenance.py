"""Analysis provenance tracking for FreeClimber.

Generates a JSON sidecar documenting software version, parameters,
dependency versions, and input video fingerprint for reproducibility.
"""

import hashlib
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

MAX_HASH_BYTES = 10 * 1024 * 1024  # 10 MB


def _file_sha256(path, max_bytes=MAX_HASH_BYTES):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            data = f.read(max_bytes)
            h.update(data)
    except Exception as e:
        logger.warning("Could not hash %s: %s", path, e)
        return None
    return h.hexdigest()


def _get_dependency_versions():
    versions = {}
    for pkg in ['numpy', 'pandas', 'scipy', 'trackpy', 'cv2', 'matplotlib', 'customtkinter']:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass
    return versions


def generate_provenance(video_path, params, slopes_df=None):
    """Generate a provenance dict for the analysis run."""
    from scripts import __version__

    prov = {
        'software_version': __version__,
        'timestamp': datetime.now().isoformat(),
        'parameters': {k: str(v) for k, v in params.items()},
        'dependency_versions': _get_dependency_versions(),
        'input_video': os.path.basename(video_path) if video_path else None,
        'input_video_sha256': _file_sha256(video_path) if video_path else None,
    }

    if slopes_df is not None:
        prov['n_vials'] = len(slopes_df)
        if 'quality' in slopes_df.columns:
            good = (slopes_df['quality'] == 'good').sum()
            prov['quality_summary'] = f"{good}/{len(slopes_df)} good"

    return prov


def save_provenance(video_path, params, slopes_df=None, output_dir=None):
    """Generate and save provenance JSON alongside the video."""
    prov = generate_provenance(video_path, params, slopes_df)

    if output_dir is None:
        output_dir = os.path.dirname(video_path) if video_path else '.'

    stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else 'analysis'
    out_path = os.path.join(output_dir, f"{stem}.provenance.json")

    with open(out_path, 'w') as f:
        json.dump(prov, f, indent=2)

    logger.info("Provenance saved: %s", out_path)
    return out_path
