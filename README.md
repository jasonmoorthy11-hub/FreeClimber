# FreeClimber

Automated quantification of climbing performance in *Drosophila* using background subtraction, particle detection, and local linear regression.

<img src="https://github.com/adamspierer/FreeClimber/blob/master/z/0-Tutorial_climbing.gif" width="600" height="200" align="center">

## Overview

FreeClimber is a Python-based pipeline for quantifying vertical velocity in rapid iterative negative geotaxis (RING) assays. It:

- **Subtracts background** to isolate moving particles from static frames
- **Detects particles** using [trackpy](https://github.com/soft-matter/trackpy) to find x, y, time coordinates of each fly
- **Fits local linear regressions** over sliding windows to find the most consistent climbing segment
- **Reports vertical velocity** (slope of best-fit segment) per vial

Version 2.0 adds: statistical analysis, quality scoring, HTML/PDF reports, experiment profiles, a SQLite database, a modern CustomTkinter GUI, and optional Raspberry Pi hardware control for automated RING assays.

## Requirements

- **Python** >= 3.10
- **OS**: macOS, Linux (including Raspberry Pi OS), Windows

Core dependencies (installed automatically):

| Package | Version |
|---------|---------|
| numpy | >= 1.26, < 2.0 |
| pandas | >= 2.0 |
| scipy | >= 1.12 |
| matplotlib | >= 3.8 |
| trackpy | >= 0.6 |
| opencv-python | >= 4.8 |
| customtkinter | >= 5.2 |

Optional extras:

| Extra | Packages | Install with |
|-------|----------|-------------|
| reports | jinja2, weasyprint | `pip install -e ".[reports]"` |
| interactive | plotly | `pip install -e ".[interactive]"` |
| excel | openpyxl | `pip install -e ".[excel]"` |
| all | all of the above | `pip install -e ".[all]"` |
| dev | pytest, ruff | `pip install -e ".[dev]"` |

## Installation

### From PyPI

```bash
pip install FreeClimber
```

### From source (recommended for development)

```bash
git clone https://github.com/adamspierer/FreeClimber.git
cd FreeClimber
pip install -e ".[all]"
```

## Usage

### GUI

Launch the graphical interface:

```bash
freeclimber
```

Load a video file, adjust detection parameters interactively, then process.

### Command line

Process videos in batch using a configuration file:

```bash
python scripts/FreeClimber_main.py --config_file ./example/example.cfg
```

Options: `--process_all`, `--process_undone`, `--process_custom`. Use `-h` for details.

### Hardware-automated RING assay (Raspberry Pi)

FreeClimber includes modules for controlling a NEMA 34 stepper motor (via 2HSS86 driver) and Raspberry Pi HQ Camera for fully automated RING assays:

```python
from scripts.hardware.motor import RINGMotor
from scripts.hardware.camera import RINGCamera
from scripts.workflow import RINGWorkflow

motor = RINGMotor()
camera = RINGCamera()
wf = RINGWorkflow(motor, camera, config={})
wf.run_assay(settle_time=60, n_taps=3, record_duration=10)
```

These modules run in mock mode on non-Pi systems for development and testing.

## Code Structure

```
scripts/
  detector.py            # Core detection pipeline (7-step)
  config.py              # Configuration file I/O
  FreeClimber_main.py    # CLI entry point
  gather_files.py        # File discovery utility
  workflow.py            # RING assay workflow orchestrator
  gui/
    app.py               # CustomTkinter GUI (entry point)
    controller.py        # GUI ↔ detector bridge
  analysis/
    metrics.py           # Climbing velocity metrics
    normalization.py     # Data normalization
    quality.py           # Quality scoring
    stats.py             # Statistical analysis
  output/
    database.py          # SQLite experiment database
    export.py            # CSV/Excel export
    figures.py           # Matplotlib figure generation
    reports.py           # HTML/PDF report generation
    video.py             # Video utilities
  hardware/
    motor.py             # NEMA 34 motor control (lgpio)
    camera.py            # Pi HQ Camera control (picamera2)
tests/                   # pytest test suite (117+ tests)
example/                 # Example video and config files
```

## Test files

The `example` folder contains test videos and configuration files. See the [tutorial](https://github.com/adamspierer/FreeClimber/blob/master/TUTORIAL.md) for a walkthrough.

**Inputs**: `.h264` / `.mov` video files, `.cfg` configuration files

**Outputs**: `.raw.csv`, `.filtered.csv`, `.slopes.csv` (data); `.ROI.png`, `.processed.csv`, `.spot_check.png`, `.diagnostic.png` (plots); `results.csv` (merged); `log/completed.log`, `log/skipped.log`

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Version releases

- **2.0.0** — Modern rewrite: CustomTkinter GUI, statistical analysis, quality scoring, HTML/PDF reports, experiment profiles, SQLite database, Raspberry Pi hardware integration, 117+ tests
- 0.4.0 — Fixed frame cropping error, added variable compatibility checks
- 0.3.2 — Release version for *Journal of Experimental Biology*

## Deployment

Tested on macOS and Raspberry Pi OS (Debian Bookworm). Should work on any platform with Python >= 3.10.

## Contributing

Fork the repository and submit a pull request. Please document your changes and include tests where applicable.

## Citing this work

If you use FreeClimber, please cite:

> A.N. Spierer, D. Yoon, CT. Zhu, and DM. Rand. (2020) FreeClimber: Automated quantification of climbing performance in *Drosophila*. *Journal of Experimental Biology*. DOI: 10.1242/jeb.229377

## License

MIT

## Authors

Originally written by [Adam Spierer](https://github.com/adamspierer) and [Lei Zhuo](https://github.com/ctzhu/) with assistance from Brown University's [Computational Biology Core](https://github.com/compbiocore/). v2.0 by [Jason Moorthy](https://github.com/jasonmoorthy11-hub) at Wayne State University.
