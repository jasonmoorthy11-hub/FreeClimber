# Changelog

## [4.0.0] — 2026-03-16

Major scientific accuracy overhaul, modern GUI with design system, and comprehensive testing.

### Fixed
- **find_threshold**: Returns signal value instead of bin index; prominence tuple indexing fix
- **Diameter oddness**: Bitwise NOT bug (`~self.diameter%2`) replaced with proper modulo check
- **Window float→int**: Prevents trackpy crash on non-integer window
- **invert_y**: Uses ROI height instead of data max for correct coordinate inversion
- **Extension handling**: `os.path.splitext` instead of hardcoded `[:-5]` slice
- **x/y swap**: Corrected crop_and_grayscale dimension assignment
- **colored_hist**: Divide-by-zero guard
- **gather_files**: Bitwise NOT bug on empty list check
- **climbing_index**: Corrected threshold comparison direction (`>=` not `<=`)
- **subtract_drift**: pandas ambiguity fix with `reset_index(drop=True)`

### Added
- **Design system**: 5-tier elevation, typography scale, spacing grid, border radius tokens
- **KPI summary cards**: Mean Velocity, Flies Tracked, R-squared, Quality, Climbing Index
- **Toast notifications**: Bottom-right stacking toasts with auto-dismiss
- **Parameter validation**: Pre-run checks for crop bounds, odd diameter, eccentricity range, frame rate
- **Cancel button**: Abort running analysis with Escape key
- **Background video loading**: Threaded video open with progress indication
- **Double-run lock**: Prevents concurrent analysis
- **Stats wiring**: Test selection, p-value correction, Dunnett's test fully connected
- **Normalization wiring**: % of Control and Z-score methods connected to UI
- **Database auto-save**: SQLite experiment storage after every analysis
- **LLR caching**: Avoids redundant linear regression computation
- **Memory warning**: Alert for videos >2GB
- **15 detector unit tests**: Covering all Phase 1 scientific fixes
- **Profile delete confirmation**: Prevents accidental deletion

### Changed
- Version bumped from 3.1 to 4.0.0
- PyInstaller spec: bundle_identifier set, hiddenimports updated
- Removed stale .pyc from git tracking

## [3.1.0] — 2026-03-16

Crash fixes, scientific accuracy improvements, and performance optimizations.

### Fixed
- Plot display crashes on missing data
- Treeview sort/copy functionality
- Version string consistency across modules

### Added
- PlotToolbar with zoom, pan, save, copy
- Tooltip and chevron components
- Quality badges in treeview

## [3.0.0] — 2026-03-15

GUI modernization with design token system.

### Added
- Design token consolidation (colors, spacing, typography, radius)
- Collapsible sidebar cards
- Menu bar with keyboard shortcuts
- Context menus on plots

### Changed
- Migrated from flat layout to card-based sidebar

## [2.0.0] — 2026-03-11

Complete rewrite of FreeClimber for modern Python with a new GUI, advanced statistics, and hardware integration.

### Added
- **GUI**: CustomTkinter dark theme (navy/teal/coral palette) replacing wxPython
- **Video Engine**: OpenCV I/O with temporal median, MOG2, and KNN background subtraction
- **Tracking**: Individual fly tracking via trackpy with per-fly metrics (AUC, straightness, completeness)
- **Statistics**: t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis, Dunnett's test, effect sizes (Cohen's d), p-value corrections (Holm, Benjamini-Hochberg)
- **Reports**: PDF reports (Jinja2 + WeasyPrint) and interactive HTML dashboards (Plotly)
- **Export**: CSV, tidy/long format, Prism-compatible, per-fly track data
- **Database**: SQLite experiment storage for managing multiple runs
- **Profiles**: Save/load parameter profiles for reproducible analysis
- **Quality Scoring**: Automated vial and video quality assessment
- **Normalization**: Control-relative, batch Z-score, and fly-count adjustment methods
- **Hardware**: Raspberry Pi HQ camera control + NEMA 34 motor control for RING assay automation
- **Tests**: 117 automated tests covering config, stats, figures, reports, profiles, database, and integration
- **CI/CD**: GitHub Actions workflows for automated testing

### Changed
- Minimum Python version: 3.10+
- Version bumped from 0.3.3 to 2.0.0

## [0.3.3] — Previous release

See git history for earlier changes.
