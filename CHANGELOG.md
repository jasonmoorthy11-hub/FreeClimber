# Changelog

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
