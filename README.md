# Planet Hunter — Exoplanet Detection Pipeline

An autonomous pipeline for detecting exoplanet candidates from NASA TESS photometric data. Runs 24/7 on a home server, processing thousands of stars automatically.

## How It Works

```
TESS Star Catalog (TIC)
        │
        ▼
  Fetch light curves          ← MAST API (NASA)
        │
        ▼
  Clean & stitch sectors      ← outlier removal, normalization
        │
        ▼
  BLS Periodogram             ← Box Least Squares (iterative, 3 signals)
        │
        ▼
  Diagnostic checks           ← secondary eclipse, odd/even depth, sinusoid test
        │
        ▼
  Multi-sector validation     ← signal must appear in multiple TESS sectors
        │
        ▼
  Classification              ← PLANET_CANDIDATE / ECLIPSING_BINARY /
                                 VARIABLE_STAR / NOISE / KNOWN_PLANET /
                                 FALSE_POSITIVE / MANUAL_REVIEW
        │
        ▼
  Planet properties           ← radius, semi-major axis, equilibrium temp
        │
        ▼
  Plots & database storage    ← SQLite + FastAPI web interface
```

## Features

- **Automated queue processing** — stars are queued for analysis and processed in the background 24/7
- **Iterative BLS** — finds up to 3 periodic signals per star, picks the most planet-like one
- **Multi-signal diagnostics** — secondary eclipse depth, odd/even transit depth comparison, sinusoid test
- **Multi-sector validation** — signal must be present in at least N independent TESS sectors
- **Known planet database** — 700+ confirmed NASA planets for training data and validation
- **Automatic cache cleanup** — FITS files removed after each analysis to save disk space
- **Web interface** — FastAPI + SQLite dashboard to browse results and queue new stars

## Classification

| Class | Description |
|---|---|
| `PLANET_CANDIDATE` | Passes all diagnostic checks, consistent across sectors |
| `ECLIPSING_BINARY` | Deep transits, secondary eclipse detected |
| `VARIABLE_STAR` | Sinusoidal brightness variation |
| `FALSE_POSITIVE` | Inconsistent signal across sectors |
| `NOISE` | No significant periodic signal found |
| `KNOWN_PLANET` | Matched against NASA confirmed planet catalog |
| `MANUAL_REVIEW` | Marginal signal, requires human inspection |

## Tech Stack

| Layer | Technology |
|---|---|
| Data source | NASA MAST API (TESS light curves) |
| Signal processing | lightkurve, BLS (Box Least Squares) |
| Pipeline | Python, custom multi-stage pipeline |
| Storage | SQLite |
| API | FastAPI |
| Infrastructure | Docker, deployed on home server (Beelink mini PC) |

## Project Structure

```
planet_hunter/
├── pipeline/
│   ├── runner.py        # Main pipeline orchestrator + background queue processor
│   ├── fetcher.py       # MAST API — light curve download
│   ├── cleaner.py       # Signal cleaning, outlier removal, sector stitching
│   ├── periodogram.py   # BLS periodogram, iterative search, diagnostic checks
│   ├── classifier.py    # Classification logic
│   ├── properties.py    # Planet radius, orbit, temperature estimation
│   └── plots.py         # Light curve, phase-fold, periodogram plots
├── scanner/
│   ├── auto_scanner.py  # Automatic TIC catalog scanning
│   └── tic_catalog.py   # TIC catalog queries
├── web/
│   └── routes.py        # FastAPI endpoints
├── db.py                # SQLite operations
├── models.py            # Data models
└── config.py            # Pipeline parameters (SNR, depth thresholds, etc.)
```

## Running

```bash
docker compose up -d
```

The pipeline starts automatically and processes the queue in the background.
Web interface available at `http://localhost:8000`.

---

> Data source: [NASA TESS Mission](https://tess.mit.edu/) via [MAST Archive](https://mast.stsci.edu/)
