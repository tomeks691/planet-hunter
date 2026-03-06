import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from planet_hunter.config import PLOT_DIR

log = logging.getLogger(__name__)

COLORS = {
    "flux": "#2196F3",
    "transit": "#F44336",
    "model": "#FF9800",
    "highlight": "#4CAF50",
}


def generate_all_plots(tic_id: int, lc, bls_result, classification: str) -> dict:
    """Generate all 4 diagnostic plots. Returns dict of plot filenames."""
    prefix = f"tic_{tic_id}"
    plots = {}

    plots["plot_lightcurve"] = _plot_lightcurve(prefix, lc)
    plots["plot_periodogram"] = _plot_periodogram(prefix, bls_result)
    plots["plot_phase_fold"] = _plot_phase_fold(prefix, lc, bls_result)
    plots["plot_diagnostic"] = _plot_diagnostic(prefix, lc, bls_result, classification)

    return plots


def _plot_lightcurve(prefix: str, lc) -> str:
    """Raw light curve with time on x-axis."""
    fname = f"{prefix}_lightcurve.png"
    path = PLOT_DIR / fname
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(lc.time.value, lc.flux.value, s=0.5, color=COLORS["flux"], alpha=0.6)
        ax.set_xlabel("Time (BTJD)")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"{prefix.replace('_', ' ').upper()} - Light Curve")
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
    except Exception as e:
        log.error("Light curve plot failed: %s", e)
    return fname


def _plot_periodogram(prefix: str, bls_result) -> str:
    """BLS power spectrum."""
    fname = f"{prefix}_periodogram.png"
    path = PLOT_DIR / fname
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        if bls_result.periods_searched is not None and bls_result.power_spectrum is not None:
            ax.plot(bls_result.periods_searched, bls_result.power_spectrum,
                    color=COLORS["flux"], lw=0.8)
            ax.axvline(bls_result.period, color=COLORS["transit"], ls="--", lw=1.5,
                       label=f"Best P = {bls_result.period:.4f} d")
            ax.legend()
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("BLS Power")
        ax.set_title(f"{prefix.replace('_', ' ').upper()} - BLS Periodogram")
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
    except Exception as e:
        log.error("Periodogram plot failed: %s", e)
    return fname


def _plot_phase_fold(prefix: str, lc, bls_result) -> str:
    """Phase-folded light curve at best period."""
    fname = f"{prefix}_phasefold.png"
    path = PLOT_DIR / fname
    try:
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        phase = ((time - bls_result.t0) % bls_result.period) / bls_result.period
        # center on transit
        phase[phase > 0.5] -= 1.0

        # bin the phase curve
        n_bins = 200
        bins = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_flux = np.zeros(n_bins)
        for i in range(n_bins):
            in_bin = (phase >= bins[i]) & (phase < bins[i + 1])
            if np.sum(in_bin) > 0:
                bin_flux[i] = np.median(flux[in_bin])
            else:
                bin_flux[i] = np.nan

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(phase, flux, s=0.3, color=COLORS["flux"], alpha=0.3, label="Data")
        ax.plot(bin_centers, bin_flux, color=COLORS["transit"], lw=2, label="Binned")

        # Mark transit region
        half_dur = (bls_result.duration / 24) / bls_result.period / 2
        ax.axvspan(-half_dur, half_dur, color=COLORS["transit"], alpha=0.1)

        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(
            f"{prefix.replace('_', ' ').upper()} - Phase Fold "
            f"(P={bls_result.period:.4f} d, depth={bls_result.depth:.5f})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
    except Exception as e:
        log.error("Phase fold plot failed: %s", e)
    return fname


def _plot_diagnostic(prefix: str, lc, bls_result, classification: str) -> str:
    """Diagnostic: odd/even transits + secondary eclipse check."""
    fname = f"{prefix}_diagnostic.png"
    path = PLOT_DIR / fname
    try:
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        phase = ((time - bls_result.t0) % bls_result.period) / bls_result.period
        transit_num = np.floor((time - bls_result.t0) / bls_result.period).astype(int)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Panel 1: Odd transits
        odd = transit_num % 2 == 1
        p_odd = phase[odd].copy()
        p_odd[p_odd > 0.5] -= 1.0
        near_transit_odd = (p_odd > -0.1) & (p_odd < 0.1)
        axes[0].scatter(p_odd[near_transit_odd], flux[odd][near_transit_odd],
                        s=1, color=COLORS["flux"], alpha=0.5)
        axes[0].set_title("Odd Transits")
        axes[0].set_xlabel("Phase")
        axes[0].set_ylabel("Flux")

        # Panel 2: Even transits
        even = transit_num % 2 == 0
        p_even = phase[even].copy()
        p_even[p_even > 0.5] -= 1.0
        near_transit_even = (p_even > -0.1) & (p_even < 0.1)
        axes[1].scatter(p_even[near_transit_even], flux[even][near_transit_even],
                        s=1, color=COLORS["model"], alpha=0.5)
        axes[1].set_title("Even Transits")
        axes[1].set_xlabel("Phase")

        # Panel 3: Secondary eclipse region (phase ~0.5)
        near_secondary = (phase > 0.35) & (phase < 0.65)
        axes[2].scatter(phase[near_secondary], flux[near_secondary],
                        s=1, color=COLORS["highlight"], alpha=0.5)
        axes[2].axvline(0.5, color=COLORS["transit"], ls="--", alpha=0.5)
        axes[2].set_title("Secondary Eclipse Region")
        axes[2].set_xlabel("Phase")

        fig.suptitle(f"{prefix.replace('_', ' ').upper()} - Diagnostics [{classification}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
    except Exception as e:
        log.error("Diagnostic plot failed: %s", e)
    return fname
