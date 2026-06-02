#!/opt/anaconda3/envs/p3.12/bin/python
"""Build a merged 1D spectrum from APERO-like 2D echelle products.

This script ports the APERO merge logic in:
apero/science/extract/gen_ext.py::e2ds_to_s1d

Science reference:
Cook et al. 2022, PASP, section 7.6 and figure 21.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline
import yaml


# Default config file inside this repository.
DEFAULT_CONFIG_PATH = Path("config/s1drama.yaml")


def load_config(config_path: Path) -> dict:
    """Load YAML configuration.

    The YAML stores all tunable constants that were previously hard-coded.
    """
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} is not a dictionary.")
    return cfg


def _safe_iuv_spline(x: np.ndarray, y: np.ndarray, k: int, ext: int):
    """APERO-like robust spline helper.

    APERO uses splines extensively. In real data, some orders can have
    too few valid points, repeated wavelengths, or NaNs. This helper keeps
    the exact behavior simple and safe: return None instead of crashing.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # InterpolatedUnivariateSpline requires strictly increasing x, so we
    # drop duplicate x values after sorting.
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if x_unique.size < 2:
        return None

    k_use = min(k, x_unique.size - 1)
    if k_use < 1:
        return None
    return InterpolatedUnivariateSpline(x_unique, y_unique, k=k_use, ext=ext)


def _get_magic_grid(wave0: float, wave1: float, dv_grid_ms: float) -> np.ndarray:
    """APERO magic velocity grid from aperocore.math.gen_math.get_magic_grid.

    This is a logarithmic wavelength grid with constant velocity spacing.
    The speed of light comes from scipy.constants.c (m/s).
    """
    logwaveratio = np.log(wave1 / wave0)
    len_magic = int(np.floor(logwaveratio * c / dv_grid_ms))
    wave1_new = np.exp(len_magic * dv_grid_ms / c) * wave0
    logwaveratio = np.log(wave1_new / wave0)
    plen_magic = np.arange(len_magic)
    return np.exp((plen_magic / len_magic) * logwaveratio) * wave0


def e2ds_to_s1d_apero_logic(
    wavemap: np.ndarray,
    e2ds: np.ndarray,
    blaze: np.ndarray,
    *,
    wgrid: str = "wave",
    wavestart: float,
    waveend: float,
    binwave: float,
    binvelo: float,
    smooth_size: int,
    blazethres: float,
    e2dserr: np.ndarray | None = None,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Port of APERO e2ds_to_s1d logic for 2D order merging.

    Returns:
    - wavelength, flux, eflux, weight: final merged 1D vectors
    - debug_slopes: per-order edge taper vectors for diagnostics/plots
    """
    # Number of spectral orders and pixels/order.
    nord, npix = e2ds.shape

    # APERO accepts optional error arrays. If unavailable, APERO uses a
    # placeholder and ultimately sets eflux to zero.
    if e2dserr is None:
        e2dserr = np.tile(np.arange(npix, dtype=float), nord).reshape((nord, npix))
        has_errors = False
    else:
        has_errors = True

    # Build the output 1D wavelength grid exactly as APERO does.
    if wgrid == "wave":
        wavegrid = np.arange(wavestart, waveend + binwave / 2.0, binwave)
    elif wgrid == "velocity":
        wavegrid = _get_magic_grid(wavestart, waveend, binvelo * 1000.0)
    else:
        raise ValueError("wgrid must be 'wave' or 'velocity'")

    # APERO smooth order edges to avoid sharp discontinuities where adjacent
    # orders overlap.
    xker = np.arange(-smooth_size * 3, smooth_size * 3, 1)
    ker = np.exp(-0.5 * (xker / smooth_size) ** 2)
    edges = np.ones(npix, dtype=bool)
    edges[: int(3 * smooth_size)] = False
    edges[-int(3 * smooth_size) :] = False

    slopevector = np.zeros_like(blaze, dtype=float)
    debug_slopes: list[np.ndarray] = []
    for order_num in range(nord):
        # blaze and data must both be finite, and blaze must be above the
        # configured fraction of its peak for this order.
        oblaze = np.array(blaze[order_num], dtype=float)
        cond1 = np.isfinite(oblaze) & np.isfinite(e2ds[order_num])
        with np.errstate(invalid="ignore"):
            cond2 = oblaze > (blazethres * np.nanmax(oblaze))
        valid = cond1 & cond2 & edges
        # Gaussian-smooth the binary mask into soft edge weights.
        oweight = np.convolve(valid.astype(float), ker, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            oweight = oweight - np.nanmin(oweight)
            denom = np.nanmax(oweight)
            if np.isfinite(denom) and denom != 0:
                oweight = oweight / denom
            else:
                oweight[:] = 0.0
        slopevector[order_num] = oweight
        debug_slopes.append(np.array(oweight, dtype=float))

    # Apply the edge taper to blaze, flux, and error arrays.
    sblaze = np.array(blaze, dtype=float) * slopevector
    se2ds = np.array(e2ds, dtype=float) * slopevector
    se2dserr = np.array(e2dserr, dtype=float) * slopevector

    # Storage for weighted accumulation on the common 1D grid.
    out_spec = np.zeros_like(wavegrid, dtype=float)
    out_spec_err = np.zeros_like(wavegrid, dtype=float)
    weight = np.zeros_like(wavegrid, dtype=float)

    for order_num in range(nord):
        # Reject invalid wavelength pixels (can happen in real extractions).
        wavemask = np.isfinite(wavemap[order_num])
        valid = np.isfinite(se2ds[order_num]) & np.isfinite(sblaze[order_num])
        valid &= wavemask
        if np.sum(valid) == 0:
            continue

        owave = wavemap[order_num]
        oe2ds = se2ds[order_num, valid]
        oe2dserr = se2dserr[order_num, valid]
        oblaze = sblaze[order_num, valid]

        # APERO strategy: spline both flux and blaze onto common wavegrid,
        # then combine orders through blaze-based weighting.
        spline_sp = _safe_iuv_spline(owave[valid], oe2ds, k=5, ext=1)
        spline_bl = _safe_iuv_spline(owave[valid], oblaze, k=1, ext=1)
        spline_sperr = _safe_iuv_spline(owave[valid], oe2dserr, k=5, ext=1)
        if spline_sp is None or spline_bl is None or spline_sperr is None:
            continue

        # APERO avoids interpolation across large invalid gaps by requiring
        # local validity around each candidate wavegrid pixel.
        valid_float = valid.astype(float)
        valid_float = np.convolve(valid_float, np.ones(3) / 3.0, mode="same")
        spline_valid = _safe_iuv_spline(owave[wavemask], valid_float[wavemask], k=1, ext=1)
        if spline_valid is None:
            continue

        useful_range = wavegrid > np.nanmin(owave[valid])
        useful_range &= wavegrid < np.nanmax(owave[valid])

        maskvalid = np.zeros_like(wavegrid, dtype=bool)
        maskvalid[useful_range] = spline_valid(wavegrid[useful_range]) > 0.9
        useful_range &= maskvalid

        # Add this order contribution to the running weighted sums.
        weight[useful_range] += spline_bl(wavegrid[useful_range])
        out_spec[useful_range] += spline_sp(wavegrid[useful_range])
        out_spec_err[useful_range] += spline_sperr(wavegrid[useful_range])

    # Division by zero protection: APERO marks zero-weight bins as NaN.
    zeroweights = weight == 0
    weight[zeroweights] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        w_out_spec = out_spec / weight
        w_out_spec_err = out_spec_err / weight

    # If no input errors were supplied, mimic APERO and set output eflux to 0.
    if not has_errors:
        w_out_spec_err = np.zeros_like(w_out_spec)

    return {
        "wavelength": wavegrid,
        "flux": w_out_spec,
        "eflux": w_out_spec_err,
        "weight": weight,
        "debug_slopes": debug_slopes,
    }


def make_debug_plots(
    input_path: Path,
    fig_dir: Path,
    wavemap: np.ndarray,
    e2ds: np.ndarray,
    blaze: np.ndarray,
    s1d: dict[str, np.ndarray | list[np.ndarray]],
    slope_order: int,
    overlap_orders: list[int],
    overlap_margin_nm: float,
) -> None:
    """Write optional PNG figures to explain how the merge works.

    These figures are intended for README documentation and student onboarding.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    # 1) Flux image overview.
    plt.figure(figsize=(10, 4.5))
    plt.imshow(
        e2ds,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=np.nanpercentile(e2ds, 5),
        vmax=np.nanpercentile(e2ds, 99),
    )
    plt.xlabel("Pixel")
    plt.ylabel("Order")
    plt.title("Input 2D flux orders (e2ds)")
    plt.colorbar(label="Flux")
    plt.tight_layout()
    plt.savefig(str(fig_dir / f"{stem}_fig1_e2ds.png"), dpi=150)
    plt.close()

    # 2) Edge taper for one order.
    use_order = int(np.clip(slope_order, 0, e2ds.shape[0] - 1))
    slope = np.array(s1d["debug_slopes"][use_order], dtype=float)
    plt.figure(figsize=(10, 4.5))
    plt.plot(blaze[use_order], lw=1.2, label=f"Blaze order {use_order}")
    plt.plot(slope * np.nanmax(blaze[use_order]), lw=1.2, label="Scaled edge taper")
    plt.xlabel("Pixel")
    plt.ylabel("Arbitrary units")
    plt.title("Order-edge taper used before order merging")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(fig_dir / f"{stem}_fig2_edge_taper.png"), dpi=150)
    plt.close()

    # 3) Final merged 1D flux + weight.
    wave = np.array(s1d["wavelength"], dtype=float)
    flux = np.array(s1d["flux"], dtype=float)
    weight = np.array(s1d["weight"], dtype=float)
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.plot(wave, flux, color="black", lw=0.8)
    ax1.set_xlabel("Wavelength [nm]")
    ax1.set_ylabel("Merged flux")
    ax1.set_title("Final merged 1D spectrum (APERO-style order merge)")

    ax2 = ax1.twinx()
    ax2.plot(wave, weight, color="tab:blue", alpha=0.4, lw=0.8)
    ax2.set_ylabel("Merge weight", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.savefig(str(fig_dir / f"{stem}_fig3_s1d.png"), dpi=150)
    plt.close(fig)

    # 4) Zoom around selected order overlaps (e.g. 59-61) with explicit
    #    color-coded order contributions and merged result.
    se2ds = np.array(e2ds, dtype=float) * np.array(s1d["debug_slopes"], dtype=float)
    sblaze = np.array(blaze, dtype=float) * np.array(s1d["debug_slopes"], dtype=float)

    wavegrid = np.array(s1d["wavelength"], dtype=float)
    merged_flux = np.array(s1d["flux"], dtype=float)
    total_weight = np.array(s1d["weight"], dtype=float)

    n_orders = e2ds.shape[0]
    use_orders = [int(np.clip(o, 0, n_orders - 1)) for o in overlap_orders]
    use_orders = sorted(list(set(use_orders)))

    # Compute overlap window from adjacent pairs. If no overlap exists,
    # fallback to the full span of selected orders.
    overlap_lows = []
    overlap_highs = []
    for i in range(len(use_orders) - 1):
        o1, o2 = use_orders[i], use_orders[i + 1]
        w1 = wavemap[o1]
        w2 = wavemap[o2]
        w1f = w1[np.isfinite(w1)]
        w2f = w2[np.isfinite(w2)]
        if w1f.size == 0 or w2f.size == 0:
            continue
        lo = max(np.nanmin(w1f), np.nanmin(w2f))
        hi = min(np.nanmax(w1f), np.nanmax(w2f))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            overlap_lows.append(lo)
            overlap_highs.append(hi)

    if len(overlap_lows) > 0:
        zoom_lo = min(overlap_lows) - overlap_margin_nm
        zoom_hi = max(overlap_highs) + overlap_margin_nm
    else:
        wmins, wmaxs = [], []
        for o in use_orders:
            wf = wavemap[o][np.isfinite(wavemap[o])]
            if wf.size > 0:
                wmins.append(np.nanmin(wf))
                wmaxs.append(np.nanmax(wf))
        zoom_lo = min(wmins) - overlap_margin_nm
        zoom_hi = max(wmaxs) + overlap_margin_nm

    zoom_mask = np.isfinite(wavegrid) & (wavegrid >= zoom_lo) & (wavegrid <= zoom_hi)

    cmap = mpl.colormaps["tab10"]
    fig, (ax_flux, ax_wfrac) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for idx, order_num in enumerate(use_orders):
        color = cmap(idx % 10)
        owave = wavemap[order_num]
        valid = np.isfinite(owave) & np.isfinite(se2ds[order_num]) & np.isfinite(sblaze[order_num])
        if np.sum(valid) < 5:
            continue

        sp_flux = _safe_iuv_spline(owave[valid], se2ds[order_num, valid], k=5, ext=1)
        sp_blaze = _safe_iuv_spline(owave[valid], sblaze[order_num, valid], k=1, ext=1)
        if sp_flux is None or sp_blaze is None:
            continue

        in_range = zoom_mask.copy()
        in_range &= wavegrid > np.nanmin(owave[valid])
        in_range &= wavegrid < np.nanmax(owave[valid])
        if not np.any(in_range):
            continue

        # Order-specific spectrum estimate in APERO formalism:
        # order_flux ~ spline(flux*taper) / spline(blaze*taper)
        with np.errstate(invalid="ignore", divide="ignore"):
            oi_flux = sp_flux(wavegrid[in_range]) / sp_blaze(wavegrid[in_range])
            oi_wfrac = sp_blaze(wavegrid[in_range]) / total_weight[in_range]

        ax_flux.plot(wavegrid[in_range], oi_flux, color=color, lw=1.1,
                     label=f"Order {order_num}")
        ax_wfrac.plot(wavegrid[in_range], oi_wfrac, color=color, lw=1.1,
                      label=f"Order {order_num}")

    ax_flux.plot(wavegrid[zoom_mask], merged_flux[zoom_mask], color="black",
                 lw=2.0, label="Merged S1D")

    ax_flux.set_ylabel("Flux")
    ax_flux.set_title("Zoom on order overlap and seamless merge (orders 59-61 style)")
    ax_flux.legend(loc="best", ncol=2, fontsize=9)

    ax_wfrac.set_xlabel("Wavelength [nm]")
    ax_wfrac.set_ylabel("Weight fraction")
    ax_wfrac.set_ylim(-0.05, 1.05)
    ax_wfrac.grid(alpha=0.25)

    # Visual cues for overlap boundaries used in this zoom.
    for lo, hi in zip(overlap_lows, overlap_highs):
        ax_flux.axvspan(lo, hi, color="gray", alpha=0.08)
        ax_wfrac.axvspan(lo, hi, color="gray", alpha=0.08)

    ax_flux.set_xlim(zoom_lo, zoom_hi)
    fig.tight_layout()
    fig.savefig(str(fig_dir / f"{stem}_fig4_overlap_zoom.png"), dpi=150)
    plt.close(fig)


def resolve_input_files(input_patterns: list[str]) -> list[Path]:
    """Resolve literal files and wildcard patterns into a sorted file list."""
    resolved: list[Path] = []
    for pattern in input_patterns:
        # If user passes a literal path, keep it directly.
        p = Path(pattern)
        if p.exists() and p.is_file():
            resolved.append(p)
            continue

        # Otherwise interpret as wildcard (supports ** patterns).
        matches = [Path(m) for m in glob.glob(pattern, recursive=True)]
        matches = [m for m in matches if m.is_file()]
        resolved.extend(matches)

    # Deduplicate while preserving deterministic order.
    unique_sorted = sorted({str(p.resolve()): p.resolve() for p in resolved}.values())
    return unique_sorted


def build_output_path(input_path: Path, output_dir: Path, output_suffix: str) -> Path:
    """Build output S1D path from input filename and configured folder/suffix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}{output_suffix}.fits"


def process_one_file(
    input_path: Path,
    output_path: Path,
    *,
    flux_ext: str,
    wave_ext: str,
    blaze_ext: str,
    wgrid: str,
    wavestart: float,
    waveend: float,
    bin_uwave: float,
    bin_uvel: float,
    edge_ssize: int,
    cut_blaze_norm: float,
    make_plots: bool,
    fig_dir: Path,
    plot_order: int,
    overlap_orders: list[int],
    overlap_margin_nm: float,
) -> tuple[int, int]:
    """Run full 2D->1D merge pipeline for one file and return quality counts."""
    print(f"[READ ] Opening input FITS: {input_path}")
    with fits.open(input_path) as hdul:
        header0 = hdul[0].header
        header1 = hdul[1].header if len(hdul) > 1 else None
        e2ds = np.array(hdul[flux_ext].data, dtype=float)
        wavemap = np.array(hdul[wave_ext].data, dtype=float)
        blaze = np.array(hdul[blaze_ext].data, dtype=float)
        
        # Read BERV and systemic velocity for zero-velocity wavelength correction
        berv_kms = header1.get('BERV', 0.0) if header1 else 0.0
        # ESO TEL TARG RADVEL is the target's radial velocity (systemic)
        rv_sys_kms = header0.get('ESO TEL TARG RADVEL', 0.0)

    print("[MERGE] Running APERO-like order merge...")
    s1d = e2ds_to_s1d_apero_logic(
        wavemap=wavemap,
        e2ds=e2ds,
        blaze=blaze,
        wgrid=wgrid,
        wavestart=wavestart,
        waveend=waveend,
        binwave=bin_uwave,
        binvelo=bin_uvel,
        smooth_size=edge_ssize,
        blazethres=cut_blaze_norm,
    )
    
    # Create zero-velocity flux by shifting the observed flux onto the common grid
    # Total velocity to correct (km/s): negative means blueshift
    total_vel_kms = berv_kms + rv_sys_kms
    beta = (total_vel_kms * 1000.0) / c  # c from scipy.constants in m/s
    # Relativistic Doppler factor
    doppler_factor = np.sqrt((1.0 - beta) / (1.0 + beta))
    
    # To shift flux to zero velocity: interpolate flux from shifted wavelengths
    # flux_zerovel(λ) = flux(λ / doppler_factor)
    wavelength_shifted = s1d["wavelength"] / doppler_factor
    
    # Interpolate flux and eflux onto the shifted wavelengths
    valid = np.isfinite(s1d["flux"]) & np.isfinite(s1d["wavelength"])
    if np.sum(valid) > 1:
        flux_zerovel = np.interp(
            wavelength_shifted,
            s1d["wavelength"][valid],
            s1d["flux"][valid],
            left=np.nan,
            right=np.nan,
        )
        eflux_zerovel = np.interp(
            wavelength_shifted,
            s1d["wavelength"][valid],
            s1d["eflux"][valid],
            left=np.nan,
            right=np.nan,
        )
    else:
        flux_zerovel = np.full_like(s1d["flux"], np.nan)
        eflux_zerovel = np.full_like(s1d["eflux"], np.nan)

    print(f"[WRITE] Writing merged S1D FITS: {output_path}")
    cols = [
        fits.Column(name="wavelength", array=s1d["wavelength"], format="D", unit="nm"),
        fits.Column(name="flux", array=s1d["flux"], format="D"),
        fits.Column(name="eflux", array=s1d["eflux"], format="D"),
        fits.Column(name="flux_zerovel", array=flux_zerovel, format="D"),
        fits.Column(name="eflux_zerovel", array=eflux_zerovel, format="D"),
        fits.Column(name="weight", array=s1d["weight"], format="D"),
    ]
    s1d_hdu = fits.BinTableHDU.from_columns(cols, name="S1D")
    s1d_hdu.header["S1DKIND"] = (wgrid, "APERO-like S1D grid type")
    s1d_hdu.header["S1DWAVE0"] = (wavestart, "Initial wavelength for s1d [nm]")
    s1d_hdu.header["S1DWAVE1"] = (waveend, "Final wavelength for s1d [nm]")
    s1d_hdu.header["S1DBWAVE"] = (bin_uwave, "S1D wavelength step [nm]")
    s1d_hdu.header["S1DBVELO"] = (bin_uvel, "S1D velocity step [km/s]")
    s1d_hdu.header["S1DSSIZE"] = (edge_ssize, "S1D order-edge smoothing scale [pix]")
    s1d_hdu.header["S1DBLAZT"] = (cut_blaze_norm, "Normalized blaze threshold")
    s1d_hdu.header["APEROALG"] = ("e2ds_to_s1d", "APERO function logic used")
    s1d_hdu.header["BERV"] = (berv_kms, "Barycentric Earth RV [km/s]")
    s1d_hdu.header["RV_SYS"] = (rv_sys_kms, "Systemic RV [km/s]")
    s1d_hdu.header["RV_TOTAL"] = (total_vel_kms, "BERV + systemic RV [km/s]")

    primary = fits.PrimaryHDU(header=header0)
    fits.HDUList([primary, s1d_hdu]).writeto(output_path, overwrite=True)

    if make_plots:
        print(f"[PLOT ] Writing diagnostics in: {fig_dir}")
        make_debug_plots(
            input_path=input_path,
            fig_dir=fig_dir,
            wavemap=wavemap,
            e2ds=e2ds,
            blaze=blaze,
            s1d=s1d,
            slope_order=plot_order,
            overlap_orders=overlap_orders,
            overlap_margin_nm=overlap_margin_nm,
        )

    total_rows = len(s1d["wavelength"])
    finite_flux = int(np.isfinite(s1d["flux"]).sum())
    print(f"[DONE ] Rows={total_rows} | finite_flux={finite_flux}/{total_rows}")
    return finite_flux, total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge NIRPS 2D orders into 1D using APERO e2ds_to_s1d logic "
            "(Cook+2022, section 7.6 / figure 21)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config path with APERO-like constants and I/O defaults",
    )
    parser.add_argument(
        "input_patterns",
        nargs="+",
        help="Input FITS path(s) and/or wildcard pattern(s), e.g. data/*.fits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single-file output path (only valid when exactly one input file is processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override YAML output directory for merged S1D FITS files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if output already exists",
    )
    parser.add_argument("--flux-ext", default=None, help="Flux extension name")
    parser.add_argument("--wave-ext", default=None, help="Wavelength extension name")
    parser.add_argument("--blaze-ext", default=None, help="Blaze extension name")
    parser.add_argument("--wgrid", choices=["wave", "velocity"], default=None)
    parser.add_argument("--wavestart", type=float, default=None)
    parser.add_argument("--waveend", type=float, default=None)
    parser.add_argument("--bin-uwave", type=float, default=None)
    parser.add_argument("--bin-uvel", type=float, default=None)
    parser.add_argument("--edge-ssize", type=int, default=None)
    parser.add_argument("--cut-blaze-norm", type=float, default=None)
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Write diagnostic PNG figures that explain the merge steps",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory to write diagnostic plots",
    )
    parser.add_argument(
        "--plot-order",
        type=int,
        default=None,
        help="Order index used for edge-taper illustration plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    io_cfg = cfg["io"]
    output_dir = args.output_dir if args.output_dir is not None else Path(io_cfg["output_dir"])
    output_suffix = str(io_cfg.get("output_suffix", "_s1d_apero"))

    merge_cfg = cfg["merge"]
    plot_cfg = cfg["plotting"]

    flux_ext = args.flux_ext or io_cfg["flux_ext"]
    wave_ext = args.wave_ext or io_cfg["wave_ext"]
    blaze_ext = args.blaze_ext or io_cfg["blaze_ext"]
    wgrid = args.wgrid or merge_cfg["wgrid"]
    wavestart = args.wavestart if args.wavestart is not None else float(merge_cfg["wavestart_nm"])
    waveend = args.waveend if args.waveend is not None else float(merge_cfg["waveend_nm"])
    bin_uwave = args.bin_uwave if args.bin_uwave is not None else float(merge_cfg["bin_uwave_nm"])
    bin_uvel = args.bin_uvel if args.bin_uvel is not None else float(merge_cfg["bin_uvel_kms"])
    edge_ssize = args.edge_ssize if args.edge_ssize is not None else int(merge_cfg["edge_ssize_pix"])
    cut_blaze_norm = (
        args.cut_blaze_norm if args.cut_blaze_norm is not None else float(merge_cfg["cut_blaze_norm"])
    )

    fig_dir = args.fig_dir if args.fig_dir is not None else Path(plot_cfg["fig_dir"])
    plot_order = args.plot_order if args.plot_order is not None else int(plot_cfg["slope_plot_order"])
    overlap_orders = [int(v) for v in plot_cfg.get("overlap_orders", [59, 60, 61])]
    overlap_margin_nm = float(plot_cfg.get("overlap_margin_nm", 0.5))

    print("[INFO ] Resolving input paths/patterns...")
    files = resolve_input_files(args.input_patterns)
    if len(files) == 0:
        raise FileNotFoundError(
            "No input FITS file found from provided paths/patterns: "
            + ", ".join(args.input_patterns)
        )

    if args.output is not None and len(files) != 1:
        raise ValueError("--output can only be used when exactly one input file is resolved.")

    print(f"[INFO ] Resolved {len(files)} file(s).")
    print(f"[INFO ] Output directory: {output_dir}")
    print(f"[INFO ] Skip existing outputs: {'NO (force mode)' if args.force else 'YES'}")

    processed = 0
    skipped = 0
    failed = 0
    for idx, input_path in enumerate(files, start=1):
        print("\n" + "=" * 78)
        print(f"[FILE ] {idx}/{len(files)}: {input_path}")

        # Avoid recursively processing previously generated S1D products.
        if input_path.stem.endswith(output_suffix):
            print(f"[SKIP ] Input appears to be an already merged output: {input_path.name}")
            skipped += 1
            continue

        if args.output is not None:
            output_path = args.output
        else:
            output_path = build_output_path(input_path, output_dir, output_suffix)

        if output_path.exists() and not args.force:
            print(f"[SKIP ] Output exists already: {output_path}")
            print("[SKIP ] Use --force to overwrite this file.")
            skipped += 1
            continue

        try:
            process_one_file(
                input_path=input_path,
                output_path=output_path,
                flux_ext=flux_ext,
                wave_ext=wave_ext,
                blaze_ext=blaze_ext,
                wgrid=wgrid,
                wavestart=wavestart,
                waveend=waveend,
                bin_uwave=bin_uwave,
                bin_uvel=bin_uvel,
                edge_ssize=edge_ssize,
                cut_blaze_norm=cut_blaze_norm,
                make_plots=args.make_plots,
                fig_dir=fig_dir,
                plot_order=plot_order,
                overlap_orders=overlap_orders,
                overlap_margin_nm=overlap_margin_nm,
            )
            print(f"[OUT  ] {output_path}")
            processed += 1
        except Exception as exc:
            print(f"[FAIL ] Could not process file: {input_path}")
            print(f"[FAIL ] Reason: {exc}")
            failed += 1

    print("\n" + "=" * 78)
    print(f"[SUM  ] Total files resolved: {len(files)}")
    print(f"[SUM  ] Processed: {processed}")
    print(f"[SUM  ] Skipped (already exists): {skipped}")
    print(f"[SUM  ] Failed: {failed}")
    print("[SUM  ] Completed.")


if __name__ == "__main__":
    main()
