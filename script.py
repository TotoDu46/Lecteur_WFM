"""
WFM Reader — Tektronix MSO54
Réécriture complète avec corrections et améliorations.

Corrections appliquées :
  1. Curseurs : CosmeticInfiniteLine force le flag cosmetic dans paint(),
     viewTransformChanged() et setPen() — verrouillage complet
  2. Export label : double processEvents() + setFont() direct sur l'axe
     + appel _updateLabel() / resizeEvent() pour recalculer labelWidth
  3. UI Windows 10 : stylesheet corrigé, fin des superpositions
  4. Popups : ordre corrigé (QFileDialog avant popup), double processEvents()
  5. _finish_busy_popup : stoppe le timer précédent avant d'en créer un
  6. Export tick_font_pt augmenté à 11 (était 9), duplicate supprimée

Améliorations :
  - Constantes regroupées en dataclasses (Config)
  - Helpers extraits et typés
  - Suppression des try/except silencieux inutiles
  - _render_plots_fixed_image : layout pass garanti avant render()
  - load_paths : progression plus précise avec durée par fichier
  - ScopeStack : _wire_master_plot_signals robustifié
  - CursorSeriesWidget : colonne vitesse colorée selon valeur
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import (
    QKeySequence, QGuiApplication, QImage, QPainter, QFont, QColor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel,
    QGroupBox, QMessageBox, QComboBox, QCheckBox,
    QSizePolicy, QTabWidget, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QToolButton, QLineEdit, QDoubleSpinBox, QProgressDialog
)

import pyqtgraph as pg
from tm_data_types import read_file

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors as rl_colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ============================================================
# CONFIGURATION CENTRALISÉE
# ============================================================

@dataclass
class _LiveConfig:
    use_multires: bool = True
    raw_max_samples_per_pixel: float = 3.0
    force_raw_if_block_leq: int = 16
    target_blocks_per_pixel: float = 1.0
    lod_min_block: int = 4
    lod_max_block: int = 131072
    lod_extra_blocks: int = 2
    raw_extra_points: int = 24
    view_margin_ratio: float = 0.12
    render_coalesce_ms: int = 0
    fallback_plot_width_px: int = 1000
    force_disable_pg_autodownsample: bool = True


@dataclass
class _CursorConfig:
    line_width: float = 1.0
    label_font_pt: int = 11
    label_xshift_ratio: float = 0.006
    label_y_from_top_ratio: float = 0.25


@dataclass
class _ExportConfig:
    width_px: int = 1600
    height_per_plot_px: int = 300
    margin_px: int = 1
    scale: int = 3
    label_font_pt: int = 11
    tick_font_pt: int = 11               # FIX 3 : augmenté de 9 à 11 (duplicate supprimée)
    cursor_label_font_pt: int = 11
    target_blocks_per_px: float = 1.0
    full_detail_factor: float = 2.0
    left_axis_width_px: int = 72         # valeur de base ; sera ajustée dynamiquement
    bottom_axis_height_px: int = 44
    hide_minor_ticks: bool = True


@dataclass
class _AutoConfig:
    default_max_front_us: float = 0.05
    default_min_amp_pct: float = 15.0
    default_smooth_points: int = 2
    default_stability_factor: float = 8.0
    default_first_second_ratio: float = 0.35
    search_win_factor: int = 4
    max_candidates: int = 32
    peak_sigma: float = 3.0
    quiet_sigma: float = 1.5
    front_width_tol: float = 1.6
    preview_max_points: int = 200_000
    preview_top_candidates: int = 5
    refine_margin_front_factor: float = 6.0
    refine_min_points: int = 400


@dataclass
class _UIConfig:
    left_panel_width: int = 270
    right_panel_width: int = 400
    analysis_strip_width: int = 18
    analysis_plot_height: int = 230
    analysis_point_size: int = 9
    analysis_point_label_font_pt: int = 9


LIVE = _LiveConfig()
CURSOR = _CursorConfig()
EXPORT = _ExportConfig()
AUTO = _AutoConfig()
UI = _UIConfig()

# Rétrocompat : conserver les anciens noms pour les usages internes
LEFT_PANEL_WIDTH = UI.left_panel_width
RIGHT_PANEL_WIDTH = UI.right_panel_width
ANALYSIS_STRIP_WIDTH = UI.analysis_strip_width


# ============================================================
# HELPERS NUMÉRIQUES
# ============================================================

def _ensure_contiguous(a: np.ndarray) -> np.ndarray:
    if isinstance(a, np.ndarray) and a.flags["C_CONTIGUOUS"]:
        return a
    return np.ascontiguousarray(a)


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _next_pow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def _slice_visible(x: np.ndarray, y: np.ndarray, x0: float, x1: float):
    i0 = max(0, int(np.searchsorted(x, x0, side="left")))
    i1 = min(len(x), int(np.searchsorted(x, x1, side="right")))
    if i1 <= i0:
        return x[:0], y[:0]
    return x[i0:i1], y[i0:i1]


def _slice_visible_indices(x: np.ndarray, x0: float, x1: float):
    i0 = max(0, int(np.searchsorted(x, x0, side="left")))
    i1 = min(len(x), int(np.searchsorted(x, x1, side="right")))
    return i0, i1


def _downsample_minmax(x: np.ndarray, y: np.ndarray, target_blocks: int):
    n = int(len(y))
    if n <= 0:
        return x, y
    target_blocks = int(max(300, target_blocks))
    if n <= 2 * target_blocks:
        return x, y
    block = int(np.ceil(n / target_blocks))
    m = (n // block) * block
    if m <= 0:
        step = max(1, block)
        return x[::step], y[::step]
    x, y = x[:m], y[:m]
    yr = y.reshape(-1, block)
    xr = x.reshape(-1, block)
    x0_blk = xr[:, 0]
    x_out = np.repeat(x0_blk, 2)
    y_out = np.empty(x_out.shape[0], dtype=y.dtype)
    y_out[0::2] = yr.min(axis=1)
    y_out[1::2] = yr.max(axis=1)
    return x_out, y_out


def _build_lod_cache(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x)
    y = np.asarray(y)
    n = int(len(y))
    levels: dict = {}
    blocks: list = []

    if n <= 0:
        return {"raw_x": x, "raw_y": y, "levels": levels, "blocks": blocks}

    block = int(max(2, LIVE.lod_min_block))
    while block < n and block <= int(LIVE.lod_max_block):
        n_full = n // block
        cut = n_full * block
        if n_full <= 0:
            break

        xr = x[:cut].reshape(n_full, block)
        yr = y[:cut].reshape(n_full, block)
        x0_b = xr[:, 0].copy()
        mn = yr.min(axis=1).copy()
        mx = yr.max(axis=1).copy()

        if cut < n:
            tail = y[cut:]
            x0_b = np.concatenate([x0_b, [x[cut]]])
            mn = np.concatenate([mn, [np.min(tail)]])
            mx = np.concatenate([mx, [np.max(tail)]])

        levels[block] = {
            "block": block,
            "x0": _ensure_contiguous(x0_b),
            "mn": _ensure_contiguous(mn),
            "mx": _ensure_contiguous(mx),
        }
        blocks.append(block)
        block *= 2

    return {"raw_x": x, "raw_y": y, "levels": levels, "blocks": blocks}


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win))
    if win <= 1:
        return y
    if win % 2 == 0:
        win += 1
    return np.convolve(y, np.ones(win, dtype=float) / float(win), mode="same")


def _enforce_non_decreasing(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return y
    return np.maximum.accumulate(y)


def _robust_mad(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def _interp_crossing_time(
    x: np.ndarray, y: np.ndarray, level: float, i0: int, i1: int
) -> Optional[float]:
    i0 = max(0, int(i0))
    i1 = min(len(y) - 1, int(i1))
    if i1 <= i0:
        return None
    ys = y[i0:i1 + 1]
    xs = x[i0:i1 + 1]
    s = ys - level
    for k in range(len(s) - 1):
        s0, s1 = float(s[k]), float(s[k + 1])
        if s0 == 0:
            return float(xs[k])
        if (s0 < 0 <= s1) or (s0 > 0 >= s1):
            y0, y1 = float(ys[k]), float(ys[k + 1])
            x0v, x1v = float(xs[k]), float(xs[k + 1])
            if y1 == y0:
                return x0v
            return x0v + (level - y0) / (y1 - y0) * (x1v - x0v)
    return None


def _pick_unit(values: np.ndarray, kind: str):
    """Retourne (factor, unit) pour avoir des ticks lisibles."""
    if values.size == 0:
        return (1e3, "ms") if kind == "time" else (1.0, "mm")
    vmax = float(np.max(np.abs(values)))
    if vmax <= 0:
        return (1e3, "ms") if kind == "time" else (1.0, "mm")

    if kind == "time":
        candidates = [("s", 1.0), ("ms", 1e3), ("µs", 1e6), ("ns", 1e9)]
    else:
        candidates = [("m", 1e-3), ("mm", 1.0), ("µm", 1e3)]

    best_unit, best_k, best_score = candidates[0][0], candidates[0][1], 1e18
    for unit, k in candidates:
        vv = vmax * k
        score = (1 / vv) if vv < 1 else (vv / 5000 if vv > 5000 else 0)
        if score < best_score:
            best_score, best_unit, best_k = score, unit, k
    return best_k, best_unit


# ============================================================
# PCHIP (interpolation monotone cubique par morceaux)
# ============================================================

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=float)
    if n == 2:
        d = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([d, d], dtype=float)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    # extrémités
    for side in ("left", "right"):
        if side == "left":
            d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
            if np.sign(d0) != np.sign(delta[0]):
                d0 = 0.0
            elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3.0 * delta[0]):
                d0 = 3.0 * delta[0]
            d[0] = d0
        else:
            dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
            if np.sign(dn) != np.sign(delta[-1]):
                dn = 0.0
            elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3.0 * delta[-1]):
                dn = 3.0 * delta[-1]
            d[-1] = dn
    return d


def _pchip_piecewise_coeffs(x: np.ndarray, y: np.ndarray) -> list:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return []
    d = _pchip_slopes(x, y)
    coeffs = []
    for i in range(n - 1):
        h = float(x[i + 1] - x[i])
        if h == 0:
            continue
        delta = float((y[i + 1] - y[i]) / h)
        c0 = float(y[i])
        c1 = float(d[i])
        c2 = float((3.0 * delta - 2.0 * d[i] - d[i + 1]) / h)
        c3 = float((d[i] + d[i + 1] - 2.0 * delta) / (h * h))
        coeffs.append({"x0": float(x[i]), "x1": float(x[i + 1]),
                        "c0": c0, "c1": c1, "c2": c2, "c3": c3})
    return coeffs


def _pchip_eval_dense(x: np.ndarray, coeffs: list, points_per_seg: int = 40):
    if not coeffs:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs, ys = [], []
    for i, c in enumerate(coeffs):
        n = max(8, int(points_per_seg))
        tt = np.linspace(c["x0"], c["x1"], n, endpoint=True)
        if i > 0:
            tt = tt[1:]
        t = tt - c["x0"]
        yy = c["c0"] + c["c1"] * t + c["c2"] * t * t + c["c3"] * t * t * t
        xs.append(tt)
        ys.append(yy)
    return np.concatenate(xs), np.concatenate(ys)


# ============================================================
# AUTO FRONT DETECTION
# ============================================================

def _make_auto_preview_signal(x: np.ndarray, y: np.ndarray, max_points: int = AUTO.preview_max_points):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= max_points:
        return x, y, 1
    block = int(np.ceil(n / float(max_points)))
    m = (n // block) * block
    if m <= 0:
        return x, y, 1
    xr = x[:m].reshape(-1, block)
    yr = y[:m].reshape(-1, block)
    x_preview = xr[:, block // 2].copy()
    y_preview = yr.mean(axis=1).copy()
    if m < n:
        x_preview = np.concatenate([x_preview, [x[m + (n - m) // 2]]])
        y_preview = np.concatenate([y_preview, [np.mean(y[m:])]])
    return x_preview, y_preview, block


def _rolling_mean_std_from_cumsum(y: np.ndarray, win: int):
    win = int(max(1, win))
    yy = np.asarray(y, dtype=float)
    n = len(yy)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if win <= 1:
        return yy.copy(), np.zeros_like(yy)
    if n < win:
        m = float(np.mean(yy))
        s = float(np.std(yy))
        return np.array([m], dtype=float), np.array([s], dtype=float)
    c1 = np.concatenate(([0.0], np.cumsum(yy)))
    c2 = np.concatenate(([0.0], np.cumsum(yy * yy)))
    s1 = c1[win:] - c1[:-win]
    s2 = c2[win:] - c2[:-win]
    mean = s1 / float(win)
    var = np.maximum(0.0, s2 / float(win) - mean * mean)
    return mean, np.sqrt(var)


def _local_maxima_indices(a: np.ndarray, min_value: float) -> np.ndarray:
    if a.size == 0:
        return np.array([], dtype=int)
    if a.size == 1:
        return np.array([0], dtype=int) if a[0] >= min_value else np.array([], dtype=int)
    if a.size == 2:
        k = int(np.argmax(a))
        return np.array([k], dtype=int) if a[k] >= min_value else np.array([], dtype=int)
    m = (a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] >= float(min_value))
    return np.where(m)[0] + 1


def _merge_close_candidates(cands: list, merge_dt: float) -> list:
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c["t_pos"])
    merged = [cands[0]]
    for c in cands[1:]:
        prev = merged[-1]
        if abs(c["t_pos"] - prev["t_pos"]) <= float(merge_dt):
            if c["score"] > prev["score"]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _crossing_time_raw_near_level(
    x: np.ndarray, y: np.ndarray, level: float, i0: int, i1: int
) -> Optional[float]:
    i0, i1 = max(0, int(i0)), min(len(y) - 1, int(i1))
    if i1 <= i0:
        return None
    ys = y[i0:i1 + 1]
    xs = x[i0:i1 + 1]
    s = ys - level
    best, best_abs = None, None
    for k in range(len(s) - 1):
        s0, s1 = float(s[k]), float(s[k + 1])
        if s0 == 0.0:
            return float(xs[k])
        if not ((s0 < 0.0 <= s1) or (s0 > 0.0 >= s1) or s1 == 0.0):
            continue
        x0v, x1v = float(xs[k]), float(xs[k + 1])
        y0v, y1v = float(ys[k]), float(ys[k + 1])
        t = x0v if y1v == y0v else x0v + (level - y0v) / (y1v - y0v) * (x1v - x0v)
        closeness = abs(s0) + abs(s1)
        if best is None or closeness < best_abs:
            best, best_abs = t, closeness
    return best


def _refine_edge_position_raw(
    x: np.ndarray, y: np.ndarray,
    y_before: float, y_after: float,
    center_idx: int, half_win: int, mode: str
):
    i0 = max(0, int(center_idx) - int(half_win))
    i1 = min(len(y) - 1, int(center_idx) + int(half_win))
    if i1 <= i0 + 1:
        return None, None, None, None

    level10 = y_before + 0.1 * (y_after - y_before)
    level50 = y_before + 0.5 * (y_after - y_before)
    level90 = y_before + 0.9 * (y_after - y_before)

    t10 = _crossing_time_raw_near_level(x, y, level10, i0, i1)
    t50 = _crossing_time_raw_near_level(x, y, level50, i0, i1)
    t90 = _crossing_time_raw_near_level(x, y, level90, i0, i1)

    dy = np.diff(y[i0:i1 + 1])
    tpk = float(x[int(np.argmax(np.abs(dy))) + i0]) if dy.size > 0 else None

    if str(mode or "50").lower() == "maxslope":
        return tpk, t10, t50, t90
    return t50, t10, t50, t90


def _detect_best_front_v3(
    sig: dict,
    max_front_s: float,
    min_amp_pct: float = 15.0,
    smooth_points: int = 5,
    stability_factor: float = 8.0,
    prefer_first: bool = True,
    ignore_first_if_weak: bool = True,
    first_second_ratio: float = 0.35,
    position_mode: str = "50",
) -> Optional[dict]:
    x = np.asarray(sig.get("x", []), dtype=float)
    y = np.asarray(sig.get("y", []), dtype=float)
    if x.size < 12 or y.size < 12 or x.size != y.size:
        return None

    dx = np.diff(x[:min(len(x), 5000)])
    dx = dx[np.isfinite(dx)]
    if dx.size <= 0:
        return None
    dt = float(np.median(dx))
    if not np.isfinite(dt) or dt <= 0:
        return None

    full_range = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(full_range) or full_range <= 0:
        return None

    amp_min_abs = float(min_amp_pct) * 0.01 * full_range
    smooth_n = min(max(1, int(smooth_points)) | 1, 9)
    ys = y if smooth_n <= 1 else _moving_average(y, smooth_n)

    front_samples = max(1, min(int(round(float(max_front_s) / dt)), max(1, len(y) // 30)))
    plateau_win = max(3, min(
        int(round(max(3.0, stability_factor) * max(1, front_samples))),
        max(3, len(y) // 16)
    ))
    gap_win = max(1, min(2, front_samples))

    if len(ys) < 2 * plateau_win + 2 * gap_win + 3:
        return None

    mean_valid, std_valid = _rolling_mean_std_from_cumsum(ys, plateau_win)
    candidates = []
    amp_trace = []
    center_indices = []

    max_sb = len(ys) - (2 * plateau_win + 2 * gap_win)
    for sb in range(0, max_sb):
        sa = sb + plateau_win + 2 * gap_win
        y_before = float(mean_valid[sb])
        y_after = float(mean_valid[sa])
        amp = abs(y_after - y_before)
        center = sb + plateau_win + gap_win
        amp_trace.append(amp)
        center_indices.append(center)
        candidates.append({
            "sb": sb, "sa": sa, "center": center,
            "y_before": y_before, "y_after": y_after,
            "std_before": float(std_valid[sb]),
            "std_after": float(std_valid[sa]),
            "amp": float(amp),
        })

    if not candidates:
        return None

    amp_trace = np.asarray(amp_trace, dtype=float)
    amp_med = float(np.median(amp_trace))
    amp_mad = _robust_mad(amp_trace)
    peak_thresh = max(amp_min_abs, amp_med + 2.0 * max(amp_mad, 1e-18))
    pk_local = _local_maxima_indices(amp_trace, min_value=peak_thresh)

    if pk_local.size == 0:
        order = np.argsort(amp_trace)[::-1][:min(AUTO.max_candidates, len(amp_trace))]
        pk_local = np.sort(order)
    if pk_local.size > AUTO.max_candidates:
        order = np.argsort(amp_trace[pk_local])[::-1][:AUTO.max_candidates]
        pk_local = np.sort(pk_local[order])

    out = []
    for ip in pk_local:
        base = candidates[int(ip)]
        amp = float(base["amp"])
        if not np.isfinite(amp) or amp < amp_min_abs:
            continue

        center = int(base["center"])
        y_before = float(base["y_before"])
        y_after = float(base["y_after"])
        refine_half_win = max(2, min(8, 3 * max(1, front_samples)))

        t_pos, t10, t50, t90 = _refine_edge_position_raw(
            x, y, y_before=y_before, y_after=y_after,
            center_idx=center, half_win=refine_half_win, mode=position_mode,
        )
        if t_pos is None:
            continue

        width = float("inf")
        if t10 is not None and t90 is not None:
            width = abs(float(t90) - float(t10)) or dt

        if np.isfinite(width) and width > max(3.0 * max_front_s, 6.0 * dt):
            continue

        i0 = max(0, center - refine_half_win)
        i1 = min(len(y) - 1, center + refine_half_win)
        dy = np.diff(y[i0:i1 + 1])
        slope_metric = float(np.max(np.abs(dy))) / dt if dy.size > 0 else 0.0
        peak_idx = int(i0 + np.argmax(np.abs(dy))) if dy.size > 0 else center

        stability_before = amp / (base["std_before"] + 1e-18)
        stability_after = amp / (base["std_after"] + 1e-18)
        amp_norm = amp / max(full_range, 1e-18)
        score = (
            0.72 * amp_norm
            + 0.14 * min(stability_before / 20.0, 1.0)
            + 0.06 * min(stability_after / 20.0, 1.0)
            + 0.08 * min(slope_metric / max(full_range / max(dt, 1e-18), 1e-18), 1.0)
        )

        out.append({
            "peak_idx": peak_idx,
            "t_center": float(x[center]),
            "t_pos": float(t_pos),
            "t10": None if t10 is None else float(t10),
            "t50": None if t50 is None else float(t50),
            "t90": None if t90 is None else float(t90),
            "width": float(width) if np.isfinite(width) else float("inf"),
            "amp": float(amp),
            "y_before": float(y_before),
            "y_after": float(y_after),
            "direction": "rising" if y_after > y_before else "falling",
            "slope_metric": float(slope_metric),
            "stability_before": float(stability_before),
            "stability_after": float(stability_after),
            "score": float(score),
        })

    if not out:
        return None

    out = _merge_close_candidates(out, merge_dt=max(4.0 * dt, 2.0 * max_front_s))
    if not out:
        return None

    out_by_score = sorted(out, key=lambda c: c["score"], reverse=True)
    best = out_by_score[0]
    best_score, best_amp = float(best["score"]), float(best["amp"])

    strong = [
        c for c in out
        if c["amp"] >= max(amp_min_abs, 0.60 * best_amp)
        and c["score"] >= 0.50 * best_score
    ] or out[:]

    strong_by_time = sorted(strong, key=lambda c: c["t_pos"])
    chosen = strong_by_time[0] if bool(prefer_first) else best

    if bool(prefer_first) and bool(ignore_first_if_weak) and len(strong_by_time) >= 2:
        c1, c2 = strong_by_time[0], strong_by_time[1]
        dt12 = float(c2["t_pos"] - c1["t_pos"])
        amp_ratio = float(c1["amp"] / max(c2["amp"], 1e-18))
        if 0.0 <= dt12 <= 10.0 * float(max_front_s) and amp_ratio < float(first_second_ratio):
            chosen = c2

    chosen["amp_min_abs"] = float(amp_min_abs)
    chosen["full_range"] = float(full_range)
    chosen["candidates"] = sorted(out, key=lambda c: c["t_pos"])
    return chosen


def _detect_front_candidates_preview(sig, max_front_s, min_amp_pct, smooth_points,
                                      stability_factor, prefer_first, ignore_first_if_weak,
                                      first_second_ratio, position_mode,
                                      top_k=AUTO.preview_top_candidates):
    x = np.asarray(sig.get("x", []), dtype=float)
    y = np.asarray(sig.get("y", []), dtype=float)
    if x.size < 20 or y.size < 20 or x.size != y.size:
        return []

    xp, yp, _block = _make_auto_preview_signal(x, y)
    sig_preview = {"x": xp, "y": yp, "display": sig.get("display", ""),
                   "plot_display": sig.get("plot_display", "")}

    best = _detect_best_front_v3(
        sig_preview, max_front_s=max_front_s, min_amp_pct=min_amp_pct,
        smooth_points=smooth_points, stability_factor=stability_factor,
        prefer_first=prefer_first, ignore_first_if_weak=ignore_first_if_weak,
        first_second_ratio=first_second_ratio, position_mode=position_mode,
    )
    if best is None:
        return []

    cands = best.get("candidates", [])
    if not cands:
        return [best]
    return sorted(cands, key=lambda c: c.get("score", 0.0), reverse=True)[:max(1, int(top_k))]


def _extract_local_sig_window(sig: dict, t_center: float, max_front_s: float) -> Optional[dict]:
    x = np.asarray(sig.get("x", []), dtype=float)
    y = np.asarray(sig.get("y", []), dtype=float)
    if x.size < 4 or y.size != x.size:
        return None

    dx = np.diff(x[:min(len(x), 5000)])
    dx = dx[np.isfinite(dx)]
    if dx.size <= 0:
        return None
    dt = float(np.median(dx))
    if not np.isfinite(dt) or dt <= 0:
        return None

    margin = max(AUTO.refine_margin_front_factor * float(max_front_s),
                 AUTO.refine_min_points * dt)
    t0 = float(t_center) - margin
    t1 = float(t_center) + margin
    i0 = max(0, int(np.searchsorted(x, t0, side="left")))
    i1 = min(len(x), int(np.searchsorted(x, t1, side="right")))

    if i1 - i0 < 8:
        return None
    return {"x": x[i0:i1], "y": y[i0:i1],
            "display": sig.get("display", ""), "plot_display": sig.get("plot_display", "")}


def _detect_best_front_v3_coarse_to_fine(
    sig: dict, max_front_s: float,
    min_amp_pct: float = 15.0, smooth_points: int = 5,
    stability_factor: float = 8.0, prefer_first: bool = True,
    ignore_first_if_weak: bool = True, first_second_ratio: float = 0.35,
    position_mode: str = "50",
) -> Optional[dict]:
    x = np.asarray(sig.get("x", []), dtype=float)

    if x.size <= AUTO.preview_max_points:
        return _detect_best_front_v3(
            sig, max_front_s=max_front_s, min_amp_pct=min_amp_pct,
            smooth_points=smooth_points, stability_factor=stability_factor,
            prefer_first=prefer_first, ignore_first_if_weak=ignore_first_if_weak,
            first_second_ratio=first_second_ratio, position_mode=position_mode,
        )

    preview_cands = _detect_front_candidates_preview(
        sig, max_front_s=max_front_s, min_amp_pct=min_amp_pct,
        smooth_points=smooth_points, stability_factor=stability_factor,
        prefer_first=prefer_first, ignore_first_if_weak=ignore_first_if_weak,
        first_second_ratio=first_second_ratio, position_mode=position_mode,
        top_k=AUTO.preview_top_candidates,
    )
    if not preview_cands:
        return None

    refined = []
    for c in preview_cands:
        t_guess = float(c.get("t_pos", c.get("t_center", 0.0)))
        local_sig = _extract_local_sig_window(sig, t_guess, max_front_s=max_front_s)
        if local_sig is None:
            continue
        r = _detect_best_front_v3(
            local_sig, max_front_s=max_front_s, min_amp_pct=min_amp_pct,
            smooth_points=smooth_points, stability_factor=stability_factor,
            prefer_first=prefer_first, ignore_first_if_weak=ignore_first_if_weak,
            first_second_ratio=first_second_ratio, position_mode=position_mode,
        )
        if r is not None:
            refined.append(r)

    if not refined:
        return None

    refined = sorted(refined, key=lambda d: d.get("score", 0.0), reverse=True)
    best = refined[0]
    best_score = float(best.get("score", 0.0))
    best_amp = float(best.get("amp", 0.0))

    strong = [
        c for c in refined
        if float(c.get("amp", 0.0)) >= max(1e-18, 0.60 * best_amp)
        and float(c.get("score", 0.0)) >= 0.50 * best_score
    ] or refined[:]

    strong_by_time = sorted(strong, key=lambda c: c.get("t_pos", 0.0))
    chosen = best

    if bool(prefer_first):
        chosen = strong_by_time[0]

    if bool(prefer_first) and bool(ignore_first_if_weak) and len(strong_by_time) >= 2:
        c1, c2 = strong_by_time[0], strong_by_time[1]
        dt12 = float(c2["t_pos"] - c1["t_pos"])
        amp_ratio = float(c1["amp"] / max(c2["amp"], 1e-18))
        if 0.0 <= dt12 <= 10.0 * float(max_front_s) and amp_ratio < float(first_second_ratio):
            chosen = c2

    chosen["coarse_to_fine"] = True
    chosen["refined_candidates"] = refined
    return chosen


# ============================================================
# AXE TEMPS
# ============================================================

class TimeAxis(pg.AxisItem):
    def _unit_for_spacing(self, spacing_s: float):
        sp = max(float(abs(spacing_s or 1.0)), 1e-30)
        if sp >= 1.0:   return "s",  1.0
        if sp >= 1e-3:  return "ms", 1e3
        if sp >= 1e-6:  return "µs", 1e6
        return "ns", 1e9

    def tickStrings(self, values, scale, spacing):
        unit, k = self._unit_for_spacing(spacing)
        return [f"{float(v) * k:.6g} {unit}" for v in values]


# ============================================================
# WFM READER
# ============================================================

def _sample_array(a: np.ndarray, max_points: int = 8192) -> np.ndarray:
    n = int(len(a))
    if n <= max_points:
        return np.asarray(a)
    return np.asarray(a[np.linspace(0, n - 1, max_points, dtype=int)])


def _sample_monotonic_score(a: np.ndarray) -> float:
    s = _sample_array(a, max_points=4096)
    if len(s) < 8:
        return 0.0
    d = np.diff(s)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(max(np.mean(d > 0), np.mean(d < 0)))


def read_wfm_as_xy(path: str):
    """
    Reader WFM optimisé pour Tektronix.
    Retourne (sid, display, x, y, meta).
    """
    t_all0 = perf_counter()
    t0 = perf_counter()
    wf = read_file(path)
    t1 = perf_counter()

    def is_numeric_1d(a):
        return (isinstance(a, np.ndarray) and a.ndim == 1
                and a.size > 20 and np.issubdtype(a.dtype, np.number))

    visited: set = set()
    arrays: list = []
    scalars: dict = {}
    x_hints: list = []

    X_KEYS = ("normalized_horizontal_values", "x_axis", "xaxis", "time", ".x", "t_axis", "horizontal")

    def walk(obj, prefix="", depth=0, max_depth=4):
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        if is_numeric_1d(obj):
            arrays.append((obj.size, prefix, obj))
            if any(k in prefix.lower() for k in X_KEYS):
                x_hints.append((obj.size, prefix, obj))
            return

        if isinstance(obj, (int, float, np.integer, np.floating)):
            scalars[prefix.strip(".")] = float(obj)
            return

        if depth >= max_depth:
            return

        for name, val in (obj.items() if isinstance(obj, dict)
                          else _iter_public(obj)):
            p = f"{prefix}.{name}" if prefix else name
            if is_numeric_1d(val):
                arrays.append((val.size, p, val))
                if any(k in name.lower() for k in X_KEYS):
                    x_hints.append((val.size, p, val))
            else:
                walk(val, p, depth + 1, max_depth)

    def _attr_safe(obj, name):
        try:
            v = getattr(obj, name)
            return None if callable(v) else v
        except Exception:
            return None

    def _iter_public(obj):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            v = _attr_safe(obj, name)
            if v is not None:
                yield name, v

    walk(wf)
    t3 = perf_counter()

    if not arrays:
        raise RuntimeError("Aucun tableau 1D numérique trouvé.")

    def score_y(arr, apath):
        a = np.asarray(arr)
        if a.size < 200:
            return -1e18
        p = apath.lower()
        var = float(np.var(_sample_array(a, 8192)))
        mono = _sample_monotonic_score(a)
        score = 0.8 * math.log10(max(a.size, 2)) + 0.25 * math.log10(var + 1e-30)
        if "normalized_vertical_values" in p: score += 40.0
        elif "y_axis_values" in p: score += 24.0
        elif any(k in p for k in ("curve", "waveform", "samples", "data", "values")): score += 8.0
        if any(k in p for k in ("time", "x_axis", "xaxis", ".x", "axis", "index",
                                  "point", "sample", "horizontal")): score -= 30.0
        if mono > 0.995: score -= 25.0
        return score

    t4 = perf_counter()
    arrays_sorted = sorted(arrays, key=lambda t: t[0], reverse=True)

    best_y, best_y_path, best_score = None, None, -1e18
    for _, apath, arr in arrays_sorted:
        s = score_y(arr, apath)
        if s > best_score:
            best_score, best_y, best_y_path = s, np.asarray(arr), apath

    if best_y is None:
        raise RuntimeError("Impossible de choisir y.")

    y = best_y
    t5 = perf_counter()

    x, x_path = None, None

    # Chercher dans les x_hints d'abord
    for size, apath, arr in sorted(x_hints, key=lambda t: t[0], reverse=True):
        if size == y.size and apath != best_y_path:
            x, x_path = np.asarray(arr), apath
            break

    # Fallback : chercher dans tous les arrays
    if x is None:
        for size, apath, arr in arrays_sorted:
            if size == y.size and apath != best_y_path:
                pl = apath.lower()
                if any(k in pl for k in X_KEYS):
                    x, x_path = np.asarray(arr), apath
                    break

    # Fallback : reconstruire depuis scalaires
    if x is None:
        dt_keys = ("sample_interval", "x_increment", "x_incr", "xincr", "dt", "delta_t", "time_increment")
        t0_keys = ("x_offset", "x_origin", "x_zero", "t0", "time_origin")

        def find_scalar(keys):
            for kk, vv in scalars.items():
                if any(kk.lower().endswith(k) for k in keys):
                    return vv, kk
            return None, None

        dt_val, _ = find_scalar(dt_keys)
        t0_val, _ = find_scalar(t0_keys)
        if t0_val is None:
            t0_val = 0.0
        if dt_val is None or dt_val == 0:
            raise RuntimeError("dt introuvable (axe temps).")
        x = float(t0_val) + np.arange(len(y), dtype=float) * float(dt_val)

    t7 = perf_counter()

    if x.size != y.size:
        raise RuntimeError(f"x/y tailles différentes ({x.size} vs {y.size}).")

    display = os.path.basename(path)
    plot_display = os.path.splitext(display)[0]
    meta = {
        "y_source": best_y_path,
        "x_source": x_path,
        "plot_display": plot_display,
        "reader_mode": "generic_optimized",
    }

    t_all1 = perf_counter()
    print(
        f"[WFM] {display} | "
        f"read={t1-t0:.3f}s walk={t3-t3:.3f}s y={t5-t4:.3f}s x={t7-t5:.3f}s | "
        f"total={t_all1-t_all0:.3f}s | n={len(y):,} | y='{best_y_path}'"
        + (f" | x='{x_path}'" if x_path else "")
    )

    return os.path.abspath(path), display, np.asarray(x), np.asarray(y), meta


# ============================================================
# COPY-PASTE TABLE
# ============================================================

class CopyPasteTableWidget(QTableWidget):
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self._copy_selection()
            return
        if event.matches(QKeySequence.Paste):
            self._paste_selection()
            return
        super().keyPressEvent(event)

    def _copy_selection(self):
        sel = self.selectedRanges()
        if not sel:
            return
        r = sel[0]
        rows = []
        for i in range(r.topRow(), r.bottomRow() + 1):
            cols = [
                "" if (item := self.item(i, j)) is None else item.text()
                for j in range(r.leftColumn(), r.rightColumn() + 1)
            ]
            rows.append("\t".join(cols))
        QGuiApplication.clipboard().setText("\n".join(rows))

    def _paste_selection(self):
        text = QGuiApplication.clipboard().text()
        if not text:
            return
        sel = self.selectedRanges()
        if not sel:
            return
        r0, c0 = sel[0].topRow(), sel[0].leftColumn()
        for i, line in enumerate(text.splitlines()):
            if r0 + i >= self.rowCount():
                break
            for j, val in enumerate(line.split("\t")):
                cc = c0 + j
                if cc >= self.columnCount():
                    break
                it = self.item(r0 + i, cc) or QTableWidgetItem()
                it.setText(val)
                self.setItem(r0 + i, cc, it)


# ============================================================
# CURSOR SERIES WIDGET
# ============================================================

class CursorSeriesWidget(QWidget):
    changed = Signal()
    reset_requested = Signal()

    # Seuils de couleur pour la colonne vitesse (m/s)
    _SPEED_THRESHOLDS = [(500, "#c8f5c8"), (1500, "#fff3cd"), (3000, "#ffd6a5"), (None, "#ffcccc")]

    def __init__(self, default_name=""):
        super().__init__()
        self.series_name = default_name
        self.mode = "ref1"
        self.rows = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Série :"))
        self.ed_name = QLineEdit(self.series_name)
        self.ed_name.setPlaceholderText("Nom (optionnel)")
        row_name.addWidget(self.ed_name, 1)
        lay.addLayout(row_name)

        lay.addWidget(QLabel("Référence vitesse :"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Vitesse par rapport au curseur 1", userData="ref1")
        self.cb_mode.addItem("Vitesse par rapport au curseur précédent", userData="prev")
        lay.addWidget(self.cb_mode)

        row_cursors = QHBoxLayout()
        row_cursors.addWidget(QLabel("Curseurs"))
        self.btn_add = QToolButton()
        self.btn_add.setText("+")
        self.btn_del = QToolButton()
        self.btn_del.setText("–")
        self.btn_reset = QPushButton("RAZ positions")
        row_cursors.addWidget(self.btn_add)
        row_cursors.addWidget(self.btn_del)
        row_cursors.addWidget(self.btn_reset)
        row_cursors.addStretch(1)
        lay.addLayout(row_cursors)

        self.table = CopyPasteTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Désignation", "Abscisse (mm)", "Temps1 (ms)", "Vitesse (m/s)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        hdr = self.table.horizontalHeader()
        for col, mode in enumerate([QHeaderView.ResizeToContents] * 4):
            hdr.setSectionResizeMode(col, mode)
        for col, w in enumerate([90, 78, 86, 78]):
            self.table.setColumnWidth(col, w)

        lay.addWidget(self.table, 1)

        self.ed_name.textChanged.connect(self._on_name_changed)
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_add.clicked.connect(self.add_cursor)
        self.btn_del.clicked.connect(self.remove_selected_cursor)
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        self.table.itemChanged.connect(self._on_item_changed)

    def _on_name_changed(self, txt):
        self.series_name = txt.strip()
        self.changed.emit()

    def _on_mode_changed(self):
        self.mode = self.cb_mode.currentData() or "ref1"
        self._refresh_table(update_only=True)
        self.changed.emit()

    def add_cursor(self):
        n = len(self.rows) + 1
        self.rows.append({"name": f"c{n}", "d_mm": 0.0, "t_s": None})
        self._refresh_table()
        self.changed.emit()

    def remove_selected_cursor(self):
        r = self.table.currentRow()
        if 0 <= r < len(self.rows):
            self.rows.pop(r)
            self._refresh_table()
            self.changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()
        if not 0 <= r < len(self.rows):
            return
        if c == 0:
            self.rows[r]["name"] = item.text().strip() or self.rows[r]["name"]
        elif c == 1:
            v = _safe_float(item.text().replace(",", "."))
            if v is None:
                self._refresh_table(update_only=True)
                return
            self.rows[r]["d_mm"] = float(v)
        self._refresh_table(update_only=True)
        self.changed.emit()

    def set_times_from_positions(self, positions_s: list):
        for i in range(min(len(self.rows), len(positions_s))):
            self.rows[i]["t_s"] = positions_s[i]
        self._refresh_table(update_only=True)

    def get_positions_s(self) -> list:
        return [row.get("t_s") for row in self.rows]

    def get_names(self) -> list:
        return [row.get("name", "") for row in self.rows]

    def get_analysis_points(self) -> list:
        pts = []
        for row in self.rows:
            t_s = row.get("t_s")
            d_mm = row.get("d_mm")
            nm = str(row.get("name", ""))
            if t_s is None or d_mm is None:
                continue
            try:
                t_s, d_mm = float(t_s), float(d_mm)
            except Exception:
                continue
            if np.isfinite(t_s) and np.isfinite(d_mm):
                pts.append((nm, t_s, d_mm))
        pts.sort(key=lambda p: p[1])
        return pts

    def _compute_speeds(self) -> list:
        n = len(self.rows)
        v = [None] * n
        for i in range(1, n):
            j = 0 if self.mode == "ref1" else (i - 1)
            t1 = self.rows[i].get("t_s")
            t0 = self.rows[j].get("t_s")
            d1 = self.rows[i].get("d_mm")
            d0 = self.rows[j].get("d_mm")
            if t1 is None or t0 is None:
                continue
            dt_ms = (float(t1) - float(t0)) * 1e3
            dd_mm = float(d1) - float(d0)
            if abs(dt_ms) < 1e-15:
                continue
            v[i] = dd_mm / dt_ms
        return v

    @staticmethod
    def _speed_bg(v: Optional[float]) -> Optional[str]:
        if v is None:
            return None
        av = abs(v)
        for threshold, color in CursorSeriesWidget._SPEED_THRESHOLDS:
            if threshold is None or av <= threshold:
                return color
        return None

    def _refresh_table(self, update_only=False):
        self.table.blockSignals(True)
        try:
            if not update_only:
                self.table.setRowCount(len(self.rows))
            speeds = self._compute_speeds()

            for r, row in enumerate(self.rows):
                def _get_or_create(col):
                    it = self.table.item(r, col)
                    if it is None:
                        it = QTableWidgetItem()
                        self.table.setItem(r, col, it)
                    return it

                it0 = _get_or_create(0)
                it0.setText(str(row.get("name", "")))
                it0.setFlags(it0.flags() | Qt.ItemIsEditable)

                it1 = _get_or_create(1)
                it1.setText(f"{float(row.get('d_mm', 0.0)):.2f}")
                it1.setFlags(it1.flags() | Qt.ItemIsEditable)

                t_s = row.get("t_s")
                it2 = _get_or_create(2)
                it2.setText("—" if t_s is None else f"{float(t_s) * 1e3:.6f}")
                it2.setFlags(it2.flags() & ~Qt.ItemIsEditable)

                it3 = _get_or_create(3)
                spd = speeds[r]
                if r == 0:
                    it3.setText("")
                    it3.setBackground(pg.mkBrush(200, 200, 200))
                else:
                    it3.setText("—" if spd is None else f"{spd:.2f}")
                    bg = self._speed_bg(spd)
                    if bg:
                        c = QColor(bg)
                        it3.setBackground(pg.mkBrush(c.red(), c.green(), c.blue()))
                    else:
                        it3.setBackground(pg.mkBrush(255, 255, 255))
                it3.setFlags(it3.flags() & ~Qt.ItemIsEditable)
        finally:
            self.table.blockSignals(False)


# ============================================================
# ANALYSIS TOGGLE STRIP
# ============================================================

class AnalysisToggleStrip(QPushButton):
    toggled_request = Signal()

    def __init__(self):
        super().__init__(">\n>\n>")
        self.setFixedWidth(ANALYSIS_STRIP_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self.toggled_request.emit)
        self.setStyleSheet("""
            QPushButton {
                border: 1px solid #8a8a8a;
                border-radius: 3px;
                padding: 0px;
                background: #efefef;
                color: #202020;
                font-size: 8pt;
            }
            QPushButton:hover { background: #e2e2e2; }
        """)


# ============================================================
# COSMETIC INFINITE LINE  — FIX 1 : pen cosmétique à largeur entière
# ============================================================

class CosmeticInfiniteLine(pg.InfiniteLine):
    """
    Curseur vertical à épaisseur fixe en pixels écran.

    Stratégie robuste en 2 parties :

    A) On arrondit la largeur à un entier (≥ 2px) pour éliminer les
       problèmes de rendu sub-pixel qui causaient le « clignotement »
       aléatoire selon la position du curseur à l'écran.
       Avec une largeur entière, Qt n'a pas d'ambiguïté sur le nombre
       de pixels à remplir → épaisseur visuellement constante.

    B) On force le flag cosmetic sur TOUS les attributs pen internes
       de pyqtgraph (self.pen, self.currentPen, self.hoverPen) à
       chaque point d'entrée possible : __init__, setPen, paint,
       viewTransformChanged, boundingRect.
    """
    def __init__(self, *args, pixel_width: float = 1.35, **kwargs):
        # Arrondir à l'entier supérieur ≥ 2 pour un rendu pixel-perfect
        self._pixel_width = float(max(2.0, math.ceil(pixel_width)))
        self._lock = False
        super().__init__(*args, **kwargs)
        self._lock = True
        self._stamp_all_pens()

    def _stamp_pen(self, pen):
        """Applique cosmetic + largeur fixe sur un QPen, in-place."""
        pen.setCosmetic(True)
        pen.setWidthF(self._pixel_width)

    def _stamp_all_pens(self):
        """Force cosmetic sur tous les pens internes connus."""
        for attr in ("pen", "currentPen", "hoverPen"):
            p = getattr(self, attr, None)
            if p is not None:
                self._stamp_pen(p)

    def setPen(self, *args, **kwargs):
        super().setPen(*args, **kwargs)
        if self._lock:
            self._stamp_all_pens()

    def setHoverPen(self, *args, **kwargs):
        super().setHoverPen(*args, **kwargs)
        if self._lock:
            self._stamp_all_pens()

    def viewTransformChanged(self):
        super().viewTransformChanged()
        if self._lock:
            self._stamp_all_pens()

    def boundingRect(self):
        br = super().boundingRect()
        if self._lock:
            self._stamp_all_pens()
        return br

    def paint(self, p, *args):
        # Dernière ligne de défense : forcer le pen juste avant le rendu
        self._stamp_all_pens()
        super().paint(p, *args)


# ============================================================
# SCOPE STACK
# ============================================================

class ScopeStack(QWidget):
    TEK_COLORS = [
        (255, 215, 0),
        (0, 190, 255),
        (255, 60, 60),
        (120, 255, 120),
    ]
    SERIES_COLORS = [
        (255, 170, 0), (0, 255, 255), (255, 0, 255),
        (120, 255, 120), (255, 60, 60), (255, 215, 0),
        (0, 190, 255), (200, 120, 255),
    ]
    CURSOR_PERC_PATTERN = [0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60]

    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self.plots: list = []
        self.curves: list = []
        self.active_map: list = []

        self.cursor_lines: list = []
        self.cursor_text_items: list = []
        self.cursor_positions: list = []
        self.cursor_labels: list = []
        self.cursor_series_color = self.SERIES_COLORS[0]
        self.cursor_labels_visible: bool = True
        self.all_series_for_export: list = []

        self._cursor_callbacks: list = []
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._refresh_visible_data)
        self._last_render_keys: list = []
        self._last_render_blocks: list = []

        self._build_plots(1)

    # ---- callbacks curseurs ----
    def on_cursor_changed(self, cb):
        self._cursor_callbacks.append(cb)

    def _emit_cursor(self):
        for cb in self._cursor_callbacks:
            cb()

    # ---- construction des plots ----
    def _make_plot(self) -> pg.PlotWidget:
        pw = pg.PlotWidget(axisItems={"bottom": TimeAxis(orientation="bottom")})
        pw.showGrid(x=True, y=True)
        pw.setMenuEnabled(False)
        pw.setAntialiasing(False)
        pw.setMouseEnabled(x=True, y=False)
        pw.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        return pw

    def _wire_master_plot_signals(self):
        if not self.plots:
            return
        vb0 = self.plots[0].getViewBox()
        vb0.sigXRangeChanged.connect(lambda *_: self._schedule_render_update())
        vb0.sigRangeChanged.connect(lambda *_: self._update_cursor_label_positions())
        vb0.sigResized.connect(self._schedule_render_update)

    def _schedule_render_update(self):
        self._render_timer.start(max(0, int(LIVE.render_coalesce_ms)))

    def set_y_unlocked(self, plot_idx: int, unlocked: bool):
        if 0 <= plot_idx < len(self.plots):
            self.plots[plot_idx].setMouseEnabled(x=True, y=bool(unlocked))

    def auto_y_for_plot(self, plot_idx: int):
        if not 0 <= plot_idx < len(self.active_map):
            return
        _ch, sig = self.active_map[plot_idx]
        y = sig.get("y")
        if y is None or len(y) == 0:
            return
        mn, mx = float(np.min(y)), float(np.max(y))
        if mn == mx:
            mn -= 0.5; mx += 0.5
        pad = 0.05 * (mx - mn)
        self.plots[plot_idx].getViewBox().setYRange(mn - pad, mx + pad, padding=0.0)

    def _clear_stack(self):
        for pw in self.plots:
            self._layout.removeWidget(pw)
            pw.deleteLater()
        self.plots.clear()
        self.curves.clear()
        self.cursor_lines.clear()
        self.cursor_text_items.clear()
        self._last_render_keys.clear()
        self._last_render_blocks.clear()

    def _build_plots(self, n: int, keep_xrange=None):
        self._clear_stack()
        for _ in range(n):
            pw = self._make_plot()
            self._layout.addWidget(pw, 1)
            self.plots.append(pw)
            self.curves.append(None)
            self._last_render_keys.append(None)
            self._last_render_blocks.append(None)

        for i in range(1, len(self.plots)):
            self.plots[i].setXLink(self.plots[0])

        if keep_xrange is not None and self.plots:
            self.plots[0].getViewBox().setXRange(*keep_xrange, padding=0.0)

        self._wire_master_plot_signals()
        self._rebuild_cursor_lines()

    def _current_xrange(self):
        if not self.plots:
            return 0.0, 1.0
        try:
            x0, x1 = self.plots[0].getViewBox().viewRange()[0]
            return float(x0), float(x1)
        except Exception:
            return 0.0, 1.0

    def _plot_width_px(self, plot_idx: int) -> int:
        if not 0 <= plot_idx < len(self.plots):
            return int(LIVE.fallback_plot_width_px)
        try:
            w = int(round(float(self.plots[plot_idx].getViewBox().sceneBoundingRect().width())))
            return max(50, w)
        except Exception:
            return int(LIVE.fallback_plot_width_px)

    def _configure_curve_item(self, curve):
        if curve is None:
            return
        try: curve.setClipToView(False)
        except Exception: pass
        if LIVE.force_disable_pg_autodownsample:
            try: curve.setDownsampling(auto=False)
            except Exception: pass
        try: curve.setSkipFiniteCheck(True)
        except Exception: pass

    def _expanded_visible_range(self, sig, x0: float, x1: float):
        x = sig["x"]
        if len(x) == 0:
            return x0, x1
        dx = float(x1 - x0)
        if dx <= 0:
            return x0, x1
        extra = float(LIVE.view_margin_ratio) * dx
        return max(float(x[0]), x0 - extra), min(float(x[-1]), x1 + extra)

    def _select_lod_block(self, sig, x0: float, x1: float, plot_width_px: int) -> int:
        if not LIVE.use_multires:
            return 1
        lod = sig.get("lod")
        if not lod:
            return 1

        raw_x = lod["raw_x"]
        i0, i1 = _slice_visible_indices(raw_x, x0, x1)
        nvis_raw = max(0, i1 - i0)
        if nvis_raw <= 0:
            return 1

        width_px = max(1, int(plot_width_px))
        samples_per_pixel = float(nvis_raw) / float(width_px)

        if samples_per_pixel <= float(LIVE.raw_max_samples_per_pixel):
            return 1

        wanted = _next_pow2(max(
            int(LIVE.lod_min_block),
            int(math.ceil(samples_per_pixel / max(1e-9, float(LIVE.target_blocks_per_pixel))))
        ))
        blocks = lod.get("blocks", [])
        if not blocks:
            return 1

        for b in blocks:
            if b >= wanted:
                return 1 if b <= int(LIVE.force_raw_if_block_leq) else int(b)
        last = int(blocks[-1])
        return 1 if last <= int(LIVE.force_raw_if_block_leq) else last

    def _get_visible_xy_for_sig(self, sig, x0: float, x1: float, plot_width_px: int):
        x0e, x1e = self._expanded_visible_range(sig, x0, x1)
        lod = sig.get("lod")

        if lod is None:
            x, y = sig["x"], sig["y"]
            i0, i1 = _slice_visible_indices(x, x0e, x1e)
            i0 = max(0, i0 - LIVE.raw_extra_points)
            i1 = min(len(x), i1 + LIVE.raw_extra_points)
            return x[i0:i1], y[i0:i1], 1, (i0, i1)

        raw_x, raw_y = lod["raw_x"], lod["raw_y"]
        block = self._select_lod_block(sig, x0e, x1e, plot_width_px)

        if block <= 1:
            i0, i1 = _slice_visible_indices(raw_x, x0e, x1e)
            i0 = max(0, i0 - LIVE.raw_extra_points)
            i1 = min(len(raw_x), i1 + LIVE.raw_extra_points)
            return raw_x[i0:i1], raw_y[i0:i1], 1, (i0, i1)

        lvl = lod["levels"].get(block)
        if lvl is None:
            i0, i1 = _slice_visible_indices(raw_x, x0e, x1e)
            i0 = max(0, i0 - LIVE.raw_extra_points)
            i1 = min(len(raw_x), i1 + LIVE.raw_extra_points)
            return raw_x[i0:i1], raw_y[i0:i1], 1, (i0, i1)

        bx, mn, mx = lvl["x0"], lvl["mn"], lvl["mx"]
        j0 = max(0, int(np.searchsorted(bx, x0e, side="left")) - int(LIVE.lod_extra_blocks))
        j1 = min(len(bx), int(np.searchsorted(bx, x1e, side="right")) + int(LIVE.lod_extra_blocks))

        if j1 <= j0:
            return raw_x[:0], raw_y[:0], int(block), (j0, j1)

        bxv, mnv, mxv = bx[j0:j1], mn[j0:j1], mx[j0:j1]
        x_out = np.empty(len(bxv) * 2, dtype=bxv.dtype)
        y_out = np.empty(len(bxv) * 2, dtype=mnv.dtype)
        x_out[0::2] = x_out[1::2] = bxv
        y_out[0::2] = mnv
        y_out[1::2] = mxv
        return x_out, y_out, int(block), (j0, j1)

    def _refresh_visible_data(self, force=False):
        if not self.plots:
            return
        x0, x1 = self._current_xrange()

        for plot_idx in range(min(len(self.curves), len(self.active_map))):
            curve = self.curves[plot_idx]
            if curve is None:
                continue
            _ch, sig = self.active_map[plot_idx]
            plot_width_px = self._plot_width_px(plot_idx)
            x_vis, y_vis, lod_block, key_slice = self._get_visible_xy_for_sig(sig, x0, x1, plot_width_px)

            render_key = (
                id(sig),
                round(float(x0), 15), round(float(x1), 15),
                int(plot_width_px), int(lod_block),
                int(key_slice[0]), int(key_slice[1]),
            )
            if not force and self._last_render_keys[plot_idx] == render_key:
                continue
            curve.setData(x_vis, y_vis)
            self._last_render_keys[plot_idx] = render_key
            self._last_render_blocks[plot_idx] = lod_block

        self._update_cursor_label_positions()

    # ---- curseurs ----
    def set_cursor_series(self, positions_s: list, series_index: int = 0,
                          labels=None, labels_visible=True):
        self.cursor_positions = list(positions_s)
        self.cursor_series_color = self.SERIES_COLORS[series_index % len(self.SERIES_COLORS)]
        self.cursor_labels = list(labels) if labels is not None else [f"c{i+1}" for i in range(len(positions_s))]
        self.cursor_labels_visible = bool(labels_visible)
        self._rebuild_cursor_lines()
        self._emit_cursor()

    def set_all_series_for_export(self, payload):
        self.all_series_for_export = payload or []

    def reset_cursors_default(self):
        if not self.plots or not self.cursor_positions:
            return
        x0, x1 = self.plots[0].getViewBox().viewRange()[0]
        if x1 == x0:
            x1 = x0 + 1.0
        for i in range(len(self.cursor_positions)):
            perc = self.CURSOR_PERC_PATTERN[i % len(self.CURSOR_PERC_PATTERN)]
            self.cursor_positions[i] = float(x0 + perc * (x1 - x0))
        self._apply_positions_to_lines()
        self._update_cursor_label_positions()
        self._emit_cursor()

    def _clear_cursor_items(self):
        for p, pw in enumerate(self.plots):
            if p < len(self.cursor_lines):
                for ln in self.cursor_lines[p]:
                    try: pw.removeItem(ln)
                    except Exception: pass
        if self.plots:
            for ti in self.cursor_text_items:
                try: self.plots[0].removeItem(ti)
                except Exception: pass
        self.cursor_lines = [[] for _ in self.plots]
        self.cursor_text_items = []

    def _apply_positions_to_lines(self):
        for p in range(len(self.plots)):
            for ci, ln in enumerate(self.cursor_lines[p]):
                if ci < len(self.cursor_positions) and self.cursor_positions[ci] is not None:
                    ln.blockSignals(True)
                    ln.setPos(float(self.cursor_positions[ci]))
                    ln.blockSignals(False)

    def _build_cursor_text_items(self):
        if not self.plots:
            return
        self.cursor_text_items = []
        font = QFont("Segoe UI", CURSOR.label_font_pt)
        font.setBold(True)
        for ci in range(len(self.cursor_positions)):
            txt = self.cursor_labels[ci] if ci < len(self.cursor_labels) else f"c{ci+1}"
            ti = pg.TextItem(text=txt, color=self.cursor_series_color, anchor=(0, 0.5))
            ti.setFont(font)
            try: ti.setZValue(20)
            except Exception: pass
            self.plots[0].addItem(ti, ignoreBounds=True)
            self.cursor_text_items.append(ti)

    def _update_cursor_label_positions(self):
        if not self.plots or not self.cursor_text_items:
            return
        try:
            (x0, x1), (y0, y1) = self.plots[0].getViewBox().viewRange()
        except Exception:
            return
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 or dy == 0:
            return
        y_text = y1 - CURSOR.label_y_from_top_ratio * dy
        x_shift = CURSOR.label_xshift_ratio * dx

        for i, ti in enumerate(self.cursor_text_items):
            if i >= len(self.cursor_positions):
                ti.setVisible(False)
                continue
            x = self.cursor_positions[i]
            if x is None or not self.cursor_labels_visible:
                ti.setVisible(False)
                continue
            ti.setVisible(True)
            ti.setPos(float(x + x_shift), float(y_text))
            if i < len(self.cursor_labels):
                ti.setText(str(self.cursor_labels[i]))

    def _make_cursor_pen(self) -> pg.mkPen:
        pen = pg.mkPen(self.cursor_series_color, width=CURSOR.line_width)
        pen.setCosmetic(True)
        return pen

    def _rebuild_cursor_lines(self):
        if not self.plots:
            return
        self._clear_cursor_items()
        ncur = len(self.cursor_positions) if self.cursor_positions else 0
        if ncur <= 0:
            return

        x0, x1 = self.plots[0].getViewBox().viewRange()[0]
        if x1 == x0:
            x1 = x0 + 1.0

        for i in range(ncur):
            if self.cursor_positions[i] is None:
                perc = self.CURSOR_PERC_PATTERN[i % len(self.CURSOR_PERC_PATTERN)]
                self.cursor_positions[i] = float(x0 + perc * (x1 - x0))

        self.cursor_lines = [[] for _ in self.plots]
        pen = self._make_cursor_pen()

        for p, pw in enumerate(self.plots):
            for ci in range(ncur):
                ln = CosmeticInfiniteLine(
                    angle=90, movable=True, pen=pen,
                    pixel_width=CURSOR.line_width,
                )
                ln.setPos(float(self.cursor_positions[ci]))
                ln.sigPositionChanged.connect(
                    lambda _=None, pi=p, idx=ci: self._cursor_moved(pi, idx)
                )
                pw.addItem(ln)
                self.cursor_lines[p].append(ln)

        self._build_cursor_text_items()
        self._update_cursor_label_positions()

    def _cursor_moved(self, plot_idx: int, cursor_idx: int):
        if not (0 <= plot_idx < len(self.cursor_lines)):
            return
        if not (0 <= cursor_idx < len(self.cursor_lines[plot_idx])):
            return
        val = float(self.cursor_lines[plot_idx][cursor_idx].value())
        self.cursor_positions[cursor_idx] = val

        for p in range(len(self.plots)):
            if p == plot_idx or p >= len(self.cursor_lines):
                continue
            if cursor_idx < len(self.cursor_lines[p]):
                other = self.cursor_lines[p][cursor_idx]
                other.blockSignals(True)
                other.setPos(val)
                other.blockSignals(False)

        self._update_cursor_label_positions()
        self._emit_cursor()

    def get_cursor_positions(self) -> list:
        return list(self.cursor_positions)

    def _y_unit_for_signal(self, sig) -> str:
        y = sig.get("y")
        if y is None or len(y) == 0:
            return "V"
        return "mV" if float(np.nanmax(np.abs(y))) < 1.0 else "V"

    def set_layout_for_channels(self, assigned_channels: list):
        keep_xrange = None
        if self.plots:
            keep_xrange = tuple(self.plots[0].getViewBox().viewRange()[0])

        if not assigned_channels:
            if len(self.plots) != 1:
                self._build_plots(1, keep_xrange=keep_xrange)
            self.active_map = []
            pw = self.plots[0]
            pw.setLabel("left", "—")
            if self.curves[0] is not None:
                pw.removeItem(self.curves[0])
                self.curves[0] = None
            self._last_render_keys = [None]
            self._last_render_blocks = [None]
            return

        n = len(assigned_channels)
        if len(self.plots) != n:
            self._build_plots(n, keep_xrange=keep_xrange)

        self.active_map = assigned_channels

        for i, (ch_index, sig) in enumerate(assigned_channels):
            pw = self.plots[i]
            unit = self._y_unit_for_signal(sig)
            pw.setLabel("left", f"<b>{sig.get('plot_display', sig['display'])}</b>  ({unit})")
            pen = pg.mkPen(self.TEK_COLORS[ch_index], width=1.6)

            if self.curves[i] is None:
                curve = pg.PlotDataItem(pen=pen)
                pw.addItem(curve)
                self.curves[i] = curve
                self._configure_curve_item(curve)
            else:
                self.curves[i].setPen(pen)
                self._configure_curve_item(self.curves[i])

            pw.setYRange(sig["ymin"], sig["ymax"], padding=0.05)
            self._last_render_keys[i] = None
            self._last_render_blocks[i] = None

        self._refresh_visible_data(force=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_render_update()


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):
    MAX_FILES = 12

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Le super lecteur de WFM")
        self.resize(1900, 980)

        # ---- FIX 3 : stylesheet Windows 10 propre ----
        self.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
                margin-top: -1px;
            }
            QTabWidget::tab-bar { alignment: left; }
            QTabBar::tab {
                background: palette(base);
                color: palette(text);
                border: 1px solid palette(mid);
                border-bottom: 2px solid palette(mid);
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: palette(window);
                border-color: palette(dark);
                border-bottom: 2px solid palette(highlight);
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected { background: palette(midlight); }
            QTabWidget QTabBar::tab {
                min-height: 22px;
            }
            QGroupBox {
                border: 1px solid #b8b8b8;
                border-radius: 5px;
                margin-top: 14px;
                padding: 6px 4px 4px 4px;
                background: palette(window);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 1px;
                padding: 0 4px;
                background: palette(window);
                color: palette(text);
            }
            QTabWidget > QWidget > QGroupBox:first-child { margin-top: 10px; }
        """)

        self.loaded: list = []
        self.series_widgets: list[CursorSeriesWidget] = []
        self.analysis_piecewise_coeffs: list = []
        self.analysis_point_text_items: list = []
        self._last_busy_popup_timer: Optional[QTimer] = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ======================================================
        # PANNEAU GAUCHE
        # ======================================================
        self.left_container = QWidget()
        self.left_container.setFixedWidth(LEFT_PANEL_WIDTH)
        left_outer = QVBoxLayout(self.left_container)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)
        root.addWidget(self.left_container, 0)

        self.left_tabs = QTabWidget()
        left_outer.addWidget(self.left_tabs)

        # --- Onglet Fichiers ---
        left_files_tab = QWidget()
        left_files = QVBoxLayout(left_files_tab)
        left_files.setContentsMargins(0, 6, 0, 0)
        left_files.setSpacing(6)
        self.left_tabs.addTab(left_files_tab, "Fichiers")

        files_box = QGroupBox("Gestion des fichiers chargés")
        fbl = QVBoxLayout(files_box)
        fbl.setSpacing(6)
        btn_row = QHBoxLayout()
        self.btn_open_files = QPushButton("Ouvrir")
        self.btn_open_folder = QPushButton("Dossier")
        self.btn_clear = QPushButton("Vider")
        for btn in (self.btn_open_files, self.btn_open_folder, self.btn_clear):
            btn.setMaximumWidth(78)
            btn_row.addWidget(btn)
        fbl.addLayout(btn_row)

        fbl.addWidget(QLabel("Fichiers chargés :"))
        self.listw = QListWidget()
        self.listw.setFixedHeight(118)
        self.listw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fbl.addWidget(self.listw)
        left_files.addWidget(files_box)

        self.meta_box = QGroupBox("Métadonnées (fichier sélectionné)")
        mg = QVBoxLayout(self.meta_box)
        self.meta_lbl = QLabel("—")
        self.meta_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.meta_lbl.setWordWrap(True)
        self.meta_lbl.setMaximumHeight(105)
        mg.addWidget(self.meta_lbl)
        left_files.addWidget(self.meta_box, 0)

        assign_box = QGroupBox("Affectation → Voies")
        ag = QGridLayout(assign_box)
        self.assign: list[QComboBox] = []
        for ch in range(4):
            ag.addWidget(QLabel(f"CH{ch+1}"), ch, 0)
            cb = QComboBox()
            cb.addItem("—", userData=None)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            ag.addWidget(cb, ch, 1)
            self.assign.append(cb)
        left_files.addWidget(assign_box)

        lock_box = QGroupBox("Y Unlock / Auto Y")
        lg = QGridLayout(lock_box)
        self.unlock_y: list[QCheckBox] = []
        self.btn_autoy: list[QPushButton] = []
        for ch in range(4):
            cb = QCheckBox(f"Y Unlock CH{ch+1}")
            cb.setChecked(False)
            btn = QPushButton("Auto Y")
            lg.addWidget(cb, ch, 0)
            lg.addWidget(btn, ch, 1)
            self.unlock_y.append(cb)
            self.btn_autoy.append(btn)
        left_files.addWidget(lock_box)
        left_files.addStretch(1)

        # --- Onglet Export ---
        left_export_tab = QWidget()
        left_export = QVBoxLayout(left_export_tab)
        left_export.setContentsMargins(0, 6, 0, 0)
        left_export.setSpacing(8)
        self.left_tabs.addTab(left_export_tab, "Export")

        export_box = QGroupBox("Export")
        eg = QGridLayout(export_box)
        eg.setColumnStretch(0, 3)
        eg.setColumnStretch(1, 3)

        self.cb_export_dark = QCheckBox("Fond noir")
        self.cb_export_dark.setChecked(False)
        self.cb_export_hide_cursors = QCheckBox("Masquer curseurs")
        self.cb_export_hide_cursors.setChecked(False)
        self.btn_copy_img = QPushButton("Copier image")
        self.btn_save_img = QPushButton("Enregistrer PNG…")
        self.btn_pdf = QPushButton("Générer PDF…")
        self.ed_export_title = QLineEdit()
        self.ed_export_title.setPlaceholderText("Titre (optionnel)")

        eg.addWidget(self.cb_export_dark, 0, 0)
        eg.addWidget(self.cb_export_hide_cursors, 0, 1)
        eg.addWidget(self.btn_copy_img, 1, 0)
        eg.addWidget(self.btn_save_img, 1, 1)
        eg.addWidget(self.ed_export_title, 2, 0)
        eg.addWidget(self.btn_pdf, 2, 1)

        left_export.addWidget(export_box)
        left_export.addStretch(1)

        # ======================================================
        # CENTRE
        # ======================================================
        self.scope = ScopeStack()
        root.addWidget(self.scope, 1)

        self.analysis_strip = AnalysisToggleStrip()
        root.addWidget(self.analysis_strip, 0)

        # ======================================================
        # PANNEAU DROIT
        # ======================================================
        self.right_container = QWidget()
        self.right_container.setFixedWidth(RIGHT_PANEL_WIDTH)
        right_outer = QVBoxLayout(self.right_container)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(0)
        root.addWidget(self.right_container, 0)

        self.right_tabs = QTabWidget()
        right_outer.addWidget(self.right_tabs)

        # --- Onglet Curseurs ---
        right_cursor_tab = QWidget()
        right_cursor = QVBoxLayout(right_cursor_tab)
        right_cursor.setContentsMargins(0, 6, 0, 0)
        right_cursor.setSpacing(8)
        self.right_tabs.addTab(right_cursor_tab, "Curseurs")

        tools = QHBoxLayout()
        tools.addWidget(QLabel("Séries :"))
        self.btn_new_series = QToolButton()
        self.btn_new_series.setText("+")
        self.btn_new_series.setToolTip("Nouvelle série")
        self.btn_del_series = QToolButton()
        self.btn_del_series.setText("–")
        self.btn_del_series.setToolTip("Supprimer la série active")
        tools.addWidget(self.btn_new_series)
        tools.addWidget(self.btn_del_series)
        tools.addStretch(1)
        right_cursor.addLayout(tools)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        right_cursor.addWidget(self.tabs, 1)

        self.lbl_info = QLabel("—")
        right_cursor.addWidget(self.lbl_info)

        # --- Onglet Auto ---
        right_auto_tab = QWidget()
        right_auto = QVBoxLayout(right_auto_tab)
        right_auto.setContentsMargins(0, 6, 0, 0)
        right_auto.setSpacing(8)
        self.right_tabs.addTab(right_auto_tab, "Auto")

        auto_box = QGroupBox("Piquage automatique")
        auto_lay = QGridLayout(auto_box)
        auto_lay.setColumnStretch(0, 1)
        auto_lay.setColumnStretch(1, 1)

        self.auto_ch_checks: list[QCheckBox] = []
        for ch in range(4):
            cb = QCheckBox(f"CH{ch+1}")
            cb.setChecked(ch == 0)
            self.auto_ch_checks.append(cb)
            auto_lay.addWidget(cb, ch // 2, ch % 2)

        _params = [
            ("Temps max front détectable :", "sp_auto_front_us",
             QDoubleSpinBox, dict(decimals=2, range=(0.01, 100000.0),
                                  value=AUTO.default_max_front_us, suffix=" µs")),
            ("Amplitude mini du front :", "sp_auto_min_amp_pct",
             QDoubleSpinBox, dict(decimals=1, range=(0.1, 100.0),
                                  value=AUTO.default_min_amp_pct, suffix=" %")),
            ("Lissage :", "sp_auto_smooth_pts",
             QDoubleSpinBox, dict(decimals=0, range=(1, 101), step=2,
                                  value=AUTO.default_smooth_points, suffix=" pts")),
            ("Stabilité avant front :", "sp_auto_stability_factor",
             QDoubleSpinBox, dict(decimals=1, range=(1.0, 50.0),
                                  value=AUTO.default_stability_factor, suffix=" × front")),
        ]
        row = 2
        for label, attr, cls, kw in _params:
            auto_lay.addWidget(QLabel(label), row, 0)
            w = cls()
            w.setDecimals(kw.get("decimals", 2))
            lo, hi = kw["range"]
            w.setRange(lo, hi)
            w.setValue(kw["value"])
            if "suffix" in kw:
                w.setSuffix(kw["suffix"])
            if "step" in kw:
                w.setSingleStep(kw["step"])
            setattr(self, attr, w)
            auto_lay.addWidget(w, row, 1)
            row += 1

        self.cb_auto_prefer_first = QCheckBox("Privilégier le premier front")
        self.cb_auto_prefer_first.setChecked(True)
        auto_lay.addWidget(self.cb_auto_prefer_first, row, 0, 1, 2)
        row += 1

        self.cb_auto_ignore_weak_first = QCheckBox(
            "Ignorer le premier s'il est trop faible devant le second"
        )
        self.cb_auto_ignore_weak_first.setChecked(True)
        auto_lay.addWidget(self.cb_auto_ignore_weak_first, row, 0, 1, 2)
        row += 1

        auto_lay.addWidget(QLabel("Ratio mini amplitude 1er / 2e :"), row, 0)
        self.sp_auto_first_second_ratio = QDoubleSpinBox()
        self.sp_auto_first_second_ratio.setDecimals(2)
        self.sp_auto_first_second_ratio.setRange(0.01, 1.00)
        self.sp_auto_first_second_ratio.setSingleStep(0.05)
        self.sp_auto_first_second_ratio.setValue(AUTO.default_first_second_ratio)
        auto_lay.addWidget(self.sp_auto_first_second_ratio, row, 1)
        row += 1

        auto_lay.addWidget(QLabel("Position du curseur :"), row, 0)
        self.cb_auto_position_mode = QComboBox()
        self.cb_auto_position_mode.addItem("50% transition locale", userData="50")
        self.cb_auto_position_mode.addItem("Pente max", userData="maxslope")
        auto_lay.addWidget(self.cb_auto_position_mode, row, 1)
        row += 1

        self.btn_auto_pick = QPushButton("Lancer piquage auto")
        auto_lay.addWidget(self.btn_auto_pick, row, 0, 1, 2)

        self.lbl_auto_info = QLabel("—")
        self.lbl_auto_info.setWordWrap(True)
        self.lbl_auto_info.setStyleSheet("background:#f3f6fb; border:1px solid #c9d4e6; border-radius:6px; padding:6px;")
        right_auto.addWidget(auto_box)
        right_auto.addWidget(self.lbl_auto_info)
        right_auto.addStretch(1)

        # --- Onglet Analyse ---
        right_analysis_tab = QWidget()
        right_analysis = QVBoxLayout(right_analysis_tab)
        right_analysis.setContentsMargins(0, 6, 0, 0)
        right_analysis.setSpacing(6)
        self.right_tabs.addTab(right_analysis_tab, "Analyse")

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Série :"))
        self.cb_analysis_series = QComboBox()
        sel_row.addWidget(self.cb_analysis_series, 1)
        right_analysis.addLayout(sel_row)

        self.analysis_title = QLabel("—")
        self.analysis_title.setAlignment(Qt.AlignCenter)
        self.analysis_title.setStyleSheet("font-weight:600; color:#1f2d3d;")
        right_analysis.addWidget(self.analysis_title)

        self.analysis_plot = pg.PlotWidget()
        self.analysis_plot.setBackground((248, 250, 253))
        self.analysis_plot.showGrid(x=True, y=True)
        self.analysis_plot.setMenuEnabled(False)
        self.analysis_plot.setMinimumHeight(UI.analysis_plot_height)
        self.analysis_plot.setMaximumHeight(UI.analysis_plot_height)
        self.analysis_plot.setMouseEnabled(x=True, y=False)
        self.analysis_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.analysis_plot.setLabel("left", "")
        self.analysis_plot.setLabel("bottom", "")
        right_analysis.addWidget(self.analysis_plot, 0, Qt.AlignTop)

        self.analysis_info = QLabel("—")
        self.analysis_info.setWordWrap(True)
        self.analysis_info.setStyleSheet("background:#f3f6fb; border:1px solid #c9d4e6; border-radius:6px; padding:6px;")
        right_analysis.addWidget(self.analysis_info)
        right_analysis.addStretch(1)

        self.analysis_scatter = pg.ScatterPlotItem(
            size=UI.analysis_point_size,
            pen=pg.mkPen(40, 40, 40),
            brush=pg.mkBrush(0, 120, 255)
        )
        self.analysis_curve = pg.PlotDataItem(pen=pg.mkPen(24, 102, 201, width=2.4))
        self.analysis_plot.addItem(self.analysis_curve)
        self.analysis_plot.addItem(self.analysis_scatter)

        # ======================================================
        # WIRING
        # ======================================================
        self.btn_open_files.clicked.connect(self.open_files)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_clear.clicked.connect(self.clear_all)

        for cb in self.assign:
            cb.currentIndexChanged.connect(self.refresh_view)

        for ch in range(4):
            self.unlock_y[ch].stateChanged.connect(self.apply_y_locks)
            self.btn_autoy[ch].clicked.connect(lambda _=None, c=ch: self.auto_y_channel(c))

        self.listw.currentItemChanged.connect(self.update_meta_from_selection)
        self.btn_new_series.clicked.connect(self._on_new_series)
        self.btn_del_series.clicked.connect(self._on_delete_series)
        self.tabs.currentChanged.connect(self._on_series_changed)
        self.scope.on_cursor_changed(self._sync_table_from_plot)
        self.analysis_strip.toggled_request.connect(self._toggle_right_panel)
        self.cb_analysis_series.currentIndexChanged.connect(self._update_analysis_plot)
        self.btn_auto_pick.clicked.connect(self._run_auto_pick)
        self.btn_copy_img.clicked.connect(self.copy_plot_image_fixed)
        self.btn_save_img.clicked.connect(self.save_plot_image_fixed)
        self.btn_pdf.clicked.connect(self.export_pdf_report)

        self.rebuild_list()
        self.rebuild_assignments()
        self.refresh_view()
        self.update_meta_from_selection()
        self._apply_active_series_to_plot()
        self._rebuild_analysis_series_combo()
        self._update_auto_channel_enable()
        self.right_container.setVisible(True)

    # ======================================================
    # UI HELPERS
    # ======================================================
    def _toggle_right_panel(self):
        self.right_container.setVisible(not self.right_container.isVisible())

    def _series_display_name(self, idx: int, sw: CursorSeriesWidget) -> str:
        return sw.series_name.strip() or f"Série {idx+1}"

    def _next_auto_series_name(self) -> str:
        used = {self._series_display_name(i, sw).lower()
                for i, sw in enumerate(self.series_widgets)}
        k = 1
        while True:
            nm = f"auto{k}"
            if nm.lower() not in used:
                return nm
            k += 1

    def _update_auto_channel_enable(self):
        assigned_ids = [self.assign[ch].currentData() for ch in range(4)]
        for ch in range(4):
            was_enabled = self.auto_ch_checks[ch].isEnabled()
            enabled = assigned_ids[ch] is not None
            self.auto_ch_checks[ch].setEnabled(enabled)
            if not enabled:
                self.auto_ch_checks[ch].setChecked(False)
            elif not was_enabled:
                self.auto_ch_checks[ch].setChecked(True)

    # ======================================================
    # BUSY POPUP  — FIX 4 : gestion correcte des timers
    # ======================================================
    def _make_busy_popup(self, title: str, message: str,
                         width: int = 420, height: int = 120) -> QProgressDialog:
        dlg = QProgressDialog(message, None, 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setFixedSize(width, height)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _finish_busy_popup(self, dlg: Optional[QProgressDialog],
                           message: str, delay_ms: int = 700):
        if dlg is None:
            return
        # Stopper le timer précédent pour éviter les fuites
        if self._last_busy_popup_timer is not None:
            try: self._last_busy_popup_timer.stop()
            except Exception: pass

        dlg.setLabelText(message)
        QApplication.processEvents()

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(dlg.close)
        timer.start(delay_ms)
        self._last_busy_popup_timer = timer

    # ======================================================
    # SÉRIES
    # ======================================================
    def _on_new_series(self):
        idx = len(self.series_widgets) + 1
        w = CursorSeriesWidget(default_name=f"Série {idx}")
        w.changed.connect(self._on_series_widget_changed)
        w.reset_requested.connect(self._reset_cursors)
        w.add_cursor()
        w.add_cursor()
        self.series_widgets.append(w)
        self.tabs.addTab(w, w.series_name)
        self.tabs.setCurrentWidget(w)
        self._rebuild_analysis_series_combo(prefer_index=len(self.series_widgets) - 1)
        self._apply_active_series_to_plot()

    def _on_delete_series(self):
        i = self.tabs.currentIndex()
        if i < 0:
            return
        w = self.tabs.widget(i)
        self.tabs.removeTab(i)
        try: self.series_widgets.remove(w)
        except Exception: pass
        self._rebuild_analysis_series_combo(
            prefer_index=max(0, min(i, len(self.series_widgets) - 1))
        )
        if self.tabs.count() == 0:
            self.scope.set_cursor_series([], series_index=0, labels=[], labels_visible=True)
            self.lbl_info.setText("—")
            self._push_all_series_to_scope_for_export()
            self._update_analysis_plot()
            return
        self._apply_active_series_to_plot()

    def _active_series_widget(self) -> Optional[CursorSeriesWidget]:
        w = self.tabs.currentWidget()
        return w if isinstance(w, CursorSeriesWidget) else None

    def _on_series_widget_changed(self):
        w = self._active_series_widget()
        if w is None:
            return
        i = self.tabs.currentIndex()
        self.tabs.setTabText(i, w.series_name.strip() or f"Série {i+1}")
        self._apply_active_series_to_plot()
        self._update_summary_label()
        self._push_all_series_to_scope_for_export()
        self._rebuild_analysis_series_combo(prefer_index=i)
        self._update_analysis_plot()

    def _on_series_changed(self, idx: int):
        self._apply_active_series_to_plot()
        self._update_summary_label()
        self._push_all_series_to_scope_for_export()
        if 0 <= idx < self.cb_analysis_series.count():
            self.cb_analysis_series.setCurrentIndex(idx)
        self._update_analysis_plot()

    def _apply_active_series_to_plot(self):
        w = self._active_series_widget()
        if w is None:
            self.scope.set_cursor_series([], series_index=0, labels=[], labels_visible=True)
            self._push_all_series_to_scope_for_export()
            return
        self.scope.set_cursor_series(
            w.get_positions_s(),
            series_index=self.tabs.currentIndex(),
            labels=w.get_names(),
            labels_visible=True,
        )
        self._sync_table_from_plot()
        self._push_all_series_to_scope_for_export()

    def _push_all_series_to_scope_for_export(self):
        self.scope.set_all_series_for_export([
            {
                "positions": sw.get_positions_s(),
                "labels": sw.get_names(),
                "color": self.scope.SERIES_COLORS[si % len(self.scope.SERIES_COLORS)],
            }
            for si, sw in enumerate(self.series_widgets)
        ])

    def _sync_table_from_plot(self):
        w = self._active_series_widget()
        if w is None:
            return
        pos = self.scope.get_cursor_positions()
        if len(pos) != len(w.rows):
            pos = (pos + [None] * len(w.rows))[: len(w.rows)]
        w.set_times_from_positions(pos)
        self._update_summary_label()
        self._update_analysis_plot()

    def _update_summary_label(self):
        w = self._active_series_widget()
        if w is None or len(w.rows) < 2:
            self.lbl_info.setText("—")
            return
        sv = [v for v in w._compute_speeds()[1:] if v is not None and np.isfinite(v)]
        if not sv:
            self.lbl_info.setText("Vitesse série : —")
            return
        self.lbl_info.setText(
            f"Vitesse série : moy={float(np.mean(sv)):.2f} m/s   "
            f"max={float(np.max(sv)):.2f} m/s"
        )

    def _reset_cursors(self):
        self.scope.reset_cursors_default()
        self._sync_table_from_plot()

    # ======================================================
    # AUTO PICK
    # ======================================================
    def _run_auto_pick(self):
        selected = [
            (ch, self.get_sig(self.assign[ch].currentData()))
            for ch in range(4)
            if self.auto_ch_checks[ch].isChecked()
            and self.assign[ch].currentData() is not None
            and self.get_sig(self.assign[ch].currentData()) is not None
        ]
        if not selected:
            QMessageBox.information(self, "Auto", "Aucune voie sélectionnée / assignée.")
            return

        popup = self._make_busy_popup("Piquage auto", "Analyse des voies sélectionnées...")

        max_front_s = float(self.sp_auto_front_us.value()) * 1e-6
        min_amp_pct = float(self.sp_auto_min_amp_pct.value())
        smooth_points = int(round(float(self.sp_auto_smooth_pts.value())))
        stability_factor = float(self.sp_auto_stability_factor.value())
        prefer_first = self.cb_auto_prefer_first.isChecked()
        ignore_first_if_weak = self.cb_auto_ignore_weak_first.isChecked()
        first_second_ratio = float(self.sp_auto_first_second_ratio.value())
        position_mode = self.cb_auto_position_mode.currentData() or "50"

        detected, infos = [], []

        for i, (ch, sig) in enumerate(selected, start=1):
            popup.setLabelText(
                f"Piquage auto en cours…\n"
                f"Voie {i}/{len(selected)}\n"
                f"CH{ch+1}"
            )
            # FIX 4 : double processEvents pour garantir le repaint du label
            QApplication.processEvents()
            QApplication.processEvents()

            best = _detect_best_front_v3_coarse_to_fine(
                sig, max_front_s=max_front_s, min_amp_pct=min_amp_pct,
                smooth_points=smooth_points, stability_factor=stability_factor,
                prefer_first=prefer_first, ignore_first_if_weak=ignore_first_if_weak,
                first_second_ratio=first_second_ratio, position_mode=position_mode,
            )
            if best is None:
                infos.append(f"CH{ch+1}: rien de robuste détecté")
                continue

            tpick = float(best["t_pos"])
            detected.append((ch, tpick))
            width_txt = (f"{best['width'] * 1e6:.2f} µs"
                         if np.isfinite(best["width"]) else "∞")
            amp_pct = 100.0 * best["amp"] / max(best["full_range"], 1e-18)
            infos.append(
                f"CH{ch+1}: {best['direction']} @ {tpick*1e6:.3f} µs | "
                f"amp={best['amp']:.4g} ({amp_pct:.1f}% FS) | "
                f"score={best['score']:.3f} | largeur≈{width_txt}"
            )

        self.lbl_auto_info.setText("<b>Résumé des fronts détectés</b><br>" + "<br>".join(infos) if infos else "—")

        if not detected:
            self._finish_busy_popup(popup, "Piquage auto terminé\nAucun front robuste détecté", 900)
            QMessageBox.information(self, "Auto", "Aucun front robuste détecté.")
            return

        name = self._next_auto_series_name()
        w = CursorSeriesWidget(default_name=name)
        w.changed.connect(self._on_series_widget_changed)
        w.reset_requested.connect(self._reset_cursors)

        for ch, tpick in detected:
            w.rows.append({"name": f"aCH{ch+1}", "d_mm": 0.0, "t_s": float(tpick)})
        w._refresh_table()

        self.series_widgets.append(w)
        self.tabs.addTab(w, name)
        self.tabs.setCurrentWidget(w)

        self._rebuild_analysis_series_combo(prefer_index=len(self.series_widgets) - 1)
        self._apply_active_series_to_plot()
        self.right_tabs.setCurrentIndex(1)

        self._finish_busy_popup(
            popup, f"Piquage auto terminé\n{len(detected)} voie(s) piquée(s)", 800
        )

    # ======================================================
    # ANALYSE
    # ======================================================
    def _rebuild_analysis_series_combo(self, prefer_index=None):
        old = self.cb_analysis_series.currentIndex()
        if prefer_index is None:
            prefer_index = old
        self.cb_analysis_series.blockSignals(True)
        self.cb_analysis_series.clear()
        for i, sw in enumerate(self.series_widgets):
            self.cb_analysis_series.addItem(self._series_display_name(i, sw), userData=i)
        if self.cb_analysis_series.count() > 0:
            idx = max(0, min(int(prefer_index or 0), self.cb_analysis_series.count() - 1))
            self.cb_analysis_series.setCurrentIndex(idx)
        self.cb_analysis_series.blockSignals(False)
        self._update_analysis_plot()

    def _get_selected_analysis_series(self):
        idx = self.cb_analysis_series.currentData()
        if idx is None:
            idx = self.cb_analysis_series.currentIndex()
        if idx is None or idx < 0 or idx >= len(self.series_widgets):
            return None, None
        return idx, self.series_widgets[idx]

    def _clear_analysis_point_labels(self):
        for ti in self.analysis_point_text_items:
            try: self.analysis_plot.removeItem(ti)
            except Exception: pass
        self.analysis_point_text_items = []

    def _update_analysis_plot(self):
        self.analysis_scatter.setData([], [])
        self.analysis_curve.setData([], [])
        self.analysis_info.setText("—")
        self.analysis_title.setText("—")
        self.analysis_piecewise_coeffs = []
        self._clear_analysis_point_labels()

        idx, sw = self._get_selected_analysis_series()
        if sw is None:
            return

        pts = sw.get_analysis_points()
        if not pts:
            self.analysis_info.setText("Pas assez de points valides.")
            return

        names = [p[0] for p in pts]
        t_s = np.array([p[1] for p in pts], dtype=float)
        d_mm = np.array([p[2] for p in pts], dtype=float)

        t_factor, t_unit = _pick_unit(t_s, "time")
        d_factor, d_unit = _pick_unit(d_mm, "pos")
        tx = t_s * t_factor
        dy_raw = d_mm * d_factor
        dy = _enforce_non_decreasing(dy_raw)

        color = self.scope.SERIES_COLORS[idx % len(self.scope.SERIES_COLORS)]
        self.analysis_scatter.setBrush(pg.mkBrush(*color, 220))
        self.analysis_scatter.setPen(pg.mkPen(20, 20, 20, width=1.2))
        self.analysis_scatter.setData(tx, dy)

        if len(tx) >= 2 and np.all(np.diff(tx) > 0):
            coeffs = _pchip_piecewise_coeffs(tx, dy)
            self.analysis_piecewise_coeffs = coeffs
            xx, yy = _pchip_eval_dense(tx, coeffs, points_per_seg=50)
            self.analysis_curve.setPen(pg.mkPen(*color, width=2))
            self.analysis_curve.setData(xx, yy)
            self.analysis_info.setText(
                f"<b>{self._series_display_name(idx, sw)}</b> — interpolation cubique monotone (PCHIP), contrainte croissante\n"
                f"Segments: {len(coeffs)}"
            )
        else:
            self.analysis_curve.setData(tx, dy)
            self.analysis_piecewise_coeffs = []
            self.analysis_info.setText(
                f"<b>{self._series_display_name(idx, sw)}</b> — segments simples (abscisses non strictement croissantes)."
            )

        self.analysis_title.setText(f"Time ({t_unit}) vs Position ({d_unit})")

        y_min, y_max = float(np.min(dy)), float(np.max(dy))
        if y_max == y_min: y_min -= 0.5; y_max += 0.5
        y_pad = 0.08 * (y_max - y_min)
        x_min, x_max = float(np.min(tx)), float(np.max(tx))
        if x_max == x_min: x_min -= 0.5; x_max += 0.5

        font = QFont("Segoe UI", UI.analysis_point_label_font_pt)
        for i, nm in enumerate(names):
            ti = pg.TextItem(text=str(nm), color=(30, 30, 30), anchor=(0, 1))
            ti.setFont(font)
            ti.setPos(float(tx[i]), float(dy[i] + 0.03 * (y_max - y_min)))
            self.analysis_plot.addItem(ti, ignoreBounds=True)
            self.analysis_point_text_items.append(ti)

        vb = self.analysis_plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        self.analysis_plot.setMouseEnabled(x=True, y=False)
        vb.setLimits(
            xMin=None, xMax=None,
            yMin=float(y_min - y_pad), yMax=float(y_max + y_pad),
            minYRange=float(y_max - y_min + 2 * y_pad),
            maxYRange=float(y_max - y_min + 2 * y_pad),
        )
        vb.setXRange(x_min, x_max, padding=0.05)
        vb.setYRange(y_min - y_pad, y_max + y_pad, padding=0.0)

    # ======================================================
    # EXPORT OFFSCREEN  — FIX 2 : layout pass garanti + _updateLabel
    # ======================================================
    def _render_plots_fixed_image(
        self,
        width_px: int = EXPORT.width_px,
        height_per_plot_px: int = EXPORT.height_per_plot_px,
        margin_px: int = EXPORT.margin_px,
    ) -> Optional[QImage]:
        if not self.scope.plots:
            return None
        app = QApplication.instance()
        if app is None:
            return None

        scale = max(1, int(EXPORT.scale))
        width_px *= scale
        height_per_plot_px *= scale
        margin_px *= scale

        black_bg = self.cb_export_dark.isChecked()
        bg_color = Qt.black if black_bg else Qt.white
        tick_pen = (220, 220, 220) if black_bg else (30, 30, 30)
        label_color = "#E0E0E0" if black_bg else "#202020"

        try:
            x0, x1 = self.scope.plots[0].getViewBox().viewRange()[0]
        except Exception:
            x0, x1 = 0.0, 1.0

        hide_all_cursors = self.cb_export_hide_cursors.isChecked()
        all_series = self.scope.all_series_for_export or []

        n = len(self.scope.plots)
        total_h = n * height_per_plot_px + (n - 1) * margin_px
        out = QImage(width_px, total_h, QImage.Format_ARGB32)
        out.fill(bg_color)

        painter_out = QPainter(out)
        try:
            yoff = 0
            for plot_idx, pw in enumerate(self.scope.plots):
                try:
                    (_sx0, _sx1), (sy0, sy1) = pw.getViewBox().viewRange()
                except Exception:
                    sy0, sy1 = 0.0, 1.0

                # ---- Créer le widget offscreen ----
                glw = pg.GraphicsLayoutWidget()
                glw.setAttribute(Qt.WA_DontShowOnScreen, True)
                glw.setBackground("k" if black_bg else "w")

                # FIX 2a : resize AVANT show, puis 2 passes processEvents
                glw.resize(width_px, height_per_plot_px)
                glw.show()
                app.processEvents()
                app.processEvents()

                axis = TimeAxis(orientation="bottom")
                pi = glw.addPlot(axisItems={"bottom": axis})
                pi.showGrid(x=True, y=True)
                pi.setMenuEnabled(False)
                pi.setMouseEnabled(x=True, y=False)

                # Calcul dynamique de la largeur axe gauche :
                # on estime la largeur du texte le plus large des ticks Y
                # pour ne pas laisser un blanc énorme à gauche.
                tick_font = QFont("Segoe UI", max(6, int(EXPORT.tick_font_pt * scale)))
                from PySide6.QtGui import QFontMetrics as _QFM
                fm = _QFM(tick_font)
                # Valeurs typiques visibles sur l'axe Y
                if plot_idx < len(self.scope.active_map):
                    _ch_tmp, sig_tmp = self.scope.active_map[plot_idx]
                    yrange = abs(sig_tmp["ymax"] - sig_tmp["ymin"])
                    tick_sample = f"{sig_tmp['ymax']:.4g}"
                else:
                    tick_sample = "1000.0"
                tick_text_w = fm.horizontalAdvance(tick_sample)
                # label vertical = texte rotaté → sa contribution à la largeur
                # est sa hauteur (fm.height) une fois rotaté
                label_font = QFont("Segoe UI", max(8, int(EXPORT.label_font_pt * scale)))
                label_font.setBold(True)
                fm_lbl = _QFM(label_font)
                # largeur totale = ticks + label + marges
                left_w = tick_text_w + fm_lbl.height() + int(20 * scale)
                left_w = max(int(EXPORT.left_axis_width_px * scale), left_w)

                pi.getAxis("left").setWidth(left_w)
                pi.getAxis("bottom").setHeight(int(EXPORT.bottom_axis_height_px * scale))

                # Couleurs des axes — AVANT le label pour que setLabel hérite bien
                for ax_name in ("left", "bottom"):
                    ax = pi.getAxis(ax_name)
                    ax.setPen(pg.mkPen(tick_pen))
                    ax.setTextPen(pg.mkPen(tick_pen))

                # FIX 2b : label gauche — appliquer la font directement sur l'objet
                # label de l'axe, PAS via size= en pt (instable en offscreen)
                if plot_idx < len(self.scope.active_map):
                    _ch, sig = self.scope.active_map[plot_idx]
                    unit = "mV" if abs(sig["ymax"] - sig["ymin"]) < 2.0 else "V"
                    pi.setLabel(
                        "left",
                        f"{sig.get('plot_display', sig['display'])}  ({unit})",
                        color=label_color,
                    )
                    left_ax = pi.getAxis("left")
                    left_ax.label.setFont(label_font)

                    # FIX 2 : forcer le recalcul de labelWidth avec la bonne font
                    # _updateLabel() recalcule la taille interne du label,
                    # resizeEvent(None) force un relayout de l'axe
                    try:
                        left_ax._updateLabel()
                    except Exception:
                        pass
                    try:
                        left_ax.resizeEvent(None)
                    except Exception:
                        pass

                # Appliquer tickFont APRÈS le label (layout pass plus stable)
                for ax_name in ("left", "bottom"):
                    ax = pi.getAxis(ax_name)
                    ax.setStyle(
                        tickFont=tick_font,
                        maxTickLevel=0 if EXPORT.hide_minor_ticks else 2,
                    )

                # ---- Données ----
                if plot_idx < len(self.scope.active_map):
                    ch_index, sig = self.scope.active_map[plot_idx]
                    pen = pg.mkPen(self.scope.TEK_COLORS[ch_index], width=1.6 * scale)
                    x_arr = sig["x"]
                    y_arr = sig["y"]
                    if (isinstance(x_arr, np.ndarray) and isinstance(y_arr, np.ndarray)
                            and len(x_arr) == len(y_arr) > 0):
                        xv, yv = _slice_visible(
                            _ensure_contiguous(x_arr),
                            _ensure_contiguous(y_arr),
                            x0, x1,
                        )
                        if len(yv) == 0:
                            step = max(1, len(y_arr) // max(1, width_px))
                            pi.plot(x_arr[::step], y_arr[::step], pen=pen, clear=True)
                        else:
                            nvis = len(yv)
                            target_blocks = int(width_px * float(EXPORT.target_blocks_per_px))
                            if nvis <= EXPORT.full_detail_factor * 2.0 * target_blocks:
                                pi.plot(xv, yv, pen=pen, clear=True)
                            else:
                                xds, yds = _downsample_minmax(xv, yv, target_blocks=target_blocks)
                                pi.plot(xds, yds, pen=pen, clear=True)

                # ---- Curseurs export ----
                if not hide_all_cursors:
                    for s in all_series:
                        col = s.get("color", (255, 170, 0))
                        pen_c = pg.mkPen(col, width=max(1.0, CURSOR.line_width) * scale)
                        positions = s.get("positions") or []
                        labels_list = s.get("labels") or []
                        for ci, xc in enumerate(positions):
                            if xc is None:
                                continue
                            # Utiliser CosmeticInfiniteLine aussi à l'export
                            ln = CosmeticInfiniteLine(
                                angle=90, movable=False, pen=pen_c,
                                pixel_width=max(1.0, CURSOR.line_width) * scale,
                            )
                            ln.setPos(float(xc))
                            pi.addItem(ln)
                            if plot_idx == 0:
                                nm = labels_list[ci] if ci < len(labels_list) else f"c{ci+1}"
                                ti = pg.TextItem(text=str(nm), color=col, anchor=(0, 0.5))
                                tf = QFont("Segoe UI", max(10, int(EXPORT.cursor_label_font_pt * scale)))
                                tf.setBold(True)
                                ti.setFont(tf)
                                pi.addItem(ti, ignoreBounds=True)
                                dx_range = x1 - x0
                                ti.setPos(
                                    float(xc + CURSOR.label_xshift_ratio * dx_range),
                                    float(sy1 - CURSOR.label_y_from_top_ratio * (sy1 - sy0)),
                                )

                vb = pi.getViewBox()
                vb.setXRange(x0, x1, padding=0.0)
                vb.setYRange(sy0, sy1, padding=0.0)

                # FIX 2c : double repaint + processEvents avant render()
                glw.repaint()
                app.processEvents()
                app.processEvents()

                img_part = QImage(width_px, height_per_plot_px, QImage.Format_ARGB32)
                img_part.fill(bg_color)
                p2 = QPainter(img_part)
                try:
                    glw.render(p2, QRectF(0, 0, width_px, height_per_plot_px))
                finally:
                    p2.end()

                glw.close()
                glw.deleteLater()

                if img_part.isNull():
                    return None

                painter_out.drawImage(0, yoff, img_part)
                yoff += height_per_plot_px + margin_px

        finally:
            painter_out.end()

        return out

    def copy_plot_image_fixed(self):
        popup = self._make_busy_popup("Export image", "Création de l'image…")
        img = self._render_plots_fixed_image()
        if img is None:
            popup.close()
            QMessageBox.warning(self, "Erreur", "Impossible de capturer l'image.")
            return
        QGuiApplication.clipboard().setImage(img)
        self._finish_busy_popup(popup, "Image copiée dans le presse-papiers", 700)

    def save_plot_image_fixed(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer image", "", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        popup = self._make_busy_popup("Export PNG", "Création du PNG…")
        img = self._render_plots_fixed_image()
        if img is None:
            popup.close()
            QMessageBox.warning(self, "Erreur", "Impossible de capturer l'image.")
            return
        img.save(path, "PNG")
        self._finish_busy_popup(popup, "PNG enregistré", 700)

    def export_pdf_report(self):
        if not REPORTLAB_OK:
            QMessageBox.information(
                self, "PDF indisponible",
                "La librairie 'reportlab' n'est pas installée.\n\npip install reportlab"
            )
            return

        # FIX 4 : QFileDialog EN PREMIER — popup seulement si chemin confirmé
        path, _ = QFileDialog.getSaveFileName(self, "Exporter rapport PDF", "", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        popup = self._make_busy_popup("Export PDF", "Génération du PDF…")

        img = self._render_plots_fixed_image(width_px=1400, height_per_plot_px=300, margin_px=1)
        if img is None:
            popup.close()
            QMessageBox.warning(self, "Erreur", "Impossible de capturer l'image des graphes.")
            return

        tmp_png = os.path.join(os.path.dirname(path), "_tmp_scope_capture.png")
        img.save(tmp_png, "PNG")

        c = rl_canvas.Canvas(path, pagesize=A4)
        W, H = A4

        title_opt = self.ed_export_title.text().strip()
        title = "Signaux WFM" if not title_opt else f"Signaux WFM - {title_opt}"

        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(W / 2, H - 15 * mm, title)

        img_w = W - 30 * mm
        img_h = img_w * (img.height() / max(1, img.width()))
        y_top = H - 25 * mm
        y_img = y_top - img_h
        if y_img < 70 * mm:
            img_h = H - 95 * mm
            img_w = img_h * (img.width() / max(1, img.height()))
            y_img = y_top - img_h

        c.drawImage(tmp_png, 15 * mm, y_img, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        y_cursor = y_img - 8 * mm

        for si, w in enumerate(self.series_widgets):
            if y_cursor < 70 * mm:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawCentredString(W / 2, H - 15 * mm, title)
                y_cursor = H - 25 * mm

            tab_name = w.series_name.strip() or f"Série {si+1}"
            ref = "curseur 1" if w.mode == "ref1" else "curseur précédent"

            sq = 5 * mm
            x_sq = 15 * mm
            y_sq = y_cursor - 3.5 * mm
            col = self.scope.SERIES_COLORS[si % len(self.scope.SERIES_COLORS)]
            r, g, b = col
            c.setFillColorRGB(r / 255.0, g / 255.0, b / 255.0)
            c.rect(x_sq, y_sq, sq, sq, fill=1, stroke=0)

            c.setFillColor(rl_colors.black)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_sq + sq + 4 * mm, y_cursor,
                         f"{tab_name}  (référence vitesse : {ref})")
            y_cursor -= 8 * mm

            data = [["Désignation", "Abscisse (mm)", "Temps1 (ms)", "Vitesse (m/s)"]]
            speeds = w._compute_speeds()
            for i, row in enumerate(w.rows):
                t_s = row.get("t_s")
                t_ms = "" if t_s is None else f"{float(t_s) * 1e3:.6f}"
                v = ("" if i == 0 else
                     ("—" if speeds[i] is None else f"{speeds[i]:.2f}"))
                data.append([
                    str(row.get("name", "")),
                    f"{float(row.get('d_mm', 0.0)):.2f}",
                    t_ms or "—", v,
                ])

            tbl = Table(data, colWidths=[55 * mm, 35 * mm, 35 * mm, 35 * mm])
            tbl.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.grey),
                ("FONT", (0, 1), (-1, -1), "Helvetica"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("BACKGROUND", (3, 1), (3, 1), rl_colors.whitesmoke),
            ]))
            w_tbl, h_tbl = tbl.wrapOn(c, W - 30 * mm, y_cursor)
            tbl.drawOn(c, 15 * mm, y_cursor - h_tbl)
            y_cursor = y_cursor - h_tbl - 8 * mm

        c.save()
        self._finish_busy_popup(popup, "PDF généré", 800)
        try:
            os.remove(tmp_png)
        except Exception:
            pass

    # ======================================================
    # FICHIERS
    # ======================================================
    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choisir des fichiers WFM", "", "Tektronix WFM (*.wfm);;Tous (*.*)"
        )
        if paths:
            self.load_paths(paths)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir un dossier contenant des WFM")
        if not folder:
            return
        paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".wfm")
        )
        if not paths:
            QMessageBox.information(self, "Info", "Aucun fichier .wfm trouvé.")
            return
        self.load_paths(paths)

    def load_paths(self, paths: list):
        if not paths:
            return

        progress = QProgressDialog("", None, 0, len(paths), self)
        progress.setWindowTitle("Chargement WFM")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setFixedSize(430, 140)
        progress.setValue(0)
        progress.setLabelText("Préparation du chargement…")
        progress.show()
        QApplication.processEvents()

        total_t0 = perf_counter()
        loaded_count = 0

        for idx, p in enumerate(paths, start=1):
            if len(self.loaded) >= self.MAX_FILES:
                QMessageBox.warning(self, "Limite",
                                    f"Max {self.MAX_FILES} fichiers. Le reste est ignoré.")
                break

            base = os.path.basename(p)
            progress.setValue(idx - 1)
            progress.setLabelText(
                f"Chargement en cours…\nFichier {idx}/{len(paths)}\n{base}"
            )
            QApplication.processEvents()

            file_t0 = perf_counter()
            try:
                sid, display, x, y, meta = read_wfm_as_xy(p)
                if any(s["id"] == sid for s in self.loaded):
                    continue

                ymin, ymax = float(np.min(y)), float(np.max(y))
                if ymin == ymax:
                    ymin -= 1.0; ymax += 1.0

                dt = fs = duration = None
                if x is not None and len(x) > 2:
                    dx = np.diff(x[:min(len(x), 2000)])
                    dx = dx[np.isfinite(dx)]
                    if dx.size:
                        dt = float(np.median(dx))
                        if dt != 0:
                            fs = 1.0 / dt
                        duration = float(x[-1] - x[0])

                lod = None
                try: lod = _build_lod_cache(x, y)
                except Exception: pass

                self.loaded.append({
                    "id": sid,
                    "display": display,
                    "plot_display": meta.get("plot_display", display),
                    "x": x, "y": y, "lod": lod,
                    "ymin": ymin, "ymax": ymax,
                    "meta": meta,
                    "dt": dt, "fs": fs, "duration": duration,
                    "n": int(len(y)),
                    "load_time_s": perf_counter() - file_t0,
                })
                loaded_count += 1

            except Exception as e:
                QMessageBox.warning(self, "Erreur lecture WFM", f"{base}\n{e}")

            progress.setValue(idx)
            QApplication.processEvents()

        self.rebuild_list()
        self.rebuild_assignments()
        self.refresh_view()

        if self.listw.count() and self.listw.currentRow() < 0:
            self.listw.setCurrentRow(self.listw.count() - 1)
        self.update_meta_from_selection()

        dt_total = perf_counter() - total_t0
        progress.setValue(len(paths))
        progress.setLabelText(
            f"Chargement terminé\n"
            f"{loaded_count} fichier(s) ajouté(s)\n"
            f"Temps total : {dt_total:.2f} s"
        )
        QApplication.processEvents()

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(progress.close)
        timer.start(900)
        self._last_busy_popup_timer = timer

    def clear_all(self):
        self.loaded.clear()
        self.rebuild_list()
        self.rebuild_assignments()
        self.refresh_view()
        self.update_meta_from_selection()

    def rebuild_list(self):
        self.listw.clear()
        for s in self.loaded:
            it = QListWidgetItem(s["display"])
            it.setData(Qt.UserRole, s["id"])
            self.listw.addItem(it)

    def rebuild_assignments(self):
        current_ids = [cb.currentData() for cb in self.assign]
        for ch, cb in enumerate(self.assign):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("—", userData=None)
            for s in self.loaded:
                cb.addItem(s["display"], userData=s["id"])
            if current_ids[ch] is not None:
                idx = cb.findData(current_ids[ch])
                cb.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                cb.setCurrentIndex(0)
            cb.blockSignals(False)

    def get_sig(self, sid) -> Optional[dict]:
        return next((s for s in self.loaded if s["id"] == sid), None)

    def update_meta_from_selection(self, *_):
        it = self.listw.currentItem()
        if it is None:
            self.meta_lbl.setText("—")
            return
        sig = self.get_sig(it.data(Qt.UserRole))
        if sig is None:
            self.meta_lbl.setText("—")
            return

        lines = [f"Fichier : {sig['display']}"]
        if sig.get("fs") is not None:
            lines.append(f"Fech : {sig['fs']:.6g} Hz")
        if sig.get("dt") is not None:
            lines.append(f"dt : {sig['dt']:.6g} s")
        lines.append(f"Nb points : {sig.get('n', len(sig.get('y', []))):,}")
        if sig.get("duration") is not None:
            lines.append(f"Durée : {sig['duration']:.6g} s")
        lines.append(f"Y min/max : {sig['ymin']:.6g} / {sig['ymax']:.6g}")
        self.meta_lbl.setText("\n".join(lines))

    # ======================================================
    # VUE
    # ======================================================
    def refresh_view(self):
        assigned = [
            (ch, sig)
            for ch in range(4)
            if (sid := self.assign[ch].currentData()) is not None
            and (sig := self.get_sig(sid)) is not None
        ]
        self.scope.set_layout_for_channels(assigned)
        self.apply_y_locks()
        self._apply_active_series_to_plot()
        self._update_auto_channel_enable()

    def apply_y_locks(self):
        for i, (ch_index, _sig) in enumerate(self.scope.active_map):
            self.scope.set_y_unlocked(i, self.unlock_y[ch_index].isChecked())

    def auto_y_channel(self, ch_index: int):
        for plot_idx, (ch, _sig) in enumerate(self.scope.active_map):
            if ch == ch_index:
                self.scope.auto_y_for_plot(plot_idx)
                break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
