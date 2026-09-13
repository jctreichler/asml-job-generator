#!/usr/bin/env python3.12
"""PyQt6 GUI for the ASML PAS5500 job-file generator (asml18.py backend).

Install:
    python3.12 -m ensurepip
    python3.12 -m pip install PyQt6 matplotlib gdstk
Run:
    python3.12 asml18_gui.py
"""

import csv
import json
import math
import os
import sys
import threading
import traceback
from typing import Any

import numpy as np

# ── Backend import ─────────────────────────────────────────────────────────────
try:
    from asml18 import (
        FOUR_INCH, SIX_INCH, EIGHT_INCH, WAFER_SIZES, FLAT, CELL_SIZE,
        Point, Mask, Image, Distrib, Layer, Expo, Illume, Alignment, Mark, Wafer,
        read_gds,
    )
    BACKEND_OK    = True
    BACKEND_ERROR = ""
except Exception as _e:
    BACKEND_OK    = False
    BACKEND_ERROR = str(_e)
    FOUR_INCH, SIX_INCH, EIGHT_INCH = 0, 1, 2
    WAFER_SIZES = [100.0, 150.0, 200.0]
    FLAT        = [47.2857, 69.0, 100.0]
    CELL_SIZE   = 22000
    class Point:
        def __init__(self, x: float = 0.0, y: float = 0.0):
            self.x = x; self.y = y
    class Illume:
        DEFAULT = 0; CONVENTIONAL = 1; QUADRUPOLE = 2; ANNULAR = 3
        def __init__(self, illume_type=1, na=0.57, sigin=0.0, sigout=0.5):
            pass

try:
    import gdstk
    GDSTK_OK = True
except ImportError:
    GDSTK_OK = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle as MplRect
    from matplotlib.transforms import Affine2D
    from matplotlib.colors import to_rgb
    MATPLOTLIB_OK = True
    # Axes use `aspect='equal', adjustable='datalim'` so the plot box always
    # fills its slot and the data range is padded instead; matplotlib logs a
    # WARNING every time it does that padding — drop just that one message.
    import logging as _logging
    _logging.getLogger('matplotlib.axes._base').addFilter(
        lambda r: 'to fulfill fixed data aspect' not in r.getMessage())
    import warnings as _warnings
    _warnings.filterwarnings('ignore', message='Tight layout not applied')
except ImportError:
    MATPLOTLIB_OK = False

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QDialog, QDialogButtonBox,
    QFileDialog, QMessageBox, QPlainTextEdit, QRadioButton,
    QGroupBox, QSplitter, QListWidget, QListWidgetItem, QStackedWidget,
    QScrollArea,
)

# ── Display constants ──────────────────────────────────────────────────────────
WAFER_LABELS     = ['4 inch (100 mm)', '6 inch (150 mm)', '8 inch (200 mm)']
COVER_MODES      = ['W – Whole wafer']
PLACE_MODES      = ['O – Origin', 'C – Centred']
PREALIGN_METHODS = ['STANDARD', 'NONE']
ILLUME_LABELS    = ['Conventional', 'Annular', 'Quadrupole', 'Default']
ILLUME_CODES     = [1, 3, 2, 0]          # Illume.CONVENTIONAL / ANNULAR / QUADRUPOLE / DEFAULT
LEVEL_METHODS    = ['D – Default', 'G – Global', 'L – Local', 'N – None']
PM_GENERATED     = 'PM (generated)'
_PREVIEW_COLORS  = ['#4fc3f7', '#81c784', '#ffb74d', '#f06292', '#ce93d8']


# ── AppState ───────────────────────────────────────────────────────────────────
class AppState:
    """Central data store. All tabs share one instance; they mutate it in-place."""

    def __init__(self):
        self.cell = dict(
            wafer_w=FOUR_INCH, cell_x=22.0, cell_y=22.0,
            round_edge=2.0, flat_edge=0.0, edge_excl=3.0,
            cover_mode='W', placement_mode='O', prealign_method='STANDARD',
            combine_zero_first=False, output_name='output',
            pm_reticle='4544020*',
            rotated_marks=False, rotated_marks_sign=-1,
            dies_x=1, dies_y=1, min_dies=0,
        )
        self.masks:      list[dict] = []
        self.images:     list[dict] = []
        self.layers:     list[dict] = []
        self.marks:      list[dict] = []
        self.alignments: list[dict] = []
        self.exposures:  list[dict] = []
        self.pattern_mode:   dict[str, str]        = {}
        self.pattern_manual: dict[str, list[dict]] = {}
        self.pattern_code:   dict[str, str]        = {}
        self.patterns:       dict[str, list[dict]] = {}
        self.layer_process:  dict[str, dict]       = {}
        self.pattern_grid:     dict[str, set]  = {}
        self.pattern_csv_file: dict[str, str]  = {}
        self.pattern_step:     dict[str, dict] = {}

    def mask_ids(self)      -> list[str]: return [m['id'] for m in self.masks]
    def image_ids(self)     -> list[str]: return [i['id'] for i in self.images]
    def layer_ids(self)     -> list[str]: return [l['id'] for l in self.layers]
    def mark_ids(self)      -> list[str]: return [m['id'] for m in self.marks]
    def alignment_ids(self) -> list[str]: return [a['name'] for a in self.alignments]


# ── Pattern eval ───────────────────────────────────────────────────────────────
_PATTERN_MODULES = {'math': math, 'numpy': np, 'np': np}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__ for pattern code: `math` and `numpy` only (both are
    already bound as `math` / `np`, but users still write `import math`)."""
    root = name.split('.')[0]
    if level == 0 and root in _PATTERN_MODULES:
        return _PATTERN_MODULES[root]
    raise ImportError(
        f"cannot import {name!r} in pattern code — only 'math' and 'numpy' "
        f"are available (already bound as math / np)")


def _eval_pattern(code: str) -> list[dict] | str:
    """Evaluate pattern code in a restricted namespace.

    Accepts a single expression that returns a list, or statements that assign
    to 'points' or 'result'. Returns a list of {x, y} dicts or an error string.
    """
    safe: dict = {
        '__builtins__': {'__import__': _safe_import},
        'math': math, 'np': np,
        'range': range, 'list': list, 'zip': zip,
        'enumerate': enumerate, 'abs': abs, 'round': round,
        'len': len, 'float': float, 'int': int,
        'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
    }
    if BACKEND_OK:
        safe['Point'] = Point

    try:
        result = eval(code, safe, {})
    except SyntaxError:
        ns = dict(safe)
        try:
            exec(code, ns)
            result = ns.get('points') or ns.get('result')
            if result is None:
                return "Assign output to 'points' or 'result'"
        except Exception as exc:
            return str(exc)
    except Exception as exc:
        return str(exc)

    if not isinstance(result, (list, tuple)):
        return "Expression must produce a list"
    out = []
    for p in result:
        if BACKEND_OK and isinstance(p, Point):
            out.append({'x': float(p.x), 'y': float(p.y)})
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            out.append({'x': float(p[0]), 'y': float(p[1])})
        elif isinstance(p, dict) and 'x' in p and 'y' in p:
            out.append({'x': float(p['x']), 'y': float(p['y'])})
        else:
            return f"Unsupported point type: {type(p).__name__}"
    return out


# ── State serialization ────────────────────────────────────────────────────────
def _serialize_state(state: 'AppState') -> dict:
    return {
        'cell':             state.cell,
        'masks':            state.masks,
        'images':           state.images,
        'layers':           state.layers,
        'marks':            state.marks,
        'alignments':       state.alignments,
        'exposures':        state.exposures,
        'pattern_mode':     state.pattern_mode,
        'pattern_manual':   state.pattern_manual,
        'pattern_code':     state.pattern_code,
        'patterns':         state.patterns,
        'layer_process':    state.layer_process,
        'pattern_grid':     {k: [[p[0], p[1]] for p in v]
                             for k, v in state.pattern_grid.items()},
        'pattern_csv_file': state.pattern_csv_file,
        'pattern_step':     state.pattern_step,
    }


def _deserialize_state(d: dict) -> 'AppState':
    s = AppState()
    s.cell             = d.get('cell', s.cell)
    s.masks            = d.get('masks', [])
    s.images           = d.get('images', [])
    s.layers           = d.get('layers', [])
    s.marks            = d.get('marks', [])
    s.alignments       = d.get('alignments', [])
    s.exposures        = d.get('exposures', [])
    s.pattern_mode     = d.get('pattern_mode', {})
    s.pattern_manual   = d.get('pattern_manual', {})
    s.pattern_code     = d.get('pattern_code', {})
    s.patterns         = d.get('patterns', {})
    s.layer_process    = d.get('layer_process', {})
    s.pattern_csv_file = d.get('pattern_csv_file', {})
    s.pattern_step     = d.get('pattern_step', {})
    s.pattern_grid     = {
        k: {(float(p[0]), float(p[1])) for p in v}
        for k, v in d.get('pattern_grid', {}).items()
    }
    return s


# ── PM mark generation ─────────────────────────────────────────────────────────
def _get_pm_image(reticle_id: str | None = None):
    """The PM alignment-mark image.

    Delegates to the backend, which generates the geometry in code and caches it
    in its own module state (so the implicit layer-0 marks exposure,
    _prepare_job and write_file's mark visualisation all see the same image).
    `reticle_id` overrides which plate exposes the marks (default: COMBI plate).
    """
    if not (GDSTK_OK and BACKEND_OK):
        return None
    import asml18 as _backend
    return _backend.get_pm_image(reticle_id)


# ── FormDialog ─────────────────────────────────────────────────────────────────
class FormDialog(QDialog):
    """Base dialog with a label-widget grid and OK/Cancel buttons."""

    def __init__(self, title: str, parent=None, apply_callback=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._outer   = QVBoxLayout(self)
        self._form    = QGridLayout()
        self._row     = 0
        self._widgets: dict[str, QWidget] = {}
        self._outer.addLayout(self._form)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        if apply_callback is not None:
            self._apply_callback = apply_callback
            _ab = btns.addButton('Apply', QDialogButtonBox.ButtonRole.ApplyRole)
            _ab.clicked.connect(self._on_apply)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self._outer.addWidget(btns)

    def _on_apply(self):
        if hasattr(self, '_apply_callback'):
            self._apply_callback(self.result())

    def _add(self, key: str, label: str, widget: QWidget) -> QWidget:
        self._form.addWidget(QLabel(label + ':'), self._row, 0)
        self._form.addWidget(widget, self._row, 1)
        self._widgets[key] = widget
        self._row += 1
        return widget

    def _line(self, key: str, label: str, value: str = '') -> QLineEdit:
        return self._add(key, label, QLineEdit(value))

    def _dspin(self, key: str, label: str, value: float = 0.0,
               lo: float = -9999.0, hi: float = 9999.0, dec: int = 4) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi); w.setDecimals(dec); w.setValue(value)
        return self._add(key, label, w)

    def _combo(self, key: str, label: str, items: list[str], current: str = '') -> QComboBox:
        w = QComboBox(); w.addItems(items)
        if current in items:
            w.setCurrentText(current)
        elif items:
            w.setCurrentIndex(0)
        return self._add(key, label, w)

    def _check(self, key: str, label: str, value: bool = False) -> QCheckBox:
        w = QCheckBox(); w.setChecked(value)
        return self._add(key, label, w)

    def _dspin_xy(self, key_prefix: str, label: str,
                  vx: float = 0.0, vy: float = 0.0,
                  lo: float = -9999.0, hi: float = 9999.0, dec: int = 4):
        _NoBtn = QAbstractSpinBox.ButtonSymbols.NoButtons
        wx = QDoubleSpinBox(); wx.setRange(lo, hi); wx.setDecimals(dec)
        wx.setValue(vx); wx.setButtonSymbols(_NoBtn)
        wy = QDoubleSpinBox(); wy.setRange(lo, hi); wy.setDecimals(dec)
        wy.setValue(vy); wy.setButtonSymbols(_NoBtn)
        row_w = QWidget(); h = QHBoxLayout(row_w); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel('X:')); h.addWidget(wx)
        h.addSpacing(8)
        h.addWidget(QLabel('Y:')); h.addWidget(wy)
        h.addStretch()
        self._form.addWidget(QLabel(label + ':'), self._row, 0)
        self._form.addWidget(row_w, self._row, 1)
        self._widgets[key_prefix + '_x'] = wx
        self._widgets[key_prefix + '_y'] = wy
        self._row += 1

    def val(self, key: str) -> Any:
        w = self._widgets[key]
        if isinstance(w, QLineEdit):      return w.text().strip()
        if isinstance(w, QDoubleSpinBox): return w.value()
        if isinstance(w, QSpinBox):       return w.value()
        if isinstance(w, QComboBox):      return w.currentText()
        if isinstance(w, QCheckBox):      return w.isChecked()
        return None


# ── GDS cell inspection (hierarchy-aware, no flattening) ──────────────────────
def _hierarchy_layers(cell) -> list:
    """Layer numbers used anywhere under `cell` — walks the dependency cells and
    reads each one's own geometry, so it never materialises the (possibly
    millions of) flattened instances."""
    layers: set = set()
    for c in (cell, *cell.dependencies(True)):
        for p in c.polygons:
            layers.add(p.layer)
        for pth in c.paths:
            try:
                layers.update(pth.layers)
            except Exception:
                pass
    return sorted(layers)


def _approx_instances(cell, _cache: dict | None = None) -> int:
    """Rough count of placed instances under `cell` (arrays counted in full).
    Cheap — recurses over the ~cell hierarchy, not the flattened geometry."""
    if _cache is None:
        _cache = {}
    if cell.name in _cache:
        return _cache[cell.name]
    _cache[cell.name] = 1          # guard against cycles
    total = 1
    for r in cell.references:
        rep = getattr(r, 'repetition', None)
        k = rep.size if (rep is not None and rep.size) else 1
        total += k * _approx_instances(r.cell, _cache)
    _cache[cell.name] = total
    return total


_MAX_MASK_INSTANCES = 10_000_000   # above any real reticle; blocks flattening a
                                   # pathological top cell (e.g. UtilOut 'top' ~47M)

_RASTER_POLY_THRESHOLD = 50_000   # per-layer polygon count above which we give up
                                   # on drawing individual shapes and switch to a
                                   # density raster (uniform stride-sampling that
                                   # many polygons down to a cap just drops most of
                                   # them, which reads as noise, not density)
_RASTER_RES = 512                 # density-raster resolution (long side, px)


_RASTER_STAMP_BUDGET = 2_000_000   # cap on individually-stamped array elements
_RASTER_VISIT_CAP    = 400_000     # cap on reference visits in the walk
_WINDOW_COUNT_VISIT_CAP = 200_000  # same, for the windowed re-count on zoom —
                                   # it runs on the interactive path and re-walks
                                   # shared subtrees once per path through them,
                                   # so it needs its own bound


def _norm2(v):
    return float(v[0] * v[0] + v[1] * v[1])


def _raster_frame(bbox, res):
    """Pixel grid for a world bbox `((x0,y0),(x1,y1))`: returns
    `(extent, nx, ny, px)` with ~`res` px on the long side and `px` = world
    µm per pixel."""
    (x0, y0), (x1, y1) = bbox
    w, h = max(float(x1 - x0), 1e-6), max(float(y1 - y0), 1e-6)
    s = res / max(w, h)
    nx, ny = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return (float(x0), float(y0), float(x1), float(y1)), nx, ny, max(w / nx, h / ny)


def _poly_world_rects(wp):
    """`wp` (list of (N,2) world-point arrays) -> (K,4) array of
    `[minx, miny, maxx, maxy]`, one row per polygon."""
    lens = np.array([len(p) for p in wp], dtype=np.intp)
    allp = np.vstack(wp)
    if lens.size and bool((lens == lens[0]).all()):
        a = allp.reshape(-1, int(lens[0]), 2)
        return np.column_stack([a[:, :, 0].min(1), a[:, :, 1].min(1),
                                a[:, :, 0].max(1), a[:, :, 1].max(1)])
    st = np.concatenate(([0], np.cumsum(lens)[:-1]))
    return np.column_stack([np.minimum.reduceat(allp[:, 0], st),
                            np.minimum.reduceat(allp[:, 1], st),
                            np.maximum.reduceat(allp[:, 0], st),
                            np.maximum.reduceat(allp[:, 1], st)])


def _world_rect_indices(rects, extent, nx, ny):
    """(K,4) world `[x0,y0,x1,y1]` rects -> inclusive pixel index arrays
    `(r0, r1, c0, c1)`, keeping only rects that touch the frame. Row 0 = y1."""
    ex0, ey0, ex1, ey1 = extent
    ew, eh = max(ex1 - ex0, 1e-9), max(ey1 - ey0, 1e-9)
    x0, y0, x1, y1 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    keep = (x1 >= ex0) & (x0 <= ex1) & (y1 >= ey0) & (y0 <= ey1)
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    c0 = np.clip(((x0 - ex0) / ew * nx).astype(np.intp), 0, nx - 1)
    c1 = np.clip(((x1 - ex0) / ew * nx).astype(np.intp), 0, nx - 1)
    r0 = np.clip(((ey1 - y1) / eh * ny).astype(np.intp), 0, ny - 1)
    r1 = np.clip(((ey1 - y0) / eh * ny).astype(np.intp), 0, ny - 1)
    return r0, r1, c0, c1


def _stamp_rects(grid, r0, r1, c0, c1):
    """Mark every pixel of each `[r0..r1] x [c0..c1]` rect in `grid` (uint8, set
    to 255) — vectorised via a 2-D difference array + prefix sum, corner deltas
    accumulated with `bincount` (a tight C loop, unlike `np.add.at`)."""
    ny, nx = grid.shape
    fw = nx + 1
    n = (ny + 1) * fw
    diff = (np.bincount(r0 * fw + c0,           minlength=n)
            - np.bincount(r0 * fw + (c1 + 1),   minlength=n)
            - np.bincount((r1 + 1) * fw + c0,   minlength=n)
            + np.bincount((r1 + 1) * fw + (c1 + 1), minlength=n)).reshape(ny + 1, fw)
    cov = diff.cumsum(0).cumsum(1)[:ny, :nx]
    np.maximum(grid, np.where(cov > 0, np.uint8(255), np.uint8(0)), out=grid)


def _layer_poly_counts(cell, _cache=None):
    """`{layer: flattened polygon count}` for `cell`, array sizes multiplied
    through the hierarchy — no geometry touched, so it's cheap even when the
    real flatten would be millions of polygons."""
    if _cache is None:
        _cache = {}
    if cell.name in _cache:
        return _cache[cell.name]
    _cache[cell.name] = {}
    tally: dict = {}
    for p in cell.polygons:
        tally[p.layer] = tally.get(p.layer, 0) + 1
    # Paths: one polygon per path element, counted from `pth.layers` rather
    # than materialising `to_polygons()`. An estimate, but this count only
    # picks the raster-vs-vector threshold, and paths are never the dense case.
    for pth in cell.paths:
        for lyr in _path_layers_seq(pth):
            tally[lyr] = tally.get(lyr, 0) + 1
    for r in cell.references:
        rep = r.repetition
        k = rep.size if (rep is not None and rep.size) else 1
        for lyr, n in _layer_poly_counts(r.cell, _cache).items():
            tally[lyr] = tally.get(lyr, 0) + k * n
    _cache[cell.name] = tally
    return tally


def _window_layer_counts(cell, extent, m=None, t=None, acc=None, _lc=None,
                         _st=None):
    """Like `_layer_poly_counts` but only counting geometry whose bbox meets
    `extent` — a regular array contributes just the `(i,j)` range that lands
    in the window (`_regular_index_range`), so a deep zoom onto a label array
    reports a few hundred polygons for that layer, not the whole cell's
    hundreds of thousands. Lets the caller keep such a layer as real vector
    shapes instead of a coverage wash. The child's per-instance layer counts
    are taken whole (not re-clipped) — a safe over-estimate for the raster
    threshold decision."""
    if m is None:
        m, t, acc, _lc = np.eye(2), np.zeros(2), {}, {}
        _st = dict(visits=0)
    ebb = ((extent[0], extent[1]), (extent[2], extent[3]))
    cbb = cell.bounding_box()
    if cbb is not None and not _bbox_intersects(_transform_bbox(m, t, cbb), ebb):
        return acc

    for p in cell.polygons:
        mn, mx = p.points.min(0), p.points.max(0)
        if _bbox_intersects(_transform_bbox(m, t, ((mn[0], mn[1]), (mx[0], mx[1]))), ebb):
            acc[p.layer] = acc.get(p.layer, 0) + 1

    for pth in cell.paths:
        pbb = pth.bounding_box()
        if pbb is None or not _bbox_intersects(_transform_bbox(m, t, pbb), ebb):
            continue
        for lyr in _path_layers_seq(pth):
            acc[lyr] = acc.get(lyr, 0) + 1

    frame = _Frame(m, t, ebb)

    for ref in cell.references:
        _st['visits'] += 1
        plan = _plan_reference(ref, frame)
        if plan is None:
            continue

        if plan.kind == _REF_SINGLE and _st['visits'] <= _WINDOW_COUNT_VISIT_CAP:
            m2, t2 = plan.base_transform(m, t)
            _window_layer_counts(ref.cell, extent, m2, t2, acc, _lc, _st)
            continue

        # A repeated child (or a single one we've run out of visits for) is
        # taken whole from the cached per-cell counts rather than descended.
        # `n_visible` is the viewport-pruned instance count for a regular grid
        # and the full repetition otherwise — an over-estimate in that case,
        # which biases toward the density raster, the safe way to be wrong.
        n = plan.n_visible if plan.repeated else 1
        sub = _lc.get(ref.cell.name)
        if sub is None:
            sub = _layer_poly_counts(ref.cell)
            _lc[ref.cell.name] = sub
        for lyr, c in sub.items():
            acc[lyr] = acc.get(lyr, 0) + n * c
    return acc


def _path_layers_seq(pth) -> tuple:
    """One layer per element of `pth` (a path can carry several parallel
    elements on different layers). Read straight off the path — nothing is
    materialised."""
    try:
        return tuple(pth.layers)
    except Exception:
        return ()


def _path_layers(cell) -> set:
    """Layers `cell`'s own paths draw on — read from `pth.layers`, so no
    geometry is materialised."""
    s: set = set()
    for pth in cell.paths:
        s.update(_path_layers_seq(pth))
    return s


def _cell_path_polys(cell, _cache=None) -> dict:
    """`{layer: [(N,2) local point arrays]}` for `cell`'s own PATHS.

    gdstk keeps paths separate from polygons. `cell.get_polygons()` — what the
    pre-rewrite renderer called — includes them; the reference walk that
    replaced it read `cell.polygons` only, so path geometry silently rendered
    as nothing while `_hierarchy_layers` still offered its layers in the mask
    dialog. `to_polygons()` materialises, so results are cached per cell."""
    if _cache is not None and cell.name in _cache:
        return _cache[cell.name]
    g: dict = {}
    for pth in cell.paths:
        try:
            polys = pth.to_polygons()
        except Exception:
            continue
        for poly in polys:
            g.setdefault(poly.layer, []).append(poly.points)
    if _cache is not None:
        _cache[cell.name] = g
    return g


def _leaf_layers(cell, _cache=None):
    """Set of layers any polygon or path anywhere under `cell` draws on."""
    if _cache is None:
        _cache = {}
    if cell.name in _cache:
        return _cache[cell.name]
    _cache[cell.name] = set()
    s = {p.layer for p in cell.polygons} | _path_layers(cell)
    for r in cell.references:
        s |= _leaf_layers(r.cell, _cache)
    _cache[cell.name] = s
    return s


def _group_own_world(cell, m, t, path_cache=None):
    """`{layer: [world-point arrays]}` for `cell`'s *own* polygons and paths."""
    g: dict = {}
    for p in cell.polygons:
        g.setdefault(p.layer, []).append(p.points @ m.T + t)
    for lyr, pts_list in _cell_path_polys(cell, path_cache).items():
        g.setdefault(lyr, []).extend(q @ m.T + t for q in pts_list)
    return g


def _rc_push(ctx, lyr, idx):
    r0, r1, c0, c1 = idx
    if len(r0):
        ctx['pending'].setdefault(lyr, []).append(idx)


def _tile_into(main, local, dcol, drow):
    """OR `local` (uint8) into `main` (uint8), top-left at `(drow, dcol)`,
    clipped to `main`'s bounds."""
    mny, mnx = main.shape
    lny, lnx = local.shape
    r0, c0 = max(0, drow), max(0, dcol)
    r1, c1 = min(mny, drow + lny), min(mnx, dcol + lnx)
    if r1 > r0 and c1 > c0:
        np.maximum(main[r0:r1, c0:c1],
                   local[r0 - drow:r1 - drow, c0 - dcol:c1 - dcol],
                   out=main[r0:r1, c0:c1])


def _tile_non_pure(ctx, child, m0, t0, base_bb, ox, oy, budget, depth):
    """A repeated reference whose child has its own sub-references: rasterise
    ONE instance into a local grid (same pixel pitch as the main frame) and OR
    it in at every offset, instead of walking the subtree once per element — a
    7x7 die array of a 200-reference cell goes from 49 walks to 1."""
    px = ctx['px']
    (bx0, by0), (bx1, by1) = _transform_bbox(m0, t0, base_bb)
    snx = max(1, int(math.ceil((bx1 - bx0) / px)))
    sny = max(1, int(math.ceil((by1 - by0) / px)))
    if snx * sny > 4 * ctx['nx'] * ctx['ny'] or len(ox) <= 1:
        for k in range(len(ox)):
            _raster_accum(child, m0, np.array([t0[0] + ox[k], t0[1] + oy[k]]),
                          ctx, budget, depth + 1)
        return
    sub = dict(raster_set=ctx['raster_set'],
               grids={l: np.zeros((sny, snx), np.uint8) for l in ctx['raster_set']},
               extent=(bx0, by0, bx1, by1), nx=snx, ny=sny, px=px,
               vec={}, pending={}, leaf_cache=ctx['leaf_cache'],
               path_cache=ctx['path_cache'], visits=0)
    _raster_accum(child, m0, t0, sub, budget, 0)
    for lyr, chunks in sub['pending'].items():
        _stamp_rects(sub['grids'][lyr],
                     *[np.concatenate([c[i] for c in chunks]) for i in range(4)])
    grids = {l: g for l, g in sub['grids'].items() if g.any()}
    mex = ctx['extent']
    budget[0] -= len(ox)
    for k in range(len(ox)):
        dcol = int(round((bx0 + ox[k] - mex[0]) / px))
        drow = int(round((mex[3] - (by1 + oy[k])) / px))
        for lyr, g in grids.items():
            dst = ctx['grids'].setdefault(
                lyr, np.zeros((ctx['ny'], ctx['nx']), np.uint8))
            _tile_into(dst, g, dcol, drow)
        if sub['vec']:
            d = np.array([ox[k], oy[k]])
            for lyr, polys in sub['vec'].items():
                ctx['vec'].setdefault(lyr, []).extend(p + d for p in polys)


_VEC_LAYER_CAP = 120_000   # a "kept" layer that collects more real polygons
                            # than this mid-walk is sealed to a coverage raster
                            # after all (guards a mis-estimated zoom window)


def _seal_layer(ctx, lyr):
    """Promote a layer that was being kept as vector polygons to a coverage
    raster — stamp what it has collected so far, then treat it as raster."""
    ctx['raster_set'].add(lyr)
    ctx['grids'].setdefault(lyr, np.zeros((ctx['ny'], ctx['nx']), np.uint8))
    got = ctx['vec'].pop(lyr, None)
    if got:
        _rc_push(ctx, lyr, _world_rect_indices(_poly_world_rects(got),
                                               ctx['extent'], ctx['nx'], ctx['ny']))


def _emit_polys(ctx, lyr, wp):
    """A batch of world polygons on one layer -> pixel-bbox stamps (raster
    layer) or decimatable vector polygons (kept layer)."""
    if not wp:
        return
    if lyr in ctx['raster_set']:
        _rc_push(ctx, lyr, _world_rect_indices(_poly_world_rects(wp),
                                               ctx['extent'], ctx['nx'], ctx['ny']))
    else:
        dst = ctx['vec'].setdefault(lyr, [])
        dst.extend(wp)
        if len(dst) > _VEC_LAYER_CAP:
            _seal_layer(ctx, lyr)


def _emit_rect(ctx, lyr, wbb):
    """A single world bbox (an array envelope filled analytically) -> stamp."""
    (x0, y0), (x1, y1) = wbb
    if lyr in ctx['raster_set']:
        _rc_push(ctx, lyr, _world_rect_indices(
            np.array([[x0, y0, x1, y1]], float), ctx['extent'], ctx['nx'], ctx['ny']))
    else:
        ctx['vec'].setdefault(lyr, []).append(
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))


def _emit_tiled(ctx, lyr, wp, ox, oy):
    """Base world polygons `wp` replicated at every `(ox, oy)` offset."""
    if not wp or len(ox) == 0:
        return
    if lyr in ctx['raster_set']:
        base = _poly_world_rects(wp)                       # (P, 4)
        off  = np.stack([ox, oy, ox, oy], axis=1)          # (T, 4)
        rects = (base[:, None, :] + off[None, :, :]).reshape(-1, 4)
        _rc_push(ctx, lyr, _world_rect_indices(rects, ctx['extent'],
                                               ctx['nx'], ctx['ny']))
    else:
        dst = ctx['vec'].setdefault(lyr, [])
        for k in range(len(ox)):
            d = np.array([ox[k], oy[k]])
            dst.extend(p + d for p in wp)
        if len(dst) > _VEC_LAYER_CAP:
            _seal_layer(ctx, lyr)


def _raster_accum(cell, m, t, ctx, budget, depth=0):
    """Walk `cell`'s reference tree, feeding `ctx` a coverage stamp or a vector
    polygon per leaf — WITHOUT expanding arrays. A regular grid finer than one
    raster pixel is filled over its envelope in O(1); a coarser one is tiled
    with a single vectorised stamp; only a non-leaf array is descended per
    element."""
    ex = ctx['extent']
    ebb = ((ex[0], ex[1]), (ex[2], ex[3]))
    cbb = cell.bounding_box()
    if cbb is not None and not _bbox_intersects(_transform_bbox(m, t, cbb), ebb):
        return

    frame = _Frame(m, t, ebb)

    # Skip the affine only when it really IS the identity. This used to test
    # `depth == 0`, but `_tile_non_pure` re-enters the walk at depth 0 with the
    # reference's own transform — so a repeated ref to a non-pure child placed
    # that child's own polygons at their raw local coordinates, off by the whole
    # reference transform. Deriving the flag from `m`/`t` can't drift.
    ident = (m[0, 0] == 1.0 and m[1, 1] == 1.0 and m[0, 1] == 0.0
             and m[1, 0] == 0.0 and t[0] == 0.0 and t[1] == 0.0)
    groups: dict = {}
    for p in cell.polygons:
        groups.setdefault(p.layer, []).append(
            p.points if ident else p.points @ m.T + t)
    for lyr, pts_list in _cell_path_polys(cell, ctx['path_cache']).items():
        groups.setdefault(lyr, []).extend(
            pts_list if ident else [q @ m.T + t for q in pts_list])
    for lyr, wp in groups.items():
        _emit_polys(ctx, lyr, wp)

    for ref in cell.references:
        ctx['visits'] += 1
        over_budget = ctx['visits'] > _RASTER_VISIT_CAP or budget[0] <= 0
        plan = _plan_reference(ref, frame)
        if plan is None:
            continue

        child_layers = _leaf_layers(ref.cell, ctx['leaf_cache'])
        # Envelope collapse only ever applies to layers that end up as a
        # coverage bitmap — a kept (vector) layer always needs its real
        # polygons, and it's < the raster threshold so materialising it is
        # bounded anyway.
        can_fill = bool(child_layers) and child_layers <= ctx['raster_set']

        if plan.kind == _REF_SINGLE:
            if over_budget and can_fill:
                for lyr in child_layers:
                    _emit_rect(ctx, lyr, plan.whole_wb)
            else:
                m2, t2 = plan.base_transform(m, t)
                _raster_accum(ref.cell, m2, t2, ctx, budget, depth + 1)
            continue

        child_pure = not ref.cell.references

        # Fill the whole envelope (O(1)) only when it would read as solid: a
        # regular grid whose *repeated* axes step less than a pixel AND whose
        # leaf is big enough that adjacent copies abut (a via/contact fill).
        # A sparse grid — small marks on a coarse pitch, or a 1×N row of
        # digits whose unused axis carries a spurious tiny step — must be
        # tiled, not filled, or the gaps get painted over.
        fill = over_budget and can_fill
        if not can_fill:
            pass                       # keep every real polygon for vector layers
        elif not fill and plan.kind == _REF_GRID and child_pure:
            V1, V2 = m @ plan.v1, m @ plan.v2
            bwp    = _transform_bbox(m, np.zeros(2), plan.base_wb)
            leaf_w = bwp[1][0] - bwp[0][0]
            leaf_h = bwp[1][1] - bwp[0][1]
            col_step = math.sqrt(_norm2(V1))
            row_step = math.sqrt(_norm2(V2))
            steps, dense = [], True
            if (plan.rep.columns or 0) > 1:
                steps.append(col_step)
                dense &= leaf_w >= 0.8 * col_step
            if (plan.rep.rows or 0) > 1:
                steps.append(row_step)
                dense &= leaf_h >= 0.8 * row_step
            if steps and dense and min(steps) < ctx['px']:
                fill = True
        if can_fill and not fill and child_pure and plan.kind == _REF_GRID:
            # Too many surviving elements to stamp one by one.
            fill = plan.n_visible > min(budget[0], _RASTER_STAMP_BUDGET)

        # An irregular repetition too large to enumerate has no instance list
        # to walk, so it collapses to its envelope whatever `can_fill` says.
        if fill or plan.kind == _REF_UNBOUNDED:
            env = _bbox_clip(plan.whole_wb, ebb) or plan.whole_wb
            for lyr in child_layers:
                _emit_rect(ctx, lyr, env)
            continue

        ox, oy = plan.world_offsets(m)
        if ox.size == 0:
            continue
        budget[0] -= ox.size
        if child_pure:
            m2, t2 = plan.base_transform(m, t)
            for lyr, wp in _group_own_world(ref.cell, m2, t2,
                                            ctx['path_cache']).items():
                _emit_tiled(ctx, lyr, wp, ox, oy)
        else:
            _tile_non_pure(ctx, ref.cell, m @ plan.mr, t + m @ plan.tr,
                           plan.base_bb, ox, oy, budget, depth)


def _build_layer_payload(cell, clip_bbox=None) -> dict:
    """Preview payload for `cell` WITHOUT ever flattening its arrays.

    A single reference walk (`_raster_accum`) decides per layer, from an
    analytic polygon count (`_layer_poly_counts` — array sizes multiplied
    through, no geometry), whether to keep real vertex-decimated polygons in
    `by_layer` or accumulate a coverage raster in `density`. Regular-grid
    arrays finer than one raster pixel — the via/contact fields that dominate
    real reticles — are filled over their envelope in O(1); coarser grids get
    one vectorised tile-stamp. `cell.get_polygons()` (which expands every
    array: ~1.1M polygons and ~3s on `ExShan.gds`'s `mask`) is never called.

    `density[lyr]` is `(uint8 [ny,nx] 0/255, (x0,y0,x1,y1))` exactly as before
    — the render sites' alpha wash (`arr/255 * 0.5`) is unchanged.
    `clip_bbox` (world `((x0,y0),(x1,y1))`) restricts the raster frame to that
    window at full resolution, to sharpen detail on zoom-in.
    """
    cbb = cell.bounding_box()
    if cbb is None:
        return dict(by_layer={}, density={}, total=0, n_layers=0, truncated=False)
    frame_bbox = (tuple(clip_bbox[0]), tuple(clip_bbox[1])) if clip_bbox is not None \
        else ((cbb[0][0], cbb[0][1]), (cbb[1][0], cbb[1][1]))
    extent, nx, ny, px = _raster_frame(frame_bbox, _RASTER_RES)

    # Whole-cell counts pick raster-vs-vector normally; a windowed re-raster
    # (zoom-in) counts only what's IN the window, so a label array that's
    # dense across the whole reticle but sparse in this view is kept as real
    # glyph shapes instead of a wash of rectangles.
    counts = (_window_layer_counts(cell, extent) if clip_bbox is not None
              else _layer_poly_counts(cell))
    raster_set = {l for l, n in counts.items() if n > _RASTER_POLY_THRESHOLD}

    ctx = dict(raster_set=raster_set,
               grids={l: np.zeros((ny, nx), np.uint8) for l in raster_set},
               extent=extent, nx=nx, ny=ny, px=px,
               vec={}, pending={}, leaf_cache={}, path_cache={}, visits=0)
    _raster_accum(cell, np.eye(2), np.zeros(2), ctx, [_RASTER_STAMP_BUDGET])

    for lyr, chunks in ctx['pending'].items():   # one bincount per layer
        grid = ctx['grids'].setdefault(lyr, np.zeros((ny, nx), np.uint8))
        _stamp_rects(grid,
                     *[np.concatenate([c[i] for c in chunks]) for i in range(4)])

    density = {l: (g, (extent[0], extent[1], extent[2], extent[3]))
               for l, g in ctx['grids'].items() if g.any()}

    MAX_VERTS = 256
    by_layer: dict = {}
    for lyr, lst in ctx['vec'].items():
        dec = []
        for pts in lst:
            pts = np.asarray(pts)
            if len(pts) > MAX_VERTS:
                step = max(1, len(pts) // MAX_VERTS)
                pts  = np.vstack([pts[::step], pts[:1]])
            dec.append(pts)
        by_layer[lyr] = dec

    return dict(by_layer=by_layer, density=density,
                total=int(sum(counts.values())),
                n_layers=len(by_layer) + len(density),
                truncated=bool(density))


def _crop_density(arr, extent, bx0, by0, bx1, by1):
    """Crop a `_build_layer_payload` density raster to a local (bx0,by0)-(bx1,by1)
    box. Returns `(sub_array, (cx0,cx1,cy0,cy1))` clipped to the actual overlap,
    or None if the box misses the raster or the overlap is empty of coverage."""
    x0, y0, x1, y1 = extent
    if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
        return None
    cx0, cx1 = max(bx0, x0), min(bx1, x1)
    cy0, cy1 = max(by0, y0), min(by1, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    ny, nx = arr.shape
    col0 = max(0, min(nx - 1, int((cx0 - x0) / (x1 - x0) * nx)))
    col1 = max(col0 + 1, min(nx, int(math.ceil((cx1 - x0) / (x1 - x0) * nx))))
    # Row 0 is the top of the raster (= y1); row index grows as y falls.
    row0 = max(0, min(ny - 1, int((1.0 - (cy1 - y0) / (y1 - y0)) * ny)))
    row1 = max(row0 + 1, min(ny, int(math.ceil((1.0 - (cy0 - y0) / (y1 - y0)) * ny))))
    sub = arr[row0:row1, col0:col1]
    if sub.size == 0 or not sub.any():
        return None
    return sub, (cx0, cx1, cy0, cy1)


# ── Viewport-bounded GDS query ───────────────────────────────────────────────
# For a cell too large to ever flatten in full (e.g. a reticle with tens of
# millions of placed instances), pull only the geometry that overlaps the
# currently-visible window — pruning subtrees (including huge regular-grid
# arrays, analytically) by bounding box, so the cost tracks what's on screen,
# not the size of the whole file.
_REP_EXPAND_CAP = 200_000   # irregular repetitions above this are skipped, not enumerated


def _affine_for_ref(ref):
    """2x2 matrix + translation mapping `ref.cell`'s local frame into the
    frame `ref` is placed in (its parent cell's frame)."""
    rot = ref.rotation or 0.0
    mag = ref.magnification or 1.0
    ca, sa = math.cos(rot), math.sin(rot)
    r = np.array([[ca, -sa], [sa, ca]])
    s = mag * (np.array([[1.0, 0.0], [0.0, -1.0]]) if ref.x_reflection else np.eye(2))
    return r @ s, np.array(ref.origin, dtype=float)


def _compose(m1, t1, m2, t2):
    """Compose two (matrix, translation) transforms: apply (m2,t2) first,
    then (m1,t1) — i.e. m2/t2 is the child (inner) transform."""
    return m1 @ m2, m1 @ t2 + t1


def _bbox_intersects(a, b):
    return not (a[1][0] < b[0][0] or a[0][0] > b[1][0] or a[1][1] < b[0][1] or a[0][1] > b[1][1])


def _transform_bbox(m, t, bbox):
    (x0, y0), (x1, y1) = bbox
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    world = corners @ m.T + t
    return ((float(world[:, 0].min()), float(world[:, 1].min())),
            (float(world[:, 0].max()), float(world[:, 1].max())))


def _repetition_vectors(rep, mr):
    """Step vectors (v1, v2) of a regular-grid repetition, in the same
    parent-local frame as the reference's own offsets. gdstk exposes
    `.v1`/`.v2` directly (already reflecting the ref's rotation/reflection)
    once that transform is non-trivial; for an identity transform it only
    exposes `.spacing` (the pre-transform step) — apply `mr` to that
    ourselves in that case. Returns None if not a plain regular grid
    (columns x rows, evenly spaced)."""
    if rep.columns is None or rep.rows is None:
        return None
    if rep.v1 is not None and rep.v2 is not None:
        return np.array(rep.v1), np.array(rep.v2)
    if rep.spacing is not None:
        sx, sy = rep.spacing
        return mr @ np.array([sx, 0.0]), mr @ np.array([0.0, sy])
    return None


def _regular_index_range(rep, mr, base_wb, target_local_bbox):
    """The small range of (i, j) array indices whose instance bbox
    (`base_wb` shifted by i*v1 + j*v2) can overlap `target_local_bbox` —
    O(1) regardless of `rep.size`, so a multi-million-instance array costs
    the same to prune as a small one. None if `rep` isn't a regular grid."""
    vecs = _repetition_vectors(rep, mr)
    if vecs is None:
        return None
    v1, v2 = vecs
    v = np.array([v1, v2]).T
    (bx0, by0), (bx1, by1) = base_wb
    (tx0, ty0), (tx1, ty1) = target_local_bbox
    # Minkowski displacement range for overlap: d in [target.min-base.max, target.max-base.min]
    dxlo, dxhi = tx0 - bx1, tx1 - bx0
    dylo, dyhi = ty0 - by1, ty1 - by0
    try:
        vinv = np.linalg.inv(v)
    except np.linalg.LinAlgError:
        return None
    corners = np.array([[dxlo, dylo], [dxhi, dylo], [dxlo, dyhi], [dxhi, dyhi]])
    ij = corners @ vinv.T
    i_lo = max(0, int(math.floor(ij[:, 0].min())) - 1)
    i_hi = min(rep.columns - 1, int(math.ceil(ij[:, 0].max())) + 1)
    j_lo = max(0, int(math.floor(ij[:, 1].min())) - 1)
    j_hi = min(rep.rows - 1, int(math.ceil(ij[:, 1].max())) + 1)
    if i_lo > i_hi or j_lo > j_hi:
        return range(0, 0), range(0, 0)
    return range(i_lo, i_hi + 1), range(j_lo, j_hi + 1)


# ── Shared traversal core ────────────────────────────────────────────────────
# Four walks over the same hierarchy live in this file — `_raster_accum`
# (coverage/vector accumulation), `_query_cell_bbox` (viewport-bounded exact
# query), `_window_layer_counts` (per-layer counts inside a window) and
# `_placeholder_boxes` (collapsed-subtree boxes). They differ entirely in what
# they do at a leaf, but each has to answer the same three questions about a
# reference first: where does it land in world space, is it repeated, and if so
# which of its instances can touch what we're looking at.
#
# Each used to answer them itself — four copies of the frame inversion and of
# the regular-grid / irregular / too-big-to-enumerate classification, including
# four chances to get `Repetition.spacing`'s pre-transform gotcha wrong. That
# divergence is what put a reference transform on the wrong side of
# `_tile_non_pure`. `_plan_reference` answers them once and returns a
# `_RefPlan`; the walkers keep their own policies and loops, but share this
# arithmetic.

_REF_SINGLE    = 'single'      # no repetition (or a repetition of one)
_REF_GRID      = 'grid'        # regular columns x rows, pruned analytically
_REF_OFFSETS   = 'offsets'     # irregular, small enough to enumerate
_REF_UNBOUNDED = 'unbounded'   # irregular and too large to enumerate


def _local_target_bbox(m, t, target):
    """`target` (a world bbox) expressed in the local frame of a cell placed at
    `(m, t)`."""
    try:
        minv = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        minv = np.eye(2)
    (x0, y0), (x1, y1) = target
    c = (np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) - t) @ minv.T
    return ((float(c[:, 0].min()), float(c[:, 1].min())),
            (float(c[:, 0].max()), float(c[:, 1].max())))


class _Frame:
    """Where a walker currently is: the transform `(m, t)` of the cell being
    walked and the world bbox it is looking at. One per cell visit.

    `local` — that same bbox pulled back into this cell's frame — is what
    prunes a regular grid analytically, and it is computed on FIRST USE rather
    than up front. Both matter: hoisting it out of the reference loop stops a
    7x7 die array of a 200-reference cell inverting `m` ~20k times, and
    deferring it stops every cell with no repeated child (the overwhelming
    majority, and `_query_cell_bbox` recurses per instance) paying for an
    inverse it never reads."""

    __slots__ = ('m', 't', 'world', '_local')

    def __init__(self, m, t, world):
        self.m, self.t, self.world = m, t, world
        self._local = None

    @property
    def local(self):
        if self._local is None:
            self._local = _local_target_bbox(self.m, self.t, self.world)
        return self._local


class _RefPlan:
    """One reference, normalised for traversal.

    `kind` is `_REF_SINGLE` / `_REF_GRID` / `_REF_OFFSETS` / `_REF_UNBOUNDED`.
    `whole_wb` is the world bbox of the *entire* placement (the array envelope
    for a repeated reference, not just its base instance) — what every walker
    prunes and draws placeholders against. `base_bb` is the child cell's own
    local bbox and `base_wb` that bbox in the parent's frame. For `_REF_GRID`,
    `irange`/`jrange` are the only array indices whose instance can touch the
    target, computed in O(1) regardless of `rep.size`.
    """

    __slots__ = ('cell', 'kind', 'rep', 'mr', 'tr', 'base_bb', 'base_wb',
                 'whole_wb', 'v1', 'v2', 'irange', 'jrange')

    def __init__(self, cell, kind, rep, mr, tr, base_bb, base_wb, whole_wb,
                 v1=None, v2=None, irange=None, jrange=None):
        self.cell     = cell
        self.kind     = kind
        self.rep      = rep
        self.mr       = mr
        self.tr       = tr
        self.base_bb  = base_bb
        self.base_wb  = base_wb
        self.whole_wb = whole_wb
        self.v1, self.v2         = v1, v2
        self.irange, self.jrange = irange, jrange

    @property
    def repeated(self) -> bool:
        return self.kind != _REF_SINGLE

    @property
    def n_visible(self) -> int:
        """Instances that can touch the target: the pruned `(i, j)` count for a
        grid, the whole repetition otherwise, 1 for a lone placement."""
        if self.kind == _REF_GRID:
            return len(self.irange) * len(self.jrange)
        if self.kind == _REF_SINGLE:
            return 1
        return self.rep.size

    def base_transform(self, m, t):
        """`(m2, t2)` placing the child's own frame — the base instance of a
        repeated reference, the only one of a single."""
        return _compose(m, t, self.mr, self.tr)

    def local_offsets(self):
        """`(K, 2)` parent-local translations, one per instance to visit, or
        None when the repetition is too large to enumerate."""
        if self.kind == _REF_SINGLE:
            return np.zeros((1, 2))
        if self.kind == _REF_GRID:
            ii, jj = self._ij()
            return np.column_stack([
                (ii * self.v1[0] + jj * self.v2[0]).ravel(),
                (ii * self.v1[1] + jj * self.v2[1]).ravel()])
        if self.kind == _REF_OFFSETS:
            return np.asarray(self.rep.get_offsets(), float)
        return None

    def world_offsets(self, m):
        """The same translations in world space, as `(ox, oy)`. A grid
        transforms its two step vectors and combines them, rather than
        transforming every offset."""
        if self.kind == _REF_GRID:
            V1, V2 = m @ self.v1, m @ self.v2
            ii, jj = self._ij()
            return ((ii * V1[0] + jj * V2[0]).ravel(),
                    (ii * V1[1] + jj * V2[1]).ravel())
        offs = self.local_offsets()
        if offs is None:
            return None
        d = offs @ m.T
        return d[:, 0], d[:, 1]

    def _ij(self):
        return np.meshgrid(np.fromiter(self.irange, np.float64),
                           np.fromiter(self.jrange, np.float64), indexing='ij')


def _plan_reference(ref, frame):
    """Normalise `ref` for a walker positioned at `frame`.

    Returns None when the reference provably cannot touch the target, so every
    caller's rejection test is this one test. Only the regular-grid branch
    touches `frame.local`, which is why that is computed lazily."""
    m, t = frame.m, frame.t
    mr, tr = _affine_for_ref(ref)
    base_bb = ref.cell.bounding_box()
    if base_bb is None:
        return None
    rep = ref.repetition

    if rep is None or not rep.size or rep.size <= 1:
        m2, t2 = _compose(m, t, mr, tr)
        whole_wb = _transform_bbox(m2, t2, base_bb)
        if not _bbox_intersects(whole_wb, frame.world):
            return None
        return _RefPlan(ref.cell, _REF_SINGLE, rep, mr, tr, base_bb,
                        _transform_bbox(mr, tr, base_bb), whole_wb)

    # Repeated: reject against the whole array envelope before doing any
    # per-instance work.
    whole_wb = _transform_bbox(m, t, ref.bounding_box())
    if not _bbox_intersects(whole_wb, frame.world):
        return None
    base_wb = _transform_bbox(mr, tr, base_bb)

    vecs = _repetition_vectors(rep, mr)
    if vecs is not None:
        idx = _regular_index_range(rep, mr, base_wb, frame.local)
        if idx is not None:
            return _RefPlan(ref.cell, _REF_GRID, rep, mr, tr, base_bb, base_wb,
                            whole_wb, v1=vecs[0], v2=vecs[1],
                            irange=idx[0], jrange=idx[1])
    kind = _REF_OFFSETS if rep.size <= _REP_EXPAND_CAP else _REF_UNBOUNDED
    return _RefPlan(ref.cell, kind, rep, mr, tr, base_bb, base_wb, whole_wb)


_MAX_ZOOM_OUT_STEP = 50.0   # cap on how far one drag-up gesture can pull back,
                            # so a stray 2-pixel box doesn't fling the view out


def _zoom_out_limits(xlim, ylim, box, home_xlim=None, home_ylim=None):
    """New axis limits for a drag-up gesture — the inverse of drag-down.

    Drag-down sets the view TO the box; drag-up shrinks the current view INTO
    the box. The factor is how many times the box fits in the view, and the new
    centre is placed so the data you can see right now lands where you drew the
    box — draw it to the right and the new space opens up on the left.

    Clamped to the home extent, so repeated drag-ups converge on the full view
    and stop there rather than drifting off into empty space; the Home button
    is still the one-shot reset."""
    (x0, x1), (y0, y1) = xlim, ylim
    w, h = x1 - x0, y1 - y0
    bw = max(box[2] - box[0], w * 1e-3)
    bh = max(box[3] - box[1], h * 1e-3)
    f  = min(max(max(w / bw, h / bh), 1.02), _MAX_ZOOM_OUT_STEP)

    nw, nh = w * f, h * f
    if home_xlim is not None and home_ylim is not None:
        if nw >= home_xlim[1] - home_xlim[0] or nh >= home_ylim[1] - home_ylim[0]:
            return tuple(home_xlim), tuple(home_ylim)

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ncx = cx - ((box[0] + box[2]) / 2.0 - cx) * f
    ncy = cy - ((box[1] + box[3]) / 2.0 - cy) * f
    return (ncx - nw / 2.0, ncx + nw / 2.0), (ncy - nh / 2.0, ncy + nh / 2.0)


_SUBTREE_INSTANCE_CAP = 5_000   # cap on how many elements of ONE regular array are
                                 # enumerated individually; above it the array gets a
                                 # placeholder box. Distinct from _subtree_too_dense,
                                 # which decides whether a subtree is worth drawing
                                 # at all — this is about the cost of walking the
                                 # surviving (i, j) candidates one by one

_PLACEHOLDER_EXPAND_MAX_CHILDREN = 32   # a collapsed container with more direct
                                        # child references than this is drawn as
                                        # one box; at or below it, one box per
                                        # child instead (so e.g. a container of 6
                                        # side-by-side arrays reads as 6 boxes,
                                        # not one rectangle spanning the gaps)
_PLACEHOLDER_EXPAND_MAX_DEPTH = 3       # nested-container levels to keep splitting
                                        # before falling back to a single box
_PLACEHOLDER_MAX_BOXES = 96             # soft cap (may overshoot by ~one child
                                        # subtree) on boxes emitted for one
                                        # collapsed subtree — a deep/bushy
                                        # hierarchy (low fan-out but many levels,
                                        # or fan-out near the child cap at every
                                        # level) stops splitting here and the
                                        # unfinished remainder gets one coarse box


def _bbox_area(bbox):
    (x0, y0), (x1, y1) = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


_SUBTREE_MIN_POLYS = 5_000      # a subtree at least this cheap is always drawn,
                                # however small it is on screen — otherwise an
                                # overview turns into a sea of boxes. Measured
                                # band on UtilOut.gds: below ~2.6k the 10µm
                                # SQR/BAR resolution patches (~2.5k polys each)
                                # stay boxed even zoomed in; at/above ~10.2k
                                # RESO2 draws at every zoom and the full-reticle
                                # view goes 0.17s -> 1.13s. 5k sits between.
_PREVIEW_PIXELS    = 700 * 700  # rough drawable area of the mask panel
_POLYS_PER_PIXEL   = 0.25       # draw real shapes only while each polygon gets
                                # ~4 pixels or more; below that a labelled box
                                # says the same thing faster and more honestly


def _screen_fraction(sub_bbox, view_bbox):
    """Fraction of the VIEWPORT that `sub_bbox` covers.

    The complement of `_bbox_overlap_fraction`, which measures how much of the
    SUBTREE is in view. One says how much work descending costs, the other how
    much of the screen there is to show it on; the gate needs both."""
    total = _bbox_area(view_bbox)
    if total <= 0.0:
        return 1.0
    (ax0, ay0), (ax1, ay1) = sub_bbox
    (bx0, by0), (bx1, by1) = view_bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) / total


def _subtree_too_dense(cell, sub_bbox, view_bbox, cost_cache, lpc_cache):
    """Would descending into this subtree cost more polygons than the view can
    actually show?

    Two terms, because either alone gets a common case wrong:

    * `visible` — the subtree's analytic flattened polygon count (no geometry
      touched, cached) scaled by how much of it is in view. This is what lets a
      multi-million-element array resolve once you zoom INSIDE it: the total
      count never shrinks, but the visible share does.
    * `allowance` — how many polygons are worth drawing into the screen area
      the subtree actually occupies. This is what keeps a wide view fast: a
      240µm cell in a 31mm view gets ~28 pixels, and 10k polygons in 28 pixels
      is a slower and less honest picture than one labelled box.

    Replaces an earlier gate on `_approx_instances` alone, which measured the
    wrong thing (instances, not cost) and could only be escaped by zooming
    until *less than half* the subtree was visible — so a 10k-polygon cell like
    `UtilOut`'s `RESO2` could be seen cropped, but never whole."""
    cost = cost_cache.get(cell.name)
    if cost is None:
        cost = sum(_layer_poly_counts(cell, lpc_cache).values())
        cost_cache[cell.name] = cost
    visible   = cost * _bbox_overlap_fraction(sub_bbox, view_bbox)
    allowance = max(_SUBTREE_MIN_POLYS,
                    _PREVIEW_PIXELS * _screen_fraction(sub_bbox, view_bbox)
                    * _POLYS_PER_PIXEL)
    return visible > allowance


def _bbox_overlap_fraction(subtree_bbox, viewport_bbox):
    """How much of `subtree_bbox` the current `viewport_bbox` still covers —
    1.0 while zoomed out to (or past) its full extent, falling toward 0 as
    the user zooms in on just a piece of it."""
    (ax0, ay0), (ax1, ay1) = subtree_bbox
    (bx0, by0), (bx1, by1) = viewport_bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    total = _bbox_area(subtree_bbox)
    if total <= 0.0:
        return 1.0
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) / total


def _bbox_clip(a, b):
    """`a` clipped to `b`, or None when they don't overlap."""
    c = ((max(a[0][0], b[0][0]), max(a[0][1], b[0][1])),
         (min(a[1][0], b[1][0]), min(a[1][1], b[1][1])))
    return c if c[0][0] < c[1][0] and c[0][1] < c[1][1] else None


def _placeholder_boxes(cell, m, t, bbox, _depth=0, _acc=None, _lc=None):
    """Stand-in bounding boxes for a subtree that's too expensive to flatten.

    A pure *container* cell (no geometry of its own, few child references) is
    broken into one box per child — each child's own world bbox (a repeated
    child keeps its whole array envelope as a single box), clipped to `bbox` —
    recursing through nested containers up to `_PLACEHOLDER_EXPAND_MAX_DEPTH`.
    So a container holding 6 side-by-side arrays yields 6 boxes over the arrays
    rather than one rectangle that also covers the gaps between them.

    Splitting stops at the first of: a cell with its own polygons/paths or no
    children (its own box); more than `_PLACEHOLDER_EXPAND_MAX_CHILDREN` direct
    children (its own box — too bushy to be worth splitting); depth
    `_PLACEHOLDER_EXPAND_MAX_DEPTH` (its own box — deep enough); or
    `_PLACEHOLDER_MAX_BOXES` boxes already emitted for this subtree (the
    unfinished remainder collapses to one coarse box). So a fractal-ish tree —
    low fan-out but many levels — terminates on depth, and a tree that stays
    just under the child cap at every level terminates on the box budget;
    either way the result is bounded. Boxes that miss `bbox` are dropped; a
    node that expands to nothing still emits its own box."""
    root = _acc is None
    if root:
        _acc, _lc = [], {}
    bb = cell.bounding_box()
    if bb is None:
        return _acc if root else None
    fb = _bbox_clip(_transform_bbox(m, t, bb), bbox)

    if (_depth >= _PLACEHOLDER_EXPAND_MAX_DEPTH
            or len(_acc) >= _PLACEHOLDER_MAX_BOXES
            or cell.polygons or cell.paths
            or not cell.references
            or len(cell.references) > _PLACEHOLDER_EXPAND_MAX_CHILDREN):
        if fb:
            _acc.append((fb, cell.name, _leaf_layers(cell, _lc)))
        return _acc if root else None

    produced = False
    hit_cap = False
    frame = _Frame(m, t, bbox)
    for ref in cell.references:
        if len(_acc) >= _PLACEHOLDER_MAX_BOXES:
            hit_cap = True
            break
        plan = _plan_reference(ref, frame)
        if plan is None:
            continue
        if plan.repeated:
            # A repeated child: keep the whole array envelope as one box —
            # never split an array into per-element boxes.
            cb = _bbox_clip(plan.whole_wb, bbox)
            if cb:
                _acc.append((cb, ref.cell.name, _leaf_layers(ref.cell, _lc)))
                produced = True
        else:
            m2, t2 = plan.base_transform(m, t)
            before = len(_acc)
            _placeholder_boxes(ref.cell, m2, t2, bbox, _depth + 1, _acc, _lc)
            produced = produced or len(_acc) > before
            if len(_acc) >= _PLACEHOLDER_MAX_BOXES:
                hit_cap = True
                break

    # Nothing came back (all children clipped out), or we ran out of box
    # budget partway: fall back to one coarse box covering this whole node.
    if (not produced or hit_cap) and fb:
        _acc.append((fb, cell.name, _leaf_layers(cell, _lc)))
    return _acc if root else None


def _add_skip(stats, bbox, cell):
    """Record a subtree left as a placeholder box, tagged with the layers it
    draws on so the renderer can tint it in the layer colour when that layer
    is enabled (rather than a neutral grey hatch)."""
    stats['skipped'].append(
        (bbox, cell.name,
         _leaf_layers(cell, stats.setdefault('_leaf_cache', {}))))


def _query_cell_bbox(cell, bbox, budget=300_000, m=None, t=None, out=None, stats=None,
                      exact=False):
    """Collect (world-space) polygons of `cell` that intersect `bbox`,
    pruning by bounding box at every level instead of flattening the whole
    tree. A subtree that's individually too expensive to flatten (more than
    polygons than the view can show — see `_subtree_too_dense` — or an
    irregular repetition too big to enumerate) is skipped *on its own* — recorded in `stats['skipped']` as
    `(world_bbox, cell_name)` — while every cheaper sibling elsewhere in the
    tree still gets fully resolved. The global `budget` is only a backstop
    against many individually-cheap subtrees adding up; hitting it skips
    whatever's left in the same way, rather than aborting outright.

    `exact=True` is for real mask geometry, not preview: every density/budget
    shortcut that exists only to keep the screen redraw fast is disabled, so a
    subtree is fully resolved regardless of cost. The only thing that can
    still land in `stats['skipped']` is a repetition that is structurally
    impossible to enumerate (`_REF_UNBOUNDED` — irregular and unbounded) —
    a real "can't do this", not a speed trade-off — so callers in exact mode
    should treat any non-empty `skipped` as a hard error, not a partial result.

    Whole shapes are kept as soon as their bounding box touches `bbox` — there
    is no clip at the window edge. That is deliberate: a bbox test is one
    comparison per shape, clipping is a boolean op per shape, and the former
    is what this function already does for preview, so exact mode reuses it
    unchanged rather than paying for precision nothing asked for.

    Returns `(by_layer: dict[layer -> list[ndarray]], stats)`; `incomplete`
    is set whenever anything was skipped — the result is a valid, honest
    partial sample, not wrong, just not exhaustive."""
    if m is None:
        m, t = np.eye(2), np.zeros(2)
        out, stats = {}, dict(visited=0, incomplete=False, leaf_polys=0,
                              bailed_refs=0, skipped=[],
                              _cost_cache={}, _lpc_cache={})
    if exact:
        budget = float('inf')   # no display-speed backstop for real mask geometry

    for p in cell.polygons:
        pts = p.points
        mn, mx = pts.min(0), pts.max(0)
        if not _bbox_intersects(_transform_bbox(m, t, ((mn[0], mn[1]), (mx[0], mx[1]))), bbox):
            continue
        out.setdefault(p.layer, []).append(pts @ m.T + t)
        stats.setdefault('layer_dtype', {})[p.layer] = p.datatype
        stats['leaf_polys'] += 1
        if stats['leaf_polys'] >= budget:
            stats['incomplete'] = True
            break   # this cell's own remaining shapes are dropped, but its
                     # children (below) still get a chance — as placeholders

    # Same treatment for this cell's paths (gdstk keeps them separate from
    # `cell.polygons`, and they were previously invisible to this query).
    for lyr, pts_list in _cell_path_polys(
            cell, stats.setdefault('_path_cache', {})).items():
        if stats['leaf_polys'] >= budget:
            stats['incomplete'] = True
            break
        for pts in pts_list:
            mn, mx = pts.min(0), pts.max(0)
            if not _bbox_intersects(
                    _transform_bbox(m, t, ((mn[0], mn[1]), (mx[0], mx[1]))), bbox):
                continue
            out.setdefault(lyr, []).append(pts @ m.T + t)
            stats['leaf_polys'] += 1
            if stats['leaf_polys'] >= budget:
                stats['incomplete'] = True
                break

    frame = _Frame(m, t, bbox)

    for ref in cell.references:
        stats['visited'] += 1
        plan = _plan_reference(ref, frame)
        if plan is None:
            continue

        # Draw this subtree's real geometry when the view can actually show it,
        # a labelled box when it can't. Zooming in always eventually reaches
        # detail, because both terms move the right way as the view shrinks.
        too_dense = False if exact else _subtree_too_dense(
            ref.cell, plan.whole_wb, bbox, stats['_cost_cache'], stats['_lpc_cache'])

        if plan.kind == _REF_SINGLE:
            m2, t2 = plan.base_transform(m, t)
            if stats['leaf_polys'] >= budget or too_dense:
                stats['incomplete'] = True
                boxes = _placeholder_boxes(ref.cell, m2, t2, bbox)
                stats['skipped'].extend(boxes or [
                    (plan.whole_wb, ref.cell.name,
                     _leaf_layers(ref.cell, stats.setdefault('_leaf_cache', {})))])
                continue
            _query_cell_bbox(ref.cell, bbox, budget, m2, t2, out, stats, exact)
            continue

        if stats['leaf_polys'] >= budget or too_dense:
            stats['incomplete'] = True
            _add_skip(stats, plan.whole_wb, ref.cell)
            continue

        if plan.kind == _REF_UNBOUNDED:
            # Irregular and too large to enumerate exactly — structurally
            # impossible, not a speed trade-off, so this is the one case
            # exact mode still records as skipped; callers must treat that
            # as a hard failure rather than a partial mask.
            stats['incomplete'] = True
            stats['bailed_refs'] += 1
            _add_skip(stats, plan.whole_wb, ref.cell)
            continue

        if not exact and plan.kind == _REF_GRID and plan.n_visible > _SUBTREE_INSTANCE_CAP:
            # Even after pruning, this many array elements survive in the
            # current view — still too fine-grained to enumerate one by one.
            # Record the array's *true* world bbox (the renderer clips it to
            # the axes): zoomed inside the array it then fills the whole view,
            # and it still stops exactly at the array's edge.
            stats['incomplete'] = True
            _add_skip(stats, plan.whole_wb, ref.cell)
            continue

        # Few enough instances to visit individually. The two repetition kinds
        # prune in different frames on purpose: a grid's `(i, j)` rectangle is
        # already expressed in the parent's frame, so its instances are tested
        # against `target_local` there; an irregular repetition transforms all
        # its offsets to world in one matmul and tests against `bbox`. Testing
        # a grid in world space (or offsets in local) would mean a 4-corner
        # transform per instance for no gain.
        budget_hit = False
        mr, tr = plan.mr, plan.tr

        if plan.kind == _REF_GRID:
            # Kept as an integer double loop over the pruned (i, j) rectangle
            # with the instance bbox built inline: this is the hot path when a
            # zoom lands inside a large array, and iterating a materialised
            # (K, 2) offsets array here instead cost ~35%.
            v1, v2 = plan.v1, plan.v2
            (bx0, by0), (bx1, by1) = plan.base_wb
            local_target = frame.local
            for i in plan.irange:
                for j in plan.jrange:
                    offset = i * v1 + j * v2
                    ox, oy = offset[0], offset[1]
                    if not _bbox_intersects(((bx0 + ox, by0 + oy),
                                             (bx1 + ox, by1 + oy)), local_target):
                        continue
                    m2, t2 = m @ mr, t + m @ (tr + offset)
                    if stats['leaf_polys'] >= budget:
                        stats['incomplete'] = True
                        if not budget_hit:
                            budget_hit = True
                            _add_skip(stats, _transform_bbox(m2, t2, plan.base_bb),
                                      ref.cell)
                        continue
                    _query_cell_bbox(ref.cell, bbox, budget, m2, t2, out, stats, exact)
        else:
            offsets    = plan.local_offsets()
            base_world = _transform_bbox(m, t, plan.base_wb)
            dxs, dys   = plan.world_offsets(m)
            for k in range(len(offsets)):
                dx, dy = dxs[k], dys[k]
                inst_wb = ((base_world[0][0] + dx, base_world[0][1] + dy),
                           (base_world[1][0] + dx, base_world[1][1] + dy))
                if not _bbox_intersects(inst_wb, bbox):
                    continue
                if stats['leaf_polys'] >= budget:
                    stats['incomplete'] = True
                    if not budget_hit:
                        budget_hit = True
                        _add_skip(stats, inst_wb, ref.cell)
                    continue
                m2, t2 = m @ mr, t + m @ (tr + offsets[k])
                _query_cell_bbox(ref.cell, bbox, budget, m2, t2, out, stats, exact)
    return out, stats


# ── Entity dialogs ─────────────────────────────────────────────────────────────
class MaskDialog(FormDialog):
    def __init__(self, d: dict | None = None, parent=None):
        super().__init__('Add Mask' if d is None else 'Edit Mask', parent)
        d = d or {}
        self._layer_checks: dict[int, QCheckBox] = {}
        self._line('id',       'Mask ID',  d.get('id', ''))
        self._line('bar_code', 'Bar Code', d.get('bar_code', ''))
        # GDS file row with browser button
        wrap = QWidget(); row = QHBoxLayout(wrap); row.setContentsMargins(0, 0, 0, 0)
        self._gds = QLineEdit(d.get('gds_file', ''))
        btn = QPushButton('Browse…'); btn.setFixedWidth(70)
        btn.clicked.connect(self._browse_gds)
        row.addWidget(self._gds); row.addWidget(btn)
        self._form.addWidget(QLabel('GDS File:'), self._row, 0)
        self._form.addWidget(wrap, self._row, 1); self._row += 1
        # Cell name: editable combo, auto-populated from GDS file
        self._cell_combo = QComboBox()
        self._cell_combo.setEditable(True)
        self._widgets['cell_name'] = self._cell_combo
        self._form.addWidget(QLabel('Cell Name:'), self._row, 0)
        self._form.addWidget(self._cell_combo, self._row, 1)
        self._row += 1
        existing = d.get('cell_name', '')
        if existing:
            self._cell_combo.addItem(existing)
            self._cell_combo.setCurrentText(existing)

        # ── Layer selection group ──────────────────────────────────────────────
        layer_box = QGroupBox('Layer Selection')
        lv = QVBoxLayout(layer_box)
        lv.setContentsMargins(6, 4, 6, 4)
        btn_row = QHBoxLayout()
        self._all_on_btn  = QPushButton('All On')
        self._all_off_btn = QPushButton('All Off')
        self._all_on_btn.setFixedWidth(72)
        self._all_off_btn.setFixedWidth(72)
        self._all_on_btn.clicked.connect(self._all_on)
        self._all_off_btn.clicked.connect(self._all_off)
        self._layer_hint = QLabel('(load a GDS file and select a cell to see layers)')
        self._layer_hint.setStyleSheet('color: #888888; font-size: 10px;')
        btn_row.addWidget(self._all_on_btn)
        btn_row.addWidget(self._all_off_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(self._layer_hint)
        btn_row.addStretch()
        lv.addLayout(btn_row)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFixedHeight(130)
        self._layer_container = QWidget()
        self._layer_layout = QVBoxLayout(self._layer_container)
        self._layer_layout.setContentsMargins(4, 2, 4, 2)
        self._layer_layout.setSpacing(2)
        self._layer_layout.addStretch()
        self._scroll.setWidget(self._layer_container)
        lv.addWidget(self._scroll)
        # Insert before the button box (last item in _outer)
        self._outer.insertWidget(self._outer.count() - 1, layer_box)

        # ── Area window group (for cells too large to flatten whole) ──────────
        win_box = QGroupBox('Limit to Area (cell-local coords)')
        wv = QVBoxLayout(win_box)
        wv.setContentsMargins(6, 4, 6, 4)
        self._win_enable = QCheckBox('Only use geometry inside this window — lets a '
                                      'huge cell (e.g. a full reticle) be used as a '
                                      'mask by pulling just a sub-area through its '
                                      'hierarchy instead of flattening everything.')
        self._win_enable.setWordWrap(True)
        wv.addWidget(self._win_enable)
        def _spin() -> QDoubleSpinBox:
            w = QDoubleSpinBox(); w.setRange(-1e6, 1e6); w.setDecimals(4)
            return w
        grid = QGridLayout()
        self._win_x0 = _spin()
        self._win_y0 = _spin()
        self._win_x1 = _spin()
        self._win_y1 = _spin()
        grid.addWidget(QLabel('x0:'), 0, 0); grid.addWidget(self._win_x0, 0, 1)
        grid.addWidget(QLabel('y0:'), 0, 2); grid.addWidget(self._win_y0, 0, 3)
        grid.addWidget(QLabel('x1:'), 1, 0); grid.addWidget(self._win_x1, 1, 1)
        grid.addWidget(QLabel('y1:'), 1, 2); grid.addWidget(self._win_y1, 1, 3)
        wv.addLayout(grid)
        note = QLabel('Whole shapes whose bounding box touches the window are kept '
                       '(not clipped at the edge) — faster, and geometry can extend '
                       'slightly past the boundary you set.')
        note.setWordWrap(True)
        note.setStyleSheet('color: #888888; font-size: 10px;')
        wv.addWidget(note)
        self._outer.insertWidget(self._outer.count() - 1, win_box)
        win = d.get('window')
        self._win_enable.setChecked(win is not None)
        if win is not None:
            self._win_x0.setValue(win[0]); self._win_y0.setValue(win[1])
            self._win_x1.setValue(win[2]); self._win_y1.setValue(win[3])
        for w in (self._win_x0, self._win_y0, self._win_x1, self._win_y1):
            w.setEnabled(self._win_enable.isChecked())
        self._win_enable.toggled.connect(
            lambda on: [w.setEnabled(on) for w in
                        (self._win_x0, self._win_y0, self._win_x1, self._win_y1)])

        # Populate after UI is built
        if d.get('gds_file'):
            self._populate_cells(d['gds_file'], existing)
            self._populate_layers(d['gds_file'], existing or self._cell_combo.currentText(),
                                  d.get('sel_layers'))
        self._gds.textChanged.connect(self._on_gds_changed)
        self._cell_combo.currentTextChanged.connect(self._on_cell_changed)

    def _browse_gds(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select GDS File', '', 'GDS Files (*.gds *.GDS)')
        if path:
            self._gds.setText(path)

    def _on_gds_changed(self, path: str):
        path = path.strip()
        self._populate_cells(path, '')
        self._populate_layers(path, self._cell_combo.currentText(), None)

    def _on_cell_changed(self, cell_name: str):
        self._populate_layers(self._gds.text().strip(), cell_name, None)

    def _populate_cells(self, gds_path: str, select: str):
        self._cell_combo.blockSignals(True)
        self._cell_combo.clear()
        if not gds_path or not GDSTK_OK:
            self._cell_combo.blockSignals(False)
            return
        try:
            glib  = read_gds(gds_path)
            names = sorted(c.name for c in glib.cells)
            self._cell_combo.addItems(names)
            if select and select in names:
                self._cell_combo.setCurrentText(select)
            elif names:
                self._cell_combo.setCurrentIndex(0)
        except Exception:
            pass
        self._cell_combo.blockSignals(False)

    def _populate_layers(self, gds_path: str, cell_name: str, sel_layers):
        # Clear existing checkboxes (keep the trailing stretch)
        while self._layer_layout.count() > 1:
            item = self._layer_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._layer_checks.clear()

        if not gds_path or not cell_name or not GDSTK_OK:
            self._layer_hint.setText('(load a GDS file and select a cell to see layers)')
            return
        try:
            glib   = read_gds(gds_path)
            cname  = cell_name or (glib.cells[0].name if glib.cells else '')
            if not cname:
                return
            cell   = glib[cname]
            layers = _hierarchy_layers(cell)
            if not layers:
                self._layer_hint.setText('(no polygon layers found in cell)')
                return
            sel_set = set(sel_layers) if sel_layers is not None else None
            self._layer_hint.setText(f'{len(layers)} layer(s) found')
            for lyr in layers:
                checked = (sel_set is None or lyr in sel_set)
                cb = QCheckBox(f'Layer {lyr}')
                cb.setChecked(checked)
                self._layer_layout.insertWidget(self._layer_layout.count() - 1, cb)
                self._layer_checks[lyr] = cb
        except Exception:
            self._layer_hint.setText('(could not read layers)')

    def _all_on(self):
        for cb in self._layer_checks.values():
            cb.setChecked(True)

    def _all_off(self):
        for cb in self._layer_checks.values():
            cb.setChecked(False)

    def result(self) -> dict:
        sel = None
        if self._layer_checks:
            sel = [lyr for lyr, cb in sorted(self._layer_checks.items()) if cb.isChecked()]
        window = None
        if self._win_enable.isChecked():
            window = (self._win_x0.value(), self._win_y0.value(),
                      self._win_x1.value(), self._win_y1.value())
        return dict(id=self.val('id'), bar_code=self.val('bar_code'),
                    gds_file=self._gds.text().strip(), cell_name=self.val('cell_name'),
                    sel_layers=sel, window=window)


class ImageDialog(FormDialog):
    def __init__(self, d: dict | None = None, mask_ids: list[str] | None = None,
                 parent=None, title: str | None = None):
        super().__init__(title or ('Add Image' if d is None else 'Edit Image'), parent)
        d = d or {}
        self._line('id', 'Image ID', d.get('id', ''))
        self._combo('mask_id', 'Mask', mask_ids or [], d.get('mask_id', ''))
        self._dspin_xy('size',  'Size (mm)',   d.get('size_x', 10.0),  d.get('size_y', 10.0),  0.001, 200.0)
        self._dspin_xy('shift', 'Shift (mm)',  d.get('shift_x', 0.0),  d.get('shift_y', 0.0),  -100.0, 100.0)
        default_ox = d.get('origin_x', d.get('shift_x', 0.0))
        default_oy = d.get('origin_y', d.get('shift_y', 0.0))
        self._dspin_xy('origin', 'Origin (mm)', default_ox, default_oy, -100.0, 100.0)
        # Origin tracks shift while they remain equal; user can break the link by editing origin
        self._origin_tracking = (default_ox == d.get('shift_x', 0.0) and
                                  default_oy == d.get('shift_y', 0.0))
        self._widgets['shift_x'].valueChanged.connect(self._on_shift_changed)
        self._widgets['shift_y'].valueChanged.connect(self._on_shift_changed)
        self._widgets['origin_x'].valueChanged.connect(self._on_origin_changed)
        self._widgets['origin_y'].valueChanged.connect(self._on_origin_changed)

    def _on_shift_changed(self, _=None):
        if self._origin_tracking:
            sx = self._widgets['shift_x'].value()
            sy = self._widgets['shift_y'].value()
            for key, val in (('origin_x', sx), ('origin_y', sy)):
                w = self._widgets[key]
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)

    def _on_origin_changed(self, _=None):
        if self._widgets['origin_x'].value() != self._widgets['shift_x'].value() or \
                self._widgets['origin_y'].value() != self._widgets['shift_y'].value():
            self._origin_tracking = False

    def result(self) -> dict:
        return dict(id=self.val('id'), mask_id=self.val('mask_id'),
                    size_x=self.val('size_x'), size_y=self.val('size_y'),
                    shift_x=self.val('shift_x'), shift_y=self.val('shift_y'),
                    origin_x=self.val('origin_x'), origin_y=self.val('origin_y'))


class LayerDialog(FormDialog):
    def __init__(self, d: dict | None = None, parent=None, title: str | None = None):
        super().__init__(title or ('Add Layer' if d is None else 'Edit Layer'), parent)
        d = d or {}
        self._line ('id',       'Layer ID',          d.get('id', ''))
        self._dspin('rotation', 'Rotation (°)',       d.get('rotation', 0.0), -360.0, 360.0, 1)
        self._dspin('na',       'NA',                 d.get('na', 0.57), 0.0, 1.0, 3)
        self._dspin('sigma_in', 'σ inner (0 = none)', d.get('sigma_in', 0.0), 0.0, 1.0, 3)
        self._dspin('sigma_out','σ outer',            d.get('sigma_out', 0.5), 0.0, 1.0, 3)
        self._combo('illume',   'Illumination',       ILLUME_LABELS, d.get('illume', 'Conventional'))

    def result(self) -> dict:
        return dict(id=self.val('id'), rotation=self.val('rotation'), na=self.val('na'),
                    sigma_in=self.val('sigma_in'), sigma_out=self.val('sigma_out'),
                    illume=self.val('illume'))


class MarkDialog(FormDialog):
    def __init__(self, d: dict | None = None, image_ids: list[str] | None = None, parent=None):
        super().__init__('Add Mark' if d is None else 'Edit Mark', parent)
        d = d or {}
        imgs = [PM_GENERATED] + (image_ids or [])
        self._line ('id',       'Mark ID', d.get('id', ''))
        self._combo('image_id', 'Image',   imgs, d.get('image_id', PM_GENERATED))
        self._dspin('x', 'X (mm)', d.get('x', 0.0), -200.0, 200.0)
        self._dspin('y', 'Y (mm)', d.get('y', 0.0), -200.0, 200.0)

    def result(self) -> dict:
        return dict(id=self.val('id'), image_id=self.val('image_id'),
                    x=self.val('x'), y=self.val('y'))


class StrategyDialog(QDialog):
    def __init__(self, d: dict | None = None, mark_ids: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Strategy' if d is None else 'Edit Strategy')
        d = d or {}; mark_ids = mark_ids or []
        layout = QVBoxLayout(self)

        form = QGridLayout()
        self._name = QLineEdit(d.get('name', ''))
        form.addWidget(QLabel('Strategy ID:'), 0, 0); form.addWidget(self._name, 0, 1)
        self._req = QSpinBox(); self._req.setRange(0, 20); self._req.setValue(d.get('required', 4))
        form.addWidget(QLabel('Required marks:'), 1, 0); form.addWidget(self._req, 1, 1)
        layout.addLayout(form)

        layout.addWidget(QLabel('Select marks (Ctrl+click for multiple):'))
        self._mlist = QListWidget()
        self._mlist.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        sel = set(d.get('mark_ids', []))
        for mid in mark_ids:
            item = QListWidgetItem(mid); self._mlist.addItem(item)
            if mid in sel: item.setSelected(True)
        layout.addWidget(self._mlist)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def result(self) -> dict:
        return dict(name=self._name.text().strip(), required=self._req.value(),
                    mark_ids=[i.text() for i in self._mlist.selectedItems()])


class ExposureDialog(FormDialog):
    def __init__(self, d: dict | None = None,
                 layer_ids: list[str] | None = None,
                 image_ids: list[str] | None = None, parent=None):
        super().__init__('Add Exposure' if d is None else 'Edit Exposure', parent)
        d = d or {}
        self._combo('layer_id', 'Layer',          layer_ids or [], d.get('layer_id', ''))
        self._combo('image_id', 'Image',          image_ids or [], d.get('image_id', ''))
        self._dspin('dose',     'Dose (mJ/cm²)',  d.get('dose', 20.0),  0.0,  9999.0, 2)
        self._dspin('focus',    'Focus (µm)',     d.get('focus', 0.0), -99.0,   99.0, 3)

    def result(self) -> dict:
        return dict(layer_id=self.val('layer_id'), image_id=self.val('image_id'),
                    dose=self.val('dose'), focus=self.val('focus'))


class _BindDialog(FormDialog):
    def __init__(self, layer_ids: list[str], align_ids: list[str],
                 current: tuple | None = None, parent=None):
        super().__init__('Layer–Strategy Binding', parent)
        c = current or ('', '', 'A')
        self._combo('layer',  'Layer',    layer_ids, c[0])
        self._combo('strat',  'Strategy', align_ids, c[1])
        usages = ['A – Active', 'N – None']
        self._combo('usage', 'Usage', usages,
                    'A – Active' if c[2] == 'A' else 'N – None')

    def result(self) -> tuple:
        return (self.val('layer'), self.val('strat'), self.val('usage').split()[0])


# ── ListPanel ──────────────────────────────────────────────────────────────────
class ListPanel(QWidget):
    """A QTableWidget with Add / Edit / [Copy] / Delete buttons below it."""

    def __init__(self, columns: list[str], parent=None, with_copy: bool = False):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table)
        h = QHBoxLayout()
        self.add_btn  = QPushButton('Add')
        self.edit_btn = QPushButton('Edit')
        self.copy_btn = QPushButton('Copy') if with_copy else None
        self.del_btn  = QPushButton('Delete')
        buttons = (self.add_btn, self.edit_btn, self.copy_btn, self.del_btn)
        for b in buttons:
            if b is not None:
                h.addWidget(b)
        h.addStretch(); v.addLayout(h)

    def set_rows(self, rows: list[list[str]]):
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)

    def current_row(self) -> int:
        return self.table.currentRow()


# ── WaferPreview ───────────────────────────────────────────────────────────────
class WaferPreview(QWidget):
    def __init__(self, parent=None, figsize=(3.5, 3.5)):
        super().__init__(parent)
        self.setMinimumSize(280, 300)
        layout = QVBoxLayout(self); layout.setContentsMargins(4, 4, 4, 4)
        if MATPLOTLIB_OK:
            self._fig    = Figure(figsize=figsize, dpi=90)
            self._ax     = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._fig.patch.set_facecolor('#1e1e1e')
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(QLabel('Install matplotlib for wafer preview'))

    def update_preview(self, state: 'AppState'):
        if not MATPLOTLIB_OK:
            return
        ax = self._ax
        ax.cla()
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        ax.set_facecolor('#1e1e1e')

        w     = state.cell.get('wafer_w', FOUR_INCH)
        r     = WAFER_SIZES[w] / 2
        flat_y     = -FLAT[w]
        flat_half  = math.sqrt(max(0.0, r * r - flat_y * flat_y))
        theta1 = math.atan2(flat_y, flat_half)
        theta2 = math.atan2(flat_y, -flat_half)
        if theta2 <= theta1:
            theta2 += 2 * math.pi
        thetas = np.linspace(theta1, theta2, 300)
        ax.fill(r * np.cos(thetas), r * np.sin(thetas),
                color='#3c3c3c', edgecolor='#888888', linewidth=1.5, zorder=1)

        for ci, img_d in enumerate(state.images):
            pts = state.patterns.get(img_d['id'], [])
            if not pts:
                continue
            color = _PREVIEW_COLORS[ci % len(_PREVIEW_COLORS)]
            sx  = img_d.get('size_x', 10.0)
            sy  = img_d.get('size_y', 10.0)
            shx = img_d.get('shift_x', 0.0)
            shy = img_d.get('shift_y', 0.0)
            for pt in pts:
                cx = pt['x'] + shx; cy = pt['y'] + shy
                ax.add_patch(MplRect(
                    (cx - sx / 2, cy - sy / 2), sx, sy,
                    linewidth=0.8, edgecolor=color,
                    facecolor=color, alpha=0.3, zorder=3,
                ))

        arm = r * 0.05
        for mk in state.marks:
            mx, my = mk.get('x', 0.0), mk.get('y', 0.0)
            ax.plot([mx - arm, mx + arm], [my,      my     ], color='#ffff44', lw=1.2, zorder=5)
            ax.plot([mx,       mx      ], [my - arm, my + arm], color='#ffff44', lw=1.2, zorder=5)

        ax.set_xlim(-r * 1.05, r * 1.05)
        ax.set_ylim(-r * 1.05, r * 1.05)
        ax.set_title(f'{WAFER_SIZES[w]:.0f} mm', color='#aaaaaa', fontsize=8, pad=2)
        self._fig.tight_layout(pad=0.2)
        self._canvas.draw()


# ── CellTab ────────────────────────────────────────────────────────────────────
class CellTab(QWidget):
    def __init__(self, state: AppState, on_change, parent=None):
        super().__init__(parent)
        self._state     = state
        self._on_change = on_change
        self._block     = False
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        grid = QGridLayout(self)
        grid.setColumnStretch(1, 1)
        row = [0]

        def add(label: str, widget: QWidget) -> QWidget:
            grid.addWidget(QLabel(label), row[0], 0)
            grid.addWidget(widget, row[0], 1)
            row[0] += 1
            return widget

        self._wafer = add('Wafer size', QComboBox())
        self._wafer.addItems(WAFER_LABELS)

        # Cell size X and Y on a single row
        self._cell_x = QDoubleSpinBox()
        self._cell_y = QDoubleSpinBox()
        _cw = QWidget(); _cr = QHBoxLayout(_cw); _cr.setContentsMargins(0, 0, 0, 0)
        _cr.addWidget(QLabel('X:')); _cr.addWidget(self._cell_x)
        _cr.addSpacing(8)
        _cr.addWidget(QLabel('Y:')); _cr.addWidget(self._cell_y)
        _cr.addStretch()
        add('Cell size (mm)', _cw)

        self._round_edge = add('Round edge clearance (mm)',   QDoubleSpinBox())
        self._flat_edge  = add('Flat edge clearance (mm)',    QDoubleSpinBox())
        self._edge_excl  = add('Edge exclusion (mm)',         QDoubleSpinBox())

        # Dies per cell X/Y on a single row, same layout as Cell size above
        self._dies_x = QSpinBox()
        self._dies_y = QSpinBox()
        _dw = QWidget(); _dr = QHBoxLayout(_dw); _dr.setContentsMargins(0, 0, 0, 0)
        _dr.addWidget(QLabel('X:')); _dr.addWidget(self._dies_x)
        _dr.addSpacing(8)
        _dr.addWidget(QLabel('Y:')); _dr.addWidget(self._dies_y)
        _dr.addStretch()
        add('Dies per cell', _dw)
        self._min_dies = add('Minimum dies per cell', QSpinBox())
        self._min_dies.setToolTip(
            "How many of a field's dies_x × dies_y sub-die quadrant centers "
            "must fall inside the edge exclusion for the whole field/cell to still "
            "be exposed. 0 = expose if the field merely intersects the wafer at "
            "all (loosest). One shared setting so every pattern (grid/python/csv) "
            "uses the same wafer coverage rule.")

        self._cover      = add('Cover mode',                  QComboBox())
        self._cover.addItems(COVER_MODES)
        self._place      = add('Placement mode',              QComboBox())
        self._place.addItems(PLACE_MODES)
        self._prealign   = add('Prealign method',             QComboBox())
        self._prealign.addItems(PREALIGN_METHODS)
        self._outname    = add('Output file name',            QLineEdit())
        self._pm_reticle = add('PM reticle ID',               QLineEdit())
        self._pm_reticle.setPlaceholderText('4544020*')
        grid.setRowStretch(row[0], 1)

        _NoBtn = QAbstractSpinBox.ButtonSymbols.NoButtons
        for spin in (self._cell_x, self._cell_y, self._round_edge,
                     self._flat_edge, self._edge_excl):
            spin.setRange(0.0, 200.0)
            spin.setDecimals(3)
            spin.setButtonSymbols(_NoBtn)
        for spin in (self._dies_x, self._dies_y):
            spin.setRange(1, 99)
            spin.setButtonSymbols(_NoBtn)
        self._min_dies.setRange(0, 99 * 99)
        self._min_dies.setButtonSymbols(_NoBtn)

        self._wafer.currentIndexChanged.connect(self._sync)
        for w in (self._cell_x, self._cell_y, self._round_edge,
                  self._flat_edge, self._edge_excl):
            w.valueChanged.connect(self._sync)
        for w in (self._dies_x, self._dies_y, self._min_dies):
            w.valueChanged.connect(self._sync)
        for w in (self._cover, self._place, self._prealign):
            w.currentIndexChanged.connect(self._sync)
        self._outname.textChanged.connect(self._sync)
        self._pm_reticle.textChanged.connect(self._sync)

    def _sync(self, _=None):
        if self._block:
            return
        c = self._state.cell
        c['wafer_w']           = self._wafer.currentIndex()
        c['cell_x']            = self._cell_x.value()
        c['cell_y']            = self._cell_y.value()
        c['round_edge']        = self._round_edge.value()
        c['flat_edge']         = self._flat_edge.value()
        c['edge_excl']         = self._edge_excl.value()
        c['dies_x']            = self._dies_x.value()
        c['dies_y']            = self._dies_y.value()
        c['min_dies']          = min(self._min_dies.value(),
                                      self._dies_x.value() * self._dies_y.value())
        c['cover_mode']        = self._cover.currentText().split()[0]
        c['placement_mode']    = self._place.currentText().split()[0]
        c['prealign_method']   = self._prealign.currentText()
        c['output_name']       = self._outname.text().strip()
        c['pm_reticle']        = self._pm_reticle.text().strip() or '4544020*'
        self._on_change()

    def refresh(self):
        self._block = True
        c = self._state.cell
        self._wafer.setCurrentIndex(c.get('wafer_w', FOUR_INCH))
        self._cell_x.setValue(c.get('cell_x', 22.0))
        self._cell_y.setValue(c.get('cell_y', 22.0))
        self._round_edge.setValue(c.get('round_edge', 2.0))
        self._flat_edge.setValue(c.get('flat_edge', 0.0))
        self._edge_excl.setValue(c.get('edge_excl', 3.0))
        self._dies_x.setValue(c.get('dies_x', 1))
        self._dies_y.setValue(c.get('dies_y', 1))
        self._min_dies.setValue(c.get('min_dies', 0))
        self._outname.setText(c.get('output_name', 'output'))
        self._pm_reticle.setText(c.get('pm_reticle', '4544020*'))
        self._block = False


# ── ImageTab ───────────────────────────────────────────────────────────────────
class ImageTab(QWidget):
    def __init__(self, state: AppState, on_change, parent=None):
        super().__init__(parent)
        self._state     = state
        self._on_change = on_change
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        QHBoxLayout(self).addWidget(splitter)

        self._masks_panel = ListPanel(['ID', 'Bar Code', 'GDS File', 'Cell'])
        masks_box = QGroupBox('Masks'); QVBoxLayout(masks_box).addWidget(self._masks_panel)
        splitter.addWidget(masks_box)

        self._imgs_panel = ListPanel(['ID', 'Mask', 'Size X', 'Size Y', 'Shift X', 'Shift Y', 'Origin X', 'Origin Y'],
                                      with_copy=True)
        imgs_box = QGroupBox('Images'); QVBoxLayout(imgs_box).addWidget(self._imgs_panel)
        splitter.addWidget(imgs_box)

        self._masks_panel.add_btn.clicked.connect(self._add_mask)
        self._masks_panel.edit_btn.clicked.connect(self._edit_mask)
        self._masks_panel.del_btn.clicked.connect(self._del_mask)
        self._imgs_panel.add_btn.clicked.connect(self._add_image)
        self._imgs_panel.edit_btn.clicked.connect(self._edit_image)
        self._imgs_panel.copy_btn.clicked.connect(self._copy_image)
        self._imgs_panel.del_btn.clicked.connect(self._del_image)

    # ── masks ──────────────────────────────────────────────────────────────────
    def _add_mask(self):
        dlg = MaskDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d['id']:
                self._state.masks.append(d); self._on_change()

    def _edit_mask(self):
        row = self._masks_panel.current_row()
        if row < 0: return
        dlg = MaskDialog(d=self._state.masks[row], parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._state.masks[row] = dlg.result(); self._on_change()

    def _del_mask(self):
        row = self._masks_panel.current_row()
        if row >= 0:
            self._state.masks.pop(row); self._on_change()

    # ── images ─────────────────────────────────────────────────────────────────
    def _add_image(self):
        dlg = ImageDialog(mask_ids=self._state.mask_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d.get('id'):
                self._state.images.append(d)
                self._on_change()

    def _edit_image(self):
        row = self._imgs_panel.current_row()
        if row < 0: return
        dlg = ImageDialog(d=self._state.images[row],
                          mask_ids=self._state.mask_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._state.images[row] = dlg.result()
            self._on_change()

    def _copy_image(self):
        row = self._imgs_panel.current_row()
        if row < 0:
            return
        src = dict(self._state.images[row])
        src['id'] = self._unique_image_id(src.get('id', 'image'))
        dlg = ImageDialog(d=src, mask_ids=self._state.mask_ids(), parent=self,
                          title='Copy Image')
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d.get('id'):
                self._state.images.append(d)
                self._on_change()

    def _unique_image_id(self, base: str) -> str:
        existing = set(self._state.image_ids())
        candidate = f'{base}_copy'
        n = 2
        while candidate in existing:
            candidate = f'{base}_copy{n}'
            n += 1
        return candidate

    def _del_image(self):
        row = self._imgs_panel.current_row()
        if row >= 0:
            iid = self._state.images[row]['id']
            self._state.images.pop(row)
            self._state.patterns.pop(iid, None)
            self._on_change()

    def refresh(self):
        self._masks_panel.set_rows([
            [m['id'], m['bar_code'], m['gds_file'], m['cell_name']]
            for m in self._state.masks
        ])
        self._imgs_panel.set_rows([
            [i['id'], i['mask_id'],
             f"{i['size_x']:.3f}", f"{i['size_y']:.3f}",
             f"{i['shift_x']:.3f}", f"{i['shift_y']:.3f}",
             f"{i.get('origin_x', i['shift_x']):.3f}",
             f"{i.get('origin_y', i['shift_y']):.3f}"]
            for i in self._state.images
        ])


# ── Dies-per-cell wafer coverage rule ────────────────────────────────────────
# Shared by the interactive grid picker (GridCanvas) and the pre-generate
# compliance check: one field is "on the wafer" if enough of its
# dies_x x dies_y sub-die quadrant centers fall inside the edge-exclusion
# boundary. A single global setting (CellTab, next to Cell size) rather than
# per-image, since fields at different sizes/steppings still share the same
# wafer and should agree on what counts as coverable.
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _point_on_wafer(cx: float, cy: float, wafer_w: int,
                     edge_excl: float, flat_edge: float) -> bool:
    r        = WAFER_SIZES[wafer_w] / 2
    r_eff    = r - edge_excl
    flat_lim = -(FLAT[wafer_w] - flat_edge)
    return math.sqrt(cx * cx + cy * cy) <= r_eff and cy >= flat_lim


def _rect_intersects_wafer(cx: float, cy: float, size_x_mm: float, size_y_mm: float,
                            wafer_w: int, edge_excl: float, flat_edge: float) -> bool:
    """Does the field's own footprint (centered at cx, cy) touch the
    edge-exclusion-adjusted wafer/flat at all? The min_dies == 0 case."""
    hw, hh   = size_x_mm / 2, size_y_mm / 2
    r_eff    = WAFER_SIZES[wafer_w] / 2 - edge_excl
    flat_lim = -(FLAT[wafer_w] - flat_edge)
    y_lo = max(cy - hh, flat_lim)          # part of the field at/above the flat
    if y_lo > cy + hh:
        return False                       # whole field is below the flat line
    near_x = _clamp(0.0, cx - hw, cx + hw)
    near_y = _clamp(0.0, y_lo,    cy + hh)
    return math.hypot(near_x, near_y) <= r_eff


def _dies_compliant(cx: float, cy: float, size_x_mm: float, size_y_mm: float,
                     wafer_w: int, edge_excl: float, flat_edge: float,
                     dies_x: int, dies_y: int, min_dies: int) -> bool:
    """Whole-field exposure test: divide the field into a dies_x x dies_y grid
    of quadrant sub-die centers and count how many land inside the wafer.
    min_dies == 0 is looser still — exposed if the field's footprint touches
    the wafer at all, regardless of any sub-die position."""
    if min_dies <= 0:
        return _rect_intersects_wafer(cx, cy, size_x_mm, size_y_mm,
                                       wafer_w, edge_excl, flat_edge)
    count = 0
    for i in range(dies_x):
        for j in range(dies_y):
            sx = cx + ((i + 0.5) / dies_x - 0.5) * size_x_mm
            sy = cy + ((j + 0.5) / dies_y - 0.5) * size_y_mm
            if _point_on_wafer(sx, sy, wafer_w, edge_excl, flat_edge):
                count += 1
                if count >= min_dies:
                    return True
    return count >= min_dies


def _move_inward_to_compliant(cx: float, cy: float, size_x_mm: float, size_y_mm: float,
                               wafer_w: int, edge_excl: float, flat_edge: float,
                               dies_x: int, dies_y: int, min_dies: int
                               ) -> tuple[float, float] | None:
    """Smallest inward nudge (toward the wafer center) that makes a
    non-compliant field compliant, by bisection along the line to the origin.
    None if even the wafer center itself wouldn't be compliant (edge_excl or
    min_dies too strict for any placement to satisfy)."""
    if not _dies_compliant(0.0, 0.0, size_x_mm, size_y_mm, wafer_w, edge_excl,
                            flat_edge, dies_x, dies_y, min_dies):
        return None
    lo, hi = 0.0, 1.0   # t=0 -> wafer center (compliant), t=1 -> original point
    for _ in range(40):
        mid = (lo + hi) / 2
        if _dies_compliant(cx * mid, cy * mid, size_x_mm, size_y_mm, wafer_w,
                            edge_excl, flat_edge, dies_x, dies_y, min_dies):
            lo = mid
        else:
            hi = mid
    return cx * lo, cy * lo


# ── GridCanvas ─────────────────────────────────────────────────────────────────
class GridCanvas(QWidget):
    """Interactive wafer grid: click/drag to select or deselect die cells."""

    selectionChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._step_x    = 22.0
        self._step_y    = 22.0
        self._wafer_w   = FOUR_INCH
        self._edge_excl = 3.0
        self._flat_edge = 0.0
        self._selected: set = set()

        self._drag_active  = False
        self._drag_select  = True
        self._drag_start:   tuple | None = None
        self._drag_current: tuple | None = None
        self._drag_rect_patch = None
        self._blit_bg         = None

        if MATPLOTLIB_OK:
            self._fig    = Figure(figsize=(4.5, 4.5), dpi=90)
            self._ax     = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._fig.patch.set_facecolor('#1e1e1e')
            layout.addWidget(self._canvas)
            self._canvas.mpl_connect('button_press_event',   self._on_press)
            self._canvas.mpl_connect('motion_notify_event',  self._on_motion)
            self._canvas.mpl_connect('button_release_event', self._on_release)
            self._redraw()
        else:
            layout.addWidget(QLabel('Install matplotlib for grid view'))

    # ── public API ─────────────────────────────────────────────────────────────
    def configure(self, step_x: float, step_y: float,
                  wafer_w: int, edge_excl: float, flat_edge: float):
        self._step_x    = max(step_x,    0.001)
        self._step_y    = max(step_y,    0.001)
        self._wafer_w   = wafer_w
        self._edge_excl = edge_excl
        self._flat_edge = flat_edge
        self._redraw()

    def set_selected(self, pts: set):
        self._selected = set(pts)
        self._redraw()

    def get_selected(self) -> set:
        return set(self._selected)

    def clear_selection(self):
        self._selected.clear()
        self._redraw()
        self.selectionChanged.emit()

    # ── private ────────────────────────────────────────────────────────────────
    def _is_on_wafer(self, cx: float, cy: float) -> bool:
        r        = WAFER_SIZES[self._wafer_w] / 2
        r_eff    = r - self._edge_excl
        flat_lim = -(FLAT[self._wafer_w] - self._flat_edge)
        return math.sqrt(cx * cx + cy * cy) <= r_eff and cy >= flat_lim

    def _cell_at(self, x_data: float, y_data: float):
        i  = round(x_data / self._step_x)
        j  = round(y_data / self._step_y)
        cx = i * self._step_x
        cy = j * self._step_y
        return (cx, cy) if self._is_on_wafer(cx, cy) else None

    def _on_press(self, event):
        if event.button != 1 or event.inaxes != self._ax or event.xdata is None:
            return
        cell = self._cell_at(event.xdata, event.ydata)
        self._drag_active  = True
        self._drag_select  = (cell not in self._selected) if cell is not None else True
        self._drag_start   = (event.xdata, event.ydata)
        self._drag_current = (event.xdata, event.ydata)
        color = '#4fc3f7' if self._drag_select else '#f06292'
        self._drag_rect_patch = MplRect(
            (event.xdata, event.ydata), 0.0, 0.0,
            linewidth=1.0, edgecolor=color, linestyle='--',
            facecolor=color, alpha=0.15, zorder=10, animated=True,
        )
        self._ax.add_patch(self._drag_rect_patch)
        self._canvas.draw()
        self._blit_bg = self._canvas.copy_from_bbox(self._fig.bbox)

    def _on_motion(self, event):
        if not self._drag_active or event.xdata is None or event.inaxes != self._ax:
            return
        self._drag_current = (event.xdata, event.ydata)
        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata
        self._drag_rect_patch.set_bounds(
            min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self._canvas.restore_region(self._blit_bg)
        self._ax.draw_artist(self._drag_rect_patch)
        self._canvas.blit(self._fig.bbox)

    def _on_release(self, event):
        if not self._drag_active:
            return
        self._drag_active = False
        if self._drag_rect_patch is not None:
            self._drag_rect_patch.remove()
            self._drag_rect_patch = None
        self._blit_bg = None
        x0, y0 = self._drag_start
        x1 = event.xdata if event.xdata is not None else self._drag_current[0]
        y1 = event.ydata if event.ydata is not None else self._drag_current[1]
        self._apply_drag_selection(x0, y0, x1, y1)
        self._drag_start   = None
        self._drag_current = None
        self._redraw()
        self.selectionChanged.emit()

    def _apply_drag_selection(self, x0: float, y0: float, x1: float, y1: float):
        """Select or deselect all on-wafer cells whose rectangles intersect the drag box."""
        rx0, rx1 = min(x0, x1), max(x0, x1)
        ry0, ry1 = min(y0, y1), max(y0, y1)
        r   = WAFER_SIZES[self._wafer_w] / 2
        n_x = int(r / self._step_x) + 2
        n_y = int(r / self._step_y) + 2
        hw  = self._step_x / 2
        hh  = self._step_y / 2
        for i in range(-n_x, n_x + 1):
            for j in range(-n_y, n_y + 1):
                cx = i * self._step_x
                cy = j * self._step_y
                if not self._is_on_wafer(cx, cy):
                    continue
                if cx - hw <= rx1 and cx + hw >= rx0 and cy - hh <= ry1 and cy + hh >= ry0:
                    if self._drag_select:
                        self._selected.add((cx, cy))
                    else:
                        self._selected.discard((cx, cy))

    def _redraw(self):
        if not MATPLOTLIB_OK:
            return
        ax = self._ax
        ax.cla()
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        ax.set_facecolor('#1e1e1e')

        w         = self._wafer_w
        r         = WAFER_SIZES[w] / 2
        flat_y    = -FLAT[w]
        flat_half = math.sqrt(max(0.0, r * r - flat_y * flat_y))
        theta1    = math.atan2(flat_y,  flat_half)
        theta2    = math.atan2(flat_y, -flat_half)
        if theta2 <= theta1:
            theta2 += 2 * math.pi
        thetas = np.linspace(theta1, theta2, 300)
        ax.fill(r * np.cos(thetas), r * np.sin(thetas),
                color='#2a2a2a', edgecolor='#666666', linewidth=1.2, zorder=1)

        n_x = int(r / self._step_x) + 2
        n_y = int(r / self._step_y) + 2
        for i in range(-n_x, n_x + 1):
            for j in range(-n_y, n_y + 1):
                cx = i * self._step_x
                cy = j * self._step_y
                if not self._is_on_wafer(cx, cy):
                    continue
                selected = (cx, cy) in self._selected
                fc   = '#4fc3f7' if selected else '#3a3a3a'
                alph = 0.80     if selected else 0.60
                ax.add_patch(MplRect(
                    (cx - self._step_x / 2, cy - self._step_y / 2),
                    self._step_x, self._step_y,
                    linewidth=0.5, edgecolor='#555555',
                    facecolor=fc, alpha=alph, zorder=2,
                ))

        ax.set_xlim(-r * 1.08, r * 1.08)
        ax.set_ylim(-r * 1.08, r * 1.08)
        n = len(self._selected)
        ax.set_title(
            f'{n} cell{"s" if n != 1 else ""} selected',
            color='#aaaaaa', fontsize=8, pad=2,
        )
        self._fig.tight_layout(pad=0.2)
        self._canvas.draw()


# ── PatternTab ─────────────────────────────────────────────────────────────────
class PatternTab(QWidget):
    def __init__(self, state: AppState, on_change, parent=None):
        super().__init__(parent)
        self._state         = state
        self._on_change     = on_change
        self._current_image: str | None = None
        self._block         = False
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        main = QVBoxLayout(self)

        # ── Image selector ────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel('Image:'))
        self._img_combo = QComboBox()
        self._img_combo.setMinimumWidth(160)
        self._img_combo.currentTextChanged.connect(self._on_image_changed)
        top.addWidget(self._img_combo)
        top.addStretch()
        main.addLayout(top)

        # ── Stepping ──────────────────────────────────────────────────────────
        step_box = QGroupBox('Stepping')
        step_row = QHBoxLayout(step_box)
        step_row.addWidget(QLabel('Mode:'))
        self._step_mode = QComboBox()
        self._step_mode.addItems(['Cell size', 'Image size', 'Custom'])
        self._step_mode.currentTextChanged.connect(self._on_step_mode_changed)
        step_row.addWidget(self._step_mode)
        step_row.addSpacing(12)
        step_row.addWidget(QLabel('X (mm):'))
        self._step_x = QDoubleSpinBox()
        self._step_x.setRange(0.001, 9999.0)
        self._step_x.setDecimals(4)
        self._step_x.setValue(22.0)
        self._step_x.valueChanged.connect(self._on_step_changed)
        step_row.addWidget(self._step_x)
        step_row.addSpacing(8)
        step_row.addWidget(QLabel('Y (mm):'))
        self._step_y = QDoubleSpinBox()
        self._step_y.setRange(0.001, 9999.0)
        self._step_y.setDecimals(4)
        self._step_y.setValue(22.0)
        self._step_y.valueChanged.connect(self._on_step_changed)
        step_row.addWidget(self._step_y)
        step_row.addStretch()
        main.addWidget(step_box)

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_box = QGroupBox('Point input mode')
        mode_row = QHBoxLayout(mode_box)
        self._grid_rb   = QRadioButton('Interactive grid')
        self._python_rb = QRadioButton('Python expression')
        self._csv_rb    = QRadioButton('CSV file')
        self._grid_rb.setChecked(True)
        for rb in (self._grid_rb, self._python_rb, self._csv_rb):
            mode_row.addWidget(rb)
        self._grid_rb.toggled.connect(self._on_mode_changed)
        self._python_rb.toggled.connect(self._on_mode_changed)
        self._csv_rb.toggled.connect(self._on_mode_changed)
        main.addWidget(mode_box)

        # ── Stacked panels ────────────────────────────────────────────────────
        self._stack = QStackedWidget()

        # Page 0: interactive grid
        self._grid_canvas = GridCanvas()
        self._grid_canvas.selectionChanged.connect(self._on_grid_changed)
        self._stack.addWidget(self._grid_canvas)

        # Page 1: python expression
        py_panel = QWidget()
        pv = QVBoxLayout(py_panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(QLabel('Expression or statements — assign output to "result":'))
        self._code_edit = QPlainTextEdit()
        self._code_edit.setFont(QFont('monospace', 10))
        pv.addWidget(self._code_edit)
        eval_row = QHBoxLayout()
        self._eval_btn    = QPushButton('Evaluate')
        self._default_btn = QPushButton('Reset to default')
        self._err_label   = QLabel()
        self._err_label.setStyleSheet('color: #ff6b6b;')
        self._eval_btn.clicked.connect(self._evaluate_code)
        self._default_btn.clicked.connect(self._reset_to_default)
        eval_row.addWidget(self._eval_btn)
        eval_row.addWidget(self._default_btn)
        eval_row.addWidget(self._err_label)
        eval_row.addStretch()
        pv.addLayout(eval_row)
        self._stack.addWidget(py_panel)

        # Page 2: CSV
        csv_panel = QWidget()
        cv = QVBoxLayout(csv_panel)
        cv.setContentsMargins(0, 0, 0, 0)
        csv_row = QHBoxLayout()
        self._csv_path = QLineEdit()
        self._csv_path.setPlaceholderText('Select a CSV file with x,y columns…')
        self._csv_path.setReadOnly(True)
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self._browse_csv)
        load_btn = QPushButton('Load')
        load_btn.clicked.connect(self._load_csv)
        csv_row.addWidget(self._csv_path)
        csv_row.addWidget(browse_btn)
        csv_row.addWidget(load_btn)
        cv.addLayout(csv_row)
        self._csv_status = QLabel('No file loaded')
        self._csv_status.setStyleSheet('color: #888888;')
        cv.addWidget(self._csv_status)
        cv.addStretch()
        self._stack.addWidget(csv_panel)

        main.addWidget(self._stack, stretch=1)

        self._count_label = QLabel('0 points')
        main.addWidget(self._count_label)

    # ── Step helpers ───────────────────────────────────────────────────────────
    def _computed_steps(self) -> tuple:
        mode = self._step_mode.currentText()
        if mode == 'Cell size':
            return (self._state.cell.get('cell_x', 22.0),
                    self._state.cell.get('cell_y', 22.0))
        if mode == 'Image size':
            img = next((i for i in self._state.images
                        if i['id'] == self._current_image), None)
            if img:
                return (img['size_x'], img['size_y'])
        return (self._step_x.value(), self._step_y.value())

    def _update_step_spinboxes(self):
        custom = (self._step_mode.currentText() == 'Custom')
        self._step_x.setReadOnly(not custom)
        self._step_y.setReadOnly(not custom)
        if not custom:
            sx, sy = self._computed_steps()
            self._block = True
            self._step_x.setValue(sx)
            self._step_y.setValue(sy)
            self._block = False

    def _apply_step_to_grid(self):
        sx, sy = self._computed_steps()
        c = self._state.cell
        self._grid_canvas.configure(
            sx, sy,
            c.get('wafer_w',    FOUR_INCH),
            c.get('edge_excl',  3.0),
            c.get('flat_edge',  0.0),
        )

    def _on_step_mode_changed(self):
        self._update_step_spinboxes()
        self._apply_step_to_grid()
        if not self._block and self._current_image:
            self._state.pattern_step[self._current_image] = {
                'mode': self._step_mode.currentText(),
                'x':    self._step_x.value(),
                'y':    self._step_y.value(),
            }

    def _on_step_changed(self):
        if self._block:
            return
        self._apply_step_to_grid()
        if self._current_image:
            self._state.pattern_step[self._current_image] = {
                'mode': self._step_mode.currentText(),
                'x':    self._step_x.value(),
                'y':    self._step_y.value(),
            }

    # ── Mode helpers ───────────────────────────────────────────────────────────
    def _current_mode(self) -> str:
        if self._python_rb.isChecked(): return 'python'
        if self._csv_rb.isChecked():    return 'csv'
        return 'grid'

    def _on_mode_changed(self):
        mode = self._current_mode()
        self._stack.setCurrentIndex({'grid': 0, 'python': 1, 'csv': 2}[mode])
        if not self._block and self._current_image:
            self._state.pattern_mode[self._current_image] = mode

    # ── Persist / restore ──────────────────────────────────────────────────────
    def _save_current(self):
        iid = self._current_image
        if not iid:
            return
        mode = self._current_mode()
        self._state.pattern_mode[iid] = mode
        self._state.pattern_step[iid] = {
            'mode': self._step_mode.currentText(),
            'x':    self._step_x.value(),
            'y':    self._step_y.value(),
        }
        if mode == 'grid':
            sel = self._grid_canvas.get_selected()
            self._state.pattern_grid[iid] = sel
            self._state.patterns[iid]     = [{'x': x, 'y': y} for x, y in sel]
        elif mode == 'python':
            self._state.pattern_code[iid] = self._code_edit.toPlainText()
        elif mode == 'csv':
            self._state.pattern_csv_file[iid] = self._csv_path.text()

    def _load_current(self):
        self._block = True
        iid = self._current_image

        if not iid:
            self._grid_canvas.set_selected(set())
            self._code_edit.setPlainText('')
            self._csv_path.setText('')
            self._csv_status.setText('No file loaded')
            self._count_label.setText('0 points')
            self._block = False
            return

        # Step settings
        step_d = self._state.pattern_step.get(iid, {'mode': 'Cell size'})
        sm     = step_d.get('mode', 'Cell size')
        idx    = self._step_mode.findText(sm)
        self._step_mode.setCurrentIndex(idx if idx >= 0 else 0)
        custom = (sm == 'Custom')
        self._step_x.setReadOnly(not custom)
        self._step_y.setReadOnly(not custom)
        if custom:
            self._step_x.setValue(step_d.get('x', 22.0))
            self._step_y.setValue(step_d.get('y', 22.0))
        else:
            sx, sy = self._computed_steps()
            self._step_x.setValue(sx)
            self._step_y.setValue(sy)

        # Mode (radio + stack page)
        mode = self._state.pattern_mode.get(iid, 'grid')
        self._grid_rb.setChecked(mode == 'grid')
        self._python_rb.setChecked(mode == 'python')
        self._csv_rb.setChecked(mode == 'csv')
        self._stack.setCurrentIndex({'grid': 0, 'python': 1, 'csv': 2}.get(mode, 0))

        # Grid canvas
        self._apply_step_to_grid()
        self._grid_canvas.set_selected(self._state.pattern_grid.get(iid, set()))

        # Python panel
        code = self._state.pattern_code.get(iid, '')
        if not code:
            code = self._make_default_code()
        self._code_edit.setPlainText(code)
        self._err_label.setText('')

        # CSV panel
        csv_path = self._state.pattern_csv_file.get(iid, '')
        self._csv_path.setText(csv_path)
        pts = self._state.patterns.get(iid, [])
        if csv_path:
            self._csv_status.setText(f'{len(pts)} points loaded')
        else:
            self._csv_status.setText('No file loaded')

        self._count_label.setText(f'{len(pts)} points')
        self._block = False

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _on_image_changed(self, iid: str):
        if self._block:
            return
        self._save_current()
        self._current_image = iid if iid else None
        self._load_current()

    def _on_grid_changed(self):
        if self._block:
            return
        iid = self._current_image
        if iid:
            sel = self._grid_canvas.get_selected()
            self._state.pattern_grid[iid] = sel
            self._state.patterns[iid]     = [{'x': x, 'y': y} for x, y in sel]
        n = len(self._state.patterns.get(self._current_image or '', []))
        self._count_label.setText(f'{n} points')
        self._on_change()

    # ── Python mode ───────────────────────────────────────────────────────────
    def _make_default_code(self) -> str:
        c      = self._state.cell
        w      = c.get('wafer_w', FOUR_INCH)
        r      = WAFER_SIZES[w] / 2
        excl   = c.get('edge_excl', 3.0)
        flat   = c.get('flat_edge', 0.0)
        flat_y = -(FLAT[w] - flat)
        sx, sy = self._computed_steps()
        return (
            f"step_x = {sx:.4f}\n"
            f"step_y = {sy:.4f}\n"
            f"r      = {r:.4f}\n"
            f"excl   = {excl:.4f}\n"
            f"flat_y = {flat_y:.4f}\n"
            f"\n"
            f"n_x = int(r / step_x) + 2\n"
            f"n_y = int(r / step_y) + 2\n"
            f"result = []\n"
            f"for i in range(-n_x, n_x + 1):\n"
            f"    for j in range(-n_y, n_y + 1):\n"
            f"        x = i * step_x\n"
            f"        y = j * step_y\n"
            f"        if math.sqrt(x*x + y*y) + excl <= r and y >= flat_y + excl:\n"
            f"            result.append((x, y))\n"
        )

    def _reset_to_default(self):
        self._code_edit.setPlainText(self._make_default_code())
        self._err_label.setText('')

    def _evaluate_code(self):
        if not self._current_image:
            return
        code   = self._code_edit.toPlainText()
        result = _eval_pattern(code)
        if isinstance(result, str):
            self._err_label.setText(result)
        else:
            self._err_label.setText('')
            self._state.pattern_code[self._current_image] = code
            self._state.patterns[self._current_image]     = result
            self._count_label.setText(f'{len(result)} points resolved')
            self._on_change()

    # ── CSV mode ──────────────────────────────────────────────────────────────
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select CSV file', '',
            'CSV Files (*.csv *.txt);;All Files (*)')
        if path:
            self._csv_path.setText(path)

    def _load_csv(self):
        path = self._csv_path.text().strip()
        if not path:
            self._csv_status.setText('No file selected')
            return
        try:
            pts = []
            with open(path, newline='') as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        try:
                            pts.append({'x': float(row[0]), 'y': float(row[1])})
                        except ValueError:
                            pass
            if not pts:
                self._csv_status.setText('No valid x,y pairs found')
                return
            iid = self._current_image
            if iid:
                self._state.pattern_csv_file[iid] = path
                self._state.patterns[iid]         = pts
            self._csv_status.setText(f'{len(pts)} points loaded')
            self._count_label.setText(f'{len(pts)} points')
            self._on_change()
        except Exception as exc:
            self._csv_status.setText(str(exc))

    # ── Refresh ───────────────────────────────────────────────────────────────
    def refresh(self):
        iids = self._state.image_ids()
        self._save_current()
        self._block = True
        prev = self._img_combo.currentText()
        self._img_combo.clear()
        self._img_combo.addItems(iids)
        if prev in iids:
            self._img_combo.setCurrentText(prev)
        elif iids:
            self._img_combo.setCurrentIndex(0)
        self._block = False
        new_iid = self._img_combo.currentText() or None
        if new_iid != self._current_image:
            self._current_image = new_iid
        self._load_current()


# ── LayerTab ───────────────────────────────────────────────────────────────────
class LayerTab(QWidget):
    def __init__(self, state: AppState, on_change, parent=None):
        super().__init__(parent)
        self._state     = state
        self._on_change = on_change
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        v = QVBoxLayout(self)
        self._panel = ListPanel(['ID', 'Rotation', 'NA', 'σ in', 'σ out', 'Illumination'],
                                 with_copy=True)
        v.addWidget(self._panel)
        h = QHBoxLayout()
        self._up_btn = QPushButton('▲ Move Up'); self._dn_btn = QPushButton('▼ Move Down')
        self._up_btn.clicked.connect(self._move_up); self._dn_btn.clicked.connect(self._move_down)
        h.addWidget(self._up_btn); h.addWidget(self._dn_btn); h.addStretch(); v.addLayout(h)
        self._panel.add_btn.clicked.connect(self._add)
        self._panel.edit_btn.clicked.connect(self._edit)
        self._panel.copy_btn.clicked.connect(self._copy)
        self._panel.del_btn.clicked.connect(self._del)

    def _add(self):
        dlg = LayerDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d['id']:
                self._state.layers.append(d); self._on_change()

    def _edit(self):
        row = self._panel.current_row()
        if row < 0: return
        dlg = LayerDialog(d=self._state.layers[row], parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._state.layers[row] = dlg.result(); self._on_change()

    def _copy(self):
        row = self._panel.current_row()
        if row < 0:
            return
        src = dict(self._state.layers[row])
        src['id'] = self._unique_layer_id(src.get('id', 'layer'))
        dlg = LayerDialog(d=src, parent=self, title='Copy Layer')
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d['id']:
                self._state.layers.append(d); self._on_change()

    def _unique_layer_id(self, base: str) -> str:
        existing = set(self._state.layer_ids())
        candidate = f'{base}_copy'
        n = 2
        while candidate in existing:
            candidate = f'{base}_copy{n}'
            n += 1
        return candidate

    def _del(self):
        row = self._panel.current_row()
        if row >= 0:
            lid = self._state.layers[row]['id']
            self._state.layers.pop(row)
            self._state.layer_process.pop(lid, None)
            self._on_change()

    def _move_up(self):
        row = self._panel.current_row()
        if row > 0:
            l = self._state.layers; l[row - 1], l[row] = l[row], l[row - 1]
            self._on_change(); self._panel.table.selectRow(row - 1)

    def _move_down(self):
        row = self._panel.current_row(); l = self._state.layers
        if 0 <= row < len(l) - 1:
            l[row], l[row + 1] = l[row + 1], l[row]
            self._on_change(); self._panel.table.selectRow(row + 1)

    def refresh(self):
        self._panel.set_rows([
            [lyr['id'],
             f"{lyr.get('rotation', 0.0):.1f}°",
             f"{lyr.get('na', 0.57):.3f}",
             f"{lyr.get('sigma_in', 0.0):.3f}",
             f"{lyr.get('sigma_out', 0.5):.3f}",
             lyr.get('illume', 'Conventional')]
            for lyr in self._state.layers
        ])


# ── AlignmentTab ───────────────────────────────────────────────────────────────
class AlignmentTab(QWidget):
    def __init__(self, state: AppState, on_change, parent=None):
        super().__init__(parent)
        self._state     = state
        self._on_change = on_change
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        v = QVBoxLayout(self)

        combine_row = QHBoxLayout()
        self._combine = QCheckBox('Combine zero + first layer')
        self._combine.setToolTip(
            'The first layer is exposed together with the mark layer (layer 0) '
            'and is never wafer-aligned on its own — any strategy bound to it '
            'is dropped while this is checked.')
        self._combine.stateChanged.connect(self._on_combine_changed)
        combine_row.addWidget(self._combine)
        self._combine_hint = QLabel('')
        self._combine_hint.setStyleSheet('color: #888888; font-size: 10px;')
        combine_row.addWidget(self._combine_hint)
        combine_row.addStretch()
        v.addLayout(combine_row)

        rm_row = QHBoxLayout()
        rm_row.addWidget(QLabel('Prealigner sign (rotated-mark layers):'))
        self._rot_sign = QComboBox()
        self._rot_sign.addItems(['-1', '+1'])
        self._rot_sign.setToolTip(
            'Prealigner turn direction for a layer that aligns on a rotated '
            'field. The rotated-marks job set (unaligned marks job + one lens '
            'job per rotation + a stepper-frame view per rotation) is emitted '
            'automatically whenever an alignment sits on a rotated layer, and '
            'skipped otherwise.')
        self._rot_sign.currentIndexChanged.connect(self._on_rot_sign_changed)
        rm_row.addWidget(self._rot_sign)
        rm_row.addStretch()
        v.addLayout(rm_row)

        marks_box = QGroupBox('Marks')
        self._marks_panel = ListPanel(['Mark ID', 'Image', 'X (mm)', 'Y (mm)'])
        QVBoxLayout(marks_box).addWidget(self._marks_panel)
        v.addWidget(marks_box)

        strat_box = QGroupBox('Strategies')
        self._strat_panel = ListPanel(['Strategy ID', 'Marks', 'Required'])
        QVBoxLayout(strat_box).addWidget(self._strat_panel)
        v.addWidget(strat_box)

        bind_box = QGroupBox('Layer – Strategy Bindings')
        self._bind_panel = ListPanel(['Layer', 'Strategy', 'Usage'])
        QVBoxLayout(bind_box).addWidget(self._bind_panel)
        v.addWidget(bind_box)

        self._marks_panel.add_btn.clicked.connect(self._add_mark)
        self._marks_panel.edit_btn.clicked.connect(self._edit_mark)
        self._marks_panel.del_btn.clicked.connect(self._del_mark)
        self._strat_panel.add_btn.clicked.connect(self._add_strat)
        self._strat_panel.edit_btn.clicked.connect(self._edit_strat)
        self._strat_panel.del_btn.clicked.connect(self._del_strat)
        self._bind_panel.add_btn.clicked.connect(self._add_bind)
        self._bind_panel.edit_btn.clicked.connect(self._edit_bind)
        self._bind_panel.del_btn.clicked.connect(self._del_bind)

    # ── marks ──────────────────────────────────────────────────────────────────
    def _add_mark(self):
        dlg = MarkDialog(image_ids=self._state.image_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d['id']:
                self._state.marks.append(d); self._on_change()

    def _edit_mark(self):
        row = self._marks_panel.current_row()
        if row < 0: return
        dlg = MarkDialog(d=self._state.marks[row],
                         image_ids=self._state.image_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._state.marks[row] = dlg.result(); self._on_change()

    def _del_mark(self):
        row = self._marks_panel.current_row()
        if row >= 0:
            self._state.marks.pop(row); self._on_change()

    # ── strategies ─────────────────────────────────────────────────────────────
    def _add_strat(self):
        dlg = StrategyDialog(mark_ids=self._state.mark_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d['name']:
                self._state.alignments.append(d); self._on_change()

    def _edit_strat(self):
        row = self._strat_panel.current_row()
        if row < 0: return
        dlg = StrategyDialog(d=self._state.alignments[row],
                             mark_ids=self._state.mark_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._state.alignments[row] = dlg.result(); self._on_change()

    def _del_strat(self):
        row = self._strat_panel.current_row()
        if row >= 0:
            self._state.alignments.pop(row); self._on_change()

    # ── layer–strategy bindings ─────────────────────────────────────────────────
    def _combined_first_lid(self) -> str | None:
        """The layer id that is off-limits for alignment because zero+first are
        combined (it is exposed with layer 0), or None."""
        lids = self._state.layer_ids()
        if lids and self._state.cell.get('combine_zero_first', False):
            return lids[0]
        return None

    def _drop_combined_first_binding(self) -> bool:
        """Remove any strategy bound to the combined first layer.  Returns True
        if something was removed."""
        lid = self._combined_first_lid()
        if lid is None:
            return False
        proc = self._state.layer_process.get(lid, {})
        if 'strategy_id' in proc or 'strategy_usage' in proc:
            proc.pop('strategy_id', None)
            proc.pop('strategy_usage', None)
            return True
        return False

    def _bindable_layer_ids(self) -> list[str]:
        excl = self._combined_first_lid()
        return [l for l in self._state.layer_ids() if l != excl]

    def _add_bind(self):
        lids = self._bindable_layer_ids(); aids = self._state.alignment_ids()
        if not lids or not aids:
            QMessageBox.warning(self, 'No data',
                                'Define layers and strategies first.\n'
                                '(The combined first layer cannot be aligned.)'
                                if aids or self._state.layer_ids()
                                else 'Define layers and strategies first.')
            return
        dlg = _BindDialog(layer_ids=lids, align_ids=aids, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            lid, aid, usage = dlg.result()
            self._state.layer_process.setdefault(lid, {}).update(
                strategy_id=aid, strategy_usage=usage)
            self._on_change()

    def _edit_bind(self):
        row = self._bind_panel.current_row()
        bindings = self._get_bindings()
        if row < 0 or row >= len(bindings): return
        lid0, aid0, usage0 = bindings[row]
        layer_ids = self._bindable_layer_ids()
        if lid0 not in layer_ids:            # editing an existing (now-excluded) row
            layer_ids = [lid0] + layer_ids
        dlg = _BindDialog(layer_ids=layer_ids,
                          align_ids=self._state.alignment_ids(),
                          current=(lid0, aid0, usage0), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            lid, aid, usage = dlg.result()
            if lid0 != lid:
                self._state.layer_process.get(lid0, {}).pop('strategy_id', None)
                self._state.layer_process.get(lid0, {}).pop('strategy_usage', None)
            self._state.layer_process.setdefault(lid, {}).update(
                strategy_id=aid, strategy_usage=usage)
            self._on_change()

    def _del_bind(self):
        row = self._bind_panel.current_row()
        bindings = self._get_bindings()
        if row < 0 or row >= len(bindings): return
        lid, _, _ = bindings[row]
        self._state.layer_process.get(lid, {}).pop('strategy_id', None)
        self._state.layer_process.get(lid, {}).pop('strategy_usage', None)
        self._on_change()

    def _get_bindings(self) -> list[tuple]:
        result = []
        for lid in self._state.layer_ids():
            proc = self._state.layer_process.get(lid, {})
            aid  = proc.get('strategy_id')
            if aid:
                result.append((lid, aid, proc.get('strategy_usage', 'A')))
        return result

    def _on_combine_changed(self):
        self._state.cell['combine_zero_first'] = self._combine.isChecked()
        # The first layer is now exposed with layer 0 — auto-deselect any
        # strategy bound to it (it cannot be wafer-aligned).
        self._drop_combined_first_binding()
        self._on_change()

    def _on_rot_sign_changed(self, *_):
        self._state.cell['rotated_marks_sign'] = int(self._rot_sign.currentText())
        self._on_change()

    def refresh(self):
        self._combine.blockSignals(True)
        self._combine.setChecked(self._state.cell.get('combine_zero_first', False))
        self._combine.blockSignals(False)
        # Enforce "combined first layer has no alignment" on every refresh so a
        # loaded file with a stale binding is cleaned up too.
        self._drop_combined_first_binding()
        excl = self._combined_first_lid()
        self._combine_hint.setText(
            f'— layer "{excl}" exposed with layer 0, not aligned' if excl else '')
        self._rot_sign.setCurrentText(
            str(self._state.cell.get('rotated_marks_sign', -1)))
        self._marks_panel.set_rows([
            [m['id'], m['image_id'],
             f"{m.get('x', 0.0):.3f}", f"{m.get('y', 0.0):.3f}"]
            for m in self._state.marks
        ])
        self._strat_panel.set_rows([
            [a['name'], ', '.join(a.get('mark_ids', [])), str(a.get('required', 4))]
            for a in self._state.alignments
        ])
        self._bind_panel.set_rows(
            [[lid, aid, usage] for lid, aid, usage in self._get_bindings()]
        )


class ExposureImageDialog(FormDialog):
    """Dialog for adding/editing a single image exposure within a layer."""
    def __init__(self, d: dict | None = None, image_ids: list[str] | None = None,
                 parent=None, title: str | None = None):
        super().__init__(title or ('Add Image Exposure' if d is None else 'Edit Image Exposure'), parent)
        d = d or {}
        self._combo('image_id', 'Image',         image_ids or [], d.get('image_id', ''))
        self._dspin('dose',     'Dose (mJ/cm²)', d.get('dose', 20.0),  0.0,  9999.0, 2)
        self._dspin('focus',    'Focus (µm)',     d.get('focus', 0.0), -99.0,   99.0, 3)

    def result(self) -> dict:
        return dict(image_id=self.val('image_id'),
                    dose=self.val('dose'), focus=self.val('focus'))


# ── ProcessTab ─────────────────────────────────────────────────────────────────
class ProcessTab(QWidget):
    def __init__(self, state: AppState, on_change, parent=None):
        super().__init__(parent)
        self._state     = state
        self._on_change = on_change
        self._block     = False
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        v = QVBoxLayout(self)

        # Layer selector at the top
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel('Layer:'))
        self._layer_combo = QComboBox()
        self._layer_combo.setMinimumWidth(120)
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        layer_row.addWidget(self._layer_combo)
        layer_row.addStretch()
        v.addLayout(layer_row)

        expo_box = QGroupBox('Exposures')
        ev = QVBoxLayout(expo_box)
        self._expo_panel = ListPanel(['Image', 'Dose (mJ/cm²)', 'Focus (µm)'], with_copy=True)
        ev.addWidget(self._expo_panel)
        v.addWidget(expo_box)

        level_box = QGroupBox('Leveling Methods')
        lg = QGridLayout(level_box)
        lg.addWidget(QLabel('Level Z:'), 0, 0)
        self._level_z = QComboBox(); self._level_z.addItems(LEVEL_METHODS)
        self._level_z.currentTextChanged.connect(self._sync_levels)
        lg.addWidget(self._level_z, 0, 1)
        lg.addWidget(QLabel('Level RX:'), 0, 2)
        self._level_rx = QComboBox(); self._level_rx.addItems(LEVEL_METHODS)
        self._level_rx.currentTextChanged.connect(self._sync_levels)
        lg.addWidget(self._level_rx, 0, 3)
        lg.addWidget(QLabel('Level RY:'), 0, 4)
        self._level_ry = QComboBox(); self._level_ry.addItems(LEVEL_METHODS)
        self._level_ry.currentTextChanged.connect(self._sync_levels)
        lg.addWidget(self._level_ry, 0, 5)
        lg.setColumnStretch(6, 1)
        v.addWidget(level_box)

        self._expo_panel.add_btn.clicked.connect(self._add_expo)
        self._expo_panel.edit_btn.clicked.connect(self._edit_expo)
        self._expo_panel.copy_btn.clicked.connect(self._copy_expo)
        self._expo_panel.del_btn.clicked.connect(self._del_expo)

    def _current_layer(self) -> str:
        return self._layer_combo.currentText()

    def _layer_expos_indexed(self) -> list[tuple[int, dict]]:
        lid = self._current_layer()
        return [(i, e) for i, e in enumerate(self._state.exposures)
                if e['layer_id'] == lid]

    def _on_layer_changed(self):
        self._refresh_expo_table()
        self._refresh_level_combos()

    def _refresh_expo_table(self):
        rows = self._layer_expos_indexed()
        self._expo_panel.set_rows([
            [e['image_id'], f"{e.get('dose', 20.0):.2f}", f"{e.get('focus', 0.0):.3f}"]
            for _, e in rows
        ])

    def _add_expo(self):
        lid = self._current_layer()
        if not lid:
            QMessageBox.warning(self, 'No layer', 'Select a layer first.')
            return
        dlg = ExposureImageDialog(image_ids=self._state.image_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d.get('image_id'):
                self._state.exposures.append({'layer_id': lid, **d})
                self._on_change()

    def _edit_expo(self):
        row = self._expo_panel.current_row()
        indexed = self._layer_expos_indexed()
        if row < 0 or row >= len(indexed): return
        global_idx, expo = indexed[row]
        dlg = ExposureImageDialog(d=expo, image_ids=self._state.image_ids(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            self._state.exposures[global_idx] = {'layer_id': expo['layer_id'], **d}
            self._on_change()

    def _copy_expo(self):
        row = self._expo_panel.current_row()
        indexed = self._layer_expos_indexed()
        if row < 0 or row >= len(indexed): return
        _, expo = indexed[row]
        dlg = ExposureImageDialog(d=expo, image_ids=self._state.image_ids(), parent=self,
                                  title='Copy Image Exposure')
        if dlg.exec() == QDialog.DialogCode.Accepted:
            d = dlg.result()
            if d.get('image_id'):
                self._state.exposures.append({'layer_id': expo['layer_id'], **d})
                self._on_change()

    def _del_expo(self):
        row = self._expo_panel.current_row()
        indexed = self._layer_expos_indexed()
        if row < 0 or row >= len(indexed): return
        global_idx, _ = indexed[row]
        self._state.exposures.pop(global_idx)
        self._on_change()

    def _sync_levels(self):
        if self._block: return
        lid = self._current_layer()
        if not lid: return
        proc = self._state.layer_process.setdefault(lid, {})
        proc['level_z']  = self._level_z.currentText().split()[0]
        proc['level_rx'] = self._level_rx.currentText().split()[0]
        proc['level_ry'] = self._level_ry.currentText().split()[0]
        self._on_change()

    def _refresh_level_combos(self):
        self._block = True
        lid = self._current_layer()
        proc = self._state.layer_process.get(lid, {}) if lid else {}
        for combo, key in ((self._level_z, 'level_z'),
                           (self._level_rx, 'level_rx'),
                           (self._level_ry, 'level_ry')):
            combo.blockSignals(True)
            code = proc.get(key, 'D')
            for lm in LEVEL_METHODS:
                if lm.startswith(code):
                    combo.setCurrentText(lm)
                    break
            combo.blockSignals(False)
        self._block = False

    def refresh(self):
        self._block = True
        lids = self._state.layer_ids()
        prev = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        self._layer_combo.addItems(lids)
        if prev in lids:
            self._layer_combo.setCurrentText(prev)
        elif lids:
            self._layer_combo.setCurrentIndex(0)
        self._layer_combo.blockSignals(False)
        self._refresh_expo_table()
        self._refresh_level_combos()
        self._block = False


# ── GraphicsWindow ─────────────────────────────────────────────────────────────
class GraphicsWindow(QMainWindow):
    """Separate window with Wafer Layout and Mask Layout tabs."""

    _LAYER_COLORS = [
        '#4fc3f7', '#81c784', '#ffb74d', '#f06292',
        '#ce93d8', '#80cbc4', '#fff176', '#ff8a65',
    ]

    _BUSY_SUFFIX = '   · refreshing…'

    @classmethod
    def _layer_color(cls, lyr: int) -> str:
        """Colour for a GDS layer, keyed on the layer NUMBER — never on its
        position in the payload. A layer keeps its colour across the full view,
        a zoom-detail swap (where a rasterised layer comes back as vector
        polygons and the ordering changes), a viewport query, and toggling
        other layers off. Layers 8 apart share a colour; that's the trade."""
        return cls._LAYER_COLORS[lyr % len(cls._LAYER_COLORS)]
    _mask_ready = pyqtSignal(object)    # payload dict from background loader thread
    _mask_error = pyqtSignal(str, str)  # mask_id, error text
    _mask_huge  = pyqtSignal(str, tuple, object)   # mask_id, cache_key, gdstk.Cell
    _viewport_ready = pyqtSignal(object)           # payload dict from a viewport query
    _mask_detail    = pyqtSignal(object)           # windowed re-raster for zoom-in

    def __init__(self, state: 'AppState', parent=None):
        super().__init__(parent)
        self.setWindowTitle('ASML Graphics')
        self.resize(820, 760)
        self._state        = state
        self._poly_cache:  dict         = {}   # (gds_file, cell_name) → payload dict
        self._mask_home_xlim: tuple | None = None
        self._mask_home_ylim: tuple | None = None
        self._zoom_start:   tuple | None = None
        self._zoom_patch    = None
        self._zoom_blit_bg  = None
        self._wafer_home_xlim:  tuple | None = None
        self._wafer_home_ylim:  tuple | None = None
        self._wafer_zoom_start:  tuple | None = None
        self._wafer_zoom_patch   = None
        self._wafer_zoom_blit_bg = None
        self._wafer_loading_keys: set = set()
        # Masks the wafer view has already tried and can't use (too big to
        # flatten, or the read failed). Without this the "not cached and not
        # loading" test is true forever and every redraw re-launches the load.
        self._wafer_skip: set = set()
        # Cells too large to ever flatten fully: kept around (cheap — it's a
        # reference/hierarchy handle, not flattened geometry) so pan/zoom can
        # re-query just the visible window via `_query_cell_bbox`.
        self._huge_cells:      dict = {}   # cache_key -> gdstk.Cell
        self._viewport_cache:  dict = {}   # cache_key -> last viewport payload
        self._viewport_loading: set = set()  # cache_keys with a query in flight
        # A query takes long enough that the user can zoom again while it runs.
        # `_viewport_req` stamps each request so a result that arrives after a
        # newer one was issued is discarded instead of dragging the view back to
        # its own (older) bbox; `_viewport_pending` holds the newest window
        # requested mid-flight so it is issued on completion rather than lost.
        self._viewport_seq:     int  = 0
        self._viewport_req:     dict = {}   # cache_key -> seq of newest request
        self._viewport_pending: dict = {}   # cache_key -> (mask_id, bbox) to run next
        # Zoom-in on a rasterised (truncated) non-huge mask: re-raster just the
        # visible window at full resolution so the density wash sharpens.
        self._detail_loading: set = set()  # cache_keys with a detail re-raster in flight
        self._detail_active:  set = set()  # cache_keys currently showing windowed detail
        self._mask_ready.connect(self._on_mask_ready)
        self._mask_error.connect(self._on_mask_error)
        self._mask_huge.connect(self._on_mask_huge)
        self._viewport_ready.connect(self._on_viewport_ready)
        self._mask_detail.connect(self._on_mask_detail)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        self._tabs = QTabWidget()

        # ── Wafer layout tab ──────────────────────────────────────────────────
        wafer_tab = QWidget()
        wv = QVBoxLayout(wafer_tab)
        wafer_top = QHBoxLayout()
        self._wafer_home_btn = QPushButton('Home')
        self._wafer_home_btn.setFixedWidth(60)
        self._wafer_home_btn.setToolTip('Reset view to full wafer bounds')
        self._wafer_home_btn.clicked.connect(self._wafer_home)
        wafer_top.addWidget(self._wafer_home_btn)
        self._wafer_coord_label = QLabel('X: —   Y: —   µm')
        self._wafer_coord_label.setStyleSheet('color: #888888; font-size: 11px;')
        wafer_top.addSpacing(16)
        wafer_top.addWidget(self._wafer_coord_label)
        wafer_top.addStretch()
        wv.addLayout(wafer_top)
        if MATPLOTLIB_OK:
            self._wafer_fig    = Figure(figsize=(7.0, 7.0), dpi=90)
            self._wafer_ax     = self._wafer_fig.add_subplot(111)
            self._wafer_canvas = FigureCanvas(self._wafer_fig)
            self._wafer_fig.patch.set_facecolor('#1e1e1e')
            self._wafer_canvas.mpl_connect('button_press_event',   self._wafer_zoom_press)
            self._wafer_canvas.mpl_connect('motion_notify_event',  self._wafer_zoom_motion)
            self._wafer_canvas.mpl_connect('button_release_event', self._wafer_zoom_release)
            wv.addWidget(self._wafer_canvas)
            wafer_hint = QLabel('Drag ↓ to zoom into box  ·  Drag ↑ to zoom out  ·  Home resets')
            wafer_hint.setStyleSheet('color: #555555; font-size: 10px;')
            wv.addWidget(wafer_hint)
        else:
            wv.addWidget(QLabel('Install matplotlib for wafer preview'))
        self._tabs.addTab(wafer_tab, 'Wafer Layout')

        # ── Mask layout tab ───────────────────────────────────────────────────
        mask_tab = QWidget()
        mv = QVBoxLayout(mask_tab)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel('Mask:'))
        self._mask_combo = QComboBox()
        self._mask_combo.setMinimumWidth(160)
        self._mask_combo.currentTextChanged.connect(self._draw_mask)
        sel_row.addWidget(self._mask_combo)
        self._home_btn = QPushButton('Home')
        self._home_btn.setFixedWidth(60)
        self._home_btn.setToolTip('Reset view to full image bounds')
        self._home_btn.clicked.connect(self._mask_home)
        sel_row.addWidget(self._home_btn)
        self._coord_label = QLabel('X: —   Y: —   µm')
        self._coord_label.setStyleSheet('color: #888888; font-size: 11px;')
        sel_row.addSpacing(16)
        sel_row.addWidget(self._coord_label)
        sel_row.addStretch()
        mv.addLayout(sel_row)
        if MATPLOTLIB_OK:
            self._mask_fig    = Figure(figsize=(7.0, 7.0), dpi=90)
            self._mask_ax     = self._mask_fig.add_subplot(111)
            self._mask_canvas = FigureCanvas(self._mask_fig)
            self._mask_fig.patch.set_facecolor('#1e1e1e')
            # Fixed margins, set once. tight_layout re-run per render fights
            # `aspect='equal'` — on a wide/short zoom window it collapsed the
            # axes box to a sliver. With the box pinned, `adjustable='datalim'`
            # keeps aspect by padding the data range instead.
            self._mask_fig.subplots_adjust(left=0.12, right=0.97,
                                           top=0.93, bottom=0.09)
            self._mask_canvas.mpl_connect('button_press_event',   self._zoom_press)
            self._mask_canvas.mpl_connect('motion_notify_event',  self._zoom_motion)
            self._mask_canvas.mpl_connect('button_release_event', self._zoom_release)
            mv.addWidget(self._mask_canvas)
            hint = QLabel('Drag ↓ to zoom into box  ·  Drag ↑ to zoom out  ·  Home resets')
            hint.setStyleSheet('color: #555555; font-size: 10px;')
            mv.addWidget(hint)
        else:
            mv.addWidget(QLabel('Install matplotlib for mask preview'))
        self._tabs.addTab(mask_tab, 'Mask Layout')

        v.addWidget(self._tabs)

    # ── Zoom / home / coordinate display ──────────────────────────────────────
    def _current_mask_cache_key(self, mask_id: str | None = None):
        if mask_id is None:
            mask_id = self._mask_combo.currentText()
        mask_d = next((m for m in self._state.masks if m['id'] == mask_id), None)
        if not mask_d or not mask_d.get('gds_file'):
            return None
        return (mask_d['gds_file'], mask_d.get('cell_name', ''))

    def _drop_detail_view(self, mask_id: str, cache_key,
                          keep_view: bool = False) -> bool:
        """Put the full-extent payload back if a windowed detail re-raster is
        on screen. Returns True when it re-rendered.

        With `keep_view` false (the Home button) `_render_poly_data` autoscales
        to the full data, so the caller need not set limits itself. With it true
        (zooming out a step) the caller's new limits are preserved — the detail
        payload only holds geometry from the window it was built for, so it has
        to go as soon as the view grows past that window, but where the user
        just zoomed to should not.

        Both ways out of a zoom have to come through here; when only the drag
        gesture did, Home left the zoom window's few hundred polygons stretched
        across the whole extent."""
        if cache_key in self._detail_active and cache_key in self._poly_cache:
            self._detail_active.discard(cache_key)
            self._render_poly_data(mask_id, self._poly_cache[cache_key],
                                   keep_view=keep_view)
            return True
        return False

    def _mask_home(self):
        if not MATPLOTLIB_OK:
            return
        mask_id   = self._mask_combo.currentText()
        cache_key = self._current_mask_cache_key(mask_id)
        if cache_key is not None and cache_key in self._huge_cells:
            # No cheap "full extent" for a cell we can't flatten — re-query the
            # whole bounding box instead of a real fit.
            self._viewport_cache.pop(cache_key, None)
            self._show_viewport_mask(mask_id, cache_key)
            return
        if self._drop_detail_view(mask_id, cache_key):
            return
        if self._mask_home_xlim is None:
            return
        self._mask_ax.set_xlim(self._mask_home_xlim)
        self._mask_ax.set_ylim(self._mask_home_ylim)
        self._mask_canvas.draw()

    def _zoom_press(self, event):
        if event.button != 1 or event.inaxes != self._mask_ax or event.xdata is None:
            return
        self._zoom_start = (event.xdata, event.ydata)
        color = '#4fc3f7'
        self._zoom_patch = MplRect(
            (event.xdata, event.ydata), 0.0, 0.0,
            linewidth=1.0, edgecolor=color, linestyle='--',
            facecolor=color, alpha=0.12, zorder=10, animated=True,
        )
        self._mask_ax.add_patch(self._zoom_patch)
        self._mask_canvas.draw()
        self._zoom_blit_bg = self._mask_canvas.copy_from_bbox(self._mask_fig.bbox)

    def _zoom_motion(self, event):
        if event.inaxes == self._mask_ax and event.xdata is not None:
            self._coord_label.setText(f'X: {event.xdata:.1f}   Y: {event.ydata:.1f}   µm')
        else:
            self._coord_label.setText('X: —   Y: —   µm')
        if self._zoom_start is None or event.xdata is None or event.inaxes != self._mask_ax:
            return
        x0, y0 = self._zoom_start
        x1, y1 = event.xdata, event.ydata
        self._zoom_patch.set_bounds(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self._mask_canvas.restore_region(self._zoom_blit_bg)
        self._mask_ax.draw_artist(self._zoom_patch)
        self._mask_canvas.blit(self._mask_fig.bbox)

    def _zoom_release(self, event):
        if self._zoom_start is None:
            return
        x0, y0 = self._zoom_start
        self._zoom_start  = None
        self._zoom_blit_bg = None
        if self._zoom_patch is not None:
            self._zoom_patch.remove()
            self._zoom_patch = None

        if event.xdata is None:
            self._mask_canvas.draw()
            return
        x1, y1 = event.xdata, event.ydata

        # Require a minimum drag distance to avoid accidental zoom on clicks
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        cur_w = abs(self._mask_ax.get_xlim()[1] - self._mask_ax.get_xlim()[0])
        if dx < cur_w * 0.01 and dy < cur_w * 0.01:
            self._mask_canvas.draw()
            return

        mask_id   = self._mask_combo.currentText()
        cache_key = self._current_mask_cache_key(mask_id)
        is_huge   = cache_key is not None and cache_key in self._huge_cells

        # Drag direction (data coords): down screen → y1 < y0 → zoom into box
        #                               up screen   → y1 > y0 → zoom back out
        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        prev_x, prev_y = self._mask_ax.get_xlim(), self._mask_ax.get_ylim()
        if y1 < y0:
            if box[2] <= box[0] or box[3] <= box[1]:
                self._mask_canvas.draw()
                return
            self._mask_ax.set_xlim(box[0], box[2])
            self._mask_ax.set_ylim(box[1], box[3])
        else:
            # A cell we can't flatten has no `_mask_home_xlim`; its own bounding
            # box is the natural stop instead.
            hx, hy = self._mask_home_xlim, self._mask_home_ylim
            if is_huge:
                (bx0, by0), (bx1, by1) = self._huge_cells[cache_key].bounding_box()
                hx, hy = (bx0, bx1), (by0, by1)
            nx, ny = _zoom_out_limits(self._mask_ax.get_xlim(),
                                      self._mask_ax.get_ylim(), box, hx, hy)
            self._mask_ax.set_xlim(nx)
            self._mask_ax.set_ylim(ny)

        # `aspect='equal'` pads whatever we asked for on one axis; query/raster
        # the box that will actually be *shown*, not the raw request, or the
        # result stops short of the visible edge.
        self._mask_ax.apply_aspect()
        (xmin, xmax) = self._mask_ax.get_xlim()
        (ymin, ymax) = self._mask_ax.get_ylim()

        # Already fully zoomed out and dragging up again: the clamp pinned the
        # limits where they were, so there is nothing new to query or raster.
        span = max(xmax - xmin, ymax - ymin)
        if (abs(xmin - prev_x[0]) + abs(xmax - prev_x[1])
                + abs(ymin - prev_y[0]) + abs(ymax - prev_y[1])) < span * 1e-9:
            self._mask_canvas.draw()
            return

        if is_huge:
            # Re-query just the newly-visible window; the current (coarser or
            # staler) artists stay up until that result lands.
            self._run_viewport_query(mask_id, cache_key,
                                     ((xmin, ymin), (xmax, ymax)))
        else:
            # A windowed detail payload only covers the window it was built
            # for, so zooming out has to fall back to the full-extent one —
            # but at the new limits, not back at home.
            if y1 >= y0:
                self._drop_detail_view(mask_id, cache_key, keep_view=True)
            self._maybe_detail_raster(mask_id, cache_key,
                                      (xmin, ymin, xmax, ymax))
        self._mask_canvas.draw()

    def _maybe_detail_raster(self, mask_id, cache_key, window):
        """After any zoom of a rasterised (truncated) non-huge mask, kick a
        background re-raster of just `window` (world x0,y0,x1,y1) at full
        resolution. The current wash stays up until the sharper one lands.
        Called on the way out of a zoom as well as into one, so a stepped
        zoom-out re-sharpens at each new window until the view is wide enough
        that the full-extent payload is the better answer anyway."""
        payload = self._poly_cache.get(cache_key)
        if not payload or not payload.get('truncated'):
            return
        if cache_key in self._detail_loading:
            return
        wx0, wy0, wx1, wy1 = window
        # Not worth it unless the window is a real zoom past the full extent.
        for _, (_, ext) in payload['density'].items():
            ex0, ey0, ex1, ey1 = ext
            if (wx1 - wx0) > 0.6 * (ex1 - ex0) and (wy1 - wy0) > 0.6 * (ey1 - ey0):
                return
            break
        gds_file, cell_name = cache_key
        self._detail_loading.add(cache_key)

        def _run():
            try:
                glib = read_gds(gds_file)
                cname = cell_name or glib.cells[0].name
                pl = _build_layer_payload(glib[cname],
                                          clip_bbox=((wx0, wy0), (wx1, wy1)))
                self._mask_detail.emit(dict(mask_id=mask_id, cache_key=cache_key,
                                            window=window, payload=pl))
            except Exception:
                # A swallowed failure here is indistinguishable from "this
                # window is legitimately empty" — the view just never sharpens.
                traceback.print_exc()
            finally:
                self._detail_loading.discard(cache_key)

        threading.Thread(target=_run, daemon=True).start()

    def _on_mask_detail(self, payload: dict):
        cache_key = payload['cache_key']
        if cache_key != self._current_mask_cache_key(self._mask_combo.currentText()):
            return
        base = self._poly_cache.get(cache_key)
        if not base:
            return
        # Only apply if the view is still roughly parked on the window we
        # re-rastered — tolerant of the aspect-ratio padding matplotlib adds to
        # set_xlim/set_ylim under `aspect='equal'`, but rejecting a genuine
        # re-zoom somewhere else while this was in flight.
        wx0, wy0, wx1, wy1 = payload['window']
        cx0, cx1 = self._mask_ax.get_xlim()
        cy0, cy1 = self._mask_ax.get_ylim()
        wspan = max(wx1 - wx0, wy1 - wy0)
        cspan = max(cx1 - cx0, cy1 - cy0)
        if (abs((cx0 + cx1 - wx0 - wx1) / 2) > wspan
                or abs((cy0 + cy1 - wy0 - wy1) / 2) > wspan
                or not (wspan / 3 <= cspan <= 3 * wspan)):
            return
        pl = payload['payload']
        if not pl['by_layer'] and not pl['density']:
            return
        # The windowed re-raster carries this window's real polygons (a label
        # array dense across the whole reticle is sparse here, so it comes back
        # as glyph shapes, not a wash) plus a sharper density for whatever's
        # still too dense. Swap both in; keep cname/title from the base.
        merged = dict(base, by_layer=pl['by_layer'], density=pl['density'],
                      total=pl['total'], n_layers=pl['n_layers'],
                      truncated=pl['truncated'])
        self._detail_active.add(cache_key)
        self._render_poly_data(payload['mask_id'], merged, keep_view=True)

    def update_all(self, state: 'AppState'):
        self._state = state
        self._draw_wafer_layout(state)

        mask_ids = [m['id'] for m in state.masks]
        prev = self._mask_combo.currentText()
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        self._mask_combo.addItems(mask_ids)
        if prev in mask_ids:
            self._mask_combo.setCurrentText(prev)
        elif mask_ids:
            self._mask_combo.setCurrentIndex(0)
        self._mask_combo.blockSignals(False)
        self._draw_mask(self._mask_combo.currentText())

    # ── Wafer layout rendering ────────────────────────────────────────────────
    def _draw_wafer_layout(self, state: 'AppState'):
        if not MATPLOTLIB_OK:
            return
        ax = self._wafer_ax
        ax.cla()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('#1e1e1e')
        self._wafer_fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

        w      = state.cell.get('wafer_w', FOUR_INCH)
        r_um   = WAFER_SIZES[w] * 500.0          # mm radius → µm
        flat_um = FLAT[w] * 1000.0

        # Wafer outline
        flat_y    = -flat_um
        flat_half = math.sqrt(max(0.0, r_um * r_um - flat_y * flat_y))
        theta1    = math.atan2(flat_y,  flat_half)
        theta2    = math.atan2(flat_y, -flat_half)
        if theta2 <= theta1:
            theta2 += 2 * math.pi
        thetas = np.linspace(theta1, theta2, 300)
        ax.fill(r_um * np.cos(thetas), r_um * np.sin(thetas),
                color='#2a2a2a', edgecolor='#666666', linewidth=1.2, zorder=1)

        # Alignment marks
        arm = r_um * 0.02
        for mk in state.marks:
            mx_um = mk.get('x', 0.0) * 1000.0
            my_um = mk.get('y', 0.0) * 1000.0
            ax.plot([mx_um - arm, mx_um + arm], [my_um, my_um],
                    color='#ffff44', lw=1.2, zorder=6)
            ax.plot([mx_um, mx_um], [my_um - arm, my_um + arm],
                    color='#ffff44', lw=1.2, zorder=6)

        layer_rot = {l['id']: l.get('rotation', 0.0) for l in state.layers}
        MAX_WAFER_POLYS = 150_000
        total_rendered  = 0
        # If a die's field would have to be decimated past this stride to fit
        # the wafer-wide poly budget, the leftover polygons are more misleading
        # than useful — switch that field to a labelled bounding box instead.
        STRIDE_LABEL_THRESHOLD = 8

        for ci, exp_d in enumerate(state.exposures):
            if total_rendered >= MAX_WAFER_POLYS:
                break
            lid      = exp_d.get('layer_id', '')
            iid      = exp_d.get('image_id', '')
            rotation = layer_rot.get(lid, 0.0)
            cos_a    = math.cos(math.radians(rotation))
            sin_a    = math.sin(math.radians(rotation))

            image_d = next((i for i in state.images if i['id'] == iid), None)
            if image_d is None:
                continue
            pat_pts = state.patterns.get(iid, [])
            if not pat_pts:
                continue

            shift_x_um  = image_d.get('shift_x',  0.0) * 1000.0
            shift_y_um  = image_d.get('shift_y',  0.0) * 1000.0
            origin_x_um = image_d.get('origin_x', 0.0) * 1000.0
            origin_y_um = image_d.get('origin_y', 0.0) * 1000.0
            size_x_um   = image_d.get('size_x',   0.0) * 1000.0
            size_y_um   = image_d.get('size_y',   0.0) * 1000.0
            img_x0 = shift_x_um - size_x_um / 2.0
            img_x1 = shift_x_um + size_x_um / 2.0
            img_y0 = shift_y_um - size_y_um / 2.0
            img_y1 = shift_y_um + size_y_um / 2.0

            color  = self._LAYER_COLORS[ci % len(self._LAYER_COLORS)]
            mask_d = next((m for m in state.masks
                           if m['id'] == image_d.get('mask_id', '')), None)

            field_polys  = None
            field_raster = None   # (array, (cx0,cx1,cy0,cy1)) in mask-local µm
            cell_label   = mask_d.get('cell_name', '') if mask_d else ''
            if mask_d and mask_d.get('gds_file'):
                ck = (mask_d['gds_file'], mask_d.get('cell_name', ''))
                m_sel     = mask_d.get('sel_layers')
                m_sel_set = set(m_sel) if m_sel is not None else None
                if ck in self._poly_cache:
                    cached      = self._poly_cache[ck]
                    cell_label  = cached.get('cname', cell_label)
                    raw = []
                    for lyr, pts_list in cached['by_layer'].items():
                        if m_sel_set is not None and lyr not in m_sel_set:
                            continue
                        for pts in pts_list:
                            pts = np.asarray(pts)
                            mn  = pts.min(axis=0)
                            mx2 = pts.max(axis=0)
                            if mx2[0] < img_x0 or mn[0] > img_x1 or \
                               mx2[1] < img_y0 or mn[1] > img_y1:
                                continue
                            raw.append(pts)
                    if raw:
                        field_polys = raw
                    # Layers too dense to have kept as shapes: crop the
                    # once-built coverage raster to this field's window and
                    # union any matching layers together.
                    for lyr, (arr, extent) in cached.get('density', {}).items():
                        if m_sel_set is not None and lyr not in m_sel_set:
                            continue
                        cropped = _crop_density(arr, extent, img_x0, img_y0, img_x1, img_y1)
                        if cropped is None:
                            continue
                        sub, box = cropped
                        if field_raster is None:
                            field_raster = (sub, box)
                        else:
                            prev_sub, prev_box = field_raster
                            if prev_sub.shape == sub.shape and prev_box == box:
                                field_raster = (np.maximum(prev_sub, sub), box)
                            else:
                                # All density layers share one frame, so this
                                # should be unreachable; keeping the larger
                                # crop at least never silently shows less.
                                if sub.size > prev_sub.size:
                                    field_raster = (sub, box)
                                print(f'[wafer] mixed-shape density crop on '
                                      f'{ck[1] or ck[0]} layer {lyr}: '
                                      f'{prev_sub.shape} vs {sub.shape}',
                                      file=sys.stderr)
                elif ck not in self._wafer_loading_keys and ck not in self._wafer_skip:
                    # Trigger background load so wafer view fills in automatically
                    self._wafer_loading_keys.add(ck)
                    def _load_wafer(gds=mask_d['gds_file'],
                                    cname=mask_d.get('cell_name', ''),
                                    key=ck, mid=mask_d['id']):
                        try:
                            glib   = read_gds(gds)
                            cname2 = cname or glib.cells[0].name
                            cell   = glib[cname2]
                            if _approx_instances(cell) > _MAX_MASK_INSTANCES:
                                # Nothing to cache for a cell this size, so
                                # record it as tried — otherwise every wafer
                                # redraw spawns another thread that re-reads
                                # the file and re-counts, forever.
                                self._wafer_skip.add(key)
                                self._wafer_loading_keys.discard(key)
                                return
                            payload = _build_layer_payload(cell)
                            self._mask_ready.emit(dict(
                                mask_id=mid, cache_key=key, cname=cname2, **payload,
                            ))
                        except Exception:
                            traceback.print_exc()
                            self._wafer_skip.add(key)
                            self._wafer_loading_keys.discard(key)
                    threading.Thread(target=_load_wafer, daemon=True).start()

            use_label_fallback = field_polys is None and field_raster is None
            if field_polys is not None:
                # Budget per die: share remaining capacity across remaining exposures
                n_dies     = len(pat_pts)
                remaining  = MAX_WAFER_POLYS - total_rendered
                expo_left  = max(1, len(state.exposures) - ci)
                per_die    = max(1, remaining // (n_dies * expo_left))
                if len(field_polys) > per_die:
                    step = max(1, len(field_polys) // per_die)
                    if step > STRIDE_LABEL_THRESHOLD and field_raster is None:
                        # Decimation would throw away almost everything — a
                        # labelled box is more honest than the sparse leftover.
                        use_label_fallback = True
                        field_polys = None
                    else:
                        field_polys = field_polys[::step]

            if field_raster is not None:
                # Density wash: rasterized once per exposure, stamped at every
                # die via a rotated affine transform — cheap regardless of how
                # many polygons the original layer had.
                sub, (cx0, cx1, cy0, cy1) = field_raster
                r, g, b = to_rgb(color)
                rgba = np.zeros((*sub.shape, 4), dtype=float)
                rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
                rgba[..., 3] = (sub.astype(float) / 255.0) * 0.5
                local_extent = (cx0 - origin_x_um, cx1 - origin_x_um,
                                 cy0 - origin_y_um, cy1 - origin_y_um)
                for pt in pat_pts:
                    px = pt['x'] * 1000.0
                    py = pt['y'] * 1000.0
                    trans = Affine2D().rotate_deg(rotation).translate(px, py) + ax.transData
                    ax.imshow(rgba, extent=local_extent, origin='upper',
                              transform=trans, zorder=3)
                total_rendered += 1

            if field_polys is not None:
                # Mask polygons are in GDS coords centred at image shift.
                # Subtracting origin*1000 puts the image origin at (0,0) so the
                # layer rotation pivots about the die control point, then the
                # pattern point pt*1000 translates the rotated field into place —
                # (shift + pt - origin)*1000, matching ASML.
                transformed = []
                for pt in pat_pts:
                    px = pt['x'] * 1000.0
                    py = pt['y'] * 1000.0
                    for pts in field_polys:
                        pts = np.asarray(pts, dtype=float)
                        u   = pts[:, 0] - origin_x_um
                        v   = pts[:, 1] - origin_y_um
                        transformed.append(np.column_stack([
                            u * cos_a - v * sin_a + px,
                            u * sin_a + v * cos_a + py,
                        ]))
                if transformed:
                    coll = PolyCollection(transformed, facecolors=color,
                                         edgecolors=color, alpha=0.40,
                                         linewidths=0.3, zorder=3)
                    ax.add_collection(coll)
                    total_rendered += len(transformed)
            if use_label_fallback:
                # Fallback: rotated image-field rectangles at each die, labelled
                # with the mask cell name — used when there's no cached geometry
                # yet, or when the real geometry is too dense to decimate usefully.
                rects  = []
                label  = cell_label or (mask_d.get('id', '') if mask_d else '') \
                         or image_d.get('id', '')
                for pt in pat_pts:
                    px = pt['x'] * 1000.0
                    py = pt['y'] * 1000.0
                    ox = -origin_x_um
                    oy = -origin_y_um
                    corners = np.array([
                        [img_x0 + ox, img_y0 + oy],
                        [img_x1 + ox, img_y0 + oy],
                        [img_x1 + ox, img_y1 + oy],
                        [img_x0 + ox, img_y1 + oy],
                    ])
                    rects.append(np.column_stack([
                        corners[:, 0] * cos_a - corners[:, 1] * sin_a + px,
                        corners[:, 0] * sin_a + corners[:, 1] * cos_a + py,
                    ]))
                    if label:
                        ax.text(px, py, label, ha='center', va='center',
                                fontsize=6, color='#dddddd', zorder=4,
                                clip_on=True)
                coll = PolyCollection(rects, facecolors=color, edgecolors=color,
                                     alpha=0.25, linewidths=0.8, zorder=3)
                ax.add_collection(coll)

        ax.set_xlim(-r_um * 1.05, r_um * 1.05)
        ax.set_ylim(-r_um * 1.05, r_um * 1.05)
        self._wafer_home_xlim = (-r_um * 1.05, r_um * 1.05)
        self._wafer_home_ylim = (-r_um * 1.05, r_um * 1.05)

        title = f'{WAFER_SIZES[w]:.0f} mm wafer'
        if total_rendered:
            title += f'  ({total_rendered:,} polygons)'
        elif state.exposures:
            title += '  (loading GDS…)'
        ax.set_title(title, color='#aaaaaa', fontsize=9, pad=4)
        ax.set_xlabel('µm', color='#888888', fontsize=8)
        ax.set_ylabel('µm', color='#888888', fontsize=8)
        self._wafer_fig.tight_layout(pad=0.5)
        self._wafer_canvas.draw()

    # ── Wafer zoom / home ─────────────────────────────────────────────────────
    def _wafer_home(self):
        if not MATPLOTLIB_OK or self._wafer_home_xlim is None:
            return
        self._wafer_ax.set_xlim(self._wafer_home_xlim)
        self._wafer_ax.set_ylim(self._wafer_home_ylim)
        self._wafer_canvas.draw()

    def _wafer_zoom_press(self, event):
        if event.button != 1 or event.inaxes != self._wafer_ax or event.xdata is None:
            return
        self._wafer_zoom_start = (event.xdata, event.ydata)
        color = '#4fc3f7'
        self._wafer_zoom_patch = MplRect(
            (event.xdata, event.ydata), 0.0, 0.0,
            linewidth=1.0, edgecolor=color, linestyle='--',
            facecolor=color, alpha=0.12, zorder=10, animated=True,
        )
        self._wafer_ax.add_patch(self._wafer_zoom_patch)
        self._wafer_canvas.draw()
        self._wafer_zoom_blit_bg = self._wafer_canvas.copy_from_bbox(self._wafer_fig.bbox)

    def _wafer_zoom_motion(self, event):
        if event.inaxes == self._wafer_ax and event.xdata is not None:
            self._wafer_coord_label.setText(
                f'X: {event.xdata:.1f}   Y: {event.ydata:.1f}   µm')
        else:
            self._wafer_coord_label.setText('X: —   Y: —   µm')
        if self._wafer_zoom_start is None or event.xdata is None \
                or event.inaxes != self._wafer_ax:
            return
        x0, y0 = self._wafer_zoom_start
        x1, y1 = event.xdata, event.ydata
        self._wafer_zoom_patch.set_bounds(
            min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self._wafer_canvas.restore_region(self._wafer_zoom_blit_bg)
        self._wafer_ax.draw_artist(self._wafer_zoom_patch)
        self._wafer_canvas.blit(self._wafer_fig.bbox)

    def _wafer_zoom_release(self, event):
        if self._wafer_zoom_start is None:
            return
        x0, y0 = self._wafer_zoom_start
        self._wafer_zoom_start   = None
        self._wafer_zoom_blit_bg = None
        if self._wafer_zoom_patch is not None:
            self._wafer_zoom_patch.remove()
            self._wafer_zoom_patch = None
        if event.xdata is None:
            self._wafer_canvas.draw()
            return
        x1, y1 = event.xdata, event.ydata
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        cur_w  = abs(self._wafer_ax.get_xlim()[1] - self._wafer_ax.get_xlim()[0])
        if dx < cur_w * 0.01 and dy < cur_w * 0.01:
            self._wafer_canvas.draw()
            return
        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if y1 < y0:   # dragged down → zoom into box
            if box[2] > box[0] and box[3] > box[1]:
                self._wafer_ax.set_xlim(box[0], box[2])
                self._wafer_ax.set_ylim(box[1], box[3])
        else:          # dragged up → zoom back out a step (Home still resets)
            nx, ny = _zoom_out_limits(self._wafer_ax.get_xlim(),
                                      self._wafer_ax.get_ylim(), box,
                                      self._wafer_home_xlim, self._wafer_home_ylim)
            self._wafer_ax.set_xlim(nx)
            self._wafer_ax.set_ylim(ny)
        self._wafer_canvas.draw()

    def _draw_mask(self, mask_id: str):
        if not MATPLOTLIB_OK:
            return
        ax = self._mask_ax
        ax.cla()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('#1e1e1e')
        self._mask_fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

        mask_d = next((m for m in self._state.masks if m['id'] == mask_id), None)
        if not mask_d:
            ax.text(0.5, 0.5, 'No mask selected', transform=ax.transAxes,
                    ha='center', va='center', color='#888888', fontsize=12)
            self._mask_canvas.draw()
            return

        gds_file = mask_d.get('gds_file', '')
        if not gds_file:
            ax.text(0.5, 0.5, f'Mask "{mask_id}"\nNo GDS file loaded',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#888888', fontsize=12)
            self._mask_canvas.draw()
            return

        if not GDSTK_OK:
            ax.text(0.5, 0.5, 'gdstk not installed',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#ff6b6b', fontsize=12)
            self._mask_canvas.draw()
            return

        cell_name = mask_d.get('cell_name', '')
        cache_key = (gds_file, cell_name)

        if cache_key in self._poly_cache:
            self._render_poly_data(mask_id, self._poly_cache[cache_key])
            return

        if cache_key in self._huge_cells:
            self._show_viewport_mask(mask_id, cache_key)
            return

        # Show loading indicator and start background thread
        ax.text(0.5, 0.5, 'Loading GDS…\nPlease wait',
                transform=ax.transAxes, ha='center', va='center',
                color='#aaaaaa', fontsize=13)
        self._mask_canvas.draw()

        def _load():
            try:
                glib  = read_gds(gds_file)
                cname = cell_name or glib.cells[0].name
                cell  = glib[cname]
                ninst = _approx_instances(cell)
                if ninst > _MAX_MASK_INSTANCES:
                    # Too big to ever flatten — keep the cell handle (cheap,
                    # it's the reference hierarchy, not flattened geometry)
                    # and switch to viewport-bounded live queries instead.
                    self._mask_huge.emit(mask_id, cache_key, cell)
                    return
                payload = _build_layer_payload(cell)
                self._mask_ready.emit(dict(
                    mask_id=mask_id, cache_key=cache_key, cname=cname, **payload,
                ))
            except Exception as exc:
                self._mask_error.emit(mask_id, str(exc))

        threading.Thread(target=_load, daemon=True).start()

    def _on_mask_huge(self, mask_id: str, cache_key: tuple, cell) -> None:
        self._huge_cells[cache_key] = cell
        self._wafer_loading_keys.discard(cache_key)
        if cache_key == self._current_mask_cache_key():
            self._show_viewport_mask(mask_id, cache_key)

    def _show_viewport_mask(self, mask_id: str, cache_key: tuple) -> None:
        """Display (or kick off) a viewport-bounded live query for a cell too
        large to flatten in full. Reuses the last-viewed window if we have
        one cached; otherwise queries the cell's *whole* bounding box — real
        geometry wherever a subtree is cheap enough, a labelled placeholder
        box (see `_query_cell_bbox`'s `skipped`) wherever it isn't, so the
        first view is an honest overview of the whole design rather than an
        arbitrary small fragment of it."""
        cached = self._viewport_cache.get(cache_key)
        if cached is not None:
            self._render_viewport_data(mask_id, cached)
            return
        cell = self._huge_cells[cache_key]
        self._run_viewport_query(mask_id, cache_key, cell.bounding_box())

    def _run_viewport_query(self, mask_id: str, cache_key: tuple, bbox: tuple) -> None:
        """Query `bbox` in the background. While one query is in flight the
        newest further request is remembered and issued when it lands — zooming
        (or pressing Home) during a query used to be dropped silently."""
        self._viewport_seq += 1
        seq = self._viewport_seq
        self._viewport_req[cache_key] = seq
        if cache_key in self._viewport_loading:
            self._viewport_pending[cache_key] = (mask_id, bbox)
            return
        self._viewport_loading.add(cache_key)
        cell = self._huge_cells[cache_key]
        self._note_viewport_busy()

        def _run():
            # Emit on failure too: `_on_viewport_ready` is what clears
            # `_viewport_loading`, so a swallowed exception here would wedge
            # this cell into "always loading" for the rest of the session.
            try:
                by_layer, stats = _query_cell_bbox(cell, bbox, budget=300_000)
            except Exception as exc:
                traceback.print_exc()
                by_layer = {}
                stats = dict(visited=0, incomplete=True, leaf_polys=0,
                             bailed_refs=0, skipped=[], error=str(exc))
            self._viewport_ready.emit(dict(
                mask_id=mask_id, cache_key=cache_key, bbox=bbox,
                by_layer=by_layer, stats=stats, seq=seq,
            ))

        threading.Thread(target=_run, daemon=True).start()

    def _note_viewport_busy(self, busy: bool = True) -> None:
        """Mark the panel as refreshing. The previous (coarser) artists stay up
        while a query runs, so without this there is nothing on screen to say
        anything is happening."""
        if not MATPLOTLIB_OK:
            return
        title = self._mask_ax.get_title()
        has   = title.endswith(self._BUSY_SUFFIX)
        if busy and title and not has:
            title += self._BUSY_SUFFIX
        elif not busy and has:
            title = title[:-len(self._BUSY_SUFFIX)]
        else:
            return
        self._mask_ax.set_title(title, color='#aaaaaa', fontsize=9, pad=4)
        self._mask_canvas.draw_idle()

    def _on_viewport_ready(self, payload: dict) -> None:
        cache_key = payload['cache_key']
        # Clearing the in-flight flag here (not in the worker) keeps it true for
        # the whole round trip, so `_viewport_pending` really does hold every
        # request made while the query was running.
        self._viewport_loading.discard(cache_key)
        stale = payload.get('seq') != self._viewport_req.get(cache_key)
        if not stale:
            self._viewport_cache[cache_key] = payload
            if cache_key == self._current_mask_cache_key():
                self._render_viewport_data(payload['mask_id'], payload)
        pending = self._viewport_pending.pop(cache_key, None)
        if pending is not None:
            self._run_viewport_query(pending[0], cache_key, pending[1])
        else:
            self._note_viewport_busy(False)

    def _render_viewport_data(self, mask_id: str, payload: dict) -> None:
        """Render an exact, viewport-bounded query result for a cell too
        large to ever flatten — the view's own bounds are the query's bounds
        (no data-driven autoscale), so zoom/pan stays exactly where the user
        left it."""
        if not MATPLOTLIB_OK:
            return
        ax = self._mask_ax
        ax.cla()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('#1e1e1e')
        self._mask_fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

        mask_d  = next((m for m in self._state.masks if m['id'] == mask_id), None)
        sel     = mask_d.get('sel_layers') if mask_d else None
        sel_set = set(sel) if sel is not None else None

        by_layer = payload['by_layer']
        stats    = payload['stats']
        (x0, y0), (x1, y1) = payload['bbox']
        vx, vy = x1 - x0, y1 - y0

        _lcolor = self._layer_color

        for lyr, pts_list in sorted(by_layer.items()):
            if sel_set is not None and lyr not in sel_set:
                continue
            color = _lcolor(lyr)
            # Every enabled layer's real geometry is drawn as a fill. A polygon
            # far bigger than the viewport on both axes is context (a reticle /
            # field boundary that merely encloses the view) — still filled, so
            # "layer N is here" reads, but at a lighter alpha so it doesn't
            # swamp finer content, and the too-dense-subtree placeholders sit
            # on top of it (higher zorder).
            small, big = [], []
            for p in pts_list:
                p = np.asarray(p)
                enc = (p[:, 0].max() - p[:, 0].min() > 2 * vx
                       and p[:, 1].max() - p[:, 1].min() > 2 * vy)
                (big if enc else small).append(p)
            if small:
                ax.add_collection(PolyCollection(
                    small, facecolors=color, edgecolors=color,
                    alpha=0.35, linewidths=0.5, zorder=2))
            if big:
                ax.add_collection(PolyCollection(
                    big, facecolors=color, edgecolors=color,
                    alpha=0.15, linewidths=0.8, zorder=2))

        # Subtrees too dense to resolve at this zoom: fill the region in the
        # colour of the (lowest enabled) layer it draws on — same "coverage
        # wash" read as the rasterised path, so a via array reads as "layer N
        # is solidly here" — drawn ABOVE the layer fills (zorder 3) so it's
        # never buried. A subtree whose layers are all off (or unknown) gets a
        # faint grey outline only.
        skipped = stats.get('skipped', [])
        for entry in skipped:
            bbox_s, name = entry[0], entry[1]
            lyrs = entry[2] if len(entry) > 2 else frozenset()
            on = sorted(l for l in lyrs
                        if sel_set is None or l in sel_set)
            (sx0, sy0), (sx1, sy1) = bbox_s
            if on:
                ax.add_patch(MplRect(
                    (sx0, sy0), sx1 - sx0, sy1 - sy0,
                    linewidth=1.0, edgecolor=_lcolor(on[0]), linestyle=':',
                    facecolor=_lcolor(on[0]), alpha=0.30, hatch='////', zorder=3))
            else:
                ax.add_patch(MplRect(
                    (sx0, sy0), sx1 - sx0, sy1 - sy0,
                    linewidth=1.0, edgecolor='#888888', linestyle=':',
                    facecolor='none', zorder=3))
            # Label over the *visible* part of the box (its true centre is
            # off-screen once you've zoomed inside the array).
            lx = (max(sx0, x0) + min(sx1, x1)) / 2
            ly = (max(sy0, y0) + min(sy1, y1)) / 2
            ax.text(lx, ly, name, ha='center', va='center', fontsize=6,
                    color='#dddddd', zorder=4, clip_on=True)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        n    = sum(len(v) for v in by_layer.values())
        note = f'  [{len(skipped)} subtree(s) too dense — zoom in on them for detail]' \
               if skipped else ''
        ax.set_title(f'Mask: {mask_id}  (cell too large to flatten fully — live '
                     f'viewport query, {n} polygons{note})',
                     color='#aaaaaa', fontsize=9, pad=4)
        ax.set_xlabel('µm', color='#888888', fontsize=8)
        ax.set_ylabel('µm', color='#888888', fontsize=8)
        self._mask_canvas.draw()

    def _on_mask_ready(self, payload: dict):
        self._poly_cache[payload['cache_key']] = payload
        self._wafer_loading_keys.discard(payload['cache_key'])
        if payload['cache_key'] == self._current_mask_cache_key():
            self._render_poly_data(payload['mask_id'], payload)
        if MATPLOTLIB_OK and self._tabs.currentIndex() == 0:
            self._draw_wafer_layout(self._state)

    def _on_mask_error(self, mask_id: str, error: str):
        if not MATPLOTLIB_OK:
            return
        ax = self._mask_ax
        ax.cla()
        ax.set_facecolor('#1e1e1e')
        self._mask_fig.patch.set_facecolor('#1e1e1e')
        ax.text(0.5, 0.5, error, transform=ax.transAxes,
                ha='center', va='center', color='#ff6b6b', fontsize=9)
        self._mask_canvas.draw()

    def _render_poly_data(self, mask_id: str, data: dict, keep_view: bool = False):
        """Render pre-processed polygon data via PolyCollection (fast path).
        `keep_view` preserves the current zoom/pan (used when swapping in a
        windowed detail re-raster) instead of autoscaling to the data."""
        if not MATPLOTLIB_OK:
            return
        ax = self._mask_ax
        saved_xlim = ax.get_xlim() if keep_view else None
        saved_ylim = ax.get_ylim() if keep_view else None
        ax.cla()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('#1e1e1e')
        self._mask_fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

        by_layer  = data['by_layer']
        density   = data.get('density', {})
        cname     = data['cname']
        total     = data['total']
        n_layers  = data['n_layers']
        truncated = data.get('truncated', False)

        if not by_layer and not density:
            ax.text(0.5, 0.5, 'No polygons found', transform=ax.transAxes,
                    ha='center', va='center', color='#888888', fontsize=12)
            self._mask_canvas.draw()
            return

        mask_d  = next((m for m in self._state.masks if m['id'] == mask_id), None)
        sel     = mask_d.get('sel_layers') if mask_d else None
        sel_set = set(sel) if sel is not None else None

        all_pts: list = []
        for lyr, pts_list in sorted(by_layer.items()):
            if sel_set is not None and lyr not in sel_set:
                continue
            color = self._layer_color(lyr)
            coll  = PolyCollection(pts_list, facecolors=color, edgecolors=color,
                                   alpha=0.35, linewidths=0.5, zorder=2)
            ax.add_collection(coll)
            all_pts.extend(pts_list)

        # Layers too dense to draw as shapes — a coverage-density wash instead
        # of stride-sampled leftovers (see _build_layer_payload). Same colour
        # rule as the vector layers, so a layer that switches between the two
        # representations on zoom keeps its identity.
        for lyr, (arr, extent) in sorted(density.items()):
            if sel_set is not None and lyr not in sel_set:
                continue
            color = self._layer_color(lyr)
            r, g, b = to_rgb(color)
            rgba = np.zeros((*arr.shape, 4), dtype=float)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
            rgba[..., 3] = (arr.astype(float) / 255.0) * 0.55
            x0, y0, x1, y1 = extent
            ax.imshow(rgba, extent=(x0, x1, y0, y1), origin='upper', zorder=2)
            all_pts.append(np.array([[x0, y0], [x1, y1]]))

        # ── Image field overlays (wafer scale: mm → µm) ──────────────────────
        UM = 1000.0  # mm → µm
        img_bounds: list = []
        for ci, img_d in enumerate(self._state.images):
            if img_d.get('mask_id') != mask_id:
                continue
            color = self._LAYER_COLORS[ci % len(self._LAYER_COLORS)]
            fw  = img_d.get('size_x', 0.0) * UM
            fh  = img_d.get('size_y', 0.0) * UM
            fcx = img_d.get('shift_x', 0.0) * UM
            fcy = img_d.get('shift_y', 0.0) * UM
            ax.add_patch(MplRect(
                (fcx - fw / 2, fcy - fh / 2), fw, fh,
                linewidth=1.5, edgecolor=color, linestyle='--',
                facecolor=color, alpha=0.12, zorder=4,
            ))
            ax.text(fcx, fcy, img_d['id'],
                    ha='center', va='center', color=color, fontsize=8, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e1e',
                              alpha=0.6, edgecolor='none'))
            img_bounds.append((fcx - fw / 2, fcy - fh / 2,
                               fcx + fw / 2, fcy + fh / 2))

        # Expand axis limits to include image rectangles
        if keep_view and saved_xlim is not None:
            ax.set_xlim(saved_xlim)
            ax.set_ylim(saved_ylim)
        elif all_pts or img_bounds:
            pts_arr = np.vstack(all_pts) if all_pts else np.empty((0, 2))
            if img_bounds:
                corners = np.array([[x0, y0] for x0, y0, _, _ in img_bounds] +
                                   [[x1, y1] for _, _, x1, y1 in img_bounds])
                pts_arr = np.vstack([pts_arr, corners]) if all_pts else corners
            xmin, ymin = pts_arr.min(axis=0)
            xmax, ymax = pts_arr.max(axis=0)
            m = max(xmax - xmin, ymax - ymin) * 0.03 + 0.01
            ax.set_xlim(xmin - m, xmax + m)
            ax.set_ylim(ymin - m, ymax + m)
            self._mask_home_xlim = (xmin - m, xmax + m)
            self._mask_home_ylim = (ymin - m, ymax + m)

        title = (f'Mask: {mask_id}  (cell: {cname}, '
                 f'{total} polygons, {n_layers} layers'
                 + (f'  [{len(density)} rasterized]' if truncated else '') + ')')
        ax.set_title(title, color='#aaaaaa', fontsize=9, pad=4)
        ax.set_xlabel('µm', color='#888888', fontsize=8)
        ax.set_ylabel('µm', color='#888888', fontsize=8)
        self._mask_canvas.draw()


# ── MainWindow ─────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1000, 700)
        self._current_file: str | None = None
        self._state = AppState()
        self._graphics_win = None
        self._setup_ui()
        self._update_title()
        if not BACKEND_OK:
            QMessageBox.critical(
                self, 'Import Error',
                f'Could not import asml18:\n{BACKEND_ERROR}\n\n'
                'Ensure asml18.py is in the same directory and all dependencies are installed.\n'
                '  python3.12 -m pip install gdstk'
            )

    def _setup_ui(self):
        mb = self.menuBar()
        file_menu = mb.addMenu('File')
        _acts = [
            ('New',      'Ctrl+N',       self._new_file),
            ('Open…',    'Ctrl+O',       self._open_file),
            ('Save',     'Ctrl+S',       self._save_file),
            ('Save As…', 'Ctrl+Shift+S', self._save_file_as),
        ]
        for label, shortcut, slot in _acts:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            file_menu.addAction(act)

        central = QWidget(); self.setCentralWidget(central)
        outer   = QHBoxLayout(central)

        self._tabs = QTabWidget()
        self._cell_tab    = CellTab    (self._state, self._on_state_changed)
        self._image_tab   = ImageTab   (self._state, self._on_state_changed)
        self._pattern_tab = PatternTab (self._state, self._on_state_changed)
        self._layer_tab   = LayerTab   (self._state, self._on_state_changed)
        self._align_tab   = AlignmentTab(self._state, self._on_state_changed)
        self._process_tab = ProcessTab (self._state, self._on_state_changed)
        for tab, title in (
            (self._cell_tab,    'Cell'),
            (self._image_tab,   'Image'),
            (self._pattern_tab, 'Pattern'),
            (self._layer_tab,   'Layer'),
            (self._align_tab,   'Alignment'),
            (self._process_tab, 'Process'),
        ):
            self._tabs.addTab(tab, title)
        self._tabs.currentChanged.connect(self._on_main_tab_changed)
        outer.addWidget(self._tabs)

        sb = self.statusBar()
        self._graphics_btn = QPushButton('Show Graphics')
        self._graphics_btn.clicked.connect(self._show_graphics)
        sb.addPermanentWidget(self._graphics_btn)
        self._gen_job_btn = QPushButton('Generate Job')
        self._gen_job_btn.clicked.connect(self._on_generate_job)
        sb.addPermanentWidget(self._gen_job_btn)
        self._gen_gds_btn = QPushButton('Generate GDS')
        self._gen_gds_btn.clicked.connect(self._on_generate_gds)
        sb.addPermanentWidget(self._gen_gds_btn)
        self._corr_btn = QPushButton('Correlate Marks')
        self._corr_btn.setToolTip('Rotated-marks mode: check each rotated mark '
                                  'group registers with an axis-aligned PM.')
        self._corr_btn.clicked.connect(self._on_correlate_marks)
        sb.addPermanentWidget(self._corr_btn)
        self._status = QLabel('Ready')
        sb.addWidget(self._status)

    def _on_state_changed(self):
        for tab in (self._cell_tab, self._image_tab, self._pattern_tab,
                    self._layer_tab, self._align_tab, self._process_tab):
            tab.refresh()
        if self._graphics_win is not None and self._graphics_win.isVisible():
            self._graphics_win.update_all(self._state)

    def _show_graphics(self):
        if self._graphics_win is None:
            self._graphics_win = GraphicsWindow(self._state, self)
        self._graphics_win.update_all(self._state)
        self._graphics_win.show()
        self._graphics_win.raise_()
        self._graphics_win.activateWindow()

    def _on_main_tab_changed(self, index: int):
        if self._graphics_win is None or not self._graphics_win.isVisible():
            return
        # Image tab → switch Graphics to Mask Layout; anything else → Wafer Layout
        self._graphics_win._tabs.setCurrentIndex(1 if index == 1 else 0)

    # ── title / file helpers ───────────────────────────────────────────────────
    def _update_title(self):
        base = os.path.basename(self._current_file) if self._current_file else 'Untitled'
        self.setWindowTitle(f'ASML Job File Generator — {base}')

    def _new_file(self):
        reply = QMessageBox.question(
            self, 'New', 'Discard current state and start fresh?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._state.__dict__.update(AppState().__dict__)
        self._current_file = None
        self._update_title()
        self._on_state_changed()

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open job state', '',
            'ASML Job State (*.asmlj);;JSON Files (*.json);;All Files (*)',
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            self._state.__dict__.update(_deserialize_state(d).__dict__)
            self._current_file = path
            self._update_title()
            self._on_state_changed()
        except Exception as exc:
            QMessageBox.critical(self, 'Open failed', str(exc))

    def _save_file(self):
        if not self._current_file:
            self._save_file_as()
        else:
            self._write_file(self._current_file)

    def _save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save job state', self._current_file or 'untitled.asmlj',
            'ASML Job State (*.asmlj);;JSON Files (*.json);;All Files (*)',
        )
        if not path:
            return
        if self._write_file(path):
            self._current_file = path
            self._update_title()

    def _write_file(self, path: str) -> bool:
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(_serialize_state(self._state), fh, indent=2)
            self._status.setText(f'Saved → {path}')
            return True
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return False

    # ── generate ───────────────────────────────────────────────────────────────
    def _job_output_base(self) -> str | None:
        """Directory + stem of the current job file, used as the base name for
        generated GDS and stepper-job files.  Prompts to save the job first if
        no job name is defined yet; returns None if the user cancels."""
        if not self._current_file:
            QMessageBox.information(
                self, 'Name the job',
                'Save the job file first — its name is used for the GDS and '
                'stepper-job output.')
            self._save_file_as()
            if not self._current_file:
                return None
        return os.path.splitext(self._current_file)[0]

    def _validate_for_generate(self) -> str | None:
        """Return the output base path if valid, else show a warning / prompt."""
        if not BACKEND_OK:
            QMessageBox.critical(self, 'Error', 'Backend (asml18) not available.')
            return None
        if not self._state.layers or not self._state.exposures:
            QMessageBox.warning(self, 'Incomplete',
                                'Define at least one layer and one exposure.')
            return None
        if not self._check_dies_compliance():
            return None
        return self._job_output_base()

    def _dies_violations(self) -> list[tuple[str, float, float]]:
        c = self._state.cell
        wafer_w, edge_excl, flat_edge = (c.get('wafer_w', FOUR_INCH),
                                          c.get('edge_excl', 3.0), c.get('flat_edge', 0.0))
        dies_x, dies_y, min_dies = c.get('dies_x', 1), c.get('dies_y', 1), c.get('min_dies', 0)
        out = []
        for img in self._state.images:
            sx, sy = img['size_x'], img['size_y']
            for pt in self._state.patterns.get(img['id'], []):
                if not _dies_compliant(pt['x'], pt['y'], sx, sy, wafer_w, edge_excl,
                                        flat_edge, dies_x, dies_y, min_dies):
                    out.append((img['id'], pt['x'], pt['y']))
        return out

    def _check_dies_compliance(self) -> bool:
        """Pre-generate pass: every pattern point (grid/python/csv, any image)
        must satisfy the dies-per-cell wafer-coverage rule. On a violation, let
        the user move the offending placements inward to the nearest compliant
        position, or lower minimum-dies-per-cell just enough to accept them
        as-is — either way generation can then proceed. Returns False only on
        Cancel or when neither remedy can actually make a placement compliant."""
        violations = self._dies_violations()
        if not violations:
            return True

        c = self._state.cell
        wafer_w, edge_excl, flat_edge = (c.get('wafer_w', FOUR_INCH),
                                          c.get('edge_excl', 3.0), c.get('flat_edge', 0.0))
        dies_x, dies_y, min_dies = c.get('dies_x', 1), c.get('dies_y', 1), c.get('min_dies', 0)
        img_by_id = {i['id']: i for i in self._state.images}

        r_eff = WAFER_SIZES[wafer_w] / 2 - edge_excl
        listing = '\n'.join(
            f'  {iid}  ({x:.3f}, {y:.3f}) mm — {math.hypot(x, y):.3f}mm from '
            f'center, {math.hypot(x, y) - r_eff:.3f}mm past the {r_eff:.3f}mm '
            f'effective radius'
            for iid, x, y in violations[:20])
        if len(violations) > 20:
            listing += f'\n  … and {len(violations) - 20} more'
        box = QMessageBox(self)
        box.setWindowTitle('Placement does not comply')
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"{len(violations)} placement(s) don't satisfy the minimum-dies-per-cell "
            f'rule (min_dies={min_dies} of {dies_x}×{dies_y}). Wafer radius '
            f'{WAFER_SIZES[wafer_w] / 2:.3f}mm minus edge exclusion {edge_excl:.3f}mm '
            f'= {r_eff:.3f}mm effective radius, which edge exclusion always enforces '
            f'regardless of minimum-dies-per-cell:\n\n'
            f'{listing}\n\n'
            'Move the placements inward to the nearest compliant position, or lower '
            'minimum-dies-per-cell just enough to allow them as-is?')
        move_btn   = box.addButton('Move Placements', QMessageBox.ButtonRole.AcceptRole)
        adjust_btn = box.addButton('Adjust Limit',     QMessageBox.ButtonRole.AcceptRole)
        box.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()

        if clicked is move_btn:
            for iid, x, y in violations:
                img = img_by_id[iid]
                moved = _move_inward_to_compliant(
                    x, y, img['size_x'], img['size_y'], wafer_w, edge_excl, flat_edge,
                    dies_x, dies_y, min_dies)
                if moved is None:
                    QMessageBox.critical(self, 'Cannot comply',
                        f"Image '{iid}': even the wafer center doesn't satisfy the "
                        'current edge exclusion / minimum-dies setting — lower the '
                        'limits instead.')
                    return False
                for pt in self._state.patterns.get(iid, []):
                    if (pt['x'], pt['y']) == (x, y):
                        pt['x'], pt['y'] = moved
                        break
            self._on_state_changed()
            return True

        if clicked is adjust_btn:
            def _all_pass(md: int) -> bool:
                return all(
                    _dies_compliant(x, y, img_by_id[iid]['size_x'], img_by_id[iid]['size_y'],
                                     wafer_w, edge_excl, flat_edge, dies_x, dies_y, md)
                    for iid, x, y in violations)
            # min_dies is strictly more permissive as it decreases (min_dies == 0
            # is looser still, via the rect-intersect special case) — walk down
            # from the current value to the largest one that accepts every
            # flagged placement, so the limit is loosened as little as possible.
            candidate = next((md for md in range(min_dies - 1, -1, -1) if _all_pass(md)), 0)
            if not _all_pass(candidate):
                QMessageBox.critical(self, 'Cannot comply',
                    'Even minimum-dies-per-cell = 0 (field must merely touch the wafer) '
                    "rejects at least one placement — lowering the limit alone can't fix "
                    'this; move the placement(s), or loosen edge exclusion, instead.')
                return False
            self._state.cell['min_dies'] = candidate
            self._on_state_changed()
            return True

        return False   # Cancel

    def _on_generate_job(self):
        out = self._validate_for_generate()
        if out is None:
            return
        try:
            self._status.setText('Generating job…'); QApplication.processEvents()
            self._build_and_run(out, write_job=True)
            self._status.setText(f'Done → {out}.txt')
            QMessageBox.information(self, 'Done', f'Job file written:\n  {out}.txt')
        except Exception:
            msg = traceback.format_exc()
            self._status.setText('Error — see details')
            QMessageBox.critical(self, 'Generation failed', msg)

    def _on_generate_gds(self):
        out = self._validate_for_generate()
        if out is None:
            return
        if not any(m.get('gds_file') for m in self._state.masks):
            QMessageBox.warning(self, 'No GDS geometry',
                                'No mask has a GDS file loaded — nothing to write.')
            return
        try:
            self._status.setText('Generating GDS…'); QApplication.processEvents()
            lib = self._build_and_run(out, write_job=False)
            lib.write_gds(out + '.gds')
            self._status.setText(f'Done → {out}.gds')
            QMessageBox.information(self, 'Done', f'GDS file written:\n  {out}.gds')
        except Exception:
            msg = traceback.format_exc()
            self._status.setText('Error — see details')
            QMessageBox.critical(self, 'Generation failed', msg)

    def _on_correlate_marks(self):
        # Correlation is a verification of every aligned mark and is treated
        # the same at any layer rotation (a non-rotated mark is just the
        # theta = 0 case). The "no aligned layer references marks" case is
        # handled after the run.
        out = self._validate_for_generate()
        if out is None:
            return
        try:
            self._status.setText('Correlating marks…'); QApplication.processEvents()
            wafer = self._build_wafer()
            png = out + '_correlation.png'
            res = wafer.correlate_marks(plot_path=png)
        except Exception:
            msg = traceback.format_exc()
            self._status.setText('Error — see details')
            QMessageBox.critical(self, 'Correlation failed', msg)
            return
        if not res:
            self._status.setText('No marks to correlate')
            QMessageBox.information(self, 'No marks',
                'No aligned layer references any marks.')
            return
        npass = sum(1 for r in res if r.ok)
        self._status.setText(f'{npass}/{len(res)} marks pass → {png}')
        box = QMessageBox(self)
        box.setWindowTitle('Mark correlation')
        box.setIcon(QMessageBox.Icon.Information if npass == len(res)
                    else QMessageBox.Icon.Warning)
        box.setText(f'{npass}/{len(res)} marks pass.\nMontage: {png}')
        box.setDetailedText('\n'.join(str(r) for r in res))
        box.exec()

    def _build_and_run(self, out_name: str, write_job: bool = True):
        return self._build_wafer().write_file(out_name, write_job=write_job)

    def _build_wafer(self) -> 'Wafer':
        s = self._state

        # ── Mask objects ────────────────────────────────────────────────────────
        used_mask_ids = {i['mask_id'] for i in s.images if i.get('mask_id')}
        mask_objs: dict[str, Mask] = {}
        for m in s.masks:
            if m['id'] not in used_mask_ids:
                # Not referenced by any image — skip flattening entirely so an
                # oversized unused mask (e.g. a leftover top cell) can't block
                # generation.
                continue
            if m['gds_file']:
                glib = read_gds(m['gds_file'])
                cname = m['cell_name'] or glib.cells[0].name
                src  = glib[cname]
                cell  = gdstk.Cell(m['id'] + '_cell')
                sel = m.get('sel_layers')
                sel_set = set(sel) if sel is not None else None
                window = m.get('window')
                if window is not None:
                    # Windowed selection: pull only the sub-area through the
                    # hierarchy (pruned by bbox at every level, never fully
                    # flattened) instead of enumerating every instance. Exact
                    # mode disables every display-speed shortcut, so the only
                    # way this can come back incomplete is a repetition that
                    # is structurally impossible to enumerate — a real error,
                    # not a partial mask.
                    bbox = ((window[0], window[1]), (window[2], window[3]))
                    by_layer, qstats = _query_cell_bbox(src, bbox, exact=True)
                    if qstats['skipped']:
                        raise ValueError(
                            f"Mask '{m['id']}': window {window} still contains "
                            f"an irregular repetition too large to enumerate "
                            f"exactly ({len(qstats['skipped'])} subtree(s)) — "
                            f"shrink the window.")
                    layer_dtype = qstats.get('layer_dtype', {})
                    for layer, pts_list in by_layer.items():
                        if sel_set is not None and layer not in sel_set:
                            continue
                        dtype = layer_dtype.get(layer, 0)
                        for pts in pts_list:
                            cell.add(gdstk.Polygon(pts, layer=layer, datatype=dtype))
                else:
                    ninst = _approx_instances(src)
                    if ninst > _MAX_MASK_INSTANCES:
                        raise ValueError(
                            f"Mask '{m['id']}': cell '{cname}' in "
                            f"{os.path.basename(m['gds_file'])} expands to ~{ninst:,} "
                            f"placed instances — too large to flatten as a mask. "
                            f"Pick a specific sub-cell instead, or set an area "
                            f"window on the mask to use just part of it.")
                    for poly in src.get_polygons():
                        if sel_set is None or poly.layer in sel_set:
                            cell.add(gdstk.Polygon(poly.points,
                                                   layer=poly.layer, datatype=poly.datatype))
            else:
                cell = gdstk.Cell(m['id'] + '_empty')
            # Reticle (bar-code) names are always written upper-case in the job.
            mask_objs[m['id']] = Mask(m['id'], m['bar_code'].upper(), cell)

        # ── Image objects ───────────────────────────────────────────────────────
        img_objs: dict[str, Image] = {}
        for i in s.images:
            mask = mask_objs.get(i['mask_id'])
            if mask is None:
                raise ValueError(f"Image '{i['id']}': mask '{i['mask_id']}' not found")
            ox = i.get('origin_x')
            oy = i.get('origin_y')
            origin = Point(ox, oy) if (ox is not None and oy is not None) else None
            # Image names are always written upper-case in the job; keep the
            # dict keyed on the original id so exposure / mark look-ups still hit.
            img = Image(i['id'].upper(), mask,
                        Point(i['size_x'], i['size_y']),
                        Point(i['shift_x'], i['shift_y']),
                        origin=origin)
            pts = s.patterns.get(i['id'], [])
            img.set_pattern(Distrib([Point(p['x'], p['y']) for p in pts]))
            img_objs[i['id']] = img

        # ── Alignment / Mark objects ────────────────────────────────────────────
        align_objs: dict[str, Alignment] = {}
        for a in s.alignments:
            mark_list: list[Mark] = []
            missing:   list[str]  = []
            for mid in a.get('mark_ids', []):
                mk_d = next((m for m in s.marks if m['id'] == mid), None)
                if mk_d is None:
                    missing.append(mid)      # strategy references an undefined mark
                    continue
                if mk_d['image_id'] == PM_GENERATED:
                    _pm_rid = (s.cell.get('pm_reticle') or '').upper() or None
                    pm_img = _get_pm_image(_pm_rid)
                    if pm_img is None:
                        raise ValueError('PM mark image could not be generated')
                    mark_list.append(Mark(mk_d['id'], pm_img, Point(mk_d['x'], mk_d['y'])))
                else:
                    img = img_objs.get(mk_d['image_id'])
                    if img is None:
                        raise ValueError(f"Mark '{mk_d['id']}': image '{mk_d['image_id']}' not found")
                    mark_list.append(Mark(mk_d['id'], img, Point(mk_d['x'], mk_d['y'])))
            algn = Alignment(a['name'], mark_list, a.get('required', 4))
            algn.missing = missing
            align_objs[a['name']] = algn

        # ── Layer objects (in user-defined order) ───────────────────────────────
        layer_objs: list[Layer] = []
        for lyr_d in s.layers:
            lid = lyr_d['id']
            expos_for_layer = [e for e in s.exposures if e['layer_id'] == lid]
            if not expos_for_layer:
                continue
            idx    = ILLUME_LABELS.index(lyr_d.get('illume', 'Conventional'))
            illume = Illume(ILLUME_CODES[idx],
                            lyr_d.get('na', 0.57),
                            lyr_d.get('sigma_in', 0.0),
                            lyr_d.get('sigma_out', 0.5))
            proc       = s.layer_process.get(lid, {})
            align_name = proc.get('strategy_id')
            align      = (align_objs.get(align_name) or Alignment.no_align()
                          if align_name else Alignment.no_align())
            e0  = expos_for_layer[0]
            img = img_objs.get(e0['image_id'])
            if img is None:
                raise ValueError(f"Exposure: image '{e0['image_id']}' not found")
            lyr = Layer(lid, Expo(img, e0['dose'], e0['focus']),
                        align, illume, lyr_d.get('rotation', 0.0))
            for e in expos_for_layer[1:]:
                img2 = img_objs.get(e['image_id'])
                if img2 is None:
                    raise ValueError(f"Exposure: image '{e['image_id']}' not found")
                lyr.add_expo(Expo(img2, e['dose'], e['focus']))
            layer_objs.append(lyr)

        # ── Wafer ───────────────────────────────────────────────────────────────
        wafer   = Wafer()
        wafer.w = s.cell.get('wafer_w', FOUR_INCH)
        if s.cell.get('combine_zero_first', False):
            wafer.set_zero_one_combined()
        # rotated-marks job set turns itself on when a rotated layer aligns;
        # here we only pass the prealigner sign it will use if it does.
        wafer.set_rotated_marks_sign(int(s.cell.get('rotated_marks_sign', -1)))
        for lo in layer_objs:
            wafer.add(lo)
        return wafer


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
