import gdstk
import math
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, ClassVar, List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
FOUR_INCH  = 0
SIX_INCH   = 1
EIGHT_INCH = 2

WAFER_SIZES = [100.0, 150.0, 200.0]   # mm
FLAT        = [47.2857, 69.0, 100.0]  # mm
CELL_SIZE   = 22000                    # um
IDS         = [0, 1, 3, 2]

MAX_WIDTH   = 24000                    # um
IMAGE_SEP   = 1000                     # um
IMAGE_WIDTH = (MAX_WIDTH - IMAGE_SEP) / 2.0

# Reserved GDS layers for overlay/verification geometry in the combined
# preview GDS (wafer outline, PM alignment-mark geometry, mark-location
# crosses) — kept well clear of the sequential 0, 1, 2, ... numbering
# write_file() hands out to real device layers, so neither ever collides.
WAFER_LAYER   = 900
PM_MARK_LAYER = 901
CROSS_LAYER   = 902

_pm_cell:  Optional[gdstk.Cell] = None
_pm_mask:  Optional["Mask"]     = None
_pm_image: Optional["Image"]    = None


def read_gds(file: str):
    return gdstk.read_gds(file, unit=1e-6)


def _cell(lib, name: str) -> gdstk.Cell:
    for c in lib.cells:
        if c.name == name:
            return c
    raise KeyError(f"Cell '{name}' not found in library")


# ── GDS_Layer ─────────────────────────────────────────────────────────────────
class GDS_Layer:
    def __init__(self, lyr: int, dat: int = 0):
        self.lyr = lyr
        self.dat = dat

    def __repr__(self):
        return f"GDS_Layer({self.lyr}, {self.dat})"


# ── Point ─────────────────────────────────────────────────────────────────────
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def scale(self, n: float) -> "Point":
        return Point(self.x * n, self.y * n)

    X = scale

    def __str__(self) -> str:
        return f"{self.x:.6f} {self.y:.6f}"

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


# ── Section base & value formatter ───────────────────────────────────────────
_KEY_WIDTH = 46   # key column width; 3 leading spaces + 46 = 49 chars before value


def _fmt(val: Any) -> str:
    """Format a single field value for ASML job-file output.

    Dispatch rules:
      bool  -> "Y" / "N"
      str   -> quoted string
      float -> 6 decimal places
      int   -> bare integer
      Point -> "x.xxxxxx y.yyyyyy"
      tuple -> space-joined elements (each recursively formatted)
    """
    if isinstance(val, bool):   return '"Y"' if val else '"N"'
    if isinstance(val, str):    return f'"{val}"'
    if isinstance(val, float):  return f"{val:.6f}"
    if isinstance(val, int):    return str(val)
    if isinstance(val, Point):  return str(val)
    if isinstance(val, tuple):  return " ".join(_fmt(v) for v in val)
    raise TypeError(f"No ASML formatter for {type(val).__name__}")


class Section:
    """Base class for ASML job-file sections.  Subclasses are @dataclass.

    render() iterates dataclass fields in declaration order, skips None values,
    and formats each value with _fmt().  Override render() only for sections
    that need non-standard formatting (e.g. GENERAL's multi-line COMMENT).
    """
    SECTION_NAME: ClassVar[str] = ""

    def render(self) -> str:
        # ASML job files use LF line endings (matches pas_recipe_export output).
        lines = [f"\nSTART_SECTION {self.SECTION_NAME}\n"]
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            key = f.metadata.get("key", f.name.upper())
            lines.append(f"   {key:<{_KEY_WIDTH}}{_fmt(val)}\n")
        lines.append("END_SECTION\n")
        return "".join(lines)


# ── Section dataclasses ───────────────────────────────────────────────────────

@dataclass
class GeneralSection(Section):
    """GENERAL section – wafer setup and global job parameters."""
    SECTION_NAME: ClassVar[str] = "GENERAL"
    machine_type: str             = field(default="PAS5500/300",  metadata={"key": "MACHINE_TYPE"})
    reticle_size: int             = field(default=6,              metadata={"key": "RETICLE_SIZE"})
    wfr_diameter: float           = field(default=100.0,          metadata={"key": "WFR_DIAMETER"})
    wfr_notch: bool               = field(default=False,          metadata={"key": "WFR_NOTCH"})
    cell_size: tuple              = field(default=(22.0, 22.0),   metadata={"key": "CELL_SIZE"})
    round_edge_clearance: float   = field(default=2.0,            metadata={"key": "ROUND_EDGE_CLEARANCE"})
    flat_edge_clearance: float    = field(default=0.0,            metadata={"key": "FLAT_EDGE_CLEARANCE"})
    edge_exclusion: float         = field(default=3.0,            metadata={"key": "EDGE_EXCLUSION"})
    cover_mode: str               = field(default="W",            metadata={"key": "COVER_MODE"})
    number_dies: tuple            = field(default=(2, 2),         metadata={"key": "NUMBER_DIES"})
    min_number_dies: int          = field(default=1,              metadata={"key": "MIN_NUMBER_DIES"})
    placement_mode: str           = field(default="O",            metadata={"key": "PLACEMENT_MODE"})
    matrix_shift: tuple           = field(default=(0.0, 0.0),     metadata={"key": "MATRIX_SHIFT"})
    prealign_method: str          = field(default="STANDARD",     metadata={"key": "PREALIGN_METHOD"})
    wafer_rotation: float         = field(default=0.0,            metadata={"key": "WAFER_ROTATION"})
    combine_zero_first: bool      = field(default=False,          metadata={"key": "COMBINE_ZERO_FIRST"})
    mark_clear_out: Optional[bool]= field(default=None,           metadata={"key": "MARK_CLEAR_OUT"})
    matching_set_id: str          = field(default="DEFAULT",      metadata={"key": "MATCHING_SET_ID"})

    def render(self) -> str:
        # COMMENT uses three continuation lines – handled outside the field loop
        cont = " " * (_KEY_WIDTH + 3)           # 49 spaces for continuation lines
        lines = [
            "\nSTART_SECTION GENERAL\n",
            f"   {'COMMENT':<{_KEY_WIDTH}}\"\"\n",
            f"{cont}\"\"\n",
            f"{cont}\"\"\n",
        ]
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            key = f.metadata.get("key", f.name.upper())
            lines.append(f"   {key:<{_KEY_WIDTH}}{_fmt(val)}\n")
        lines.append("END_SECTION\n")
        return "".join(lines)


@dataclass
class AlignmentMarkSection(Section):
    """ALIGNMENT_MARK section – one per physical alignment mark on the wafer."""
    SECTION_NAME: ClassVar[str] = "ALIGNMENT_MARK"
    mark_id: str             = field(default="",   metadata={"key": "MARK_ID"})
    image_id: str            = field(default="",   metadata={"key": "IMAGE_ID"})
    mark_edge_clearance: str = field(default="L",  metadata={"key": "MARK_EDGE_CLEARANCE"})
    wafer_side: str          = field(default="A",  metadata={"key": "WAFER_SIDE"})
    mark_location: Point     = field(default_factory=lambda: Point(0.0, 0.0),
                                     metadata={"key": "MARK_LOCATION"})


@dataclass
class WfrAlignStrategySection(Section):
    """WFR_ALIGN_STRATEGY section – global wafer alignment strategy parameters."""
    SECTION_NAME: ClassVar[str] = "WFR_ALIGN_STRATEGY"
    strategy_id: str                 = field(default="",        metadata={"key": "STRATEGY_ID"})
    wafer_alignment_method: str      = field(default="T",       metadata={"key": "WAFER_ALIGNMENT_METHOD"})
    nr_of_marks_to_use: int          = field(default=4,         metadata={"key": "NR_OF_MARKS_TO_USE"})
    nr_of_x_marks_to_use: int        = field(default=4,         metadata={"key": "NR_OF_X_MARKS_TO_USE"})
    nr_of_y_marks_to_use: int        = field(default=4,         metadata={"key": "NR_OF_Y_MARKS_TO_USE"})
    min_mark_distance_coarse: float  = field(default=20.0,      metadata={"key": "MIN_MARK_DISTANCE_COARSE"})
    min_mark_distance: int           = field(default=40,        metadata={"key": "MIN_MARK_DISTANCE"})
    max_80_88_mark_shift: float      = field(default=0.5,       metadata={"key": "MAX_80_88_MARK_SHIFT"})
    max_mark_residue: float          = field(default=200.0,     metadata={"key": "MAX_MARK_RESIDUE"})
    spm_mark_scan: str               = field(default="S",       metadata={"key": "SPM_MARK_SCAN"})
    corr_wafer_grid: str             = field(default="Default", metadata={"key": "CORR_WAFER_GRID"})
    err_detection_88_8: str          = field(default="M",       metadata={"key": "ERR_DETECTION_88_8"})
    grid_optimisation_algorithm: str = field(default="N",       metadata={"key": "GRID_OPTIMISATION_ALGORITHM"})
    flyer_removal_threshold: float   = field(default=0.0,       metadata={"key": "FLYER_REMOVAL_THRESHOLD"})
    alignment_monitoring: str        = field(default="D",       metadata={"key": "ALIGNMENT_MONITORING"})


@dataclass
class MarkAlignmentSection(Section):
    """MARK_ALIGNMENT section – associates a mark with a strategy."""
    SECTION_NAME: ClassVar[str] = "MARK_ALIGNMENT"
    strategy_id: str     = field(default="",  metadata={"key": "STRATEGY_ID"})
    mark_id: str         = field(default="",  metadata={"key": "MARK_ID"})
    glbl_mark_usage: str = field(default="A", metadata={"key": "GLBL_MARK_USAGE"})
    mark_preference: str = field(default="P", metadata={"key": "MARK_PREFERENCE"})


@dataclass
class ImageDefinitionSection(Section):
    """IMAGE_DEFINITION section – reticle image geometry (4x reticle coordinates)."""
    SECTION_NAME: ClassVar[str] = "IMAGE_DEFINITION"
    image_id: str              = field(default="",   metadata={"key": "IMAGE_ID"})
    reticle_id: str            = field(default="",   metadata={"key": "RETICLE_ID"})
    image_size: Point          = field(default_factory=lambda: Point(0.0, 0.0),
                                       metadata={"key": "IMAGE_SIZE"})
    image_shift: Point         = field(default_factory=lambda: Point(0.0, 0.0),
                                       metadata={"key": "IMAGE_SHIFT"})
    mask_size: Point           = field(default_factory=lambda: Point(0.0, 0.0),
                                       metadata={"key": "MASK_SIZE"})
    mask_shift: Point          = field(default_factory=lambda: Point(0.0, 0.0),
                                       metadata={"key": "MASK_SHIFT"})
    base_image_id: Optional[str] = field(default=None, metadata={"key": "BASE_IMAGE_ID"})
    variant_id: str            = field(default="",   metadata={"key": "VARIANT_ID"})


@dataclass
class ImageDistributionSection(Section):
    """IMAGE_DISTRIBUTION section – one instance placement of an image on the wafer."""
    SECTION_NAME: ClassVar[str] = "IMAGE_DISTRIBUTION"
    image_id: str           = field(default="",             metadata={"key": "IMAGE_ID"})
    instance_id: str        = field(default="001",          metadata={"key": "INSTANCE_ID"})
    # cell_selection: tuple of two strings renders as "xdie" "ydie"
    cell_selection: tuple   = field(default=("0", "0"),     metadata={"key": "CELL_SELECTION"})
    distribution_action: str= field(default="I",            metadata={"key": "DISTRIBUTION_ACTION"})
    optimize_route: bool    = field(default=False,          metadata={"key": "OPTIMIZE_ROUTE"})
    image_cell_shift: Point = field(default_factory=lambda: Point(0.0, 0.0),
                                    metadata={"key": "IMAGE_CELL_SHIFT"})


@dataclass
class InstanceDefinitionSection(Section):
    """INSTANCE_DEFINITION section – declares an instance slot."""
    SECTION_NAME: ClassVar[str] = "INSTANCE_DEFINITION"
    instance_id: str = field(default="001", metadata={"key": "INSTANCE_ID"})


@dataclass
class LayerDefinitionSection(Section):
    """LAYER_DEFINITION section – maps a layer number to a layer ID."""
    SECTION_NAME: ClassVar[str] = "LAYER_DEFINITION"
    layer_no: int   = field(default=0,   metadata={"key": "LAYER_NO"})
    layer_id: str   = field(default="",  metadata={"key": "LAYER_ID"})
    wafer_side: str = field(default="A", metadata={"key": "WAFER_SIDE"})


@dataclass
class MarksSelectionSection(Section):
    """MARKS_SELECTION section – controls mark exposure usage per layer."""
    SECTION_NAME: ClassVar[str] = "MARKS_SELECTION"
    layer_id: str        = field(default="",  metadata={"key": "LAYER_ID"})
    mark_id: str         = field(default="",  metadata={"key": "MARK_ID"})
    glbl_mark_usage: str = field(default="N", metadata={"key": "GLBL_MARK_USAGE"})


@dataclass
class StrategySelectionSection(Section):
    """STRATEGY_SELECTION section – binds an alignment strategy to a layer."""
    SECTION_NAME: ClassVar[str] = "STRATEGY_SELECTION"
    layer_id: str      = field(default="",  metadata={"key": "LAYER_ID"})
    strategy_id: str   = field(default="",  metadata={"key": "STRATEGY_ID"})
    strategy_usage: str= field(default="A", metadata={"key": "STRATEGY_USAGE"})


@dataclass
class ProcessDataSection(Section):
    """PROCESS_DATA section – all process and correction parameters for one layer.

    Fields that are None are omitted from output.  The two conditional blocks
    (corr_wafer_grid … err_detection_88_8 and alignment_method) are set to
    their values for non-zero layers and left as None for layer "0".
    """
    SECTION_NAME: ClassVar[str] = "PROCESS_DATA"
    # ── Always present ────────────────────────────────────────────────────────
    layer_id: str                       = field(default="",            metadata={"key": "LAYER_ID"})
    lens_reduction: int                 = field(default=4,             metadata={"key": "LENS_REDUCTION"})
    calibration: bool                   = field(default=False,         metadata={"key": "CALIBRATION"})
    optical_prealignment: bool          = field(default=False,         metadata={"key": "OPTICAL_PREALIGNMENT"})
    glbl_wfr_alignment: bool            = field(default=False,         metadata={"key": "GLBL_WFR_ALIGNMENT"})
    coo_reduction: str                  = field(default="D",           metadata={"key": "COO_REDUCTION"})
    min_number_pulses_in_slit: str      = field(default="D",           metadata={"key": "MIN_NUMBER_PULSES_IN_SLIT"})
    min_number_pulses: int              = field(default=21,            metadata={"key": "MIN_NUMBER_PULSES"})
    skip_coarse_wafer_align: bool       = field(default=False,         metadata={"key": "SKIP_COARSE_WAFER_ALIGN"})
    reduce_reticle_align: bool          = field(default=False,         metadata={"key": "REDUCE_RETICLE_ALIGN"})
    reduce_ra_drift: float              = field(default=5.0,           metadata={"key": "REDUCE_RA_DRIFT"})
    reduce_ra_interval: int             = field(default=2,             metadata={"key": "REDUCE_RA_INTERVAL"})
    ret_cool_corr: str                  = field(default="D",           metadata={"key": "RET_COOL_CORR"})
    ret_cool_time: int                  = field(default=0,             metadata={"key": "RET_COOL_TIME"})
    ret_cool_start_on_load: bool        = field(default=True,          metadata={"key": "RET_COOL_START_ON_LOAD"})
    ret_cool_usage: str                 = field(default="W",           metadata={"key": "RET_COOL_USAGE"})
    glbl_rtcl_alignment: bool           = field(default=False,         metadata={"key": "GLBL_RTCL_ALIGNMENT"})
    glbl_overlay_enhancement: bool      = field(default=False,         metadata={"key": "GLBL_OVERLAY_ENHANCEMENT"})
    wafer_align_repeats: bool           = field(default=False,         metadata={"key": "WAFER_ALIGN_REPEATS"})
    nr_wafer_align_repeats: int         = field(default=2,             metadata={"key": "NR_WAFER_ALIGN_REPEATS"})
    align_repeat_interval: tuple        = field(default=(10,) * 10,    metadata={"key": "ALIGN_REPEAT_INTERVAL"})
    smart_repeat_count: int             = field(default=3,             metadata={"key": "SMART_REPEAT_COUNT"})
    smart_repeat_threshold: float       = field(default=0.1,           metadata={"key": "SMART_REPEAT_THRESHOLD"})
    glbl_sym_alignment: bool            = field(default=False,         metadata={"key": "GLBL_SYM_ALIGNMENT"})
    layer_shift: tuple                  = field(default=(0.0, 0.0),    metadata={"key": "LAYER_SHIFT"})
    # ── Present only when layer_id != "0" (inserted after LAYER_SHIFT) ───────
    corr_wafer_grid: Optional[str]      = field(default=None,          metadata={"key": "CORR_WAFER_GRID"})
    nr_of_marks_to_use: Optional[int]   = field(default=None,          metadata={"key": "NR_OF_MARKS_TO_USE"})
    min_mark_distance_coarse: Optional[float] = field(default=None,    metadata={"key": "MIN_MARK_DISTANCE_COARSE"})
    min_mark_distance: Optional[int]    = field(default=None,          metadata={"key": "MIN_MARK_DISTANCE"})
    max_80_88_shift: Optional[float]    = field(default=None,          metadata={"key": "MAX_80_88_SHIFT"})
    max_mark_residue: Optional[float]   = field(default=None,          metadata={"key": "MAX_MARK_RESIDUE"})
    spm_mark_scan: Optional[str]        = field(default=None,          metadata={"key": "SPM_MARK_SCAN"})
    err_detection_88_8: Optional[str]   = field(default=None,          metadata={"key": "ERR_DETECTION_88_8"})
    # ── Always present (inter/intra-field corrections) ────────────────────────
    corr_inter_fld_expansion: tuple     = field(default=(0.0, 0.0),    metadata={"key": "CORR_INTER_FLD_EXPANSION"})
    corr_inter_fld_nonortho: float      = field(default=0.0,           metadata={"key": "CORR_INTER_FLD_NONORTHO"})
    corr_inter_fld_rotation: float      = field(default=0.0,           metadata={"key": "CORR_INTER_FLD_ROTATION"})
    corr_inter_fld_translation: tuple   = field(default=(0.0, 0.0),    metadata={"key": "CORR_INTER_FLD_TRANSLATION"})
    corr_intra_fld_magnification: float = field(default=0.0,           metadata={"key": "CORR_INTRA_FLD_MAGNIFICATION"})
    corr_intra_fld_rotation: float      = field(default=0.0,           metadata={"key": "CORR_INTRA_FLD_ROTATION"})
    corr_intra_fld_translation: tuple   = field(default=(0.0, 0.0),    metadata={"key": "CORR_INTRA_FLD_TRANSLATION"})
    corr_intra_fld_asym_rotation: float = field(default=0.0,           metadata={"key": "CORR_INTRA_FLD_ASYM_ROTATION"})
    corr_intra_fld_asym_magn: float     = field(default=0.0,           metadata={"key": "CORR_INTRA_FLD_ASYM_MAGN"})
    corr_prealign_rotation: float       = field(default=0.0,           metadata={"key": "CORR_PREALIGN_ROTATION"})
    corr_prealign_translation: tuple    = field(default=(0.0, 0.0),    metadata={"key": "CORR_PREALIGN_TRANSLATION"})
    corr_80_88_mark_shift: tuple        = field(default=(0.0,) * 4,    metadata={"key": "CORR_80_88_MARK_SHIFT"})
    corr_lens_heating: float            = field(default=1.0,           metadata={"key": "CORR_LENS_HEATING"})
    numerical_aperture: float           = field(default=0.57,          metadata={"key": "NUMERICAL_APERTURE"})
    sigma_inner: Optional[float]        = field(default=None,          metadata={"key": "SIGMA_INNER"})
    sigma_outer: float                  = field(default=0.5,           metadata={"key": "SIGMA_OUTER"})
    rtcl_check_surfaces: bool           = field(default=False,         metadata={"key": "RTCL_CHECK_SURFACES"})
    rtcl_check_limits_upper: tuple      = field(default=(50000,) * 3,  metadata={"key": "RTCL_CHECK_LIMITS_UPPER"})
    rtcl_check_limits_lower: tuple      = field(default=(50000,) * 3,  metadata={"key": "RTCL_CHECK_LIMITS_LOWER"})
    # ── Present only when layer_id != "0" (inserted after RTCL_CHECK_LIMITS) ─
    alignment_method: Optional[str]     = field(default=None,          metadata={"key": "ALIGNMENT_METHOD"})
    # ── Always present (final flags) ─────────────────────────────────────────
    close_green_laser_shutter: bool     = field(default=False,         metadata={"key": "CLOSE_GREEN_LASER_SHUTTER"})
    realignment_method: str             = field(default="D",           metadata={"key": "REALIGNMENT_METHOD"})
    image_order_optimisation: bool      = field(default=True,          metadata={"key": "IMAGE_ORDER_OPTIMISATION"})
    reticle_alignment: str              = field(default="T",           metadata={"key": "RETICLE_ALIGNMENT"})
    use_default_reticle_alignment_method: bool = field(default=False,  metadata={"key": "USE_DEFAULT_RETICLE_ALIGNMENT_METHOD"})
    critical_percentage: int            = field(default=83,            metadata={"key": "CRITICAL_PERCENTAGE"})
    share_level_info: bool              = field(default=False,         metadata={"key": "SHARE_LEVEL_INFO"})
    focus_edge_clearance: float         = field(default=3.0,           metadata={"key": "FOCUS_EDGE_CLEARANCE"})
    inline_q_above_p_calibration: str   = field(default="D",           metadata={"key": "INLINE_Q_ABOVE_P_CALIBRATION"})
    shifted_measurement_scans: bool     = field(default=False,         metadata={"key": "SHIFTED_MEASUREMENT_SCANS"})
    focus_monitoring: str               = field(default="D",           metadata={"key": "FOCUS_MONITORING"})
    focus_monitoring_scanner: str       = field(default="D",           metadata={"key": "FOCUS_MONITORING_SCANNER"})
    dyn_perf_monitoring: str            = field(default="D",           metadata={"key": "DYN_PERF_MONITORING"})
    force_meander_enabled: bool         = field(default=False,         metadata={"key": "FORCE_MEANDER_ENABLED"})


@dataclass
class ReticleDataSection(Section):
    """RETICLE_DATA section – exposure parameters for one image on one layer."""
    SECTION_NAME: ClassVar[str] = "RETICLE_DATA"
    layer_id: str                     = field(default="",   metadata={"key": "LAYER_ID"})
    image_id: str                     = field(default="",   metadata={"key": "IMAGE_ID"})
    image_usage: bool                 = field(default=True, metadata={"key": "IMAGE_USAGE"})
    reticle_id: str                   = field(default="",   metadata={"key": "RETICLE_ID"})
    image_size: Point                 = field(default_factory=lambda: Point(0.0, 0.0),
                                              metadata={"key": "IMAGE_SIZE"})
    image_shift: Point                = field(default_factory=lambda: Point(0.0, 0.0),
                                              metadata={"key": "IMAGE_SHIFT"})
    mask_size: Point                  = field(default_factory=lambda: Point(0.0, 0.0),
                                              metadata={"key": "MASK_SIZE"})
    mask_shift: Point                 = field(default_factory=lambda: Point(0.0, 0.0),
                                              metadata={"key": "MASK_SHIFT"})
    energy_actual: float              = field(default=20.0, metadata={"key": "ENERGY_ACTUAL"})
    focus_actual: float               = field(default=0.0,  metadata={"key": "FOCUS_ACTUAL"})
    focus_tilt: Point                 = field(default_factory=lambda: Point(0.0, 0.0),
                                              metadata={"key": "FOCUS_TILT"})
    numerical_aperture: float         = field(default=0.57, metadata={"key": "NUMERICAL_APERTURE"})
    sigma_inner: Optional[float]      = field(default=None, metadata={"key": "SIGMA_INNER"})
    sigma_outer: float                = field(default=0.5,  metadata={"key": "SIGMA_OUTER"})
    image_exposure_order: int         = field(default=0,    metadata={"key": "IMAGE_EXPOSURE_ORDER"})
    lithography_process: str          = field(default="Default", metadata={"key": "LITHOGRAPHY_PROCESS"})
    image_intra_fld_cor_trans: tuple  = field(default=(0.0, 0.0), metadata={"key": "IMAGE_INTRA_FLD_COR_TRANS"})
    image_intra_fld_cor_rot: float    = field(default=0.0,  metadata={"key": "IMAGE_INTRA_FLD_COR_ROT"})
    image_intra_fld_cor_mag: float    = field(default=0.0,  metadata={"key": "IMAGE_INTRA_FLD_COR_MAG"})
    image_intra_fld_cor_asym_rot: float = field(default=0.0, metadata={"key": "IMAGE_INTRA_FLD_COR_ASYM_ROT"})
    image_intra_fld_cor_asym_mag: float = field(default=0.0, metadata={"key": "IMAGE_INTRA_FLD_COR_ASYM_MAG"})
    level_method_z: str               = field(default="D",  metadata={"key": "LEVEL_METHOD_Z"})
    level_method_rx: str              = field(default="D",  metadata={"key": "LEVEL_METHOD_RX"})
    level_method_ry: str              = field(default="D",  metadata={"key": "LEVEL_METHOD_RY"})
    die_size_dependency: bool         = field(default=False, metadata={"key": "DIE_SIZE_DEPENDENCY"})
    enable_efese: bool                = field(default=False, metadata={"key": "ENABLE_EFESE"})
    cd_fec_mode: bool                 = field(default=False, metadata={"key": "CD_FEC_MODE"})
    dose_correction: bool             = field(default=False, metadata={"key": "DOSE_CORRECTION"})
    dose_critical_image: bool         = field(default=True,  metadata={"key": "DOSE_CRITICAL_IMAGE"})
    global_level_point_1: tuple       = field(default=(0.0, 0.0), metadata={"key": "GLOBAL_LEVEL_POINT_1"})
    global_level_point_2: tuple       = field(default=(0.0, 0.0), metadata={"key": "GLOBAL_LEVEL_POINT_2"})
    global_level_point_3: tuple       = field(default=(0.0, 0.0), metadata={"key": "GLOBAL_LEVEL_POINT_3"})


# ── Illume ────────────────────────────────────────────────────────────────────
class Illume:
    DEFAULT      = 0
    CONVENTIONAL = 1
    QUADRUPOLE   = 2
    ANNULAR      = 3

    def __init__(self, illume_type: int = 0, na: float = 0.57,
                 sigin: float = 0.0, sigout: float = 0.5):
        self.illume_type = illume_type
        self.na     = na
        self.sigin  = sigin
        self.sigout = sigout

    @staticmethod
    def conventional() -> "Illume":
        return Illume(Illume.CONVENTIONAL)

    def job_values(self) -> tuple:
        """(NUMERICAL_APERTURE, SIGMA_INNER, SIGMA_OUTER) as written to the job.

        DEFAULT illumination is written as -1 / -1 / -1 (the stepper takes the
        settings from the reticle).  Every other mode writes its actual NA and
        sigmas, and SIGMA_INNER is always emitted between NA and SIGMA_OUTER
        (0.000000 for conventional).
        """
        if self.illume_type == Illume.DEFAULT:
            return -1.0, -1.0, -1.0
        return float(self.na), float(self.sigin), float(self.sigout)


# ── Cross helper ─────────────────────────────────────────────────────────────
def _make_cross(cx: float, cy: float,
                arm_len: float = 2000.0, arm_width: float = 200.0,
                layer: int = CROSS_LAYER) -> list:
    """Two rectangles forming a + cross centred at (cx, cy), coordinates in um."""
    return [
        gdstk.rectangle(
            (cx - arm_len, cy - arm_width / 2),
            (cx + arm_len, cy + arm_width / 2),
            layer=layer,
        ),
        gdstk.rectangle(
            (cx - arm_width / 2, cy - arm_len),
            (cx + arm_width / 2, cy + arm_len),
            layer=layer,
        ),
    ]


# ── Distrib ───────────────────────────────────────────────────────────────────
class Distrib:
    def __init__(self, points: List[Point]):
        self.points = points


# ── Mask ──────────────────────────────────────────────────────────────────────
class Mask:
    def __init__(self, id: str, bar_code: str, cell: gdstk.Cell):
        print(f"Mask {id} {bar_code}")
        self.id       = id
        self.bar_code = bar_code
        self.cell     = cell
        self._polys   = None
        if cell is None:
            print("   Mask -> CELL IS NULL")

    def get_polys(self) -> list:
        if self._polys is None:
            self._polys = self.cell.get_polygons()
        return self._polys


# ── Image ─────────────────────────────────────────────────────────────────────
class Image:
    def __init__(self, id: str, mask: Mask, size: Point, shift: Point,
                 origin: Optional[Point] = None):
        self.id    = id
        self.mask  = mask
        self.size  = size
        self.shift = shift
        self.pat: Optional[Distrib] = None
        # origin is the pivot/reference point for pattern placement (wafer mm).
        # Defaults to (0, 0) so existing callers are unaffected.
        self._zero = origin if origin is not None else Point(0.0, 0.0)

    def set_pattern(self, pattern: Distrib):
        self.pat = pattern

    def set_zero(self, x: float, y: float):
        self._zero = Point(x, y)

    def shift_zero(self, dx: float, dy: float):
        self._zero = Point(self._zero.x + dx, self._zero.y + dy)

    def image_def_out(self) -> str:
        sz4 = self.size.scale(4)
        sh4 = self.shift.scale(4)
        return ImageDefinitionSection(
            image_id=self.id,
            reticle_id=self.mask.bar_code,
            image_size=sz4,
            image_shift=sh4,
            mask_size=sz4,
            mask_shift=sh4,
            base_image_id="PM" if self.id == "PM" else None,
        ).render()

    def instance_out(self, p: Point, inst_id: int, xdie: int, ydie: int) -> str:
        return ImageDistributionSection(
            image_id=self.id,
            instance_id=f"{inst_id:03d}",
            cell_selection=(str(xdie), str(ydie)),
            image_cell_shift=p,
        ).render()


# ── PM alignment-mark geometry ───────────────────────────────────────────────
def _build_pm_cell() -> gdstk.Cell:
    """ASML primary alignment mark, generated in code (was PM.gds).

    A 100 um centre cross surrounded by four 50 %-duty gratings — the X/Y
    capture gratings at 17.6 um pitch (upper-left, lower-left) and the X/Y fine
    gratings at 16.0 um pitch (lower-right, upper-right).  ~413 um overall, all
    on layer 1, coordinates in micrometres.  Matches PM.gds to <0.1 um.
    """
    cell = gdstk.Cell("PM")

    # centre cross: 10 um line width, 100 um vertical spine, 45 um side arms
    cell.add(gdstk.rectangle((-5.0, -50.0), (5.0, 50.0), layer=PM_MARK_LAYER))
    cell.add(gdstk.rectangle((-50.0, -5.0), (-5.0, 5.0), layer=PM_MARK_LAYER))
    cell.add(gdstk.rectangle((5.0, -5.0), (50.0, 5.0), layer=PM_MARK_LAYER))

    def grating(n, pitch, thick, first, lo, hi, vertical):
        for i in range(n):
            a = first + i * pitch
            if vertical:
                cell.add(gdstk.rectangle((a, lo), (a + thick, hi), layer=PM_MARK_LAYER))
            else:
                cell.add(gdstk.rectangle((lo, a), (hi, a + thick), layer=PM_MARK_LAYER))

    grating(11, 17.6, 8.8, -209.0,   14.0, 204.0, vertical=True)   # UL  X capture
    grating(11, 17.6, 8.8, -209.0, -209.0, -18.0, vertical=False)  # LL  Y capture
    grating(12, 16.0, 8.0,   20.0, -209.0, -18.0, vertical=True)   # LR  X fine
    grating(12, 16.0, 8.0,   20.0,   14.0, 204.0, vertical=False)  # UR  Y fine
    return cell


PM_RETICLE_ID = "4544020*"          # default: the COMBI marks plate


def get_pm_image(reticle_id: str = None) -> "Image":
    """The PM alignment-mark Image, cached in module state.

    The geometry is fixed; `reticle_id` sets RETICLE_ID — which physical plate
    exposes the marks — and defaults to the COMBI plate.  Pass a different bar
    code when the layer-0 marks come from another reticle.  The image is rebuilt
    if the requested reticle differs from the cached one.
    """
    global _pm_cell, _pm_mask, _pm_image
    rid = reticle_id or PM_RETICLE_ID
    if _pm_cell is None:
        _pm_cell = _build_pm_cell()
    if _pm_image is None or _pm_image.mask.bar_code != rid:
        _pm_mask  = Mask("COMBI", rid, _pm_cell)
        _pm_image = Image("PM", _pm_mask, Point(0.41, 0.41), Point(0.0, 0.0))
    return _pm_image


# ── Mark ──────────────────────────────────────────────────────────────────────
class Mark:
    def __init__(self, id: str, image: Image, location: Point):
        self.id           = id
        self.i            = image
        self.side         = "A"
        self.loc          = location

    @staticmethod
    def pm_mark(id: str, x_or_loc, y: float = None, reticle_id: str = None) -> "Mark":
        loc = x_or_loc if isinstance(x_or_loc, Point) else Point(x_or_loc, y)
        return Mark(id, get_pm_image(reticle_id), loc)

    def __eq__(self, other):
        return isinstance(other, Mark) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def mark_selection(self, layer_id, *, expose: bool = False) -> str:
        # layer_id must match a LAYER_DEFINITION's LAYER_ID (the layer's real
        # id string) — NOT its running LAYER_NO.  "E" only on the layer that
        # exposes the marks (layer 0); "N" (use for alignment) elsewhere.
        return MarksSelectionSection(
            layer_id=str(layer_id),
            mark_id=self.id,
            glbl_mark_usage="E" if expose else "N",
        ).render()


# ── Alignment ─────────────────────────────────────────────────────────────────
class Alignment:
    def __init__(self, name: str = "", marks: List[Mark] = None, required: int = 0):
        self.name     = name.upper() if name else ""
        self.mark: List[Mark] = marks if marks is not None else []
        self.required = required
        self.on       = bool(name)
        # Strategy mark ids that were requested but never resolved to a defined
        # Mark (set by the caller that builds the strategy).  correlate_marks
        # reports one FAIL row per entry.
        self.missing: List[str] = []

    @staticmethod
    def default4() -> "Alignment":
        marks = [
            Mark.pm_mark("PM1", -40.0,  0.0),
            Mark.pm_mark("PM2",  40.0,  0.0),
            Mark.pm_mark("PM3",   0.0, -40.0),
            Mark.pm_mark("PM4",   0.0,  40.0),
        ]
        return Alignment("def4", marks, 4)

    @staticmethod
    def no_align() -> "Alignment":
        return Alignment()

    def wfr_align_strategy(self) -> str:
        return WfrAlignStrategySection(
            strategy_id=self.name,
            nr_of_marks_to_use=self.required,
            nr_of_x_marks_to_use=self.required,
            nr_of_y_marks_to_use=self.required,
        ).render()


# ── Expo ──────────────────────────────────────────────────────────────────────
class Expo:
    def __init__(self, image: Image, dose: float, focus: float):
        self.image = image
        self.dose  = dose
        self.focus = focus

    @staticmethod
    def pm() -> "Expo":
        assert _pm_image is not None, "PM image not initialised; call Alignment.default4() first"
        return Expo(_pm_image, 20.0, 0.0)

    def __str__(self):
        return f"Expo {self.dose} , {self.focus}"


# ── Layer ─────────────────────────────────────────────────────────────────────
class Layer:
    def __init__(self, name: str, expo: "Expo" = None,
                 alignment: Alignment = None, illume: Illume = None,
                 rotation: float = 0.0):
        self.id        = name
        self.expos: List[Expo] = []
        self.side      = "A"
        self.na        = 0.0
        self.illume    = illume    if illume    is not None else Illume.conventional()
        self.alignment = alignment if alignment is not None else Alignment.no_align()
        self.rotation  = rotation
        if expo is not None:
            self.expos.append(expo)

    def add_expo(self, expo: "Expo"):
        self.expos.append(expo)

    def set_align(self, alignment: Alignment):
        self.alignment = alignment

    @staticmethod
    def zero() -> "Layer":
        return Layer("0", Expo.pm(), Alignment.no_align(), Illume.conventional())

    def layer_def_out(self, i: int) -> str:
        return LayerDefinitionSection(
            layer_no=i,
            layer_id=self.id,
            wafer_side=self.side,
        ).render()

    def strategy_selection_out(self) -> str:
        return StrategySelectionSection(
            layer_id=self.id,
            strategy_id=self.alignment.name,
            strategy_usage=self.side,
        ).render()

    def process_data_out(self) -> str:
        is_zero  = (self.id == "0")
        # The wafer-grid / mark block and ALIGNMENT_METHOD belong to a layer only
        # when that layer actually performs wafer alignment.  Keying them off the
        # layer *name* ("0" vs anything else) emitted a phantom alignment setup
        # (ALIGNMENT_METHOD "T", NR_OF_MARKS_TO_USE 0, ...) on plain layers whose
        # job has no WFR_ALIGN_STRATEGY / ALIGNMENT_MARK / STRATEGY_SELECTION,
        # which the job compiler rejects.
        aligned  = self.alignment.on
        na, sigin, sigout = self.illume.job_values()
        return ProcessDataSection(
            layer_id=self.id,
            numerical_aperture=na,
            sigma_inner=sigin,
            sigma_outer=sigout,
            inline_q_above_p_calibration="D" if is_zero else "M",
            # conditional block after LAYER_SHIFT
            corr_wafer_grid=None          if not aligned else "Default",
            nr_of_marks_to_use=None       if not aligned else self.alignment.required,
            min_mark_distance_coarse=None if not aligned else 20.0,
            min_mark_distance=None        if not aligned else 40,
            max_80_88_shift=None          if not aligned else 0.5,
            max_mark_residue=None         if not aligned else 200.0,
            spm_mark_scan=None            if not aligned else "S",
            err_detection_88_8=None       if not aligned else "M",
            # conditional field after RTCL_CHECK_LIMITS
            alignment_method=None         if not aligned else "T",
        ).render()

    def reticle_data_out(self) -> str:
        na, sigin, sigout = self.illume.job_values()
        return "".join(
            ReticleDataSection(
                layer_id=self.id,
                image_id=expo.image.id,
                reticle_id=expo.image.mask.bar_code,
                image_size=expo.image.size.scale(4),
                image_shift=expo.image.shift.scale(4),
                mask_size=expo.image.size.scale(4),
                mask_shift=expo.image.shift.scale(4),
                energy_actual=expo.dose,
                focus_actual=expo.focus,
                numerical_aperture=na,
                sigma_inner=sigin,
                sigma_outer=sigout,
            ).render()
            for expo in self.expos
        )


# ── Wafer ─────────────────────────────────────────────────────────────────────
class Wafer:
    def __init__(self, w: int = FOUR_INCH):
        self.w:     int         = w
        self.layers: List[Layer]= []
        self.zero_one_combined  = False
        # Rotated-lens mode: when on, write_file() emits a *job set* — one
        # unaligned "marks" job that exposes every alignment mark as plain
        # geometry on a flat wafer, followed by the usual one-job-per-rotation
        # lens jobs — and adds one stepper-frame GDS view per rotation.
        #
        # It is needed only when an alignment sits on a rotated layer, so
        # write_file() enables it automatically from the layers (see
        # _has_rotated_alignment).  Setting it True here is an explicit
        # override; the sign is always an explicit choice (prealigner turn
        # direction) and has no effect unless the mode is on.
        self.rotated_marks_mode = False
        self.rotated_marks_sign = -1      # prealigner turn = sign * layer.rotation

    def set_zero_one_combined(self):
        self.zero_one_combined = True

    def set_rotated_marks_sign(self, sign: int = -1):
        """Prealigner turn direction for rotated-marks mode (used only when
        that mode is active)."""
        self.rotated_marks_sign = sign

    def set_rotated_marks_mode(self, sign: int = -1):
        """Explicitly force the rotated-marks job set on.  Normally not
        needed — write_file() turns it on when a rotated layer aligns."""
        self.rotated_marks_mode = True
        self.rotated_marks_sign = sign

    def _has_rotated_alignment(self) -> bool:
        """True iff some layer both has a non-zero rotation and aligns — the
        only case the rotated-marks job set is for.  No rotated layers, or
        rotated layers that don't align, → False."""
        return any(l.rotation and l.alignment.on for l in self.layers)

    def _suppress_combined_first_alignment(self) -> None:
        """When zero + first are combined, the first layer is exposed together
        with layer 0 and is never aligned on its own.  Drop any alignment left
        on it so job output and mark correlation don't treat it as an aligned
        layer.  Idempotent; call from every entry point that reads
        layer.alignment.on (write_file, correlate_marks)."""
        if self.zero_one_combined and self.layers and self.layers[0].alignment.on:
            self.layers[0].alignment = Alignment.no_align()

    def add(self, layer: Layer):
        self.layers.append(layer)

    def _prepare_job(self, layers: List[Layer]):
        """Return (job_layers, images, alignments, marks) for a rotation group."""
        alignments: List[Alignment] = []
        images:     List[Image]     = []
        seen_align: set = set()
        seen_image: set = set()
        for layer in layers:
            if layer.alignment.on and layer.alignment.name not in seen_align:
                seen_align.add(layer.alignment.name)
                alignments.append(layer.alignment)
            for expo in layer.expos:
                if expo.image.id not in seen_image:
                    seen_image.add(expo.image.id)
                    images.append(expo.image)
        marks: List[Mark] = []
        seen_mark: set = set()
        for align in alignments:
            for mark in align.mark:
                if mark.id not in seen_mark:
                    seen_mark.add(mark.id)
                    marks.append(mark)
        job_layers = list(layers)
        if alignments:
            if _pm_image and _pm_image.id not in seen_image:
                images.insert(0, _pm_image)
            job_layers = [Layer.zero()] + job_layers
        return job_layers, images, alignments, marks

    def write_file(self, file_name: str, write_job: bool = True) -> gdstk.Library:
        self._suppress_combined_first_alignment()
        # Group layers by rotation angle; each unique angle → one job file
        rotation_groups: dict = {}
        for layer in self.layers:
            angle = layer.rotation
            if angle not in rotation_groups:
                rotation_groups[angle] = []
            rotation_groups[angle].append(layer)

        multi = len(rotation_groups) > 1

        # Auto-enable the rotated-marks job set when a rotated layer aligns
        # (an explicit set_rotated_marks_mode() still forces it on).
        if self._has_rotated_alignment():
            self.rotated_marks_mode = True

        lib      = gdstk.Library()
        top_cell = gdstk.Cell("top")

        wafer_outline = gdstk.ellipse(
            (0, 0), radius=WAFER_SIZES[self.w] * 500, tolerance=10, layer=WAFER_LAYER
        )
        flat_rect = gdstk.rectangle(
            (-400000, -400000), (400000, -FLAT[self.w] * 1000), layer=WAFER_LAYER
        )
        wafer_cell = gdstk.Cell("wafer")
        for poly in gdstk.boolean([wafer_outline], [flat_rect], "not", layer=WAFER_LAYER):
            wafer_cell.add(poly)
        top_cell.add(gdstk.Reference(wafer_cell))

        cell_size_mm  = CELL_SIZE / 1000.0

        # GDS layer numbers must be unique across the whole preview GDS, since
        # every rotation group's job (and the marks job) draws into the same
        # shared top_cell — a fresh `li` per call would let two unrelated
        # groups' geometry collide on GDS layer 0.
        li = 0

        # Rotated-lens mode: emit the unaligned marks job first, then the
        # per-rotation lens jobs below (unchanged).
        if self.rotated_marks_mode:
            li = self._emit_marks_job(file_name, top_cell, cell_size_mm, write_job, li)

        field_cells: List[gdstk.Cell] = []   # per-layer field cells, W frame
        for angle, raw_layers in rotation_groups.items():
            job_layers, images, alignments, marks = self._prepare_job(raw_layers)
            job_name = f"{file_name}_{angle:.1f}deg" if multi else file_name
            tag      = f"@{angle:.1f}" if multi else ""
            li = self._emit_job(job_name, tag, angle, job_layers, images, alignments,
                           marks, top_cell, cell_size_mm, write_job, li,
                           field_cells=field_cells if self.rotated_marks_mode else None)

        lib.add(top_cell, *top_cell.dependencies(True))

        if self.rotated_marks_mode:
            have = {id(c) for c in lib.cells}
            fresh: dict = {}
            for v in self._stepper_views(wafer_cell, field_cells,
                                         list(rotation_groups)):
                for c in (v, *v.dependencies(True)):
                    if id(c) not in have:
                        fresh.setdefault(id(c), c)
            if fresh:
                lib.add(*fresh.values())
        return lib

    def _emit_job(self, job_name: str, tag: str, angle: float,
                  job_layers: List[Layer], images: List[Image],
                  alignments: List[Alignment], marks: List[Mark],
                  top_cell: gdstk.Cell,
                  cell_size_mm: float, write_job: bool, start_li: int = 0,
                  field_cells: Optional[list] = None) -> int:
        """Render one ASML job (one wafer rotation) and its verification cells.

        If `field_cells` is given, each layer's field cell is appended to it
        (used by the rotated-lens stepper-frame views).

        `start_li` is the next free GDS layer number — shared across every
        `_emit_job`/`_emit_marks_job` call in one `write_file()` so unrelated
        rotation groups never reuse the same GDS layer in the combined preview
        GDS. Returns the next free GDS layer number after this job."""
        angle_rad = math.radians(angle)
        cos_a     = math.cos(angle_rad)
        sin_a     = math.sin(angle_rad)

        # ── Alignment mark crosses + PM geometry (verification layer 10) ──
        if marks:
            marks_vis = gdstk.Cell(f"align_marks{tag}")
            for mark in marks:
                cx, cy = mark.loc.x * 1000, mark.loc.y * 1000
                for poly in _make_cross(cx, cy):
                    marks_vis.add(poly)
                if _pm_cell is not None:
                    marks_vis.add(gdstk.Reference(_pm_cell, origin=(cx, cy)))
            top_cell.add(gdstk.Reference(marks_vis))

        # ── GENERAL section ───────────────────────────────────────────────
        gen = GeneralSection(
            wfr_diameter=WAFER_SIZES[self.w],
            wfr_notch=(WAFER_SIZES[self.w] / 2 <= FLAT[self.w]),
            wafer_rotation=angle,
            combine_zero_first=self.zero_one_combined,
            mark_clear_out=False if alignments else None,
        )
        sb = [gen.render()]

        # ── Alignment marks ───────────────────────────────────────────────
        for mark in marks:
            sb.append(AlignmentMarkSection(
                mark_id=mark.id,
                image_id=mark.i.id,
                wafer_side=mark.side,
                mark_location=mark.loc,
            ).render())

        # ── Alignment strategies ──────────────────────────────────────────
        for align in alignments:
            sb.append(align.wfr_align_strategy())

        # ── Mark–strategy bindings ────────────────────────────────────────
        for align in alignments:
            for mark in align.mark:
                sb.append(MarkAlignmentSection(
                    strategy_id=align.name,
                    mark_id=mark.id,
                ).render())

        # ── Image definitions ─────────────────────────────────────────────
        for image in images:
            sb.append(image.image_def_out())

        # ── GDS construction and per-layer job sections ───────────────────
        tsb, rsb, psb, lsb, msb, ssb = [], [], [], [], [], []
        instances = 0
        # `job_li` is this job's OWN LAYER_DEFINITION numbering, written into
        # the .txt job file — local to this call and reset per job, per ASML
        # convention (job layer "0" = marks/reference, "1" = the device layer),
        # independent of how the combined preview GDS numbers layers below.
        job_li    = 0
        # `li` is the preview-GDS layer number, shared across every
        # `_emit_job`/`_emit_marks_job` call in one `write_file()` (see
        # `start_li`) so different jobs' geometry never collides when merged
        # into one viewable top_cell. Only advances for a layer that actually
        # draws geometry — layer id "0" is a marks/reference placeholder with
        # nothing of its own to show.
        li        = start_li
        delta     = 500 #um beyond size for pneumbra and clearance.

        for layer in job_layers:
            print(f"Processing layer {layer.id}")
            # Per-die instance counter, keyed directly by (xdie, ydie) — no
            # fixed array size to outgrow (a fixed A_SIZE[wafer] x A_SIZE[wafer]
            # array here previously IndexError'd whenever a pattern point
            # stepped to a die just outside the nominal wafer-size die count,
            # e.g. a rotated mark landing on the die grid's edge).
            id_grid: dict[tuple[int, int], int] = {}
            tl = gdstk.Cell(f"{layer.id}{tag}")

            psb.append(layer.process_data_out())
            rsb.append(layer.reticle_data_out())

            if layer.id != "0":
                for expo in layer.expos:
                    size_i  = expo.image.size
                    shift_i = expo.image.shift

                    cx   = shift_i.x * 1000
                    cy   = shift_i.y * 1000
                    w_um = size_i.x  * 1000
                    h_um = size_i.y  * 1000
                    pl = gdstk.rectangle(
                        (cx - (w_um + delta) / 2, cy - (h_um + delta) / 2),
                        (cx + (w_um + delta) / 2, cy + (h_um + delta) / 2),
                        layer=100,
                        )
                    bb = pl.bounding_box()

                    tc = gdstk.Cell(f"{layer.id}_{expo.image.id}{tag}")
                    tc.add(pl)
                    bb_min_x, bb_min_y = bb[0]
                    bb_max_x, bb_max_y = bb[1]
                    inside, crossing = [], []
                    for p in expo.image.mask.get_polys():
                        pbb = p.bounding_box()
                        if pbb[1][0] < bb_min_x or pbb[0][0] > bb_max_x or \
                           pbb[1][1] < bb_min_y or pbb[0][1] > bb_max_y:
                            continue  # fully outside
                        if pbb[0][0] >= bb_min_x and pbb[1][0] <= bb_max_x and \
                           pbb[0][1] >= bb_min_y and pbb[1][1] <= bb_max_y:
                            inside.append(p)   # fully inside — no clipping needed
                        else:
                            crossing.append(p) # crosses boundary — needs boolean
                    for p in inside:
                        tc.add(gdstk.Polygon(p.points, layer=li))
                    for ap in gdstk.boolean([pl], crossing, "and", layer=li):
                        tc.add(ap)

                    for point in expo.image.pat.points:
                        px   = point.x - expo.image._zero.x
                        py   = point.y - expo.image._zero.y
                        xdie = round(px / cell_size_mm)
                        ydie = round(py / cell_size_mm)
                        shift_pt = Point(
                            px - xdie * cell_size_mm,
                            py - ydie * cell_size_mm,
                        )
                        die_key = (xdie, ydie)
                        id_grid[die_key] = id_grid.get(die_key, 0) + 1
                        tsb.append(expo.image.instance_out(
                            shift_pt, id_grid[die_key], xdie, ydie
                        ))
                        # Polygon coords in tc are centred at shift*1000.
                        # The layer rotation must pivot about this die's
                        # control point (the stepped field position), not the
                        # mask-coordinate origin.  Rotate the field about the
                        # image zero point, zero*1000 in tc coords, then land
                        # that pivot on the control point point*1000, so every
                        # die gets an identical rotated field, only translated:
                        #   world = R(theta) . (local - zero*1000) + point*1000
                        zx = expo.image._zero.x * 1000
                        zy = expo.image._zero.y * 1000
                        tl.add(gdstk.Reference(
                            tc,
                            origin=(point.x * 1000 - (zx * cos_a - zy * sin_a),
                                    point.y * 1000 - (zx * sin_a + zy * cos_a)),
                            rotation=angle_rad,
                        ))
                        if instances < id_grid[die_key]:
                            instances = id_grid[die_key]

            top_cell.add(gdstk.Reference(tl))
            if field_cells is not None:
                field_cells.append(tl)

            if job_li == 0 and not alignments:
                job_li += 1
            lsb.append(layer.layer_def_out(job_li))
            for mark in marks:
                msb.append(mark.mark_selection(layer.id, expose=(layer.id == "0")))
            if layer.alignment.name:
                ssb.append(layer.strategy_selection_out())
            job_li += 1
            if layer.id != "0":
                li += 1

        # ── Instance definitions ──────────────────────────────────────────
        for i in range(instances):
            sb.append(InstanceDefinitionSection(
                instance_id=f"{i + 1:03d}",
            ).render())

        # IMAGE_DISTRIBUTION is layer-independent (which cells an image goes in);
        # the per-layer assignment lives in RETICLE_DATA.  An image exposed on
        # several layers at the same placement is emitted once per exposure, so
        # drop the byte-identical repeats here, keeping first-seen order.
        seen_dist: set = set()
        for d in tsb:
            if d not in seen_dist:
                seen_dist.add(d)
                sb.append(d)
        sb.extend(lsb)
        sb.extend(msb)
        sb.extend(ssb)
        sb.extend(psb)
        sb.extend(rsb)

        if write_job:
            with open(f"{job_name}.txt", "w", newline="") as f:
                f.write("".join(sb))

        return li

    # ── Rotated-lens marks job ───────────────────────────────────────────────
    def _collect_marks(self) -> List[Mark]:
        """Every alignment mark referenced by any layer, de-duplicated by id,
        in first-seen order."""
        out: List[Mark] = []
        seen: set = set()
        for layer in self.layers:
            if not layer.alignment.on:
                continue
            for mark in layer.alignment.mark:
                if mark.id not in seen:
                    seen.add(mark.id)
                    out.append(mark)
        return out

    def _emit_marks_job(self, file_name: str, top_cell: gdstk.Cell,
                        cell_size_mm: float, write_job: bool, start_li: int = 0) -> int:
        """One unaligned job (wafer flat) that exposes every alignment mark as
        plain geometry — the mark groups printed once up front.  The
        per-rotation lens jobs then run against these.

        The marks are exposed as ordinary DEVICE images (from the same COMBI
        reticle), NOT as ALIGNMENT_MARK / mark images: a mark image "is not a
        device image" and the job compiler refuses to put it in an
        IMAGE_DISTRIBUTION.  No real layer 0 is written; the importer adds one.

        Scaffold: marks are stepped exactly like image placement; per-group
        rotation and anchor positioning come in the next step.

        `start_li`/return value: see `_emit_job` — shared GDS layer counter."""
        marks = self._collect_marks()
        if not marks:
            return start_li

        # one exposure per distinct mark image, stepped over that image's marks
        by_image: dict = {}          # id(mark.i) -> [source Image, [locations]]
        order: list = []
        for mark in marks:
            key = id(mark.i)
            if key not in by_image:
                by_image[key] = [mark.i, []]
                order.append(key)
            by_image[key][1].append(mark.loc)

        marks_layer = Layer("1", alignment=Alignment.no_align(),
                            illume=Illume.conventional())
        images: List[Image] = []
        for idx, key in enumerate(order):
            src, locs = by_image[key]
            # plain device-image id (no BASE_IMAGE_ID "PM"), same reticle/geometry
            dev_id = "MARKS" if len(order) == 1 else f"MARKS{idx + 1}"
            cp = Image(dev_id, src.mask, src.size, src.shift, origin=src._zero)
            cp.set_pattern(Distrib(list(locs)))
            marks_layer.add_expo(Expo(cp, 20.0, 0.0))
            images.append(cp)

        return self._emit_job(f"{file_name}_marks", "", 0.0, [marks_layer], images,
                              [], [], top_cell, cell_size_mm, write_job, start_li)

    def _stepper_views(self, wafer_cell: gdstk.Cell, field_cells: list,
                       rotation_angles: list) -> list:
        """One GDS top cell per rotation — the scene as the sensor sees it for
        that layer.  Everything is assembled in the wafer frame (marks drawn as
        the user supplied them, at their mark locations; fields already carry
        their own rotation), then each view turns the whole scene by the
        prealigner angle ``sign * theta``.  So layer θ's field and a correctly
        pre-rotated group come in axis-aligned; the wafer and every other
        rotation appear turned.  A group that was NOT pre-rotated shows up
        turned by ``sign * theta`` — the same gap correlate_marks reports."""
        sign = self.rotated_marks_sign

        marks_w = gdstk.Cell("marks_W")
        for layer in self.layers:
            if not layer.alignment.on:
                continue
            for mark in layer.alignment.mark:
                cx, cy = mark.loc.x * 1000.0, mark.loc.y * 1000.0
                for poly in _make_cross(cx, cy):
                    marks_w.add(poly)
                glyph = (mark.i.mask.cell if getattr(mark.i, "mask", None)
                         and mark.i.mask.cell is not None else _pm_cell)
                if glyph is not None:
                    marks_w.add(gdstk.Reference(glyph, origin=(cx, cy)))

        scene_w = gdstk.Cell("scene_W")
        scene_w.add(gdstk.Reference(wafer_cell))
        for tl in field_cells:
            scene_w.add(gdstk.Reference(tl))
        scene_w.add(gdstk.Reference(marks_w))

        views = []
        for theta in rotation_angles:
            v = gdstk.Cell(f"stepper@{theta:.1f}")
            v.add(gdstk.Reference(scene_w, rotation=math.radians(sign * theta)))
            views.append(v)
        return views

    # ── Rotated-lens mark correlation ───────────────────────────────────────
    def correlate_marks(self, *, sign: Optional[int] = None, pos_tol_um: float = 5.0,
                        ang_tol_deg: float = 0.5, score_tol: float = 0.4,
                        rms_tol_um: float = 0.3, find_tol_mm: float = 1.0,
                        plot_path: Optional[str] = None, **kw) -> "List[MarkCorr]":
        """For every (rotation, mark) the aligned layers reference, check that
        the mark still registers with the PM template once the wafer is turned
        ``sign * rotation`` on the prealigner.  ``sign`` defaults to
        ``self.rotated_marks_sign``.

        For each mark the physical mark must sit at ``R(-sign*rotation) .
        mark.loc`` so the prealigner turn lands it under the sensor.  Per mark:

          * an exposure is found there (within ``find_tol_mm``) — the pre-rotated
            mark, e.g. a PM printed on the unrotated layer 0/1 —
            (``source="exposed:<id>"``).  dx/dy are the **analytic** placement
            error ``R(sign*rotation)·(exposure - ideal)`` (exact, scale-agnostic);
            dθ and score come from correlating that exposure against the PM
            template (orientation + findability);
          * nothing there and the layer IS rotated → immediate FAIL, the mark
            cannot be found (``source="no exposed mark @ (x,y)mm"``);
          * nothing there and the layer is NOT rotated → the strategy's own mark
            image at the nominal location is correlated against an axis-aligned
            PM (``source="PM@nominal"``); this is the ordinary "does the layer-0
            mark register" check and normally passes.

        Mark ids a strategy requested but that never resolved to a defined mark
        (``alignment.missing``) each get one FAIL row (``source="undefined
        mark"``).

        Returns a list of MarkCorr (residual dx/dy/dθ + correlation score +
        pass/fail).  With ``plot_path`` set, also writes a PNG montage.
        """
        import numpy as np
        if sign is None:
            sign = self.rotated_marks_sign
        self._suppress_combined_first_alignment()
        ref_polys = [np.asarray(p.points, dtype=float)
                     for p in _build_pm_cell().get_polygons()]

        # Every exposed image instance's world placement (mm), for locating the
        # real pre-rotated mark.  "PM" is the alignment mark image itself, not a
        # device exposure — skip it.
        exposed: list = []          # (Point mm, Image)
        for layer in self.layers:
            for expo in layer.expos:
                img = expo.image
                if img.id == "PM" or img.pat is None:
                    continue
                for pt in img.pat.points:
                    exposed.append((pt, img))

        def _nearest(want: "Point"):
            best, bd = None, find_tol_mm
            for pt, img in exposed:
                d = math.hypot(pt.x - want.x, pt.y - want.y)
                if d <= bd:
                    bd, best = d, (img, pt)
            return best

        def _blank_fit(ref_at=None) -> "_MarkFit":
            """Placeholder fit for a row with nothing to correlate (keeps the
            montage in step with `results`)."""
            return _MarkFit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            ref_at if ref_at is not None else [], [], [])

        seen: set = set()
        fits: list = []
        results: List[MarkCorr] = []
        for layer in self.layers:
            if not layer.alignment.on:
                continue
            theta   = layer.rotation
            wr      = sign * theta
            rotated = abs(theta) > 1e-6

            # Strategy mark ids that never resolved to a defined mark.
            for mid in getattr(layer.alignment, "missing", []):
                key = (round(theta, 4), mid)
                if key in seen:
                    continue
                seen.add(key)
                results.append(MarkCorr(mid, theta, 0.0, 0.0, 0.0, 0.0, False,
                                        "undefined mark"))
                fits.append(_blank_fit())

            for mark in layer.alignment.mark:
                key = (round(theta, 4), mark.id)
                if key in seen:
                    continue
                seen.add(key)
                want = _rot_pt(mark.loc, -wr)      # where the physical mark must be
                hit  = _nearest(want)
                if hit is not None:
                    cand_img, cand_loc = hit
                    # Placement error is ANALYTIC: the exposure sits at its
                    # pattern point `cand_loc`, the ideal physical position is
                    # `want` = R(-wr)·mark.loc.  Exact, and independent of the
                    # mark's drawn scale (a hand-drawn PM may be at a different
                    # grating pitch than the generated one).  Correlation still
                    # supplies dθ (orientation) and score (findability).
                    fit = correlate_mark(cand_img, cand_loc, theta,
                                         sign=sign, ref_polys=ref_polys, **kw)
                    err = _rot_pt(Point(cand_loc.x - want.x,
                                        cand_loc.y - want.y), wr)   # sensor frame, mm
                    fit.dx_um, fit.dy_um = err.x * 1000.0, err.y * 1000.0
                    fit.method = "analytic+ncc"
                    source = f"exposed:{cand_img.id}"
                elif rotated:
                    # Nothing exposed where the prealigner turn makes the sensor
                    # look — the rotated layer has no findable mark here.
                    # Explicit FAIL; don't lean on the synthetic PM arriving
                    # rotated by chance.
                    ref_at = [q + (mark.loc.x * 1000.0, mark.loc.y * 1000.0)
                              for q in ref_polys]
                    results.append(MarkCorr(
                        mark.id, theta, 0.0, 0.0, 0.0, 0.0, False,
                        f"no exposed mark @ ({want.x:.2f},{want.y:.2f})mm"))
                    fits.append(_blank_fit(ref_at))
                    continue
                else:
                    # Non-rotated layer: the defined mark (exposed by layer 0)
                    # just has to register with an axis-aligned PM.
                    fit = correlate_mark(mark.i, mark.loc, theta,
                                         sign=sign, ref_polys=ref_polys, **kw)
                    source = "PM@nominal"
                ok = (abs(fit.dx_um) <= pos_tol_um and abs(fit.dy_um) <= pos_tol_um
                      and abs(fit.dtheta_deg) <= ang_tol_deg
                      and fit.score >= score_tol
                      and (fit.rms_um < 0 or fit.rms_um <= rms_tol_um))
                results.append(MarkCorr(
                    mark.id, theta, fit.dx_um, fit.dy_um, fit.dtheta_deg,
                    fit.score, ok, source,
                    fit.rms_um * 1000.0 if fit.rms_um >= 0 else -1.0))
                fits.append(fit)
        if plot_path:
            _plot_correlation(results, fits, plot_path)
        return results


# ── Mark correlation (rotated-lens verification) ─────────────────────────────
@dataclass
class MarkCorr:
    """Public result of one mark's correlation check."""
    mark_id: str
    rotation: float       # deg – the lens rotation this group serves
    dx_um: float          # residual placement error of the mark vs where the
    dy_um: float          #   stepper looks after the prealigner turn
    dtheta_deg: float     # residual misorientation (0 ⇒ comes in axis-aligned)
    score: float          # normalised cross-correlation peak, 0..1 (capture/robustness)
    ok: bool
    source: str = ""      # what was correlated: "exposed:<image id>" when a real
                          #   pre-rotated exposure was found, else "PM@nominal"
    rms_nm: float = -1.0  # ICP RMS residual, nm (<0 ⇒ ICP not run)

    def __str__(self) -> str:
        flag = "PASS" if self.ok else "FAIL"
        src  = f"  [{self.source}]" if self.source else ""
        rms  = f" rms={self.rms_nm:6.1f}nm" if self.rms_nm >= 0 else ""
        # dx/dy in nm — the ICP refine puts a good mark well under 1 µm
        return (f"[{flag}] mark {self.mark_id:>4} @{self.rotation:6.1f}deg  "
                f"dx={self.dx_um * 1000:+8.1f}nm dy={self.dy_um * 1000:+8.1f}nm "
                f"dtheta={self.dtheta_deg:+7.3f}deg  score={self.score:.3f}{rms}{src}")


@dataclass
class _MarkFit:
    dx_um: float
    dy_um: float
    dtheta_deg: float
    score: float
    tx_um: float
    ty_um: float
    ref_polys: list        # PM template polygons, centred at (tx, ty)
    arrived_polys: list    # candidate as it reaches the sensor (post-prealigner)
    cand_polys: list       # candidate polygons at best-fit transform
    rms_um: float = -1.0   # ICP RMS residual (<0 ⇒ ICP not run)
    method: str = "ncc"    # "ncc" (seed only) or "ncc+icp"


def _raster(polys, cx: float, cy: float, n: int, px_um: float):
    """Binary n×n raster of `polys` (list of (N,2) µm arrays) in a window
    centred at (cx, cy), pixel pitch `px_um`.  Row index increases with +y."""
    import numpy as np
    from PIL import Image as _PILImage, ImageDraw as _PILDraw
    im = _PILImage.new("L", (n, n), 0)
    dr = _PILDraw.Draw(im)
    x0 = cx - n * px_um / 2.0
    y0 = cy - n * px_um / 2.0
    for p in polys:
        xy = [((float(px) - x0) / px_um, (float(py) - y0) / px_um) for px, py in p]
        if len(xy) >= 3:
            dr.polygon(xy, fill=255)
    return (np.asarray(im, dtype=np.float32) > 0).astype(np.float32)


def _ncc_peak(ref, cand):
    """Normalised cross-correlation of two rasters via FFT.  Returns
    (score, dx_px, dy_px): the displacement of `cand` relative to `ref`,
    with parabolic sub-pixel refinement."""
    import numpy as np
    r = ref - ref.mean()
    c = cand - cand.mean()
    corr = np.fft.irfft2(np.fft.rfft2(r) * np.conj(np.fft.rfft2(c)), s=ref.shape)
    corr = np.fft.fftshift(corr)
    denom = math.sqrt(float((r * r).sum()) * float((c * c).sum())) + 1e-12
    corr /= denom
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    score = float(corr[py, px])

    def _sub(a, i):
        if 0 < i < len(a) - 1:
            l, m, rr = float(a[i - 1]), float(a[i]), float(a[i + 1])
            d = l - 2.0 * m + rr
            if abs(d) > 1e-9:
                return i + 0.5 * (l - rr) / d
        return float(i)

    cy, cx = ref.shape[0] // 2, ref.shape[1] // 2
    sy = _sub(corr[:, px], py)
    sx = _sub(corr[py, :], px)
    return score, (sx - cx), (sy - cy)


def _world_polys(img: "Image", loc: "Point"):
    """Mark geometry in wafer µm at wafer rotation 0: mask polygons with the
    pivot `img._zero` mapped onto `loc` (mirrors write_file's stepped image)."""
    import numpy as np
    dx = loc.x * 1000.0 - img._zero.x * 1000.0
    dy = loc.y * 1000.0 - img._zero.y * 1000.0
    out = []
    for p in img.mask.get_polys():
        q = np.array(p.points, dtype=float)
        q[:, 0] += dx
        q[:, 1] += dy
        out.append(q)
    return out


def _rot_pt(p: "Point", deg: float) -> "Point":
    """Rotate a Point about the wafer origin by `deg` (degrees, CCW)."""
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return Point(ca * p.x - sa * p.y, sa * p.x + ca * p.y)


def _rot_polys(polys, deg: float, pivot=(0.0, 0.0)):
    import numpy as np
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    ox, oy = pivot
    out = []
    for q in polys:
        x = q[:, 0] - ox
        y = q[:, 1] - oy
        r = np.empty_like(q)
        r[:, 0] = ox + ca * x - sa * y
        r[:, 1] = oy + sa * x + ca * y
        out.append(r)
    return out


# ── Rigid fit on matched vertices (2-D Kabsch) ─────────────────────────────────
def _rigid_fit(a, b):
    """Least-squares rigid transform mapping point set `a` onto `b` (matched,
    same length, corresponding rows).  Returns (theta_rad, tx, ty)."""
    import numpy as np
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.array([[1.0, 0.0],
                  [0.0, math.copysign(1.0, np.linalg.det(vt.T @ u.T))]])
    r = vt.T @ d @ u.T
    t = cb - r @ ca
    return math.atan2(r[1, 0], r[0, 0]), float(t[0]), float(t[1])


def correlate_mark(cand_image: "Image", loc: "Point", rotation_deg: float, *,
                   sign: int = -1, ref_polys=None, ref_prealign: bool = False,
                   ref_world: Optional[list] = None, px_um: float = 1.0,
                   window_um: float = 600.0, coarse_deg: Optional[float] = None,
                   coarse_step: float = 1.0, fine_step: float = 0.02) -> _MarkFit:
    """Does `cand_image`, exposed at `loc`, still register with the reference PM
    once the wafer is turned ``sign * rotation_deg`` on the prealigner?  The
    comparison window is centred where the prealigner turn lands the candidate,
    ``R(sign*rotation) . loc``.

    A coarse FFT normalised-cross-correlation rotation sweep (large capture,
    `coarse_step`) then a fine sweep (`fine_step`) around the peak.  Returns the
    residual (dx, dy, dθ) and a robustness ``score`` (0..1).  Correlation
    resolves dθ well but its dx/dy carry raster quantisation (~0.1·`px_um`) and
    degrade when the candidate's grating pitch differs from the reference's —
    for a discovered real mark, `correlate_marks` overrides dx/dy with the
    analytic placement error instead.

    Reference geometry, in order of precedence:

      * ``ref_world`` – explicit world-µm polygons to match against.
      * else ``ref_polys`` (default: the generated PM), placed at ``(tx, ty)``.
        ``ref_prealign``:
          - False – axis-aligned: the sensor expects a mark whose *geometry* was
            pre-rotated so the prealigner turn brings it in straight.  A mark
            not pre-rotated → dθ ≈ ``sign*rotation``.
          - True – turned by ``sign*rotation``: the mark was only *placed* on a
            rotated grid, geometry axis-aligned.
    """
    import numpy as np
    if ref_polys is None:
        ref_polys = [np.asarray(p.points, dtype=float)
                     for p in _build_pm_cell().get_polygons()]
    if coarse_deg is None:
        coarse_deg = abs(rotation_deg) + 4.0

    wr  = sign * rotation_deg
    a   = math.radians(wr)
    tx  = math.cos(a) * loc.x * 1000.0 - math.sin(a) * loc.y * 1000.0
    ty  = math.sin(a) * loc.x * 1000.0 + math.cos(a) * loc.y * 1000.0

    n = max(16, int(round(window_um / px_um)))
    if ref_world is not None:
        ref_wpol = [q for q in ref_world
                    if q[:, 0].max() >= tx - window_um and q[:, 0].min() <= tx + window_um
                    and q[:, 1].max() >= ty - window_um and q[:, 1].min() <= ty + window_um]
    else:
        ref_src  = _rot_polys(ref_polys, wr, pivot=(0.0, 0.0)) if ref_prealign else ref_polys
        ref_wpol = [q + (tx, ty) for q in ref_src]      # PM template at (tx, ty)
    ref_ras  = _raster(ref_wpol, tx, ty, n, px_um)

    # cand_image is a full flattened mask cell; keep only geometry near this
    # mark's exposure point so unrelated reticle content elsewhere on the plate
    # can't leak into the correlation window (write_file clips to the field —
    # mirror that here).
    cx0, cy0 = loc.x * 1000.0, loc.y * 1000.0
    half     = window_um
    cw = [q for q in _world_polys(cand_image, loc)
          if q[:, 0].max() >= cx0 - half and q[:, 0].min() <= cx0 + half
          and q[:, 1].max() >= cy0 - half and q[:, 1].min() <= cy0 + half]
    cand0 = _rot_polys(cw, wr, pivot=(0.0, 0.0))

    best = _MarkFit(0.0, 0.0, 0.0, -1.0, tx, ty, ref_wpol, cand0, cand0)

    def _sweep(lo, hi, step):
        nonlocal best
        k = lo
        while k <= hi + 1e-9:
            polys = _rot_polys(cand0, k, pivot=(tx, ty))
            score, dxp, dyp = _ncc_peak(ref_ras, _raster(polys, tx, ty, n, px_um))
            if score > best.score:
                # _ncc_peak returns the lag that aligns cand onto ref; the mark's
                # own placement error is the opposite sign.
                best = _MarkFit(-dxp * px_um, -dyp * px_um, -k, score,
                                tx, ty, ref_wpol, cand0, polys)
            k += step

    _sweep(-coarse_deg, coarse_deg, coarse_step)
    _sweep(-best.dtheta_deg - coarse_step, -best.dtheta_deg + coarse_step, fine_step)
    return best


def _plot_correlation(results: "List[MarkCorr]", fits: list, path: str) -> None:
    """PNG montage per mark: PM template (blue), candidate as it arrives at the
    sensor (red), and the candidate after the rigid fit (green, should sit on
    the blue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    m = len(results) or 1
    cols = min(m, 4)
    rows = (m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for i, (res, fit) in enumerate(zip(results, fits)):
        ax = axes[i // cols][i % cols]
        ax.set_axis_on()
        for q in fit.ref_polys:            # what the sensor needs
            ax.add_patch(MplPoly(q, closed=True, fill=False, ec="#1f6fb2", lw=0.8))
        for q in fit.arrived_polys:        # what the sensor actually sees
            ax.add_patch(MplPoly(q, closed=True, fill=False, ec="#d1495b", lw=0.6,
                                 alpha=0.6))
        for q in getattr(fit, "cand_polys", []):   # after the rigid fit
            ax.add_patch(MplPoly(q, closed=True, fill=False, ec="#2e7d32", lw=0.8))
        ax.set_aspect("equal")
        ax.relim(); ax.autoscale_view()
        rms = f"  rms={res.rms_nm:.0f}nm" if getattr(res, "rms_nm", -1) >= 0 else ""
        ax.set_title(f"{'PASS' if res.ok else 'FAIL'}  {res.mark_id} @{res.rotation:.0f}°\n"
                     f"d=({res.dx_um * 1000:+.0f},{res.dy_um * 1000:+.0f})nm  "
                     f"dθ={res.dtheta_deg:+.3f}°  s={res.score:.2f}{rms}",
                     fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    dose      = 20.0
    focus     = 0.0
    cell_size = 10      # mm – distribution grid spacing
    array_col = 3
    array_row = 3

    image_size   = Point(10.0,  10.0)
    image_shift  = Point(-6.0,   6.0)
    image_shift1 = Point(6, 6)
    image_shift2 = Point(6,-6)
    image_shift3 = Point(-6,-6)
    
    image_size3  = Point(.8, 1)

    lib1  = read_gds("ExShanV2.gds")
    cell1 = gdstk.Cell("mask")
    for poly in _cell(lib1, "mask").get_polygons():
        cell1.add(gdstk.Polygon(poly.points, layer=poly.layer, datatype=poly.datatype))

    m1 = Mask("CNF LEADS",   "CNF*", cell1)

    iGRIDLABEL = Image("GRIDLABEL",  m1, image_size,  image_shift)
    iFINETICK = Image("FINETICK",  m1, image_size, image_shift1)
    iDIELABEL = Image("DIELABEL", m1, image_size, image_shift2)
    iPIECE1 = Image("PIECE1", m1, Point(.5,.5), Point(-8,-8))
    iPADS = Image("PADS", m1, image_size3, image_shift3)
    iLEAD1 = Image("LEAD1", m1, Point(.6,.8), Point(-1,-2))
    iLEAD2 = Image("LEAD2", m1, Point(.6,.8), Point(-1,-4))
    iLEAD3 = Image("LEAD3", m1, Point(.6,.8), Point(-1,-6))
    iLEAD4 = Image("LEAD4", m1, Point(.6,.8), Point(-3,-2))
    iLEAD5 = Image("LEAD5", m1, Point(.6,.8), Point(-3,-4))
    iLEAD6 = Image("LEAD6", m1, Point(.6,.8), Point(-3,-6))



    points = []
    for i in range(array_col):
        for j in range(array_row):
            points.append(Point(
                cell_size * (i - (array_col - 1) / 2),
                cell_size * (j - (array_row - 1) / 2),
            ))

    pat = Distrib(points)

    points2 = []
    for i in range(array_col):
        for j in range(array_row):
            points2.append(Point(
                (cell_size + 0.01) * (i - (array_col - 1) / 2),
                (cell_size + 0.01) * (j - (array_row - 1) / 2),
            ))
  
    pat2 = Distrib(points2)

    offset = [Point(10,10)]

    iGRIDLABEL.set_pattern(pat)
    iFINETICK.set_pattern(pat)
    iDIELABEL.set_pattern(pat2)
    iPIECE1.set_pattern(Distrib(offset))
    iPADS.set_pattern(Distrib(offset))
    iLEAD1.set_pattern(Distrib(offset))
    iLEAD2.set_pattern(Distrib(offset))
    iLEAD3.set_pattern(Distrib(offset))
    iLEAD4.set_pattern(Distrib(offset))
    iLEAD5.set_pattern(Distrib(offset))
    iLEAD6.set_pattern(Distrib(offset))

    w         = Wafer()
    eGRIDLABEL = Expo(iGRIDLABEL, dose, focus)
    eFINETICK        = Expo(iFINETICK, dose, focus)
    eDIELABEL = Expo(iDIELABEL, dose, focus)
    ePIECE1   = Expo(iPIECE1, dose, focus)
    ePADS     = Expo(iPADS, dose, focus)
    eLEAD1    = Expo(iLEAD1, dose, focus)
    eLEAD2    = Expo(iLEAD2, dose, focus)
    eLEAD3    = Expo(iLEAD3, dose, focus)
    eLEAD4    = Expo(iLEAD4, dose, focus)
    eLEAD5    = Expo(iLEAD5, dose, focus)
    eLEAD6    = Expo(iLEAD6, dose, focus)
    alignment = Alignment.default4()
    illume    = Illume.conventional()

    layer1 = Layer("1", eGRIDLABEL, alignment, illume)
    # layer1.add_expo(eFINETICK) not necessary
    layer1.add_expo(eDIELABEL)
    layer1.add_expo(ePIECE1)

    layer2 = Layer("2", ePADS, alignment, illume)
    layer2.add_expo(eLEAD1)
    layer2.add_expo(eLEAD2)
    layer2.add_expo(eLEAD3)
    layer2.add_expo(eLEAD4)
    layer2.add_expo(eLEAD5)
    layer2.add_expo(eLEAD6)

    w.add(layer1)
    w.add(layer2)
    w.set_zero_one_combined()


    lib = w.write_file("ShanTest")
    lib.write_gds("ShanTest.gds")
    print("Done.")


if __name__ == "__main__":
    main()
