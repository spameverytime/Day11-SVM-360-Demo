"""Shared helpers for the Day-11 SVM/360 fisheye demo.

Schema follows the Day-11 lecture (slides 13 + 24). Two content kinds are kept
distinct per the lecture's "camera-specific" principle: dataset-native classes
(FishEye8K object classes, WoodScape polygon tags) vs classroom QA conventions
(edge_zone, ignore_reason, confidence, needs_review).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

DEMO_ROOT = Path(__file__).resolve().parent.parent
DATA = DEMO_ROOT / "data"          # build outputs + downloaded subset (git-ignored)
ASSETS = DEMO_ROOT / "assets"      # committed: locked picks + schema only (no raw data)
PICKS = ASSETS / "picks"

DEFAULT_ENV = DEMO_ROOT.parent / "cvat" / ".env"

# SVM surround cameras (WoodScape) and the FishEye8K subset cameras.
WS_CAMS = ["FV", "RV", "MVL", "MVR"]
WS_CAM_NAME = {"FV": "front", "RV": "rear", "MVL": "left", "MVR": "right"}
FE_CAMS = ["camera1", "camera2", "camera3", "camera4"]

# WoodScape instance-annotation tags -> our task geometries.
WS_FREESPACE_TAGS = {"free_space"}
WS_LINE_TAGS = {"lane_marking", "curb"}       # -> parking_curb polyline (line_type attr)
WS_IGNORE_TAGS = {"ego_vehicle"}              # -> ignore_region (reason=ego_body)

# FishEye8K object classes (from samples.json labels).
FE_CLASSES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]


def load_env(env_file: str | None = None) -> None:
    path = Path(env_file) if env_file else DEFAULT_ENV
    if path.exists():
        load_dotenv(path)
    local = DEMO_ROOT / ".env"
    if local.exists():
        load_dotenv(local, override=True)


def cvat_conn() -> tuple[str, str, str]:
    url = os.environ.get("CVAT_URL", "http://localhost:8080")
    user = os.environ.get("CVAT_ADMIN_USER", "admin")
    pw = os.environ.get("CVAT_ADMIN_PASSWORD", "")
    if not pw:
        raise SystemExit("CVAT_ADMIN_PASSWORD missing. Put it in cvat/.env or ./.env.")
    return url, user, pw


def kaggle_headers() -> dict:
    load_env()
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not tok:
        raise SystemExit("KAGGLE_API_TOKEN missing (put it in ../cvat/.env or ./.env; "
                         "get one at https://www.kaggle.com/settings/api).")
    return {"Authorization": f"Bearer {tok}"}


# ---- CVAT attribute-spec builders --------------------------------------------

def _select(name, values, mutable=False):
    return {"name": name, "mutable": mutable, "input_type": "select",
            "default_value": values[0], "values": values}


def _checkbox(name, mutable=False):
    return {"name": name, "mutable": mutable, "input_type": "checkbox",
            "default_value": "false", "values": ["false", "true"]}


# ---- label schemas (slide 13 + 24) -------------------------------------------

def _obj_attrs(cam):
    return [
        _checkbox("occluded"),
        _checkbox("truncated"),
        _checkbox("edge_zone"),
        _select("ignore_reason", ["none", "ego_body", "seam", "lens_vignette", "unreadable"]),
        _select("camera_id", [cam]),
    ]


def object_labels(cam):
    """Task A — road_object boxes. One label per FishEye8K class for IoU+class scoring."""
    palette = {"Bus": "#ff8c00", "Bike": "#00d0ff", "Car": "#7cff00",
               "Pedestrian": "#ff3b30", "Truck": "#a05aff"}
    return [{"name": c, "type": "rectangle", "color": palette[c], "attributes": _obj_attrs(cam)}
            for c in FE_CLASSES]


def freespace_labels(cam):
    """Task B — free_space polygon (drivable-near-vehicle)."""
    return [{
        "name": "free_space", "type": "polygon", "color": "#7cff00",
        "attributes": [
            _select("surface", ["road", "mixed", "unknown"]),
            _select("confidence", ["high", "medium", "low"]),
            _select("blocked_by", ["none", "parked_vehicle", "pedestrian", "curb", "unknown"]),
            _select("camera_id", [cam]),
        ],
    }]


def line_labels(cam):
    """Task C — parking_line / curb polyline."""
    return [{
        "name": "parking_curb", "type": "polyline", "color": "#ffd400",
        "attributes": [
            _select("line_type", ["lane_marking", "curb"]),
            _select("visibility", ["visible", "partially_occluded", "faded"]),
            _checkbox("edge_zone"),
            _select("camera_id", [cam]),
        ],
    }]


def ignore_labels(cam):
    """Task D — ignore_region polygon (part of the label, not 'skip' — slide 23)."""
    return [{
        "name": "ignore_region", "type": "polygon", "color": "#a05aff",
        "attributes": [
            _select("reason", ["ego_body", "lens_border", "stitch_seam",
                               "unreadable", "privacy_or_policy"]),
            _select("camera_id", [cam]),
        ],
    }]
