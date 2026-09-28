"""Tasks B/C/D assets — WoodScape per SVM camera, golden GT from instance polygons.

Downloads the locked WoodScape image + instance-annotation subset from Kaggle
(needs KAGGLE_API_TOKEN) and derives, per camera (FV/RV/MVL/MVR):

  B free_space   -> polygons  (tag free_space)             -> free_space.json (COCO 1.0)
  C parking_curb -> polylines (tags lane_marking, curb)    -> lines.xml       (CVAT 1.1)
  D ignore_region-> polygons  (tag ego_vehicle)            -> ignore.json     (COCO 1.0)

Thin lane/curb polygons are reduced to a centerline polyline (PCA axis + binned
perpendicular mean) so students' polylines can be scored against them. No
re-hosting: raw images/annotations land under data/ only.
"""
import json
import time
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import requests
from PIL import Image

from _common import (DATA, PICKS, WS_CAMS, WS_FREESPACE_TAGS, WS_IGNORE_TAGS,
                     WS_LINE_TAGS, kaggle_headers, load_env)

KAGGLE_DS = "subarnadasgupta/woodscapes"
DL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DS}"
OUT = DATA / "woodscape"


def kfetch(session, path, tries=4):
    for _ in range(tries):
        try:
            r = session.get(DL, params={"file_name": path}, timeout=90)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            time.sleep(1.5)
        time.sleep(0.4)
    raise SystemExit(f"failed to download {path}")


def poly_area(pts):
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def centerline(pts, n_bins=14):
    """Thin polygon -> ordered centerline polyline via PCA axis + binned mean."""
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]
    t = (pts - c) @ axis
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-6:
        return pts[[0, -1]]
    edges = np.linspace(lo, hi, n_bins + 1)
    line = []
    for i in range(n_bins):
        m = (t >= edges[i]) & (t <= edges[i + 1])
        if m.any():
            line.append(pts[m].mean(0))
    line = np.array(line)
    return line if len(line) >= 2 else pts[[0, -1]]


def as_polys(seg):
    """Normalize a WoodScape segmentation into a list of Nx2 point arrays."""
    if not seg:
        return []
    # seg is a list of [x,y] points, or a list of such rings
    if isinstance(seg[0][0], (int, float)):
        return [np.asarray(seg, float)]
    return [np.asarray(r, float) for r in seg if len(r) >= 3]


def coco_poly_ann(aid, img_id, cat_id, pts, attrs):
    xs, ys = pts[:, 0], pts[:, 1]
    x, y = float(xs.min()), float(ys.min())
    w, h = float(xs.max() - x), float(ys.max() - y)
    return {"id": aid, "image_id": img_id, "category_id": cat_id,
            "segmentation": [pts.reshape(-1).tolist()],
            "bbox": [x, y, w, h], "area": float(poly_area(pts)),
            "iscrowd": 0, "attributes": attrs}


def write_lines_xml(path, frames):
    """frames: list of (fid, name, w, h, [(pts, line_type)])."""
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for fid, name, w, h, shapes in frames:
        out.append(f'  <image id="{fid}" name="{escape(name)}" width="{w}" height="{h}">')
        for pts, line_type in shapes:
            ps = ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            out.append(f'    <polyline label="parking_curb" source="manual" occluded="0" '
                       f'points="{ps}" z_order="0">')
            out.append(f'      <attribute name="line_type">{line_type}</attribute>')
            out.append('      <attribute name="visibility">visible</attribute>')
            out.append('      <attribute name="edge_zone">false</attribute>')
            out.append(f'      <attribute name="camera_id">{escape(_cam_of(name))}</attribute>')
            out.append("    </polyline>")
        out.append("  </image>")
    out.append("</annotations>")
    path.write_text("\n".join(out), encoding="utf-8")


def _cam_of(name):
    for cam in WS_CAMS:
        if name.endswith(f"_{cam}.png"):
            return cam
    return "FV"


def build_camera(session, cam, ids):
    img_dir = OUT / cam / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fs = {"info": {"description": f"Day11 WoodScape {cam} free_space GT"}, "licenses": [],
          "images": [], "annotations": [],
          "categories": [{"id": 1, "name": "free_space", "supercategory": "surface"}]}
    ig = {"info": {"description": f"Day11 WoodScape {cam} ignore_region GT"}, "licenses": [],
          "images": [], "annotations": [],
          "categories": [{"id": 1, "name": "ignore_region", "supercategory": "ignore"}]}
    line_frames = []
    fs_aid = ig_aid = 1

    names = [f"{idx:05d}_{cam}.png" for idx in ids]
    for fid, (idx, name) in enumerate(zip(ids, names)):
        # image
        dst = img_dir / name
        if not (dst.exists() and dst.stat().st_size > 0):
            dst.write_bytes(kfetch(session, f"Woodscapes/rgb_images/{name}"))
        w, h = Image.open(dst).size
        # annotations
        js = json.loads(kfetch(session, f"Woodscapes/instance_annotations/{idx:05d}_{cam}.json"))
        anns = js[f"{idx:05d}_{cam}.json"]["annotation"]
        fs["images"].append({"id": fid + 1, "file_name": name, "width": w, "height": h})
        ig["images"].append({"id": fid + 1, "file_name": name, "width": w, "height": h})
        line_shapes = []
        for a in anns:
            tags = set(a.get("tags", []))
            for pts in as_polys(a.get("segmentation")):
                if len(pts) < 3:
                    continue
                if tags & WS_FREESPACE_TAGS:
                    fs["annotations"].append(coco_poly_ann(
                        fs_aid, fid + 1, 1, pts,
                        {"surface": "road", "confidence": "high", "blocked_by": "none",
                         "camera_id": cam}))
                    fs_aid += 1
                if tags & WS_IGNORE_TAGS:
                    ig["annotations"].append(coco_poly_ann(
                        ig_aid, fid + 1, 1, pts, {"reason": "ego_body", "camera_id": cam}))
                    ig_aid += 1
                if tags & WS_LINE_TAGS:
                    lt = "curb" if "curb" in tags else "lane_marking"
                    line_shapes.append((centerline(pts), lt))
        line_frames.append((fid, name, w, h, line_shapes))
        time.sleep(0.2)

    (OUT / cam / "free_space.json").write_text(json.dumps(fs))
    (OUT / cam / "ignore.json").write_text(json.dumps(ig))
    write_lines_xml(OUT / cam / "lines.xml", line_frames)
    nlines = sum(len(s) for *_, s in line_frames)
    print(f"{cam}: {len(fs['images'])} imgs | free_space={len(fs['annotations'])} "
          f"lines={nlines} ignore={len(ig['annotations'])}")


def main():
    load_env()
    load = json.loads((PICKS / "woodscape_picks.json").read_text())
    s = requests.Session(); s.headers.update(kaggle_headers())
    for cam in WS_CAMS:
        ids = load.get(cam, [])
        if ids:
            build_camera(s, cam, ids)


if __name__ == "__main__":
    main()
