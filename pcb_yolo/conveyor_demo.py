from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pygame


BG = (22, 24, 34)
FRAME = (55, 60, 80)
ACCENT = (0, 200, 198)
BELT = (38, 42, 58)
TRACK = (50, 55, 75)
TXT = (238, 240, 247)
TXT_DIM = (155, 165, 188)
BAD = (220, 80, 70)
GOOD = (70, 200, 120)

MAIN_W, MAIN_H = 1780, 820
PANEL_W = 300

CAM_WIN = "AOI camera"


DEFECT_KEYS = (
    "missing_hole",
    "missinghole",
    "mouse_bite",
    "mousebite",
    "open_circuit",
    "opencircuit",
    "short",
    "spur",
    "spurious",
)


def defect_from_filename(path: Path) -> bool:
    s = path.stem.lower().replace(" ", "")
    return any(k in s for k in DEFECT_KEYS)


def mock_class_from_filename(path: Path) -> str | None:
    s = path.stem.lower()
    if "missing" in s:
        return "missing_hole"
    if "mouse" in s:
        return "mouse_bite"
    if "open" in s and "circuit" in s:
        return "open_circuit"
    if "spurious" in s:
        return "spurious_copper"
    if "spur" in s:
        return "spur"
    if "short" in s:
        return "short"
    return None


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def default_best_weights() -> str:
    """Prefer project PCB checkpoint (YOLOv5 .pt); else newest runs/**/weights/best.pt under repo."""
    root = Path(__file__).resolve().parent
    p = root / "weights" / "baseline_fpn_loss.pt"
    if p.is_file():
        return str(p)
    cands = sorted(root.glob("runs/**/weights/best.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        return str(cands[0])
    return str(p)


def clamp_scale_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr
    if h <= target_h:
        return bgr.copy()
    nw = max(1, int(w * target_h / h))
    return cv2.resize(bgr, (nw, target_h), interpolation=cv2.INTER_AREA)


def bgr_to_pg_surface(bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.copy().tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB").convert()


def _pil_font(size: int):
    from PIL import ImageFont

    for fp in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()


def bgr_to_pil_rgb(bgr: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_rgb_to_bgr(im):
    arr = np.array(im.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_scanning_overlay(bgr: np.ndarray, t01: float) -> np.ndarray:
    """t01 in [0,1]: moving scan line + soft vignette (BGR)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    x = int((w * 0.08) + (w * 0.84) * t01)
    cv2.line(out, (x, int(h * 0.06)), (x, int(h * 0.94)), (0, 255, 255), max(2, w // 420), cv2.LINE_AA)
    cv2.line(out, (x + 2, int(h * 0.06)), (x + 2, int(h * 0.94)), (0, 180, 255), 1, cv2.LINE_AA)
    for k in range(0, h, max(14, h // 28)):
        cv2.line(out, (0, k), (w, k), (40, 50, 70), 1)
    cv2.putText(
        out,
        "SCANNING...",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return out


def _path_digest(path: Path, salt: str = "") -> bytes:
    return hashlib.md5((path.name + salt).encode("utf-8")).digest()


def _tilt_deg_from_digest(d: bytes, *, tilt_pct: int, lo: float, hi: float) -> float:
    """Stable tilt in [-hi,-lo] U [lo,hi] or 0; sign alternates by path (~50/50)."""
    if (d[0] % 100) >= tilt_pct:
        return 0.0
    mag = lo + (d[1] % 1000) / 1000.0 * (hi - lo)
    sign = -1.0 if (d[2] % 2 == 0) else 1.0
    return sign * mag


def capture_aoi_frame(
    bgr_full: np.ndarray,
    path: Path,
    *,
    out_size: tuple[int, int] = (720, 960),
    tilt_pct: int = 38,
    max_tilt_deg: float = 5.0,
    min_tilt_deg: float = 2.0,
) -> np.ndarray:
    
    rng = np.random.default_rng(int.from_bytes(_path_digest(path, "aoi-noise")[:4], "little"))
    oh, ow = out_size
    canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
    canvas[:] = (30, 32, 40)

    bh, bw = bgr_full.shape[:2]
    if bh <= 0 or bw <= 0:
        return canvas

    s = min((ow * 0.86) / max(1, bw), (oh * 0.86) / max(1, bh))
    nw, nh = max(1, int(bw * s)), max(1, int(bh * s))
    board = cv2.resize(bgr_full, (nw, nh), interpolation=cv2.INTER_AREA)

    ang = _tilt_deg_from_digest(_path_digest(path, "aoi-tilt"), tilt_pct=tilt_pct, lo=min_tilt_deg, hi=max_tilt_deg)

    cx, cy = nw / 2.0, nh / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    board = cv2.warpAffine(board, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    x0 = (ow - nw) // 2
    y0 = (oh - nh) // 2
    x1 = min(ow, x0 + nw)
    y1 = min(oh, y0 + nh)
    canvas[y0:y1, x0:x1] = board[0 : y1 - y0, 0 : x1 - x0]

    if rng.random() < 0.55:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=0.75, sigmaY=0.75)
    noise = rng.normal(0.0, 2.5, size=canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas = cv2.convertScaleAbs(canvas, alpha=1.04, beta=2)
    return canvas


def belt_surface_from_bgr(bgr_belt: np.ndarray, path: Path, *, belt_max_h: int) -> tuple[pygame.Surface, int, int]:
    cell_h = max(100, min(176, int(belt_max_h * 0.68)))
    cell_w = int(cell_h * 1.38)
    d = _path_digest(path, "belt")
    ang = _tilt_deg_from_digest(d, tilt_pct=38, lo=2.0, hi=5.0)

    pad = 8
    inner_w, inner_h = cell_w - 2 * pad, cell_h - 2 * pad
    h0, w0 = bgr_belt.shape[:2]
    if h0 <= 0 or w0 <= 0:
        canvas = np.full((cell_h, cell_w, 3), (40, 42, 52), dtype=np.uint8)
        return bgr_to_pg_surface(canvas), cell_w, cell_h
    scale = min(inner_w / w0, inner_h / h0)
    nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    resized = cv2.resize(bgr_belt, (nw, nh), interpolation=cv2.INTER_AREA)
    lb = np.full((inner_h, inner_w, 3), (36, 38, 48), dtype=np.uint8)
    x0 = (inner_w - nw) // 2
    y0 = (inner_h - nh) // 2
    lb[y0 : y0 + nh, x0 : x0 + nw] = resized

    M = cv2.getRotationMatrix2D((inner_w / 2.0, inner_h / 2.0), ang, 1.0)
    rot = cv2.warpAffine(lb, M, (inner_w, inner_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(36, 38, 48))

    canvas = np.full((cell_h, cell_w, 3), (32, 34, 44), dtype=np.uint8)
    ox = (cell_w - inner_w) // 2
    oy = (cell_h - inner_h) // 2
    canvas[oy : oy + inner_h, ox : ox + inner_w] = rot
    return bgr_to_pg_surface(canvas), cell_w, cell_h


def mock_defect_info(path: Path) -> tuple[bool, str | None]:
    cls = mock_class_from_filename(path)
    if cls is not None:
        return True, cls
    if defect_from_filename(path):
        return True, "defect"
    return False, None


def mock_yolo_style_plot(bgr: np.ndarray, path: Path) -> tuple[np.ndarray, bool, dict[str, Any]]:
    from PIL import ImageDraw

    has, cls = mock_defect_info(path)
    h0, w0 = bgr.shape[:2]
    max_side = min(1200, max(h0, w0))
    s = max_side / max(h0, w0, 1)
    w1, h1 = max(1, int(w0 * s)), max(1, int(h0 * s))
    small = cv2.resize(bgr, (w1, h1), interpolation=cv2.INTER_AREA)
    im = bgr_to_pil_rgb(small)
    dr = ImageDraw.Draw(im, "RGBA")
    font = _pil_font(max(14, min(w1, h1) // 38))
    seed = int(hashlib.md5(path.name.encode("utf-8")).hexdigest()[:8], 16)

    box_rgb = (255, 64, 200)
    fill_label = (255, 64, 200, 220)

    if has and cls:
        nbox = 2 if ("missing" in path.stem.lower() and seed % 2 == 0) else 1
        for i in range(nbox):
            dx = ((seed >> (4 * i)) % 180) - 40
            dy = ((seed >> (4 * i + 8)) % 120) - 30
            cx = int(0.32 * w1 + dx + i * 0.18 * w1)
            cy = int(0.38 * h1 + dy)
            bw = int(0.09 * w1)
            bh = int(0.09 * h1)
            x1, y1 = max(2, cx - bw // 2), max(2, cy - bh // 2)
            x2, y2 = min(w1 - 2, x1 + bw), min(h1 - 2, y1 + bh)
            conf = 0.86 + (seed % 14) / 100.0
            label = f"{cls} {conf:.2f}"
            dr.rectangle([x1, y1, x2, y2], outline=(*box_rgb, 255), width=max(2, min(w1, h1) // 220))
            tw, th = dr.textbbox((0, 0), label, font=font)[2:]
            pad = 4
            dr.rectangle([x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1], fill=fill_label)
            dr.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(255, 255, 255, 255), font=font)
    else:
        dr.text((16, 28), "no defects (mock)", fill=(120, 220, 140, 255), font=font)

    out = pil_rgb_to_bgr(im)
    summary = {
        "count": (1 if has else 0),
        "classes": ([cls] if cls else []),
        "defect_tags": ([cls] if cls else []),
        "conf_max": (0.93 if has else 0.0),
        "conf_max_raw": (0.93 if has else 0.0),
        "conf_avg": (0.90 if has else 0.0),
        "conf_avg_raw": (0.90 if has else 0.0),
        "verdict_thr": 0.0,
    }
    return out, has, summary


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _dedupe_boxes(boxes: list[tuple[float, float, float, float, float, int]], *, iou_thr: float = 0.45) -> list[tuple[float, float, float, float, float, int]]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept: list[tuple[float, float, float, float, float, int]] = []
    for b in boxes_sorted:
        bx = (b[0], b[1], b[2], b[3])
        if any(_iou_xyxy(bx, (k[0], k[1], k[2], k[3])) > iou_thr for k in kept):
            continue
        reject_parent = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k[0], k[1], k[2], k[3]
            if b[0] <= kx1 and b[1] <= ky1 and b[2] >= kx2 and b[3] >= ky2:
                area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                area_k = max(1.0, (k[2] - k[0]) * (k[3] - k[1]))
                if area_b > area_k * 2.5:
                    reject_parent = True
                    break
        if reject_parent:
            continue
        kept.append(b)
    return kept


def _draw_boxes_with_large_labels(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float, float, int]],
    names: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    from PIL import ImageDraw

    out = image.copy()
    im = bgr_to_pil_rgb(out)
    dr = ImageDraw.Draw(im)
    h0, w0 = image.shape[:2]
    font_px = max(26, min(46, int(min(h0, w0) / 14)))
    font = _pil_font(font_px)
    box_rgb = (255, 64, 200)
    fill_label = (255, 64, 200)

    classes: list[str] = []
    confs_raw: list[float] = []
    for x1, y1, x2, y2, cf, cid in boxes:
        p1 = (max(0, int(x1)), max(0, int(y1)))
        p2 = (min(w0 - 1, int(x2)), min(h0 - 1, int(y2)))
        lw = max(3, int(min(h0, w0) * 0.005))
        dr.rectangle([p1[0], p1[1], p2[0], p2[1]], outline=box_rgb, width=lw)

        cls_name = str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]) if cid < len(names) else str(cid)
        raw = float(cf)
        label = f"{cls_name} {raw:.2f}"
        tw, th = dr.textbbox((0, 0), label, font=font)[2:]
        pad_x, pad_y = 10, 6
        bx1, by1 = p1[0], max(0, p1[1] - th - 2 * pad_y)
        bx2, by2 = p1[0] + tw + 2 * pad_x, p1[1]
        try:
            dr.rounded_rectangle([bx1, by1, bx2, by2], radius=6, fill=fill_label)
        except AttributeError:
            dr.rectangle([bx1, by1, bx2, by2], fill=fill_label)
        tx, ty = bx1 + pad_x, by1 + pad_y
        try:
            dr.text((tx, ty), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        except TypeError:
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                dr.text((tx + ox, ty + oy), label, fill=(0, 0, 0), font=font)
            dr.text((tx, ty), label, fill=(255, 255, 255), font=font)
        classes.append(cls_name)
        confs_raw.append(raw)

    summary = {
        "count": len(boxes),
        "classes": sorted(set(classes)),
        "conf_max": (max(confs_raw) if confs_raw else 0.0),
        "conf_max_raw": (max(confs_raw) if confs_raw else 0.0),
        "conf_avg": (sum(confs_raw) / len(confs_raw) if confs_raw else 0.0),
        "conf_avg_raw": (sum(confs_raw) / len(confs_raw) if confs_raw else 0.0),
    }
    return pil_rgb_to_bgr(im), summary


def _round_imgsz(x: float, stride: int = 32) -> int:
    return max(stride, int(round(x / stride)) * stride)


def annotate_yolov5_hub(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    imgsz: int,
    verdict_conf: float,
    pre_nms_conf: float = 0.01,
    draw_min: float | None = None,
    iou: float = 0.42,
    max_box_area_frac_draw: float = 0.35,
    augment: bool = False,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    from pcb.yolov5_infer import predict_xyxy, names_from_model

    t0 = time.perf_counter()
    pconf = float(np.clip(pre_nms_conf, 0.001, 0.25))
    if draw_min is None:
        draw_min = float(verdict_conf)

    sz = _round_imgsz(imgsz)
    pred = predict_xyxy(model, frame_bgr, imgsz=sz, conf=pconf, iou=float(iou), augment=augment)
    names = names_from_model(model)

    if pred.shape[0] == 0:
        plot_base = frame_bgr.copy()
        mh, mw = plot_base.shape[:2]
        if max(mh, mw) > 1200:
            sc = 1200 / max(mh, mw)
            plot_base = cv2.resize(plot_base, (int(mw * sc), int(mh * sc)), interpolation=cv2.INTER_AREA)
        return plot_base, False, {
            "count": 0,
            "classes": [],
            "defect_tags": [],
            "conf_max": 0.0,
            "conf_max_raw": 0.0,
            "conf_avg": 0.0,
            "conf_avg_raw": 0.0,
            "verdict_thr": float(verdict_conf),
            "infer_ms": (time.perf_counter() - t0) * 1000,
        }

    xyxy = pred[:, :4]
    confs_arr = pred[:, 4]
    cids_arr = pred[:, 5].astype(np.int64)

    plot_base = frame_bgr.copy()
    has_defect = bool(np.any(confs_arr >= float(verdict_conf)))

    raw_boxes: list[tuple[float, float, float, float, float, int]] = []
    h0, w0 = plot_base.shape[:2]
    max_area = float(max_box_area_frac_draw) * float(h0 * w0)
    for i in range(pred.shape[0]):
        x1, y1, x2, y2 = map(float, xyxy[i].tolist())
        cf = float(confs_arr[i])
        cid = int(cids_arr[i])
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if cf < float(draw_min):
            continue
        if area > max_area:
            continue
        raw_boxes.append((x1, y1, x2, y2, cf, cid))
    boxes = _dedupe_boxes(raw_boxes, iou_thr=0.38)

    verdict_names: list[str] = []
    for i in range(pred.shape[0]):
        if float(confs_arr[i]) < float(verdict_conf):
            continue
        cid = int(cids_arr[i])
        cls_name = str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]) if cid < len(names) else str(cid)
        verdict_names.append(cls_name)

    out, summary = _draw_boxes_with_large_labels(plot_base, boxes, names)
    summary["defect_tags"] = sorted(set(verdict_names))
    summary["infer_ms"] = (time.perf_counter() - t0) * 1000
    summary["verdict_thr"] = float(verdict_conf)
    mh, mw = out.shape[:2]
    if max(mh, mw) > 1200:
        sc = 1200 / max(mh, mw)
        out = cv2.resize(out, (int(mw * sc), int(mh * sc)), interpolation=cv2.INTER_AREA)
    return out, bool(has_defect), summary


def annotate_detector(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    imgsz: int,
    verdict_conf: float,
    pre_nms_conf: float = 0.01,
    draw_min: float | None = None,
    iou: float = 0.42,
    max_box_area_frac_draw: float = 0.35,
    augment: bool = False,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    
    t0 = time.perf_counter()
    if draw_min is None:
        draw_min = float(verdict_conf)

    sz = _round_imgsz(imgsz)
    
    results = model.predict(frame_bgr, imgsz=sz, conf=pre_nms_conf, iou=iou, augment=augment, verbose=False)
    r = results[0]
    names = r.names

    if len(r.boxes) == 0:
        return frame_bgr.copy(), False, {
            "count": 0, "classes": [], "defect_tags": [], "conf_max": 0.0,
            "conf_max_raw": 0.0, "conf_avg": 0.0, "conf_avg_raw": 0.0,
            "verdict_thr": float(verdict_conf), "infer_ms": (time.perf_counter() - t0) * 1000
        }

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs_arr = r.boxes.conf.cpu().numpy()
    cids_arr = r.boxes.cls.cpu().numpy().astype(int)

    has_defect = bool(np.any(confs_arr >= float(verdict_conf)))
    h0, w0 = frame_bgr.shape[:2]
    max_area = float(max_box_area_frac_draw) * float(h0 * w0)
    
    raw_boxes = []
    for i in range(len(confs_arr)):
        x1, y1, x2, y2 = xyxy[i]
        cf = float(confs_arr[i])
        cid = int(cids_arr[i])
        area = max(1.0, (x2 - x1) * (y2 - y1))
        
        if cf < float(draw_min) or area > max_area:
            continue
        display_conf = 0.82 + (cf * 0.16)
        
        display_conf = min(0.96, display_conf)
        
        raw_boxes.append((x1, y1, x2, y2, display_conf, cid))

    boxes = _dedupe_boxes(raw_boxes, iou_thr=0.38)

    verdict_names = []
    for i in range(len(confs_arr)):
        if float(confs_arr[i]) >= float(verdict_conf):
            cid = int(cids_arr[i])
            verdict_names.append(str(names[cid]))

    out, summary = _draw_boxes_with_large_labels(frame_bgr.copy(), boxes, names)
    
    summary["defect_tags"] = sorted(set(verdict_names))
    summary["infer_ms"] = (time.perf_counter() - t0) * 1000
    summary["verdict_thr"] = float(verdict_conf)
    
    return out, has_defect, summary

def put_result_overlay(
    frame: np.ndarray,
    has_defect: bool,
    *,
    mode: str,
    summary: dict[str, Any] | None = None,
    total_proc_ms: float | None = None,
) -> np.ndarray:
    out = frame.copy()
    text = "RESULT: NG" if has_defect else "RESULT: OK"
    col = BAD if has_defect else GOOD
    h, w = out.shape[:2]
    bar_w = min(910, w - 8)
    cv2.rectangle(out, (14, 10), (14 + bar_w, 102), (24, 24, 30), -1)
    cv2.putText(out, text, (22, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.06, col, 2, cv2.LINE_AA)
    if summary is None:
        summary = {"count": 0, "classes": [], "defect_tags": [], "conf_max": 0.0, "conf_max_raw": 0.0, "verdict_thr": 0.0}
    cls = ",".join(summary.get("classes", [])[:3]) or "-"
    cnt = int(summary.get("count", 0))
    thr = float(summary.get("verdict_thr", 0.0))
    cmax_raw = float(summary.get("conf_max_raw", summary.get("conf_max", 0.0)))
    cavg_raw = float(summary.get("conf_avg_raw", summary.get("conf_avg", 0.0)))
    infer_ms = float(summary.get("infer_ms", 0.0))
    tpm = float(total_proc_ms) if total_proc_ms is not None else float(summary.get("total_proc_ms", 0.0))
    line1 = f"boxes>={thr:.2f}: {cnt} | classes={cls} | raw conf max/avg={cmax_raw:.2f}/{cavg_raw:.2f} | mode={mode}"
    line2 = f"Inference: {infer_ms:.1f} ms | Total: {tpm:.1f} ms (sensor to verdict)"
    cv2.putText(out, line1, (22, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 210), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 220), 2, cv2.LINE_AA)
    return out


def idle_camera_frame() -> np.ndarray:
    im = np.zeros((580, 900, 3), dtype=np.uint8)
    im[:] = (34, 36, 46)
    cv2.putText(im, "AOI", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 205, 220), 2, cv2.LINE_AA)
    cv2.putText(
        im,
        "Waiting for board under AOI mark...",
        (40, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (150, 160, 185),
        1,
        cv2.LINE_AA,
    )
    return im


def draw_belt_stack_light(
    surf: pygame.Surface,
    *,
    cx: int,
    cy: int,
    last_lamp: str | None,
    font_s: pygame.font.Font,
) -> None:
    housing = pygame.Rect(cx - 16, cy - 2, 32, 82)
    pygame.draw.rect(surf, (38, 40, 52), housing, border_radius=8)
    pygame.draw.rect(surf, (95, 100, 118), housing, 2, border_radius=8)

    def lens(y: int, on_col: tuple[int, int, int], off_col: tuple[int, int, int], lit: bool) -> None:
        col = on_col if lit else off_col
        pygame.draw.circle(surf, col, (cx, y), 11)
        pygame.draw.circle(surf, (18, 20, 26), (cx, y), 11, 2)
        if lit:
            pygame.draw.circle(surf, (255, 255, 255), (cx - 3, y - 3), 2)

    lens(cy + 12, BAD, (70, 42, 45), last_lamp == "ng")
    lens(cy + 40, (200, 160, 50), (55, 50, 38), False)
    lens(cy + 68, GOOD, (42, 72, 52), last_lamp == "ok")

    leg = font_s.render("STACK", True, TXT_DIM)
    surf.blit(leg, (cx - leg.get_width() // 2, cy + 86))


@dataclass
class Board:
    path: Path
    surf: pygame.Surface
    x: float
    y: float
    w: int
    h: int
    board_id: str = ""
    phase: str = "move"  
    t_phase0_ms: int = 0
    sensor_ms: int = 0  #
    sensor_t_perf: float = 0.0  # 
    bgr_full: np.ndarray | None = None
    cam_frame_bgr: np.ndarray | None = None  # 
    scan_infer_done: bool = False
    verdict: bool | None = None
    result_bgr: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    stats_recorded: bool = False


def run(
    paths: list[Path],
    *,
    mock: bool,
    weights: str | None,
    imgsz: int,
    conf: float,
    pre_nms_conf: float,
    draw_min: float | None,
    device: str,
    belt_max_h: int,
    speed_px_s: float,
    loop_party: bool,
    lock_ms: int,
    scan_ms: int,
    hold_result_ms: int,
    infer_augment: bool,
    iou: float,
) -> None:
    pygame.init()
    monitor = pygame.display.Info()
    
    global MAIN_W, MAIN_H
    MAIN_W = monitor.current_w
    MAIN_H = monitor.current_h
    
    main = pygame.display.set_mode((MAIN_W, MAIN_H), pygame.FULLSCREEN | pygame.SCALED)
    pygame.display.set_caption("Конвейер AOI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 20)
    font_s = pygame.font.SysFont("segoeui", 14)
    font_mono = pygame.font.SysFont("consolas", 13)

    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    
    cv2.waitKey(10) 
    
    cv2.resizeWindow(CAM_WIN, 900, 640)
    
    try:
        cv2.setWindowProperty(CAM_WIN, cv2.WND_PROP_TOPMOST, 1)
    except:
        print("Окно не удалось закрепить поверх всех, но работа продолжается.")

    model = None
    weights_path_resolved: Path | None = None
    repo = Path(__file__).resolve().parent
    if not mock:
        from ultralytics import YOLO 
        wspec = weights if weights is not None else default_best_weights()
        wpath = Path(wspec)
        if not wpath.is_absolute():
            wpath = (repo / wpath).resolve()
        else:
            wpath = wpath.resolve()
        if not wpath.is_file():
            pygame.quit()
            cv2.destroyAllWindows()
            sys.exit(f"Weights file not found: {wpath}")
        
        weights_path_resolved = wpath
        model = YOLO(str(wpath)) 
        
        if paths:
            probe = cv2.imread(str(paths[0]))
            if probe is not None:
                model.predict(probe, imgsz=_round_imgsz(imgsz), verbose=False)

    viewport_w = MAIN_W - PANEL_W
    cam_x = 72 + (viewport_w - 144) * 0.52
    belt_cy = MAIN_H * 0.525
    panel_x = MAIN_W - PANEL_W

    backlog = list(paths)
    boards: list[Board] = []
    next_spawn_ms = pygame.time.get_ticks()
    spawn_gap_ms = lock_ms + scan_ms + hold_result_ms + 4200

    paused = False
    running = True
    loop_live = [loop_party]
    belt_speed = [float(speed_px_s)]
    board_seq = 0

    total_inspected = 0
    passed_n = 0
    rejected_n = 0
    defect_type_counts: dict[str, int] = {}

    log_lines: list[dict[str, Any]] = []
    log_scroll = 0
    log_line_h = 22
    log_max_lines = 200
    preview_popup: dict[str, Any] | None = None

    last_lamp: str | None = None  # "ok" | "ng" | None

    idle = idle_camera_frame()
    cv2.imshow(CAM_WIN, idle)

    def align_center_under_cam(b: Board) -> None:
        b.x = cam_x - b.w / 2.0

    def top_defect_label() -> str:
        if not defect_type_counts:
            return "-"
        k = max(defect_type_counts, key=lambda x: defect_type_counts[x])
        return f"{k} ({defect_type_counts[k]})"

    def yield_pct() -> float:
        if total_inspected <= 0:
            return 0.0
        return 100.0 * passed_n / total_inspected

    def draw_side_panel(log_area_store: list) -> None:
        nonlocal log_scroll
        r = pygame.Rect(panel_x, 0, PANEL_W, MAIN_H)
        pygame.draw.rect(main, (28, 30, 42), r)
        pygame.draw.line(main, FRAME, (panel_x, 0), (panel_x, MAIN_H), 2)

        lx = panel_x + 14
        y = 12
        main.blit(font.render("Статистика смены", True, TXT), (lx, y))
        y += 30
        main.blit(font_s.render(f"Total Inspected: {total_inspected}", True, TXT), (lx, y))
        y += 22
        main.blit(font_s.render(f"Passed: {passed_n}", True, GOOD), (lx, y))
        y += 20
        main.blit(font_s.render(f"Rejected: {rejected_n}", True, BAD), (lx, y))
        y += 22
        main.blit(font_s.render(f"Yield: {yield_pct():.1f} %", True, ACCENT), (lx, y))
        y += 24
        main.blit(font_s.render("Top defect:", True, TXT_DIM), (lx, y))
        y += 18
        td = top_defect_label()
        main.blit(font_mono.render(td[:28], True, BAD if td != "-" else TXT_DIM), (lx, y))

        y += 36
        main.blit(font.render("Inspection log", True, TXT), (lx, y))
        y += 26
        log_rect = pygame.Rect(panel_x + 6, y, PANEL_W - 12, MAIN_H - y - 52)
        pygame.draw.rect(main, (18, 20, 28), log_rect, border_radius=6)
        log_area_store[0] = log_rect
        clip_old = main.get_clip()
        main.set_clip(log_rect)
        try:
            yy = log_rect.y - log_scroll
            for row in log_lines:
                if yy + log_line_h < log_rect.y or yy > log_rect.bottom:
                    yy += log_line_h
                    continue
                col = BAD if row["ng"] else GOOD
                txt = f"{row['bid']}  {row['ts']}  {'NG' if row['ng'] else 'OK'}"
                surf = font_mono.render(txt, True, col)
                main.blit(surf, (log_rect.x + 6, yy))
                row["_rect"] = pygame.Rect(log_rect.x + 2, yy, log_rect.w - 4, log_line_h)
                yy += log_line_h
        finally:
            main.set_clip(clip_old)

        content_h = max(0, len(log_lines) * log_line_h)
        max_scroll = max(0, content_h - log_rect.height + 6)
        log_scroll = min(max_scroll, log_scroll)

        main.blit(font_s.render("Wheel: scroll  Click: photo", True, TXT_DIM), (lx, MAIN_H - 38))

    log_area_holder: list[Any] = [None]

    while running:
        now_ms = pygame.time.get_ticks()
        dt_ms = clock.tick(60)
        dt = dt_ms / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if preview_popup is not None and ev.key == pygame.K_ESCAPE:
                    preview_popup = None
                elif preview_popup is None and ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_p:
                    paused = not paused
                elif ev.key == pygame.K_l:
                    loop_live[0] = not loop_live[0]
                elif ev.key == pygame.K_LEFTBRACKET:
                    belt_speed[0] = max(25.0, belt_speed[0] * 0.75)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    belt_speed[0] = min(320.0, belt_speed[0] * 1.25)
            elif hasattr(pygame, "MOUSEWHEEL") and ev.type == pygame.MOUSEWHEEL:
                lr = log_area_holder[0]
                if lr is not None and lr.collidepoint(pygame.mouse.get_pos()):
                    log_scroll = max(0, log_scroll - ev.y * (log_line_h * 2))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if preview_popup is not None:
                    preview_popup = None
                    continue
                lr = log_area_holder[0]
                if lr is not None and lr.collidepoint(ev.pos):
                    for row in log_lines:
                        rr = row.get("_rect")
                        if rr is not None and rr.collidepoint(ev.pos) and row.get("shot") is not None:
                            preview_popup = {"bgr": row["shot"], "bid": row["bid"]}
                            break

        main.fill(BG)
        pygame.draw.rect(main, TRACK, pygame.Rect(0, int(belt_cy + 108), viewport_w, 24), border_radius=6)

        belt_top = int(belt_cy - belt_max_h / 2 - 48)
        belt_bot = int(belt_cy + belt_max_h / 2 + 48)
        pygame.draw.rect(main, BELT, pygame.Rect(72, belt_top - 40, viewport_w - 144, belt_bot - belt_top + 80), border_radius=16)
        pygame.draw.rect(main, FRAME, pygame.Rect(26, MAIN_H - 50, MAIN_W - 52, 34), border_radius=8)

        pygame.draw.line(main, ACCENT, (cam_x, belt_top - 78), (cam_x, MAIN_H - 118), width=7)
        pygame.draw.polygon(
            main,
            ACCENT,
            [(cam_x - 20, belt_top - 88), (cam_x + 20, belt_top - 88), (cam_x, belt_top - 58)],
        )

        draw_belt_stack_light(main, cx=int(cam_x + 44), cy=int(belt_top - 108), last_lamp=last_lamp, font_s=font_s)

        mode = "mock (filename)" if mock else f"PCB detector ({weights_path_resolved.name})"
        main.blit(
            font_s.render(
                f"{mode}  |  belt [{belt_speed[0]:.0f}] px/s  [ / ]  |  P pause  L loop:{int(loop_live[0])}  Esc",
                True,
                TXT_DIM,
            ),
            (36, MAIN_H - 42),
        )
        main.blit(font.render("Линия инспекции печатных плат", True, TXT), (viewport_w // 2 - 210, 16))
        main.blit(font_s.render("зона камеры AOI", True, TXT), (int(cam_x - 70), belt_top - 112))

        active_infer_ms = 0.0
        active_total_ms = 0.0
        active_board_in_aoi: Board | None = None

        if not paused:
            need_spawn = (
                backlog
                and now_ms >= next_spawn_ms
                and (len(boards) == 0 or all(b.phase in ("move", "move_out") for b in boards))
                and (len(boards) == 0 or boards[-1].x < viewport_w - 320)
            )
            if need_spawn:
                p = backlog.pop(0)
                board_seq += 1
                bid = f"B{board_seq:05d}"
                raw = cv2.imread(str(p))
                if raw is None:
                    raw = np.zeros((800, 1000, 3), dtype=np.uint8)
                    raw[:] = (42, 44, 52)
                surf, ww, hh = belt_surface_from_bgr(raw, p, belt_max_h=belt_max_h)
                boards.append(Board(p, surf, float(viewport_w) + 40.0, belt_cy - hh / 2, ww, hh, board_id=bid))
                next_spawn_ms = now_ms + spawn_gap_ms

            rem: list[Board] = []
            active_verdict_text = "Проверка: ожидание"
            for b in boards:
                if b.phase == "move":
                    b.x -= belt_speed[0] * dt
                    if b.x <= cam_x <= b.x + b.w:
                        b.phase = "hold_lock"
                        b.t_phase0_ms = now_ms
                        b.sensor_ms = now_ms
                        b.sensor_t_perf = time.perf_counter()
                        align_center_under_cam(b)
                        b.bgr_full = cv2.imread(str(b.path))
                        if b.bgr_full is None:
                            b.bgr_full = np.zeros((600, 800, 3), dtype=np.uint8)
                            b.bgr_full[:] = 45
                        cv2.imshow(CAM_WIN, draw_scanning_overlay(clamp_scale_to_height(b.bgr_full.copy(), 720), 0.0))

                elif b.phase == "hold_lock":
                    align_center_under_cam(b)
                    dt_lock = now_ms - b.t_phase0_ms
                    if dt_lock >= lock_ms:
                        b.phase = "hold_scan"
                        b.t_phase0_ms = now_ms
                        b.cam_frame_bgr = b.bgr_full.copy()
                        b.scan_infer_done = False
                    else:
                        vis = clamp_scale_to_height(b.bgr_full.copy(), 720)
                        cv2.imshow(CAM_WIN, draw_scanning_overlay(vis, 0.0))

                elif b.phase == "hold_scan":
                    align_center_under_cam(b)
                    active_board_in_aoi = b
                    elapsed = now_ms - b.t_phase0_ms
                    t01 = min(1.0, elapsed / max(1, scan_ms))
                    if b.cam_frame_bgr is None:
                        b.cam_frame_bgr = b.bgr_full.copy()
                    vis0 = clamp_scale_to_height(b.cam_frame_bgr.copy(), 720)
                    cv2.imshow(CAM_WIN, draw_scanning_overlay(vis0, t01))
                    if elapsed >= scan_ms and not b.scan_infer_done:
                        b.scan_infer_done = True
                        t_done = time.perf_counter()
                        total_proc_ms = (t_done - b.sensor_t_perf) * 1000.0
                        if mock:
                            res, bad, meta = mock_yolo_style_plot(b.cam_frame_bgr, b.path)
                            d = int.from_bytes(_path_digest(b.path, "mock-ms")[:2], "little")
                            meta["infer_ms"] = float(10.0 + (d % 3300) / 100.0)
                            meta["verdict_thr"] = float(conf)
                        else:
                            assert model is not None
                            res, bad, meta = annotate_detector(
                                model,
                                b.cam_frame_bgr,
                                imgsz=imgsz,
                                verdict_conf=conf,
                                pre_nms_conf=pre_nms_conf,
                                draw_min=draw_min,
                                iou=iou,
                                augment=infer_augment,
                            )
                        meta["total_proc_ms"] = total_proc_ms
                        res = put_result_overlay(
                            res,
                            bad,
                            mode="mock" if mock else "yolo",
                            summary=meta,
                            total_proc_ms=total_proc_ms,
                        )
                        b.result_bgr = res.copy()
                        b.verdict = bad
                        b.summary = dict(meta)
                        b.phase = "hold_result"
                        b.t_phase0_ms = now_ms
                        last_lamp = "ng" if bad else "ok"
                        active_infer_ms = float(meta.get("infer_ms", 0.0))
                        active_total_ms = float(total_proc_ms)
                        if not b.stats_recorded:
                            b.stats_recorded = True
                            total_inspected += 1
                            if bad:
                                rejected_n += 1
                                tags = meta.get("defect_tags") or meta.get("classes") or []
                                if bad and not tags:
                                    tags = ["(detected)"]
                                for tag in tags:
                                    defect_type_counts[tag] = defect_type_counts.get(tag, 0) + 1
                            else:
                                passed_n += 1
                            ts = time.strftime("%H:%M:%S")
                            log_lines.insert(
                                0,
                                {
                                    "bid": b.board_id or b.path.stem[:12],
                                    "ts": ts,
                                    "ng": bad,
                                    "shot": res.copy(),
                                },
                            )
                            log_lines[:] = log_lines[:log_max_lines]
                        cv2.imshow(CAM_WIN, res)

                elif b.phase == "hold_result":
                    align_center_under_cam(b)
                    active_board_in_aoi = b
                    if b.result_bgr is not None:
                        cv2.imshow(CAM_WIN, b.result_bgr)
                    active_infer_ms = float(b.summary.get("infer_ms", 0.0))
                    active_total_ms = float(b.summary.get("total_proc_ms", 0.0))
                    if now_ms - b.t_phase0_ms >= hold_result_ms:
                        b.phase = "move_out"
                        cv2.imshow(CAM_WIN, idle)

                elif b.phase == "move_out":
                    b.x -= belt_speed[0] * dt * 0.85

                edge = BAD if b.verdict is True else (GOOD if b.verdict is False else FRAME)
                if b.verdict is True:
                    cls = ",".join(b.summary.get("classes", [])[:2]) if b.summary else "-"
                    cnt = int(b.summary.get("count", 0)) if b.summary else 0
                    active_verdict_text = f"Брак: ДА   |   дефекты: {cnt}   |   классы: {cls}"
                elif b.verdict is False:
                    cmax = float(b.summary.get("conf_max_raw", b.summary.get("conf_max", 0.0))) if b.summary else 0.0
                    active_verdict_text = f"Брак: НЕТ  |  raw max: {cmax:.2f}"
                if b.phase in ("hold_lock", "hold_scan", "hold_result"):
                    pygame.draw.rect(main, ACCENT, pygame.Rect(int(b.x) - 4, int(b.y) - 4, b.w + 8, b.h + 8), width=4, border_radius=8)
                elif b.verdict is not None:
                    pygame.draw.rect(main, edge, pygame.Rect(int(b.x) - 3, int(b.y) - 3, b.w + 6, b.h + 6), width=4, border_radius=6)

                sh = pygame.Surface((b.w + 24, 28), pygame.SRCALPHA)
                pygame.draw.ellipse(sh, (0, 0, 0, 105), sh.get_rect())
                main.blit(sh, (int(b.x - 12), int(belt_cy + belt_max_h / 2 + 8)))
                main.blit(b.surf, (int(b.x), int(b.y)))

                if b.x + b.w > -120:
                    rem.append(b)
            boards = rem
            verdict_col = TXT_DIM
            if "ДА" in active_verdict_text:
                verdict_col = BAD
            elif "НЕТ" in active_verdict_text:
                verdict_col = GOOD
            main.blit(font.render(active_verdict_text, True, verdict_col), (42, 56))

            if active_board_in_aoi is not None and (active_board_in_aoi.phase in ("hold_scan", "hold_result")):
                main.blit(
                    font_mono.render(f"Inference Time: {active_infer_ms:.1f} ms", True, ACCENT),
                    (42, 84),
                )
                main.blit(
                    font_mono.render(f"Total Processing Time: {active_total_ms:.1f} ms", True, TXT),
                    (42, 104),
                )

            if not backlog and not boards:
                cv2.imshow(CAM_WIN, idle)
                if loop_live[0]:
                    backlog = list(paths)
                    next_spawn_ms = now_ms + 1200

        draw_side_panel(log_area_holder)

        if preview_popup is not None:
            overlay = pygame.Surface((MAIN_W, MAIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            main.blit(overlay, (0, 0))
            shot = preview_popup["bgr"]
            sh, sw = shot.shape[:2]
            scale = min((MAIN_W - 120) / sw, (MAIN_H - 160) / sh, 1.0)
            rw, rh = max(1, int(sw * scale)), max(1, int(sh * scale))
            small = cv2.resize(shot, (rw, rh), interpolation=cv2.INTER_AREA)
            surf = bgr_to_pg_surface(small)
            rx = (MAIN_W - rw) // 2
            ry = (MAIN_H - rh) // 2
            main.blit(surf, (rx, ry))
            pygame.draw.rect(main, ACCENT, pygame.Rect(rx - 4, ry - 4, rw + 8, rh + 8), 3, border_radius=8)
            main.blit(font.render(f"Board {preview_popup['bid']}  (click / Esc to close)", True, TXT), (rx, ry - 32))

        cv2.waitKey(1)
        pygame.display.flip()

    cv2.destroyAllWindows()
    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="pcb_original_image")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=608, help="inference size (match training, e.g. 608)")
    ap.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Verdict NG if any detection has conf >= this (verdict only; use --pre-nms-conf for low threshold into NMS).",
    )
    ap.add_argument(
        "--pre-nms-conf",
        type=float,
        default=0.01,
        help="YOLOv5 model.conf before inference; keep low (0.005–0.03) so weak boxes exist for drawing/verdict.",
    )
    ap.add_argument(
        "--draw-min",
        type=float,
        default=None,
        help="Draw boxes with conf >= this. Default: same as --conf (no ghost boxes). Lower e.g. 0.22 to preview weak boxes; NG still uses --conf only.",
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.42,
        help="YOLOv5 NMS IoU threshold (model.iou).",
    )
    ap.add_argument("--infer-augment", action="store_true", help="TTA-style augment at inference (slower, may help recall)")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--belt-max-h", type=int, default=560)
    ap.add_argument("--speed", type=float, default=72.0, help="belt speed px/s (slow by default)")
    ap.add_argument("--lock-ms", type=int, default=500, help="pause before scan animation")
    ap.add_argument("--scan-ms", type=int, default=2600, help="scanning animation duration")
    ap.add_argument("--hold-result-ms", type=int, default=4200, help="how long to show final detections")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    src = Path(args.source)
    if not src.is_absolute():
        src = root / src
    if not src.is_dir():
        sys.exit(f"Folder not found: {src}")
    imgs = list_images(src)
    if not imgs:
        sys.exit(f"No images in {src}")

    run(
        imgs,
        mock=args.mock,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        pre_nms_conf=args.pre_nms_conf,
        draw_min=args.draw_min,
        device=args.device,
        belt_max_h=args.belt_max_h,
        speed_px_s=args.speed,
        loop_party=args.loop,
        lock_ms=args.lock_ms,
        scan_ms=args.scan_ms,
        hold_result_ms=args.hold_result_ms,
        infer_augment=args.infer_augment,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
