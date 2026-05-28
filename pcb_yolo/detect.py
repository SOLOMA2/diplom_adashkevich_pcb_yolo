"""
Legacy-compatible detection entrypoint migrated to Ultralytics YOLO12.
Keeps optional anomaly alarm support.
"""

from __future__ import annotations

import argparse

import torch

from pcb.anomaly_tracker import AlarmTracker, configure_alarm_logging


def _ordered_names(names) -> list[str]:
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else x)]
    return list(names)


def _default_best_weights() -> str:
    """
    Prefer the best checkpoint from our 100-epoch training if it exists,
    otherwise fall back to the Ultralytics YOLO12 nano checkpoint.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent
    candidates = list(root.glob("runs/train/**/weights/best.pt"))
    if not candidates:
        return "yolo12n.pt"
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=[_default_best_weights()], help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="data/images", help="source path / video / webcam index")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save detection labels")
    parser.add_argument("--save-conf", action="store_true", help="save confidence in txt labels")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class ids")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--project", default="runs/detect", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    parser.add_argument("--alarm", action="store_true", help="enable PCB trend / anomaly alarm")
    parser.add_argument("--alarm-quiet", action="store_true", help="only log critical alarm events")
    opt = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Ultralytics is not installed. Run: pip install -r requirements.txt") from e

    model = YOLO(opt.weights[0])
    names = _ordered_names(model.names)

    alarm_tracker = None
    if opt.alarm:
        configure_alarm_logging(quiet=opt.alarm_quiet)
        alarm_tracker = AlarmTracker(class_names=names, device="cpu")

    common_kw = dict(
        source=opt.source,
        imgsz=opt.img_size,
        conf=opt.conf_thres,
        iou=opt.iou_thres,
        device=opt.device,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
        augment=opt.augment,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        show=opt.view_img,
        save=True,
        save_txt=opt.save_txt,
        save_conf=opt.save_conf,
        verbose=False,
    )

    if alarm_tracker is None:
        model.predict(stream=False, **common_kw)
        print("Detection complete.")
        return

    for result in model.predict(stream=True, **common_kw):
        boxes = result.boxes
        if boxes is None or boxes.data.numel() == 0:
            alarm_tracker.step(
                torch.zeros(0, 4, device=alarm_tracker.device, dtype=torch.float32),
                torch.zeros(0, dtype=torch.long, device=alarm_tracker.device),
            )
            continue
        xywh = boxes.xywh.to(device=alarm_tracker.device, dtype=torch.float32)
        cls = boxes.cls.to(device=alarm_tracker.device).long().reshape(-1)
        alarm_tracker.step(xywh, cls)

    print("Detection complete.")


if __name__ == "__main__":
    main()
