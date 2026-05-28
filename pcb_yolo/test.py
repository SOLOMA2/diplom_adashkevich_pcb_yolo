"""
Legacy-compatible validation/testing entrypoint migrated to Ultralytics YOLO12.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument("--weights", nargs="+", type=str, default=["yolo12n.pt"], help="model.pt path(s)")
    parser.add_argument("--data", type=str, default="data/PCB_yolo12.yaml", help="dataset yaml path")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--img-size", type=int, default=608, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--task", default="val", help="'val' or 'test'")
    parser.add_argument("--device", default="", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--verbose", action="store_true", help="print class-wise metrics")
    parser.add_argument("--save-json", action="store_true", help="save COCO-style json when supported")
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    opt = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Ultralytics is not installed. Run: pip install -r requirements.txt") from e

    split = "test" if opt.task == "test" else "val"
    model = YOLO(opt.weights[0])
    metrics = model.val(
        data=opt.data,
        imgsz=opt.img_size,
        batch=opt.batch_size,
        conf=opt.conf_thres,
        iou=opt.iou_thres,
        device=opt.device,
        split=split,
        single_cls=opt.single_cls,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        save_json=opt.save_json,
        verbose=opt.verbose,
    )

    map50 = getattr(getattr(metrics, "box", None), "map50", None)
    map5095 = getattr(getattr(metrics, "box", None), "map", None)
    if map50 is not None and map5095 is not None:
        print(f"Validation complete. mAP50={map50:.4f}, mAP50-95={map5095:.4f}")
    else:
        print("Validation complete.")


if __name__ == "__main__":
    main()
